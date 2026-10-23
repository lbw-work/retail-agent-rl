# GRPO LogProb 一致性排查与修复

**2026-10-17 最终更新：训练链路已验收。** 原4090上真实smoke完成一步；最终FP32适配器版本在服务器中断后完成两步验证，首组同策略已保存轨迹重放、第二组更新后新采样。9869个客服token最大差0.000134，零超差，原0.05门槛不变；checkpoint-1/2、实际数值更新及独立CUDA重载均通过，峰值分配约17.5GiB。完整94步和业务提升评测尚未执行。下文一至十二节保留各次排查当时的状态，最终验收以第十三节和 [简明验收记录](GRPO_LogProb修复验收_20261017.md) 为准。

2026-10-16。当前结论：已修复可复现的 PEFT/Accelerate autocast 执行路径缺陷，并完成本地数值、掩码、损失与小模型参数更新验证。**尚未取得失败运行的真实 CUDA 轨迹，不能断言 1.130928 的全部来源已确定或真实 Smoke 已修好；当前不具备启动正式 GRPO 的验收条件。**

后续真实 Smoke 反馈：2026-10-16 15:00 启动的新版运行，仍在更新前报最大差 **0.718732**，轨迹0、completion token **1417**，进度条耗时5分23秒。该反馈确认真实问题仍未解决；两次生成的轨迹不同，不能把1.130928到0.718732当作修复收益。先复用本次轨迹离线复算，不继续付费重采样。

本轮只改进诊断工具：复用 Transformers Trainer 的特殊 token 对齐，并新增 `--focused`，仅比较当前SDPA/BF16下cache、prefix、full_selected、full_training四条路径。特殊 token 同步提示本身不是根因证据。7项专项和5项协议测试通过；训练代码、奖励、阈值、预算和冻结协议均未改动。新记录单独保存在 `outputs/review/logprob_followup_20261016/`，前一轮证据不覆盖。

## 一、证据和限制

用户提供的运行在第 0/1 步报错：`_generate_and_score_completions()` 检查到最大差 1.130928，轨迹 0、completion token 1468。错误位于首步反向传播/优化器更新之前。进度条显示 5 分 07 秒，这包含此前工作，不能作为一次完整参数更新的耗时。

用户还提供了另一轮的最大差约 0.445，以及一个 token 的以下复算值。它们来自用户记录，本轮没有获取相应原始运行文件，不能将不同批次混为同一个对照实验。

| 用户提供的计算 | LogProb |
| --- | ---: |
| 原始采样 | -2.881448 |
| 原始训练重算 | -3.326620 |
| 离线 KV Cache 逐 token | -3.177217 |
| 离线前缀 | -3.327325 |
| 离线全轨迹 | -3.175993 |

当前本地 `outputs/grpo/` 没有这两轮 CUDA 轨迹；对现有 AutoDL 地址的非交互只读 SSH 尝试被认证拒绝，没有连接成功、启动 GPU 作业或调用 DeepSeek。已保存本轮代码改动前副本、冻结输入/历史日志摘要和实验产物到 `outputs/review/logprob_audit_20261016/`。

## 二、实际调用链和三种概率

实际训练路径：`train_grpo.run_cuda()` → `RetailGRPOTrainer` → `_generate()` → `generate_groups()` → `run_token_episode()` → `TorchSampler.generate()` → **Transformers/PEFT `model.generate()`**。`use_vllm=false`，构造器也拒绝 vLLM，没有启用 SGLang。此处与 MLX 开发集评测是不同后端。

底座只加载一次：Qwen3-4B、NF4、double quant、BF16 compute、SDPA；`prepare_qlora()` 后加载可训练的既有 SFT LoRA，没有 merge、重建另一套 rollout 权重或训练中换 adapter。训练前检查底座文件、SFT adapter 文件及配置哈希；LoRA 存储为 FP32 不代表 LoRA 的实际矩阵乘一定以 FP32 执行。

采样显式 `eval()`，结束恢复原模式；LoRA dropout=0，Qwen3 attention_dropout=0，TRL `disable_dropout=True`。因此当前源码没有发现 dropout 导致的随机策略差异。

| 概率 | 本项目来源和用途 |
| --- | --- |
| Behavior / rollout | `generate()` 返回的每步 `scores`；它已经经过 temperature=0.8。随后以 FP32 log-softmax 选取真实采样 token，保存在轨迹 `logprobs`。此处没有再次除温度。 |
| Old policy | 更新前、同权重的训练前向概率，detach 后供 PPO/GRPO 比率分母使用。修复后从一致性检查的 `recomputed` 获取；原始 behavior 仍独立保存在 trace，仍与它比较。 |
| Current policy | `_compute_loss()` 中带梯度重新前向，使用同样的 token、温度、FP32 概率定义；是比率分子。 |

固定的 TRL 0.26.2 原生实现，在 `num_iterations=1`、`steps_per_generation=4`、`gradient_accumulation_steps=4` 且非 vLLM 时，通常不额外保存 old，而用 `current.detach()`：整组在一次更新前处理，期间权重不变。本项目有额外的采样一致性检查，所以保留更新前重算值即可，不新增 old forward。

原适配将 behavior 直接写入 `old_per_token_logps`。这在将 behavior 作为旧策略的两策略定义下有其含义，并非任何场景下都错误；但它不同于这里固定 TRL 的同步训练策略约定，会把容差内的执行差异带入初始 PPO 比率。本轮在**独立 behavior 检查通过后**才保存训练侧 old，使更新前比率接近 1。该修改在报错之后才会执行，不能用来令 1.130928 的检查通过，也不是 rollout IS correction。

奖励、组内优势、beta、学习率、组大小、clip loss 数学式均保持原样。没有引入 verl 的 IS/RS 权重或过滤机制。

## 三、已确认的执行缺陷

### 1. PEFT.generate 绕过外层 forward 的 autocast

安装源码证据：

- Accelerate 1.12.0 `prepare_model()` 在外层模型 `forward` 上包装 BF16 autocast，并将浮点输出转 FP32。
- PEFT 0.18.0 `PeftModelForCausalLM.generate()` 转调 `self.base_model.generate()`，底座生成没有经过该外层 `forward`。
- 原 `_generate()` 只 unwrap/临时关闭 gradient checkpointing，没有显式进入 `self.accelerator.autocast()`。

因此即便底座、LoRA 权重完全相同，也存在：采样侧 LoRA FP32 运算，而训练 forward 使用 BF16 autocast 的执行差异。

本地同权重小 Qwen3＋**非零** LoRA＋CPU BF16 实验，用 hook 直接观察实际执行：

| 路径 | autocast | LoRA_A 输出 |
| --- | --- | --- |
| 旧 generate | false | FP32 |
| 被 Accelerate 包装的训练 forward | true | BF16 |
| 显式进入 autocast 的 generate | true | BF16 |

固定小模型实验中，采样与前向最大差从 **0.0023140907** 降为 **0**。该数值不是用户的 4B 失败轨迹，也没有 NF4，不能据此宣称 1.130928 全部由 autocast 造成。原始实验见 `reproduction_before.json`，生产入口调用 autocast 另有回归测试。

最小修复位于 `RetailGRPOTrainer._generate()`：在原有 unwrap/checkpointing 管理中增加 `self.accelerator.autocast()`，并记录实际 autocast、adapter 与策略步号。

### 2. 概率精度定义需要固定

Transformers 4.57.6 的采样先将最后一步 logits 转 FP32，再应用 temperature processor；采样结果的 scores 是处理后的分布。

TRL 0.26.2 原生重算先 `logits / temperature`，再调用 `selective_log_softmax()`。当原始 logits 是 BF16 时，这一写法会先低精度舍入；与采样的顺序不同。相同的一组人工 BF16 logits 上，已有 **0.04740524** 的概率差。

**重要限定：**真实 Trainer 通常还有 Accelerate 的输出转 FP32 包装。因此此处是已确认的低精度路径风险和离线复算歧义，尚不能断言用户失败运行实际走到了 BF16 归一化。新工具同时对照未包装 TRL 和模拟 FP32 输出包装的 TRL，避免错归因。

在训练器增加小范围的固定 Qwen3 文本函数 `text_policy_logprobs()`：`logits.float() / temperature` → FP32 selective log-softmax。校验与 Loss 共用这条概率路径。没有更改模型 NF4 权重或全模型转 FP32；没有重复缩放采样 scores。

## 四、Token、mask 与索引核对

- `TokenTape` 先保存初始 prompt IDs；生成 token 原样追加，不解码重编码已有历史。
- `sampled()` 同步追加 token/logprob/mask=1，记录 completion 中的 `[start,end)`。
- `feedback()` 只编码新增顾客/工具片段，mask=0、行为 logprob 占位为0；这些 token 仍在 attention 中。
- TRL 输入 prompt 左补齐、completion 右补齐；当前每组同一任务、同一开场，prompt IDs 完全相同，没有组内不同长度 prompt 的 position 偏移。
- completion 长度 C，则请求 `logits_to_keep=C+1`，去掉最后一个尚未发生的预测，再保留 C 行。目标绝对位置 `P+j` 对应 logits 位置 `P+j-1`；没有发现一位偏移。
- 损失 mask 为 `completion_mask * tool_mask`；训练 attention 为 `prompt_mask + completion_mask` 对应的拼接，**不能用 action mask 作为 attention mask**。
- 新测试直接检查外部 token 的逐 token logprob 梯度为0、后续动作仍保留完整注意力；另有位置相关 logits 的 shift 测试，避免用每位置相同的假 logits 掩盖偏移。
- Qwen3 配置 `rope_scaling=null`，当前上下文上限16384，小于底座位置上限40960；没有发现主动重置多轮 position 的逻辑。

以上是源码和测试的结论；真实失败轨迹的逐项一致性仍需读取原始 JSONL 后确认。

## 五、同权重最小 Forward 对照

`audit_grpo_logprobs.py` 对同一真实 token 检查以下路径：

1. KV Cache：从该次 agent 动作的完整前缀 prefill，teacher-force 后续已采样 token，明确 attention/position/cache_position。
2. Prefix：只输入目标 token 之前的完整前缀。
3. Full selected：输入完整真实轨迹，只输出目标预测行。
4. 原生 TRL：完整轨迹按原始批次最大 completion 长度右补齐，复用安装的原生方法。
5. Prepared TRL：在上一条基础上模拟 Accelerate 的 FP32 输出转换。
6. 修复后的训练概率路径。

每条分别比较 autocast 开/关和 SDPA/Eager，共24项。只做 teacher forcing，不生成新顾客、不运行工具、没有 optimizer。每项写入耗时，CUDA 时写入 allocated/reserved 峰值；中途失败保留已经完成的矩阵行。改变 attention 实现不会重新加载权重。

工具只处理有 `sampled_policy_step=0`、`global_step=0` 和 `lora_before_sha256` 的首步失败运行；核对完整底座、SFT adapter 文件和实际 LoRA 参数摘要。缺少证据、参数不一致或依赖漂移即拒绝，不能拿另外一套模型或 MLX 数据近似代替。

本地 BF16 小模型、含两轮动作和反馈、completion 右补齐的24项矩阵已完成。相同 autocast 下的 cache/prefix/full/prepared-TRL/修复路径，SDPA 极差约 **0.00050831**，Eager 极差 **0**。详细值和 CPU 时间见 `offline_matrix_cpu.json`。这些是短序列随机小模型结果，不用于预测4B的误差范围或 GPU 性能。

仍需排查的候选包括 NF4 的形状相关 kernel，以及 SDPA/BF16 对序列长度/矩阵形状的数值敏感性。[bitsandbytes 0.48.1 源码](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.1/bitsandbytes/autograd/_functions.py) 确有单向量无梯度时选择 GEMV、其余路径使用 MatMul4Bit 的分支；**有分支不等于它造成了本次1.130928**，要以同权重实测判断。

[verl 的 rollout correction 文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) 区分 rollout、old、current，并区分分离与 bypass 模式；本轮仅参考其概率来源区分，未增加 correction。确认执行缺陷消除后仍有持续训练/推理偏差，才讨论有对照的修正方案。

## 六、修改文件和验证

| 文件 | 本次修改 |
| --- | --- |
| `src/retail_agent/grpo_trainer.py` | 采样 autocast、统一 FP32 概率、独立 behavior 检查后的 old 保存、精度元数据 |
| `src/retail_agent/audit_grpo_logprobs.py` | 无 API 的首步轨迹复算工具；哈希与参数摘要核对 |
| `tests/test_grpo.py` | 增加 behavior 与 old 分离、初始重要性比率验证 |
| `tests/test_grpo_logprobs.py` | autocast、精度、shift、mask、矩阵、BF16 小模型真实 GRPO 更新6项专项 |
| `configs/grpo_baseline_protocol.json` | 记录新的执行策略，冻结训练器/采样器代码摘要，标记 CUDA 仍待验收；保留历史协议副本 |
| README、操作手册、本文 | 标明阻塞状态和安全复算顺序 |

没有修改 `train_sft.py`、既有 `grpo_rollout.py` token 构建、数据集、奖励算法、优势、概率阈值、API 预算或历史训练轨迹。进入本次任务时已有的94步配置保持不变。

本轮实测：

- GRPO相关回归52项：49通过、3项因 macOS/CUDA/MLX 适用条件跳过；见 `training_regression.log`。这个批次含当时的5项新专项。
- 随后补充并修正测试夹具后，最终6项概率专项全部通过；见 `logprob_tests_final.log`。首次夹具使用真实 tokenizer 的高位 pad ID 却配128词表，导致测试越界；已仅修正夹具词表，失败日志另存，不是通过修改生产 token 解决。
- 观测/冻结协议5项全部通过；只读 preflight 通过，94条训练任务，0次 optimizer、0次顾客API。
- BF16 小模型真实 GRPO 更新1步：LoRA发生变化，冻结底座逐参数不变，梯度有限；仅用接口奖励夹具验证框架数学，不代表电商业务改善。
- 语法检查通过。没有执行真实4B NF4参数更新，没有 GPU 显存/吞吐的前后数据。

以上测试存在交叉，不将它们相加冒充独立测试数量。

## 七、性能影响与剩余验收

生产路径没有增加模型 forward：原本已有一次组内校验重算和各微批 Loss 前向；直接复用校验结果作 old，不再额外重算。autocast 改变实际运算精度而非权重存储。

当 Accelerate 本就输出 FP32 时，新 `.float()` 不额外复制 logits，概率内存阶数与原路径相同；直接遇到 BF16 输出时，FP32 logits 确有额外内存需求。当前完整词表152k、长轨迹的 logits 本身成本很高，不能声称显存没有变化。CPU矩阵单点计时有冷启动/顺序影响，不作为加速结论；GPU allocated/reserved、前向与完整更新耗时均待实测。

正式训练的剩余阻塞项：

1. 获取这次失败运行的 `run.json`、`protocol.json` 和 `observations/*.jsonl`，确认运行版本、SFT起点及目标 token。
2. 现有 AutoDL 环境执行一次离线矩阵，先复算轨迹0/token1468；若还有偏差，用同份权重和 token 定位，不先重采样。
3. 解释 autocast 修正后剩余差异，原阈值0.05仍须通过；需要时另做有限的前向/梯度模式对照。
4. 离线证据可接受后，说明资源和费用，获得用户确认，再做一组真实顾客交互和1步 CUDA Smoke；检查新轨迹、梯度、LoRA前后摘要及保存重载。
5. Smoke 验收通过，再根据有效组、显存和整步耗时决定正式运行；测试集不用于挑选检查点。

## 八、在服务器执行离线复算

上传本次新包后，复用原 `.venv-train`。先填写真实失败目录和 observations 文件名；这里没有自动选择“最新”目录，以免误用别次试验。

```bash
cd /root/autodl-tmp/retail-agent-rl
source .venv-train/bin/activate
python scripts/prepare_grpo_upload.py --verify
python -m pip install --no-deps -e .
python -m retail_agent.train_grpo --preflight

retail_failed_run="outputs/grpo/替换成这次失败批次名"
retail_failed_trace="$retail_failed_run/observations/替换成实际JSONL文件名.jsonl"
retail_audit_run="outputs/review/logprob_cuda_$(date +%Y%m%d_%H%M%S)"
timeout --signal=INT --kill-after=30s 8m \
  python -u -m retail_agent.audit_grpo_logprobs \
    --trace-file "$retail_failed_trace" \
    --run-json "$retail_failed_run/run.json" \
    --row 0 --completion-token 1417 --focused \
    --output-dir "$retail_audit_run"
```

该工具会使用 GPU 前向，计入实例运行费用，8分钟是中断保护而不是耗时预测；不会访问 DeepSeek 或启动训练。`report.json` 是单点归因结果，不能替代全组 token 校验或真实 Smoke。此次未执行上述远端命令。

## 九、原服务器真实 CUDA 定位与 FP16 执行修复

晚间使用现有专用 SSH 密钥连接原实例，未改登录配置、依赖版本、模型文件、SFT adapter、任务或奖励。代码改动前备份与报告保存在 `outputs/review/cuda_logprob_repair_20261016/`；服务器另有 `source_before/`。

### 真实定位证据

- 原失败运行是 `smoke_logprob_v2_20261016_150014`，首步更新前、行0/token1417，最大差0.718732。
- 同权重 cache 复算精确重现原行为概率；完整轨迹的 selected/prepared-TRL/training 概率相互接近，差异主要在模型前向，不能靠最后加 `.float()` 解决。
- prefix 与 full 两条路径都实际调用 PyTorch Flash Attention，不是两条路径选用了不同的 SDPA 后端。
- 第一层 K/V 投影在目标位置的输入完全相同，BF16 输出仍随序列形状变化；后续36层将小差异放大。该投影包含基座与 LoRA，未声称已逐项分离到底层哪一个 CUDA kernel。
- 关闭 BF16 reduced-precision reduction 后，目标 cache/training 差从0.718732降至0.176763，仍未通过。
- 单独让 NF4 单向量快分支改走 MatMul4Bit，在原基线目标上仅改变约0.000026，不能解释主要差异。
- Eager 完整轨迹在 FP32 attention softmax 申请额外8.07 GiB时OOM，24项矩阵仅完成14项；没有用未完成矩阵宣称Eager有效。

### 最小执行修复

保留原 BF16→NF4 的量化输入起点；将冻结的非量化参数、NF4 compute 和采样/训练 autocast 统一为 FP16，关闭 FP16/BF16 reduced-precision reduction。既有 SFT LoRA 仍以 FP32 存储，概率仍在 FP32 中按温度归一化。

`train_grpo.py` 只为 GRPO 增加精度准备，未改 `train_sft.py`。`grpo_trainer.py` 切换 CUDA AMP 并记录实际精度。离线审计按 run.json 的精度重放，缺少新字段的历史运行继续采用 BF16。原始 behavior 检查、0.05阈值、token/mask、原生TRL优势和损失均保留。

### 已验证与待验证

| 验证 | 结果 |
| --- | --- |
| 原失败token，统一FP16的四路径极差 | 0.014733 |
| 原组4条轨迹，全部4604个客服token | 最大差0.048171；0个超0.05 |
| 真实4B，8227 token长轨迹反向，不执行optimizer | 288个梯度张量有限且有非零值；默认缩放65536通过；LoRA摘要不变；峰值分配12691806208 bytes |
| 本地相关回归 | 概率8、GRPO27、CLI5、观测5，均通过 |
| Linux概率专项及冻结输入预检 | 8项通过，94条任务与原SFT起点通过 |
| 重新采样的真实smoke、重载、正式模式短程 | 运行中，完成后以运行记录补充 |

4604-token验证重新计算了相同保存token在FP16下的cache与训练概率；没有拿新概率覆盖旧BF16行为记录，也没有用旧轨迹进行off-policy参数更新。它只支持启动新的on-policy smoke，不能证明业务能力提升或所有94个任务均通过。


## 十、FP16 新 smoke 失败后的继续定位（2026-10-16 晚间）

`smoke_fp16_fix_20261016_215353` 在第 0 步仍失败：4133 个客服 token 中有 2 个超过 0.05，最大差 0.0582098961，row 1 / completion 504。因此原组 FP16 离线通过不能当作新采样 smoke 验收，也不能通过换批次或提高容差掩盖问题。

进一步固定同一权重、同一 token 的真实 CUDA 检查：

| 执行条件 | row 0 / 1184 cache–training 绝对差 | row 1 / 504 cache–training 绝对差 |
| --- | ---: | ---: |
| FP32 + Efficient SDPA，保留 NF4 单向量 GEMV | 0.01870918 | 0.03355789 |
| FP32 + Efficient SDPA，统一 MatMul4Bit 矩阵路径 | 0.00004339 | 0.00001955 |

这说明剩余问题有两个层面：低精度的形状相关舍入会沿网络放大；量化层单 token 解码的 GEMV 与整段训练的反量化矩阵路径也会造成差异。不能仅凭旧 BF16 单点上的 GEMV 差值很小，就排除它对更高精度执行方案的影响。

当前候选共享执行入口 `grpo_execution_math()`：只在单进程 CUDA FP32 下生效，暂时将 NF4 乘法统一为原 bitsandbytes `MatMul4Bit.apply`；由 Transformers 显式展开 GQA，强制使用支持 FP32 的 Efficient SDPA。退出时恢复两个依赖函数及注意力后端上下文。NF4 权重、BF16 量化输入、SFT LoRA、奖励、学习率、温度与 0.05 阈值保留。没有引入新依赖。

真实 4B 反向检查：最长现有轨迹 8348 token，288 个 LoRA 梯度张量有限且包含非零梯度，峰值分配 15330706944 bytes（约 14.3 GiB），没有 optimizer/API 调用。两条完整轨迹共 1854 个客服 token 最大差 0.000084281，零超差。为优先验证真实训练，剩余两条离线重放被主动中断，原 report 保留 KeyboardInterrupt；这不是完整四条重放通过。新采样 smoke 正在验收，尚不声明训练完成。

正式训练另修正一处保护逻辑：允许合法同分组产生有限零梯度；smoke 仍要求非零梯度。缺失梯度、非有限梯度、非有限 loss 继续中止，整个运行没有参数变化时仍不标记完成。保留真实奖励，不制造组内差异。

本次证据位于 `outputs/review/cuda_logprob_repair_20261016/server_artifacts/` 和服务器 `outputs/review/logprob_fp32_matrix_full_replay_20261016/`。Linux 概率专项 10/10 通过，包含真实 CUDA NF4 分支与异常恢复检查；本地 CPU 检查不能替代 4B CUDA smoke。

## 十一、新采样 smoke 已通过，补齐原 SFT 精度保护

`outputs/grpo/smoke_fp32_matrix_20261016_221823` 已完成真实一步更新：4 条轨迹、3695 个客服 token 全部通过原 0.05 概率检查，最大差 0.0021147728，真实奖励 `[0, 1, 0, 1]`。4 个微批次、优化器及 checkpoint-1 保存成功，训练循环 541.2861 秒，峰值分配 14926017536 bytes，顾客 API 新增记账上界 0.0134588 元。

重载核查发现 TRL 0.26.2 在 QLoRA 初始化时无条件把可训练 LoRA 转成 BF16，即使 GRPOConfig 已关闭混合精度。该 smoke 的核心 NF4/注意力是 FP32，LoRA 实际为 BF16。PEFT 重载默认升回 FP32，因此不能直接比较两种存储精度的原始字节摘要。

已在独立 CUDA 进程按实际 BF16 恢复该 checkpoint：与保存后的摘要完全相符；将 SFT 起点先归一到同样 BF16 后，288 个 LoRA 张量仍存在真实数值变化；288 份优化器状态有限且包含有效动量；前向有限。这个检查证明保存/重载成功，没有声称已验证断点续训。

最终代码在 TRL 初始化前保留原 FP32 适配器张量引用，初始化后恢复，避免先降 BF16 再升 FP32造成不可逆舍入。起点摘要移动到 Trainer 初始化后计算，确保前后摘要基于相同实际存储精度。新增回归测试专门模拟 TRL 降精度，逐值检查恢复后的精确 SFT 原值。

最终版本 Linux 概率专项 11/11 通过；本地概率/CLI 共 16 项，15 通过、1 项 CUDA 检查跳过。正式入口正以完整 94 条 TRAIN 池验证两步，manifest 已确认 288 个 FP32 LoRA 参数、无 autocast、无 TF32，初始化摘要为原 SFT 的 `96d4f897…e0ea4`。这两步尚未完成，完整 94 步未执行。

第一组正式更新已通过：TRAIN `retail_301`，奖励 `[1, 1, 1, 0]`，3935 个客服 token 的最大差 0.0001337528，零超差；最长完整轨迹 9449 token。实际梯度范数 0.4879018664，第一步成功，第二组正在使用更新后的权重采样。

GRPO 首次前向的 loss 接近 0 不等于没有训练：初始重要性比率为 1，组内优势零均值，标量 loss 可以相互抵消。是否发生有效更新要结合非零有限梯度、相同存储精度下的权重数值变化和保存/重载证据判断。本次不把 loss 接近 0 解释为训练失败，也不将它作为业务能力提升证据。

## 十二、2026-10-17 服务器重新连接后的恢复验证

前一运行在第二组最后一条轨迹中途被服务器中断。重连后训练进程不存在，日志确认完成了一次优化器更新，但 run.json 尚未执行 finally 更新，仍是旧的 running/step0 快照。不能据此声称两步已经完成。该短程正式运行沿用每五步保存，故未留下 checkpoint。

只补一处保存逻辑：少于五步的短程验证每步保存，正式长程仍每五步保存。恢复验证在 tmux 独立会话中运行，SSH 断线不会直接结束会话；服务器重启仍可能结束进程，因此实际保存与本地备份仍必要。

恢复入口是 outputs/review 下的一次性脚本，不添加生产 CLI 选项。它只在原 SFT 参数摘要、实际 FP32 适配器精度、数学代码哈希、任务组和配置均一致时，重放此前已保存的第0步真实四条轨迹。原概率门槛、原生损失与优化器都正常执行，没有重编码历史 token，也没有制造奖励。下一组使用更新后权重新采样。这不是精确 RNG/断点续训，不承诺复现之前未保存的第二组。

重放首组 3935 个客服 token 再次通过，最大差 0.0001337528；第一步更新和 checkpoint-1 已重新完成，本地备份检查确认 288 个 FP32 LoRA 张量、288 份有限优化器状态及 step=1。服务器恢复运行目录为 `outputs/grpo/train_fp32_recovery_20261017_075933`，第二组仍在验收。

恢复运行包含一组已保存轨迹，不能用该运行汇总的 token/生成时间比率估计全新训练吞吐；原始 smoke 完整新采样的速度记录仍是独立证据。

## 十三、最终两步验证和重载通过

`train_fp32_recovery_20261017_075933` 已 completed，global_step=2，退出码0。step0在同一原SFT参数下重放3935个客服token，最大差0.0001337528；step1使用更新后参数新采样5934个客服token，最大差0.0001053810。两组零超差，未修改0.05门槛。

首组 `[1,1,1,0]` 有有效梯度，第二组 `[1,1,1,1]` 合法零优势组也正常完成，实际验证了正式模式不会误拦截同分组。两个完整checkpoint和最终adapter均保存并备份到本地。峰值分配18767398400 bytes，峰值保留23072866304 bytes，最长轨迹9478 token。

独立CUDA进程实际重载checkpoint-2并核对：288个FP32 LoRA张量发生真实数值变化，288份优化器状态有限且step=2；参数摘要匹配 `6b85ab0b10e4586101696c56435348f4c13950f04c007c59912dc04fff329740`，前向有限。验证本身零API/optimizer调用。没有声称验证了精确断点续训。

相关本地检查48项通过、1项CUDA在Mac跳过；Linux概率专项11项通过。完整94步未执行，16k长度显存上限未全面验证，训练池成功结果不证明业务能力提升。摘要和来源见 [最终验收记录](GRPO_LogProb修复验收_20261017.md)。
