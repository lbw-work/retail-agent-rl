# 阶段 05B：多轮 GRPO 接口与异常验收

日期：2026-10-12。结论：离线接口准备通过；下一步为 05C 真实 SFT 组内诊断。没有新 API 请求、4B 模型加载、GPU 租赁或优化器更新，新增费用 0 元。不能据此承诺 GRPO 提升。

## 1. 本阶段交付

| 文件 | 职责 |
| --- | --- |
| `src/retail_agent/grpo_rollout.py` | 追加式实际 token 轨迹、Torch 随机采样、真实 Retail 多轮控制与任务分组 |
| `src/retail_agent/grpo_trainer.py` | 固定 TRL 适配，复用原生优势/裁剪/损失；核验旧概率、损失与更新前梯度 |
| `src/retail_agent/train_grpo.py` | SFT LoRA 继续训练加载函数、只读 `--preflight`；正式 GPU CLI 尚未启用 |
| `configs/grpo.toml` | 已核对的实际 SFT 路径、30 条训练池、奖励 A、组大小与随机采样参数 |
| `requirements/grpo.txt` | 原训练栈加项目已有 Retail/顾客运行依赖，不安装完整上游服务栈 |
| `requirements/grpo-macos.lock.txt` | 本机实际依赖快照，不能当作 CUDA wheel 安装清单 |
| `tests/test_grpo.py` | 24 项离线接口、真实小模型概率及反例测试 |

所有旧 SFT、开发集评测与官方业务评分路径保持原有口径。没有改变原始六个划分、四份数据库、首批 30 条任务、SFT 成果、旧成绩或费用账本。

## 2. 固定框架的真实限制

核对本机 TRL 0.26.2 源码和[官方对应版本源码](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_trainer.py)：非 vLLM 分支不会使用 `rollout_func`；`env_mask` 不是这个版本会直接使用的外部反馈掩码；原生 `tool_mask` 的应用受 `self.tools` 控制。内置工具解析还要求 Transformers 5，本项目固定为 4.57.6。

因此不升级训练栈、不启用额外 vLLM 副本，也不重写 GRPO 优化器。`RetailGRPOTrainer` 覆盖生成入口，使用现成 Retail 环境全权执行工具；初始化后保留真实 Retail schema，使原生 `tool_mask` 分支生效。禁止回退到内置单轮生成。优势、标准差归一化、裁剪损失、微批拆分和优化器仍由 TRL 处理。此适配依赖固定内部接口，版本漂移会拒绝启动，升级必须重新验收。

另一个重要限制：该版本的 Liger GRPO 路径只接收 `completion_mask`，未按本适配应用外部反馈 `tool_mask`。本阶段明确关闭 GRPO Liger，不能因为 SFT 用过 Liger 就直接启用。CUDA 长轨迹显存需 05D 实测。

## 3. 数据流与同组初始条件

```text
复核 TRAIN 池 → TRL 重复任务 ID 为一组 4 条
             → 每组生成一次顾客开场，固定同组初始消息
             → 每条重新建立独立 Retail 数据库与顾客实例
             → 实际客服采样 token → 工具/顾客反馈 → 下一客服回合
             → 正常结束或明确终止原因
             → 独立重放得到业务分、门控、训练奖励 A
             → 完整组先计算优势 → 一条轨迹一个微批 → 累积一次更新
```

数据集的 `prompt` 列是任务 ID，仅用于路由；不会直接作为模型输入。客服输入来自公开政策、工具 schema 和实际顾客/工具消息。隐藏金标准保留在评分端；顾客工厂只接收 `user_scenario`。

每组四条初始 prompt token 必须完全一致。后续顾客回复与工具状态可随客服行为不同而不同，这正是多轮环境。每条轨迹有独立数据库，不会沿用上一条已修改状态。服务虽然温度为零，也不保证跨请求绝对确定；共享开场只控制初始条件，不消除后续模拟器噪声。

## 4. 保留实际 token，不重写历史

Qwen3 原模板在重新渲染历史时，可能删除或改变先前的空思考段，并规范化工具 JSON。解码再编码也不能证明得到原采样 token；这种方式会把采样时和更新时的序列混为一谈。

`TokenTape` 首次用固定 tokenizer、工具和非思考模板创建提示。之后只追加：

1. 生成器实际返回的客服 token 和对应 log-prob；保留它的空格、JSON 排版与结束符。
2. 新增工具或顾客消息的模板片段；只编码这个片段，不重渲染过去的客服历史。
3. 需要继续时追加下一回合非思考提示头。

这是一条固定 tokenizer 下的追加式序列，保留历史非思考占位段；不声称它与每轮重新套完整原模板的字符串逐字一致。客服文字/工具协议仍与现有环境相同，后续业务诊断须观察历史长度与表现。

掩码对应：初始提示不进入 completion 损失；实际客服 token（包括采样结束符）为 1；反馈消息、插入的模板头和 padding 为 0。`completion_mask` 只用于有效上下文与 padding，原生损失使用 `completion_mask × tool_mask`。**外部反馈保留注意力**，不能把损失掩码当作注意力掩码。

测试直接对原生损失反向传播，外部位置的概率损失梯度为零，客服位置非零。客服损失仍可通过它关注的上下文影响网络参数，这是预期行为；并非要求反馈对后续预测没有影响。

## 5. 随机采样和旧策略概率

初始配方：组大小 4、temperature=0.8、top_p=1、top_k=0、min_p=None、repetition_penalty=1、num_iterations=1、beta=0、奖励 A、learning_rate=5e-6。这些是起步设置，没有调参最优性的证据。

关闭词表裁剪和重复惩罚，是因为固定 TRL 的更新概率采用 `log_softmax(logits / temperature)`；不能使用另一种裁剪后分布的采样概率，却按未裁剪分布更新。`GenerationConfig` 显式建立，避免继承 Qwen 默认 top-k/top-p。

Torch 采样返回实际生成序列与每步处理后分布的 log-prob，检查提示前缀未被生成器改写。随后在更新前，以完整真实上下文用 TRL 重算概率，只比较实际客服位置；差异超限中止，不能伪造或静默替换概率。旧概率保存为脱离梯度的张量，微批拆分时与对应 token、掩码、优势一起处理。

CPU 随机小 Qwen3 的实际采样通过一致性测试，完整采样→评分→TRL 路径的最大概率差测试要求小于 1e-4。配置的 0.05 是未来 BF16 路径的中止阈值，尚未由 CUDA 测量确定；若触发应查 dtype、位置、模板和生成设置，不能为跑通随意放宽。

## 6. 已排查的异常与行为

| 异常/风险 | 当前行为及验证 |
| --- | --- |
| 空生成、token/概率数量不符、无效 ID、NaN/Inf/正 log-prob | 中止，不补造数据 |
| EOS 后仍有 token | 中止，不偷偷裁掉 |
| 初始提示或外部反馈超长 | 不截断；无客服 token 时拒绝整组，有采样时明确记录 context_limit |
| 没有自然采样结束符 | generation_limit，保留原 token，不补 EOS；业务失败分为 0 |
| 思考输出、混合文字与工具、非法角色分隔符 | invalid_model_output，业务失败分为 0 |
| 工具业务错误 | 真实错误反馈进入上下文，允许纠正；连续三次终止，最后返回也保留 |
| 顾客开场为空或已 STOP | 整组中止，没有伪造客服完成样本 |
| 顾客/API、预算、工具内部、判分故障 | 异常传播并中止组，不把故障包装成零分 |
| 同组混入不同任务、不完整组 | 拒绝分组，不比较无关任务 |
| 轨迹之间数据库串扰 | 每条独立初始化；四次实际脚本业务均成功的反例验收 |
| 重复任务共享样本字典 | 逐条复制后再交给 TRL 写入奖励；避免最后奖励覆盖整组 |
| 采样与更新概率不符 | 更新前拒绝 |
| 全零、全一或相同部分奖励 | 原生优势为零，不加入随机分制造信号；真实诊断需报告同分组 |
| 非有限损失/梯度、整批无有效梯度 | 损失检查或优化器更新前回调中止 |
| 误开 vLLM/Liger、错误批次、重新建 LoRA | 固定适配拒绝；正式启动前只读核对配置 |

共享字典问题由混合奖励反例实际复现：四个引用若写入同一字典，末条奖励会覆盖前面的字段，优势误变为全零。边界逐条复制后，原生 TRL 对 `[0,0,1,1]` 正确计算正负优势，直接梯度检查也通过。

不把基础设施异常过滤成单条后继续计算三条优势；整组中止以维持组大小与比较条件。真实付费诊断的中断轨迹记录、检查点与预算退出仍需 05C/05D 的正式运行流程完成。

## 7. SFT 起点与实际验收

路径为 `outputs/downloads/sft_first_20261011_200837/outputs/sft/sft_first_20261011_200837/adapter`。检查点的模型/revision、run.json、适配器权重、准备数据清单与已公布开发评测记录一致；固定原底座逐文件 SHA256、tokenizer 文件也重新核验。没有把外层下载目录或空 `outputs/sft` 当成适配器。

`load_sft_adapter` 使用 `PeftModel.from_pretrained(..., is_trainable=True)`，不传新 `peft_config`，不 merge/unload。小模型保存/重载验证证明：加载已有 LoRA 数值、只让 LoRA 可训练、底座权重不因加载而变化；原 SFT 文件只读，后续训练另存新输出。

| 检查 | 实际结果 |
| --- | --- |
| GRPO 专项 | 24/24 通过 |
| 完整训练环境 | 共 97 项，94 通过；3 项 MLX 专项跳过 |
| 完整审计/MLX 环境 | 共 97 项，79 通过；8 项 SFT + 10 项 GRPO 框架专项跳过 |
| 两环境联合覆盖 | 97 项均在至少一个环境实际通过 |
| 依赖检查 | pip check 无冲突；固定主依赖与运行依赖核对通过 |
| CLI preflight / 骨架 | 通过，日志与哈希见 verification.json |
| 原始数据、快照、首批池、SFT、旧成绩、API 账本 | 只读核验，未改变 |
| 4B 真实组内采样 / CUDA 更新 / 训练提升 | 未执行、未验证 |

专项中有脚本控制轨迹与受控奖励 fixture，也有随机小 Qwen3 的真实 token 采样；fixture 概率不冒充模型概率。小模型仅验收接口，没有优化器更新，不能称为已完成 GRPO 训练。

日志与证据在 `outputs/logs/phase05b_*.log`、`outputs/review/phase05b/verification.json`。项目根目录复查：

```bash
.venv-train/bin/python -m pip install -r requirements/grpo.txt
.venv-train/bin/python -m pip check
PYTHONPATH=src .venv-train/bin/python -m retail_agent.train_grpo --preflight
PYTHONPATH=src .venv-train/bin/python -m unittest discover -s tests -p 'test_grpo.py' -v
PYTHONPATH=src .venv-train/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## 8. 剩余验证与下一阶段

05C 在新 API 阶段授权与费用核对后，先从复核 TRAIN 中选 8～12 条，组内采样 4 条，不更新参数。记录 A/B、业务成功、门控原因、同分/混合组、长度、费用和耗时。真实模拟器的语义、语言确认和政策解释仍要审查；本次门控不是完整政策合规证明。

Mac 的 MLX 组内采样只可作为初步诊断；本阶段的 Torch 适配不意味着 16GB Mac 可加载 4B PyTorch 底座。MLX 诊断仍须使用实际随机采样与 token 记录，其轨迹不能用于 CUDA on-policy 更新。

05D 必须用训练后端重新采样，实测 NF4/BF16 下概率差、显存、耗时、梯度与一次真实更新，并保存和重载新适配器。16,384 token 是完整轨迹上限，关闭 Liger 后原生全词表损失可能消耗较多显存，不能以 SFT 峰值显存预测 GRPO 峰值。若受限，先基于真实长度设完整任务上限或验证保持掩码语义的显存方案，不静默截断历史/反馈。

beta=0 尚未约束相对 SFT 的 KL；先采用很小更新次数和学习率，并用固定 dev 检查退步。不能保证预算内出现学习信号，也不能保证 RL 超过 SFT；没有有效信号时，应停下查原因，而不是增加奖励噪声或盲目扩大训练。
