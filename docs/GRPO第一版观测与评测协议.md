# GRPO 第一版观测与评测协议

2026-10-16更新：GRPO默认不设API费用上限，仍记录费用并检查余额；历史评测/诊断限额不变。协议：`grpo_baseline_v1`。这是 CUDA 第一版的观测和对照规则；尚无正式 GRPO 训练或新模型成绩。

## 固定输入与实验条件

机器可读协议为 [grpo_baseline_protocol.json](../configs/grpo_baseline_protocol.json)。当前冻结 train94/dev16/test22、v3 清单、三个 RL 文件、模型 revision、SFT 适配器、工具 schema、系统政策、tokenizer 模板、顾客规则来源及训练/评测配置的哈希。`experiment_protocol.check_protocol()` 核对这些文件；`train_grpo --preflight` 和当前 v3 开发评测在调用 API 前执行该检查。运行产物记录协议 SHA256，不能仅凭文件名认为条件相同。

- 训练从已验收 SFT LoRA 开始，Qwen3-4B、非思考模式。奖励 A、有限国家等价、组内标准化、group=4、beta=0 等设置保持原配置。
- 训练顾客按既定 `scenario_locked_v1`；正式训练入口实现时必须按此规则构造顾客 factory。目前 Trainer 接收外部 factory，协议文件不能代替实际 factory 的核验。
- 开发对照保留当前项目顾客 `upstream` 规则、DeepSeek Flash 非思考、temperature=0；所有参评模型保持相同。若改用训练侧顾客规则，须建立新协议并重新评测各组，不能沿用历史成绩。
- dev 按协议列出的 16 个任务 ID 和顺序全量执行，每模型每任务一次，客服贪心生成，24 个动作、每动作512 token、完整上下文16384 token。不得静默截断，业务失败与服务故障分开。
- 当前推理后端固定为已验收的 MLX 4-bit 副本，适配器按已有 PEFT 转换加载。若采用 CUDA/NF4 或 SGLang，必须在该后端重新评测全部对照模型。
- 只用 train 采样和更新；dev 用于开发决策；test 留到方案确定后一次最终比较，不用于选任务、奖励、超参数或检查点。

当前 dev16/test22 较小、类别不均，协议只能支持探索性比较。补强评测集后应发布新的数据/协议版本，并重跑相同条件下的基线。旧 v2 的 3/14 → 8/14 不作为本轮成绩。测试目标已用于质量复核，不能称为完全未查看内容的盲测。

## 训练前必须确定的运行预算

先做一次更新验证；单独的短程试跑为可选。根据实测耗时和实际剩余金额，在正式批次开始前记录计划步数、GPU时间预算和停止条件，API费用按实际记录。2026-10-16按用户要求将正式首轮配置上限改为94步，对应全池一轮任务采样、计划376条轨迹；奖励与优势计算保持原样，不代表已收敛。协议不授权租卡或调用 API。

第一版使用预先确定的正式结束步数，保存最终检查点；不根据 test 选择。1步/短程验证的结果仅作为启动验收，不宣称收敛。若后续采用 dev 选优，应预先声明候选检查点和选择指标，不能不断查看 test 后挑选结果。

## 耗时文件与统计口径

Trainer 自动在输出目录创建：

```text
observations/<run_id>.jsonl   # 逐阶段事件、组内观测、概率检查、Trainer日志
observations/<run_id>.json    # 当前运行汇总
```

每次新 Trainer 有独立 run_id；恢复运行应保留各段记录，不把两段墙钟时间误认为一次连续运行。阶段结束即追加事件，汇总用临时文件替换。捕获到的异常保留失败阶段和错误类型，继续传播原异常；强制杀进程/断电可能留下未结束阶段，不能把缺失耗时解释为0。

| 阶段名 | 范围 |
| --- | --- |
| `offline_preflight` | 只读数据、依赖、底座与适配器检查；单独存在 outputs/preflight，不算 GPU 训练时间 |
| `sft_adapter_load` | 已有 LoRA 加载函数；可传入共用 StageTimer |
| `trainer_setup` | TRL Trainer 构造，底座实际加载不在该范围内 |
| `training_run` | 实际 `Trainer.train()` 总范围，包括准备、rollout、更新和内部开销 |
| `update_step` | 一次更新的整个范围，含组采样、微批与优化器；记录 step 索引 |
| `rollout_batch` | 任务组采样，总体包含认证检查、共享开场、各条完整交互和奖励 |
| `task_identity_validation` | 任务身份信息复核 |
| `customer_opening` / `customer_reply` | 一次顾客方法调用的墙钟时间，含预算/余额检查及内部重试；共享开场只计一次 |
| `episode_setup` / `episode` | 独立环境、token 初始提示及一条训练轨迹 |
| `assistant_generation` | 客服生成，包括当前采样器执行与采样概率提取 |
| `tool_execution` | 运行轨迹中的工具执行；奖励重放不重复归入此阶段 |
| `reward_evaluation` | 业务评分、门控、奖励所需的环境重放等整体开销 |
| `batch_prepare_and_advantage` | TRL 批处理、采样、奖励汇集、优势及概率检查的总体范围 |
| `policy_logprob_forward` | 完整上下文概率计算，含验证或 loss 中调用的概率前向 |
| `forward_backward_microbatch` | 一个微批的整体训练步骤；可能嵌套首次生成/批处理，不能全当成反向传播时间 |
| `loss_forward` | 策略损失前向，包含其调用的概率前向 |
| `optimizer_step` | 框架 pre/post optimizer 回调之间的更新窗口，不含前面的梯度裁剪/检查或后面的 scheduler |
| `save_model` / `save_checkpoint` | 模型导出 / 完整检查点；后者可包含前者 |
| `trainer_evaluation` | 实际调用 Trainer.evaluate 的总体耗时；不是已实现业务评测的证明 |

每个阶段同时记录 `inclusive_seconds` 与 `self_seconds`：前者包含子阶段，后者扣除直接子阶段。**不能把所有 inclusive 相加。**用总墙钟衡量端到端成本；用 self 时间解释无重复的阶段分布；`unaccounted_or_active_seconds` 包含尚在进行的范围、范围外工作与观测文件写入等开销。

GPU 阶段在正常结束时同步 CUDA 边界，再读取单调时钟；CPU/API 阶段不做 CUDA 同步。它是加入同步后的墙钟剖析，包含仪表开销，不是纯 CUDA kernel 时间。失败阶段未必完成 GPU 同步，事件会明确标为未同步。未来异步或分布式实现不能直接沿用当前串行计时解释。

没有显式增加预热任务。第一个更新/第一次生成含实际冷启动与延迟初始化，按 step 和单次事件报告首步及后续步骤，不伪造单独的预热耗时。未执行的阶段不生成记录，不能在报告里填0。

正式 CUDA CLI 已接入，共用 StageTimer 从只读预检开始记录，`base_model_load`、`sft_adapter_load`、`cuda_run` 包住真实加载与训练，耗时汇总写入 run.json。preflight 之前的进程导入、安装、上传和服务器空闲计费不在该范围。GPU 实测尚未完成；本机只验证编排与CPU小模型更新，不能宣称已取得完整GPU耗时。服务器命令见 [4090 GRPO操作手册](4090_GRPO训练与传输操作手册.md)。

## 训练观测

- 每个完整任务组保存原始奖励、是否有组内差异、终止原因、各轨迹客服/反馈/完整上下文 token、客服动作数、工具调用与错误数；不修改奖励或过滤同分组。
- 汇总完整组数量、全0/全1组、有效组比例及各业务覆盖。有效组定义为奖励存在差异，不等于已经产生有益更新。中断组的已完成阶段保留，但不补造完整组或故障零分。
- `wall_seconds_per_effective_group` 包含本 timer 生命周期内的开销，不能叫纯采样耗时。token/s 的分子仅来自完整组，分母为记录到的全部客服生成时间，含被中断组的生成；它是保守的完整组产出指标，不是引擎纯解码吞吐。
- 保存采样/重算概率最大差、优势范围与均值。直接保存 TRL 的 loss、grad_norm、learning_rate、entropy、clip_ratio 等实际日志；不编造没有计算的参考策略 KL。beta=0 保持不变。
- CUDA 下记录进程累计显存峰值 allocated/reserved，不重置已有峰值；CPU 测试没有 GPU 指标。记录 GPU 型号、Torch/CUDA/框架版本、实际步数上限、学习率、任务ID和运行代码哈希。
- 成功返回的顾客 API 元数据关联到该运行；汇总 completed usage 费用上界。该值不包含所有失败预留和重试费用，完整账单仍以原 API ledger 为准。网络、HTTP和返回解析失败均保留请求耗时；不记录 Key、原始异常正文或请求密钥头。

## 评测产物与指标

当前业务评测入口仍仅支持 `original` 与 `sft`；第三组 `sft_grpo` 和最终 test 的实际入口尚未启用。本次没有通过修改两组汇总函数来假装第三组已实现。最终三组比较必须使用同一协议，记录各检查点哈希。

开发评测新产物包括 `manifest.json`、协议副本 `protocol.json`、`traces.jsonl`、`summary.json` 和 `observations/`。主 manifest 记录协议哈希、实际配置、任务顺序、后端和代码；每条轨迹另有本任务阶段计时，整体计时记录模型/适配器加载、来源核查与评测 episode。整体 episode 和轨迹子阶段属于不同层级，不能重复相加。评测阶段耗时只在未来实际调用评测时产生，本次没有付费调用。

主指标采用上游严格完整任务成功率；训练国家等价分不能替代它。所有计划任务均已运行且没有服务/判分故障时，才填写 `primary_success_rate`。否则主值为 null、`comparison_complete=false`，保留 attempted/eligible/system_errors/not_attempted 和部分结果。旧 `success_rate_planned` 字段仅为计划分母的进度诊断，不作为未完成实验的最终成绩。

同任务比较只纳入两组均有可评分轨迹的任务，报告改善/退步数量及实际分母。重复记录、非计划任务、当前未实现模型组会被拒绝。工具执行无错误率不是业务准确率。按业务的 `whole_task_successes` 统计“涉及该业务的整条任务成功”，不能解释为该业务的原子子目标准确率；一条组合任务会进入多类统计。

服务/余额/环境内部/评分故障中止批次，保留证据，不把它当训练零分。动作、长度或输出协议限制按现有行为判失败并分别计数。若需重跑故障实验，使用新输出目录、相同清单和条件并关联原故障记录，不覆盖历史或只补成功样本。完整逐笔确认及政策违规率尚未可靠实现，不能宣称数据库成功等于政策完全合规。

第一版至少交付：SFT与GRPO在同一dev上的完整对照、逐任务改善/退步、失败原因、有效组比例、端到端与阶段耗时、采样/API成本、显存峰值和可重载检查点。没有足够证据时如实报告未改善，不能依据训练奖励上升直接宣称业务泛化提升。

## 本机验证方式

```bash
PYTHONPATH=src .venv-train/bin/python -m retail_agent.train_grpo --preflight
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_observability.py' -v
PYTHONPATH=src .venv-train/bin/python -m unittest discover -s tests -p 'test_grpo.py' -v
```

测试包含嵌套时间对账、异常事件持久化、同步钩子、协议输入变更拒绝、真实小模型反向/优化器/保存路径、评测完整性和重复记录检查。小模型使用接口 fixture，无业务顾客 API，不证明4B能力或CUDA性能。完整验证记录见 `outputs/review/observability_20261015/`。

本轮结果：审计环境123项通过、19项跳过；训练环境136项通过、6项跳过；互补覆盖全部142项。只读预检确认94条训练任务与冻结协议一致，并实际生成耗时文件。SFT与原始数据哈希保持不变。详见 [验收记录](../outputs/review/observability_20261015/completion.json)。
