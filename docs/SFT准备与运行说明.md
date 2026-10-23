# SFT 准备与运行说明

日期：2026-10-11。当前完成 SFT 前准备，正式 Qwen3-4B 训练尚未开展。

## 1. 实际准备数据

| 项目 | train | dev |
| --- | ---: | ---: |
| 当前目标回合 | 5,682 | 1,215 |
| 完整序列 token | 37,683,631 | 8,076,585 |
| 参与损失的目标 token | 679,469 | 145,168 |
| 最大完整长度 | 13,913 | 12,169 |

输入是 `data/versions/retail_full_v2/processed/sft_train.jsonl`、`sft_dev.jsonl`。
编码数据位于 `data/prepared/retail_full_v2_sft/{train,dev}`，约 349 MiB。
最终测试侧 1,214 条 SFT 示范没有进入预处理，不用于本次调试或选择检查点。

16,384 是允许的完整长度上限，batch 只补齐到当批最长样本。批量大小 1，梯度累积 8：累积八个完整回合的梯度，不把八段对话拼成一个上下文。
约 68 万监督 token 仍需要计算约 3,768 万输入 token，不能只按监督 token 数估算耗时。

## 2. 标签逻辑

每条数据包含 `prompt`（完整政策和历史）、`completion`（恰好一个当前客服目标）、`tools`（公开工具 schema）。
`src/retail_agent/sft_data.py` 的 `encode_target` 依次完成：

1. 使用固定 Qwen 模板编码 prompt，添加生成前缀，关闭 thinking。
2. 编码 prompt+当前客服目标。
3. 检查两者在 prompt 区间逐 token 相同；失败即报错，不按字符串长度猜边界。
4. 超过完整长度上限即拒绝，不截断历史或答案。
5. 构造 `labels = [-100] * prompt_tokens + full_tokens[prompt_tokens:]`。
6. 验证来源计数、监督区间和 `<|im_end|>` 结束符。

例如 token 为 `[11,12,13,21,22,23]`，前三个来自上下文，则 labels 为 `[-100,-100,-100,21,22,23]`。
因果模型内部把 logits 与下一位置 label 对齐：历史最后一个位置预测目标第一个 token。
掩码只屏蔽损失，完整历史仍参与注意力计算。

工具调用属于客服生成：工具名、参数、`<tool_call>` 包装和目标结束标记受到监督；工具执行结果只作为上下文。非思考生成前缀属于 prompt，不监督独立 thinking 内容。

`TargetTurnCollator` 保留 labels，只补右侧 padding；padding 的 attention mask 为 0、labels 为 `-100`。
TRL 接收预编码数据，使用 `skip_prepare_dataset=True`、`max_length=None`、`packing=False`，避免再次套模板、改边界或截断。

## 3. 阅读顺序

| 顺序 | 文件 | 核心内容 |
| --- | --- | --- |
| 1 | `configs/sft.toml` | 固定模型、输入、长度、LoRA、学习率 |
| 2 | `src/retail_agent/sft_data.py` | 精确前缀、labels、padding |
| 3 | `scripts/prepare_sft.py` | train/dev 约束、流式编码、保存、全量验收 |
| 4 | `data/prepared/retail_full_v2_sft/manifest.json` | 输入/输出 SHA256、计数、依赖 |
| 5 | `src/retail_agent/train_sft.py` | 准备数据校验、完整长度桶、TRL/LoRA、保存 |
| 6 | `scripts/smoke_sft.py` | 实际参数更新、底座冻结、保存与重载 |
| 7 | `tests/test_sft.py` | 真实工具目标、长样本、错误标签和 padding |

## 4. 本机验收范围

- 全量 train 5,682、dev 1,215 条使用真实 tokenizer 编码，计数与独立数据审计一致。
- Arrow 保存后逐条重新检查掩码、长度、唯一 ID、监督数量；编码期间输入未改变。
- 八项专项测试覆盖真实文本/工具目标、超过 12K 历史、拒绝超限、padding 和损坏标签。
- 随机小 Qwen3 约 972 万参数，用同一个 `make_trainer` 在 CPU 做两步实际 LoRA 更新。
- 手工移位交叉熵与模型损失相同，TRL 数据加载器保留监督 token 数。
- LoRA 实际变化、底座保持冻结；适配器保存后重载 logits 一致，训练/开发损失为有限值。
- 训练依赖导入与 `pip check` 通过。

证据在 `outputs/reports/sft_preparation.json`、`sft_smoke.json` 和 `outputs/logs/`。
小模型只用于验收，临时权重测试结束即清除，没有替换客服底座或被当成 4B 训练成果。
Qwen3-4B CUDA 训练、16K 显存/吞吐和业务能力提升尚未验证。

## 5. 环境重建

Python 3.12；`.venv` 的审计/MLX 依赖保持原样，`.venv-train` 用于训练。
主版本：Torch 2.8.0、Transformers 4.57.6、TRL 0.26.2、PEFT 0.18.0、Accelerate 1.12.0、Datasets 4.4.1。
完整本机锁为 `requirements/train-macos.lock.txt`；CUDA 使用 `requirements/train.txt`，解析并锁定服务器平台依赖。
接口参考 [TRL v0.26.2 SFT 文档](https://huggingface.co/docs/trl/v0.26.2/en/sft_trainer)，实际兼容性以运行证据为准。

新环境的准备命令：

```bash
python -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements/train.txt
.venv-train/bin/python -m pip install --no-deps -e .
.venv-train/bin/python -m pip check
PYTHONPATH=src .venv-train/bin/python -m unittest discover -s tests -p 'test_sft.py' -v
.venv-train/bin/python scripts/smoke_sft.py
```

本机数据已准备，勿重复生成。需要重建时核查输入变化，移走旧准备目录，再运行 `scripts/prepare_sft.py`。
输入/输出哈希或主依赖改变时，训练入口拒绝使用旧准备产物；同一 PyTorch 发布版允许 CUDA wheel 的 +cu128 本地后缀。

## 6. 下一阶段：CUDA 资源试跑

正式入口只允许 CUDA，显式提供样本数、优化步数和新输出目录。以下是下一阶段示例，本次没有执行，也不表示已经授权租卡或付费训练。

```bash
PYTHONPATH=src .venv-train/bin/python -m retail_agent.train_sft \
  --max-samples 8 --max-steps 1 --max-example-tokens 8192 \
  --output-dir outputs/sft/cuda_probe_8k
```

筛选 train 中完整长度 ≤8K 的样本，固定种子打乱后取最多八条，以八次梯度累积完成一次更新。
`--max-example-tokens` 是筛选，不切短样本；`--max-steps` 决定优化步数，不自动跑满一轮。
当前已配置 CUDA 4-bit NF4 QLoRA、BF16、Liger 融合损失及中间检查点。默认筛选完整长度 ≤8K；由于这会减少复杂业务示范，正式冷启动应先验证全长桶。显存与吞吐尚需实际 GPU 测量。全部清理、上传、训练、恢复和下载指令见 [4090 操作手册](4090_24G_SFT训练操作手册.md)。

输出 `run.json`（样本 ID、模型 revision、配置、数据清单哈希、设备、依赖、状态、损失和峰值显存）以及 `adapter/`。
资源试跑不自动全量评估 dev，避免额外大规模费用；后续开发评估与检查点选择需单独实现/执行。
按实际服务器账单核算费用，结合显存、吞吐和预算决定训练量；步数限制不能替代服务器的时长计费。

后续按顺序完成：

1. CUDA 上重跑小模型验收；测真实 4B 在 8K/12K/16K 长度桶的显存、吞吐和费用。
2. 根据预算确定首批训练规模，检查业务与长度覆盖，不改固定数据划分。
3. 开展 Qwen3-4B LoRA SFT，保存并加载实际适配器。
4. 接入适配器客服推理，复用政策、15 个工具与开发任务，检查合法输出、确认步骤和业务成功。
5. 冷启动有实测依据后接 GRPO，再做三组同协议评估。

默认地址修改只有一个 SFT 训练目标；数据库奖励不完整检验身份与逐笔确认。在开发评估中单独记录这两点，不能因训练链路通过就认定七类业务均已学会。
