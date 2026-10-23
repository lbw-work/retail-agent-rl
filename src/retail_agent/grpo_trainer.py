"""固定 TRL 0.26.2 的 Retail 采样适配；优势与优化器继续由框架处理。"""

from collections import Counter
from copy import deepcopy
from importlib.metadata import version
import hashlib

import torch
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import entropy_from_logits, selective_log_softmax
from transformers import TrainerCallback

from retail_agent.grpo_rollout import RolloutError, TorchSampler, generate_groups
from retail_agent.environment import ROOT, RetailSession
from retail_agent.observability import StageTimer, timed
from retail_agent.experiment_protocol import check_protocol


class FiniteGradientCallback(TrainerCallback):
    def __init__(self, require_nonzero=True):
        self.require_nonzero = require_nonzero

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        nonzero, present = False, False
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            present = True
            if not torch.isfinite(param.grad).all():
                raise RolloutError("优化器更新前发现非有限梯度：" + name)
            nonzero = nonzero or bool(torch.any(param.grad != 0))
        if not present or (self.require_nonzero and not nonzero):
            raise RolloutError("整批没有有效梯度，停止更新并复查组内奖励信号")
        return control


class TrainingTimingCallback(TrainerCallback):
    def __init__(self, timing):
        self.timing, self.optimizer_span, self.step_span = timing, None, None

    def on_step_begin(self, args, state, control, **kwargs):
        span = self.timing.measure("update_step", gpu=True, step=state.global_step)
        span.__enter__()
        self.step_span = span

    def finish_step(self, error=None):
        if self.step_span is not None:
            span, self.step_span = self.step_span, None
            span.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)

    def on_step_end(self, args, state, control, **kwargs):
        self.finish_step()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        span = self.timing.measure("optimizer_step", gpu=True, step=state.global_step)
        span.__enter__()
        self.optimizer_span = span

    def finish_optimizer(self, error=None):
        if self.optimizer_span is not None:
            span, self.optimizer_span = self.optimizer_span, None
            span.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.finish_optimizer()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.timing.record("trainer_log", step=state.global_step, metrics=logs or {})
        if args.device.type == "cuda":
            self.timing.record("cuda_memory", step=state.global_step,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                scope="process_peak_no_reset")
        self.timing.save()


def business_training_reward(prompts, completions, training_reward, **kwargs):
    if len(training_reward) != len(prompts) or any(
            r is None or not torch.isfinite(torch.tensor(r)) or not 0 <= r <= 1
            for r in training_reward):
        raise RolloutError("训练奖励缺失、非有限或越界，不能用 NaN/None 伪装故障")
    return training_reward


def make_grpo_args(config, output, max_steps, *, cpu=False):
    return GRPOConfig(output_dir=str(output), max_steps=max_steps,
        learning_rate=config["learning_rate"], per_device_train_batch_size=1,
        gradient_accumulation_steps=config["num_generations"],
        generation_batch_size=config["num_generations"],
        num_generations=config["num_generations"], num_iterations=1,
        max_completion_length=config["max_trajectory_tokens"],
        temperature=config["temperature"], top_p=1.0, top_k=0, min_p=None,
        repetition_penalty=1.0, use_vllm=False, use_liger_kernel=False,
        loss_type="grpo", scale_rewards="group", beta=0.0,
        importance_sampling_level="token", disable_dropout=True,
        mask_truncated_completions=False, remove_unused_columns=False,
        gradient_checkpointing=not cpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=cpu, bf16=False, fp16=False, report_to="none",
        optim="adamw_torch", lr_scheduler_type="constant", warmup_steps=0,
        save_strategy="no", eval_strategy="no", logging_steps=1,
        seed=config["seed"], data_seed=config["seed"], dataloader_pin_memory=not cpu)


def text_policy_logprobs(model, input_ids, attention_mask, logits_to_keep, temperature,
                        batch_size=None, compute_entropy=False):
    """固定 Qwen3 文本路径；与 generate 一样先转 FP32，再做温度和归一化。"""
    if not 0 < logits_to_keep < input_ids.shape[1]:
        raise ValueError("completion 必须有非空的前置 prompt")
    logps, entropies = [], []
    batch_size = batch_size or input_ids.shape[0]
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start:start + batch_size]
        outputs = model(input_ids=ids, attention_mask=attention_mask[start:start + batch_size],
                        logits_to_keep=logits_to_keep + 1, use_cache=False)
        # 预测位置 P-1 ... P+C-2 对应目标 P ... P+C-1；不能拿最后一行预测已知 token。
        logits = outputs.logits[:, :-1, :][:, -logits_to_keep:, :].float() / temperature
        logps.append(selective_log_softmax(logits, ids[:, -logits_to_keep:]))
        if compute_entropy:
            with torch.no_grad():
                entropies.append(entropy_from_logits(logits))
    return torch.cat(logps), torch.cat(entropies) if compute_entropy else None


class RetailGRPOTrainer(GRPOTrainer):
    def __init__(self, *, rows, rollout_config, customer_factory, timing=None,
                 require_nonzero_gradient=True, **kwargs):
        if version("trl") != "0.26.2":
            raise ValueError("本适配只验收过 TRL 0.26.2；禁止依赖漂移")
        if kwargs.get("tools") or kwargs.get("rollout_func") or kwargs.get("peft_config"):
            raise ValueError("使用已加载 SFT LoRA 与 Retail 自定义采样，不混用内置工具或新 LoRA")
        self.rows, self.rollout_config, self.customer_factory = rows, rollout_config, customer_factory
        self.timing = timing if timing is not None else StageTimer(kwargs["args"].output_dir)
        # TRL 0.26.2 会无条件把 QLoRA 降为 BF16；保留 FP32 策略的 SFT 原值。
        initial_adapter_data = [(p, p.data) for p in kwargs["model"].parameters() if p.requires_grad] \
            if kwargs["model"].get_input_embeddings().weight.dtype == torch.float32 else []
        with self.timing.measure("trainer_setup"):
            super().__init__(reward_funcs=business_training_reward, **kwargs)
            for parameter, original_data in initial_adapter_data:
                parameter.data = original_data
        if self.accelerator.device.type == "cuda":
            self.timing.synchronize = lambda: torch.cuda.synchronize(self.accelerator.device)
        if (self.accelerator.num_processes != 1 or self.use_vllm or self.use_liger_kernel
                or self.num_generations != rollout_config["num_generations"]
                or self.temperature != rollout_config["temperature"] or self.beta != 0
                or self.num_iterations != 1 or self.loss_type != "grpo" or self.scale_rewards != "group"
                or self.top_p != 1 or self.top_k != 0 or self.min_p is not None
                or self.repetition_penalty != 1 or self.mask_truncated_completions):
            raise ValueError("当前只支持已验收的单进程、单组、无词表裁剪 GRPO 配方")
        if (self.args.per_device_train_batch_size != 1
                or self.args.gradient_accumulation_steps != self.num_generations
                or self.args.generation_batch_size != self.num_generations
                or self.args.steps_per_generation != self.num_generations):
            raise ValueError("必须先对完整任务组计算优势，再以单轨迹微批累积一次更新")
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n in trainable):
            raise ValueError("必须从可训练 SFT LoRA 副本继续训练，底座冻结")
        self.add_callback(FiniteGradientCallback(require_nonzero_gradient))
        self.timing_callback = TrainingTimingCallback(self.timing)
        self.add_callback(self.timing_callback)
        # 0.26.2 仅在 self.tools 为真时应用 tool_mask。这里只有真实 Retail schema；
        # 自定义 _generate 全权执行工具，不走要求 Transformers 5 的内置解析/工具循环。
        self.tools = RetailSession(next(iter(rows.values()))).tool_schemas
        self.last_group_diagnostics = []
        protocol = check_protocol()
        self.timing.record("run_manifest", protocol_sha256=protocol["sha256"],
            rollout_config=rollout_config, device=str(self.accelerator.device),
            actual_max_steps=self.args.max_steps, actual_learning_rate=self.args.learning_rate,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            model_hidden_size=self.model.config.hidden_size,
            train_task_ids=sorted(rows), torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            dependencies={name: version(name) for name in ["transformers", "trl", "peft", "accelerate"]},
            runtime_code_sha256={name: hashlib.sha256((ROOT / "src/retail_agent" / name).read_bytes()).hexdigest()
                for name in ["grpo_trainer.py", "grpo_rollout.py", "observability.py", "training_reward.py",
                             "customer.py", "environment.py", "reward.py"]},
            precision_policy="transformers_peft_fp32_nf4_matrix_efficient_sdpa_v4",
            nf4_matmul_policy="MatMul4Bit.apply",
            sdpa_kernel_policy="efficient_attention_explicit_gqa",
            tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
            mixed_precision=self.accelerator.mixed_precision,
            require_nonzero_gradient=require_nonzero_gradient,
            fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            attention_backend=self.model.config._attn_implementation,
            lora_parameter_dtypes=dict(Counter(str(p.dtype) for n, p in self.model.named_parameters()
                                              if p.requires_grad)),
            active_adapters=list(self.model.active_adapters),
            dropout_modules_with_nonzero_p=[n for n, m in self.model.named_modules()
                                            if isinstance(m, torch.nn.Dropout) and m.p != 0],
            gpu_name=torch.cuda.get_device_name(self.accelerator.device)
                if self.accelerator.device.type == "cuda" else None)

    def _generate_single_turn(self, prompts):
        raise RolloutError("Retail 必须走完整多轮采样，禁止回退到内置单轮生成")

    @timed("rollout_batch", gpu=True)
    def _generate(self, prompts):
        if any(p not in self.rows for p in prompts):
            raise RolloutError("只允许复核 TRAIN 任务 ID；ID 不直接作为客服输入")
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as model:
            # PEFT.generate 转调底座，绕过 Accelerate 包在外层 forward 上的 autocast。
            with disable_gradient_checkpointing(model, self.args.gradient_checkpointing_kwargs), self.accelerator.autocast():
                device_type = self.accelerator.device.type
                self.timing.record("sampling_precision", step=self.state.global_step,
                    device_type=device_type, autocast_enabled=torch.is_autocast_enabled(device_type),
                    autocast_dtype=str(torch.get_autocast_dtype(device_type)),
                    active_adapters=list(model.active_adapters),
                    policy_logprob="fp32_logits_then_temperature_then_logsoftmax")
                results = generate_groups(prompts, self.rows,
                    TorchSampler(model, self.processing_class, self.temperature),
                    self.processing_class, self.customer_factory, self.rollout_config, timing=self.timing)
        self._sampled_batch = results
        for result in results:
            self.timing.record("rollout_trace", sampled_policy_step=self.state.global_step, trace=result)
        size = self.num_generations
        self.last_group_diagnostics = []
        for start in range(0, len(results), size):
            group = results[start:start + size]
            values = [r["scores"][self.rollout_config["reward_mode"]] for r in group]
            self.last_group_diagnostics.append({"task_id": group[0]["task_id"], "rewards": values,
                "zero_variance": len(set(values)) == 1,
                "terminations": dict(Counter(r["termination"] for r in group))})
        masks = [r["assistant_mask"] for r in results]
        return ([r["prompt_ids"] for r in results], [r["completion_ids"] for r in results], masks,
            self.processing_class.batch_decode([r["completion_ids"] for r in results], skip_special_tokens=False),
            torch.tensor(sum(map(sum, masks)), device=self.accelerator.device),
            [r["logprobs"] for r in results],
            {"training_reward": [r["scores"][self.rollout_config["reward_mode"]] for r in results]})

    @timed("batch_prepare_and_advantage", gpu=True)
    def _generate_and_score_completions(self, inputs):
        # TRL 写入每条样本的奖励字段；逐条复制，防止重复任务共享字典后互相覆盖。
        inputs = [deepcopy(item) for item in inputs]
        output = super()._generate_and_score_completions(inputs)
        masks = output["completion_mask"] * output["tool_mask"]
        old = torch.zeros_like(output["completion_mask"], dtype=torch.float32)
        for i, result in enumerate(self._sampled_batch):
            old[i, :len(result["logprobs"])] = torch.tensor(result["logprobs"], device=old.device)
        # 在更新之前用完整真实上下文重算概率，发现采样/更新策略不一致即中止。
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            recomputed, _ = self._get_per_token_logps_and_entropies(self.model,
                torch.cat([output["prompt_ids"], output["completion_ids"]], 1),
                torch.cat([output["prompt_mask"], output["completion_mask"]], 1),
                output["completion_ids"].shape[1], batch_size=1)
        delta = (recomputed.float() - old).abs()[masks.bool()]
        if not delta.numel() or not torch.isfinite(delta).all():
            raise RolloutError("真实采样概率与 TRL 更新概率为空或非有限，不能继续训练")
        worst = int(delta.argmax())
        row, token = masks.bool().nonzero()[worst].tolist()
        max_delta = float(delta[worst])
        self.timing.record("logprob_check", step=self.state.global_step,
            max_delta=max_delta, tolerance=self.rollout_config["logprob_tolerance"],
            over_tolerance=int((delta > self.rollout_config["logprob_tolerance"]).sum()),
            checked_tokens=delta.numel(), row=row, completion_token=token,
            sampled_logprob=float(old[row, token]), recomputed_logprob=float(recomputed[row, token]),
            recomputed_dtype=str(recomputed.dtype))
        if max_delta > self.rollout_config["logprob_tolerance"]:
            raise RolloutError(f"真实采样概率与 TRL 更新概率不一致：最大差 {max_delta:.6f}，"
                               f"轨迹 {row} 的 completion token {token}，不能继续训练")
        # behavior 概率保存在原始 trace；PPO 分母使用更新前的训练策略概率。
        # 保留上方严格校验：绝不能用重算值替换行为值后再与自身比较。
        output["old_per_token_logps"] = recomputed.detach()
        self.last_logprob_max_delta = float(delta.max())
        if not torch.isfinite(output["advantages"]).all():
            raise RolloutError("组内优势非有限")
        advantages = output["advantages"].detach().float()
        self.timing.record("advantage_check", step=self.state.global_step,
            logprob_max_delta=self.last_logprob_max_delta,
            advantage_min=float(advantages.min()), advantage_max=float(advantages.max()),
            advantage_mean=float(advantages.mean()))
        return output

    @timed("loss_forward", gpu=True)
    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if not torch.isfinite(loss):
            raise RolloutError("GRPO 损失非有限，停止更新")
        return loss

    @timed("policy_logprob_forward", gpu=True)
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                         logits_to_keep, batch_size=None, compute_entropy=False, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise RolloutError("当前概率重算只验收 Qwen3 文本轨迹")
        return text_policy_logprobs(model, input_ids, attention_mask, logits_to_keep,
                                    self.temperature, batch_size, compute_entropy)

    @timed("forward_backward_microbatch", gpu=True)
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def train(self, *args, **kwargs):
        try:
            with self.timing.measure("training_run", gpu=True):
                try:
                    return super().train(*args, **kwargs)
                except BaseException as error:
                    self.timing_callback.finish_optimizer(error)
                    self.timing_callback.finish_step(error)
                    raise
        finally:
            self.timing.save()

    @timed("save_model", gpu=True)
    def save_model(self, *args, **kwargs):
        return super().save_model(*args, **kwargs)

    @timed("save_checkpoint", gpu=True)
    def _save_checkpoint(self, *args, **kwargs):
        return super()._save_checkpoint(*args, **kwargs)

    @timed("trainer_evaluation", gpu=True)
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)
