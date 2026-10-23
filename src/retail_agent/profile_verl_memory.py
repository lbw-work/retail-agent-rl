"""固定轨迹的 dense/verl 分块概率计算对照；只做反向探针，不执行 GRPO 或优化器。"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import tomllib
import traceback

import torch

from retail_agent.environment import ROOT


def read_trace(path, index=None):
    traces = [event["trace"] for event in map(json.loads, path.read_text().splitlines())
              if event.get("event") == "rollout_trace"]
    if not traces:
        raise ValueError("日志中没有 rollout_trace")
    trace = traces[index] if index is not None else max(traces,
        key=lambda row: len(row["prompt_ids"]) + len(row["completion_ids"]))
    prompt, completion, mask = trace["prompt_ids"], trace["completion_ids"], trace["assistant_mask"]
    if (not prompt or not completion or len(mask) != len(completion) or not any(mask)
            or any(type(m) is not int or m not in (0, 1) for m in mask)
            or any(type(t) is not int or t < 0 for t in prompt + completion)):
        raise ValueError("固定轨迹的 token/mask 无效")
    return trace


def forward_scores(model, trace, temperature, fused_linear=None):
    base = model.get_base_model()
    ids = torch.tensor([trace["prompt_ids"] + trace["completion_ids"]], device=next(model.parameters()).device)
    hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    start = len(trace["prompt_ids"]) - 1
    hidden = hidden[:, start:-1]
    labels = ids[:, len(trace["prompt_ids"]):]
    if fused_linear is not None:
        logps, _ = fused_linear(hidden, base.lm_head.weight, labels, temperature=temperature)
        return logps
    # 与 verl Torch 融合实现保持相同的调温顺序，比较时只改变分块方式。
    logits = (hidden @ base.lm_head.weight.t() / temperature).float()
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--trace-index", type=int)
    parser.add_argument("--backend", choices=["dense", "fused"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("4B 显存探针需要 CUDA；Mac 只运行专项小模型测试")
    trace = read_trace(args.trace_log, args.trace_index)
    baseline = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs/verl"):
        raise ValueError("探针输出必须在 outputs/verl 下")
    output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "fixed_trace_logprob_backward_probe_not_grpo", "backend": args.backend,
        "prompt_tokens": len(trace["prompt_ids"]), "completion_tokens": len(trace["completion_ids"]),
        "action_tokens": sum(trace["assistant_mask"]), "api_calls": 0, "optimizer_steps": 0}
    report["trace_sha256"] = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    model = None
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        import verl
        from verl.utils.experimental.torch_functional import FusedLinearForPPO
        from retail_agent.train_verl import preflight
        from retail_agent.train_sft import verify_model
        _, _, project, checked = preflight(Path(verl.__file__).resolve().parents[1])
        verify_model({"model_path": "models/qwen3-4b"}, project)
        report["verl_source"] = checked["verl_source"]
        report["sft_adapter_sha256"] = checked["sft_adapter_sha256"]
        base = AutoModelForCausalLM.from_pretrained(ROOT / "models/qwen3-4b", torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa").to("cuda")
        model = PeftModel.from_pretrained(base, ROOT / baseline["sft_adapter_dir"],
                                         is_trainable=True, local_files_only=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        report["allocated_after_load_bytes"] = torch.cuda.memory_allocated()
        started = time.perf_counter()
        logps = forward_scores(model, trace, float(baseline["temperature"]),
                               FusedLinearForPPO() if args.backend == "fused" else None)
        mask = torch.tensor([trace["assistant_mask"]], device=logps.device)
        # 固定目标的负对数似然仅作为反向显存探针；不构造奖励或冒充 GRPO 更新。
        loss = -(logps * mask).sum() / mask.sum()
        if not torch.isfinite(loss):
            raise ValueError("反向探针 loss 非有限")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("反向探针梯度缺失或非有限")
        torch.cuda.synchronize()
        report.update(status="probe_completed", seconds=time.perf_counter() - started,
            loss=float(loss.detach()), max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name())
        torch.save(logps.detach().cpu(), output / "logps.pt")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        (output / "failure.txt").write_text(traceback.format_exc())
        if torch.cuda.is_initialized():
            report.update(max_allocated_bytes=torch.cuda.max_memory_allocated(),
                          max_reserved_bytes=torch.cuda.max_memory_reserved())
        raise
    finally:
        (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
