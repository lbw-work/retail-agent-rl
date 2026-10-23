"""单卡 verl LoRA checkpoint 导出与独立重载；不采样顾客、不执行优化器。"""

import argparse
import json
from pathlib import Path
import shutil
import tomllib

import torch
from safetensors.torch import load_file, save_file

from retail_agent.environment import ROOT


def export_adapter(checkpoint, original, output):
    # 本入口仅支持第一版单卡 NO_SHARD 保存出的普通 Tensor；不近似拼接多卡分片。
    weights = torch.load(checkpoint / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=True)
    normalized = {}
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or "lora_" not in name:
            raise ValueError("需要 save_lora_only 的单卡 Tensor checkpoint")
        if tensor.dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("LoRA dtype 超出本次 BF16/FP32 验收范围")
        key = name.replace("_fsdp_wrapped_module.", "").replace(".default.", ".")
        if key in normalized or not torch.isfinite(tensor).all():
            raise ValueError("LoRA 名称冲突或存在非有限权重")
        normalized[key] = tensor.contiguous()
    reference = load_file(str(original / "adapter_model.safetensors"))
    if normalized.keys() != reference.keys():
        raise ValueError("checkpoint 与原 SFT LoRA 参数集合不一致")
    changed = 0
    for name, tensor in normalized.items():
        if tensor.shape != reference[name].shape:
            raise ValueError("LoRA 形状不一致：" + name)
        # 同时排除 FP32→BF16→FP32 的往返转换，不能把加载舍入当更新。
        direct = reference[name].to(tensor.dtype)
        rounded = reference[name].to(torch.bfloat16).to(tensor.dtype)
        changed += int(not torch.equal(tensor, direct) and not torch.equal(tensor, rounded))
    if not changed:
        raise ValueError("没有超出 dtype 转换的参数变化，不能验收有效更新")
    optimizer = torch.load(checkpoint / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=True)
    states = optimizer.get("state", {})
    if not states:
        raise ValueError("缺少优化器状态")
    steps = []
    for state in states.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("优化器状态非有限")
        if "step" in state:
            steps.append(float(state["step"]))
    if not steps or max(steps) <= 0:
        raise ValueError("优化器状态没有实际更新步")
    output.mkdir(parents=True, exist_ok=False)
    save_file(normalized, str(output / "adapter_model.safetensors"))
    shutil.copyfile(original / "adapter_config.json", output / "adapter_config.json")
    return {"lora_tensor_count": len(normalized), "changed_tensors_after_dtype_alignment": changed,
            "optimizer_state_count": len(states), "optimizer_max_step": max(steps),
            "lora_dtypes": sorted({str(t.dtype) for t in normalized.values()}),
            "cuda_reload_verified": False, "api_calls": 0, "optimizer_calls": 0}


def reload_cuda(adapter, model_path=None):
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("独立重载验收需要 CUDA")
    base = AutoModelForCausalLM.from_pretrained(model_path or ROOT / "models/qwen3-4b", torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(model)
    if actual.keys() != saved.keys():
        raise ValueError("独立重载后的 LoRA 参数集合不一致")
    for name, tensor in actual.items():
        if not torch.equal(tensor.detach().cpu(), saved[name].to(tensor.dtype)):
            raise ValueError("独立重载的参数值不一致：" + name)
    model.eval()
    with torch.no_grad():
        result = model(torch.tensor([[151643, 151645]], device="cuda"), use_cache=False, logits_to_keep=1)
    if not torch.isfinite(result.logits).all():
        raise ValueError("独立重载前向非有限")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="global_step_N/actor")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的导出目录")
    parser.add_argument("--reload-cuda", action="store_true")
    parser.add_argument("--profile", choices=['legacy', 'mmopd'], default='legacy')
    args = parser.parse_args()
    baseline = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    model_path = ROOT / 'models/qwen3-4b'
    identity = {'experiment_profile': 'legacy'}
    if args.profile == 'mmopd':
        from retail_agent.mmopd import profile, verify_snapshot
        identity = verify_snapshot(require_adapter=True)
        native = profile()
        baseline['sft_adapter_dir'] = native['sft_adapter_dir']
        model_path = ROOT / native['model_path']
    report = export_adapter(args.checkpoint.resolve(), ROOT / baseline["sft_adapter_dir"], args.output_dir.resolve())
    report.update(identity, checkpoint=str(args.checkpoint.resolve()))
    try:
        if args.reload_cuda:
            reload_cuda(args.output_dir.resolve(), model_path)
            report["cuda_reload_verified"] = True
        report["status"] = "cuda_reload_verified" if args.reload_cuda else "export_audited_not_cuda_reload"
    except BaseException as error:
        report.update(status="reload_failed", error_type=type(error).__name__)
        raise
    finally:
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
