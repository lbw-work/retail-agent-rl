"""显式启动 CUDA SFT 小规模试跑；预处理验收不触发此入口。"""

import argparse
import json
import tomllib
import time
from importlib.metadata import version
from pathlib import Path

from retail_agent.sft_data import TargetTurnCollator, file_sha256

ROOT = Path(__file__).resolve().parents[2]


def make_trainer(model, tokenizer, train, dev, config, output, max_steps, *, cpu=False):
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    use_bf16 = not cpu and torch.cuda.is_bf16_supported(including_emulation=False)
    args = SFTConfig(
        output_dir=str(output), max_steps=max_steps,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None, packing=False, padding_free=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False, assistant_only_loss=False,
        eos_token="<|im_end|>", report_to="none", use_cpu=cpu,
        bf16=use_bf16, fp16=not cpu and not use_bf16,
        use_liger_kernel=not cpu and config["use_liger_kernel"],
        # 只替换融合损失，保留原注意力、RoPE、MLP 和 norm 实现。
        liger_kernel_config={"rope": False, "cross_entropy": False,
            "fused_linear_cross_entropy": True, "rms_norm": False, "swiglu": False},
        # 一两步资源试跑不做 warmup，避免唯一优化步的学习率为零。
        optim="adamw_torch", lr_scheduler_type="cosine",
        warmup_ratio=0.03 if max_steps > 2 else 0.0,
        eval_strategy="no", save_strategy="steps", save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"], logging_steps=1,
        seed=config["seed"], data_seed=config["seed"],
        # 自定义 collator 需要保留原始 labels，不让框架重新套模板或截断。
        remove_unused_columns=True, dataloader_pin_memory=not cpu,
    )
    lora = config["lora"]
    return SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
        processing_class=tokenizer, data_collator=TargetTurnCollator(tokenizer.pad_token_id),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=lora["r"], lora_alpha=lora["alpha"],
                              lora_dropout=lora["dropout"], target_modules=lora["target_modules"]))


def load_prepared(config):
    from datasets import load_from_disk

    path = ROOT / config["prepared_dir"]
    manifest = json.loads((path / "manifest.json").read_text())
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    if manifest["status"] != "passed" or manifest["test_used"] or (
        manifest["model_id"], manifest["model_revision"], manifest["dataset_version"]) != (
        project["model_id"], project["model_revision"], project["data"]["dataset_version"]):
        raise ValueError("准备数据的底座、版本或验收状态不符")
    for relative, expected in manifest["input_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("SFT 准备输入已改变："+relative)
    for relative, expected in manifest["output_sha256"].items():
        if file_sha256(path / relative) != expected:
            raise ValueError("SFT 准备产物已改变："+relative)
    if any(not dependency_matches(n, version(n), expected)
           for n, expected in manifest["dependencies"].items()):
        raise ValueError("训练依赖与已验收组合不符")
    return load_from_disk(str(path / "train")), load_from_disk(str(path / "dev")), manifest


def dependency_matches(name, actual, expected):
    # 同一 PyTorch 发布版允许 CUDA wheel 的本地后缀，如 2.8.0+cu128。
    if name == "torch":
        return actual.split("+", 1)[0] == expected.split("+", 1)[0]
    return actual == expected


def prepare_qlora(model):
    import torch
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # PEFT 会把非量化权重升到 FP32。固定底座原为 BF16，恢复其存储精度，
    # 防止 Qwen3 的 RMSNorm/残差把 Q/K/V 升成 FP32，影响 SDPA 和显存。
    # Qwen3 原生 RMSNorm 内部仍用 FP32 计算方差；不改 4-bit 量化参数。
    for param in model.parameters():
        if param.dtype == torch.float32 and param.__class__.__name__ != "Params4bit":
            param.data = param.data.to(torch.bfloat16)
    return model


def training_time_limit(minutes):
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        expired = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - self.started >= minutes * 60:
                self.expired = True
                control.should_training_stop = True
                control.should_save = True
            return control

    return TimeLimit()


def verify_model(config, manifest):
    model_path = ROOT / config["model_path"]
    if model_path.resolve() != (ROOT / "models/qwen3-4b").resolve():
        raise ValueError("只允许已验收的固定原模型目录")
    model_manifest = json.loads((ROOT / "models/manifest.json").read_text())
    if (model_manifest["model_id"], model_manifest["model_revision"]) != (
            manifest["model_id"], manifest["model_revision"]):
        raise ValueError("本机权重与固定模型/revision 不符")
    for item in model_manifest["original_files"]:
        if file_sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("模型权重校验失败："+item["path"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if file_sha256(model_path / name) != file_sha256(ROOT / "data/raw/tokenizer" / name):
            raise ValueError("模型 tokenizer 与 SFT 编码 tokenizer 不一致")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--max-example-tokens", type=int,
                        help="过滤超长完整样本用于长度桶试跑，不截断样本")
    parser.add_argument("--output-dir", required=True, help="新的输出目录；不覆盖已有实验")
    parser.add_argument("--resume-from-checkpoint", help="本次输出目录内的完整 checkpoint-N")
    parser.add_argument("--max-minutes", type=float, default=60,
                        help="训练循环时限；步结束后保存退出，不包含安装/加载/上传的计费时间")
    parser.add_argument("--probe-longest", action="store_true",
                        help="资源试跑优先取长度桶内最长样本；正式训练不使用此选项")
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_samples < 1 or args.max_minutes <= 0 or (
        args.max_example_tokens is not None and args.max_example_tokens < 1):
        parser.error("步数、样本数及完整长度上限必须为正数")
    config = tomllib.loads((ROOT / "configs/sft.toml").read_text())
    if args.max_example_tokens is None:
        args.max_example_tokens = config["max_example_tokens"]
    output = (ROOT / args.output_dir).resolve()
    resume = Path(args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None
    if resume and (resume.parent != output or not (resume / "trainer_state.json").is_file()
                   or not (resume / "optimizer.pt").is_file()):
        raise ValueError("恢复必须使用本次输出目录内、含训练状态和优化器的完整检查点")
    if output.exists() and not resume:
        raise FileExistsError("实验输出目录已存在；不会覆盖检查点")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("正式入口需要 CUDA；Mac 只验收数据与小模型训练链路")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("当前 4090 配方要求原生 BF16；请使用支持该精度的 GPU")
    if torch.cuda.device_count() != 1:
        raise SystemExit("本入口为单卡配置；请用 CUDA_VISIBLE_DEVICES 只暴露一张显卡")
    train, dev, manifest = load_prepared(config)
    full_train_count = len(train)
    train = train.filter(lambda r: len(r["input_ids"]) <= args.max_example_tokens)
    eligible_count = len(train)
    if not len(train):
        raise ValueError("当前长度桶没有完整样本")
    if args.probe_longest:
        indices = sorted(range(len(train)), key=lambda i: len(train[i]["input_ids"]), reverse=True)
        train = train.select(indices[:args.max_samples])
    else:
        train = train.shuffle(seed=config["seed"]).select(range(min(args.max_samples, len(train))))
    plan = {"model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
            "data_manifest_sha256": file_sha256(ROOT / config["prepared_dir"] / "manifest.json"),
            "train_ids": list(train["id"]), "max_steps": args.max_steps,
            "max_example_tokens": args.max_example_tokens, "config": config,
            "training_code_sha256": file_sha256(Path(__file__))}
    if resume:
        previous = json.loads((output / "run.json").read_text())
        if any(previous.get(k) != v for k, v in plan.items()):
            raise ValueError("恢复参数、代码或数据与原实验不一致；禁止混用检查点")
    model_path = verify_model(config, manifest)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    q = config["quantization"]
    if not q["load_in_4bit"] or q["compute_dtype"] != "bfloat16":
        raise ValueError("本入口固定使用 BF16 计算的 4-bit QLoRA")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type=q["quant_type"], bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_qlora(model)
    model.config.use_cache = False
    trainer = make_trainer(model, tokenizer, train, dev, config, output, args.max_steps)
    timer = training_time_limit(args.max_minutes)
    trainer.add_callback(timer)
    output.mkdir(parents=True, exist_ok=True)
    run = {**plan, "status":"running", "full_train_rows": full_train_count,
           "eligible_train_rows": eligible_count, "selected_train_rows": len(train),
           "selected_total_tokens": sum(len(r["input_ids"]) for r in train),
           "selected_max_tokens": max(len(r["input_ids"]) for r in train),
           "max_minutes": args.max_minutes, "resume_from_checkpoint": str(resume) if resume else None,
           "device":torch.cuda.get_device_name(),
           "dependencies":{n:version(n) for n in (*manifest["dependencies"], "bitsandbytes", "liger-kernel")},
           "budget_note":"只记录训练步骤；实际服务器费用须按供应商计费及总预算人工核对"}
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n")
    try:
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        run.update(status="stopped_time_limit" if timer.expired and
                   trainer.state.global_step < args.max_steps else "completed",
                   global_step=trainer.state.global_step, metrics=result.metrics,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except (Exception, KeyboardInterrupt) as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
