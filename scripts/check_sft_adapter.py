"""CUDA 上重新加载保存的真实适配器，生成一个 dev 回合；不调用顾客 API。"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from retail_agent.data import read_jsonl
    from retail_agent.model import parse_action
    from retail_agent.sft_data import file_sha256
    from retail_agent.train_sft import load_prepared, verify_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("此验收需要原生 BF16 的 CUDA GPU")
    config = tomllib.loads((ROOT / "configs/sft.toml").read_text())
    directory = (ROOT / args.run_dir).resolve()
    run = json.loads((directory / "run.json").read_text())
    if run["status"] not in {"completed", "stopped_time_limit"} or run.get("global_step", 0) < 1:
        raise ValueError("实验没有保存成功的真实训练结果")
    _, _, manifest = load_prepared(config)
    if run["data_manifest_sha256"] != file_sha256(ROOT / config["prepared_dir"] / "manifest.json"):
        raise ValueError("训练与当前数据清单不一致")
    if (run["model_id"], run["model_revision"]) != (manifest["model_id"], manifest["model_revision"]):
        raise ValueError("训练与当前固定底座不一致")
    model_path = verify_model(config, manifest)
    tokenizer = AutoTokenizer.from_pretrained(directory / "adapter", local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, directory / "adapter", local_files_only=True)
    parameters = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if not parameters or any(not torch.isfinite(p).all().item() for p in parameters.values()):
        raise ValueError("适配器为空或含非有限值")
    if not any(torch.count_nonzero(p).item() for n, p in parameters.items() if ".lora_B." in n):
        raise ValueError("LoRA B 仍全部为初始化零值，没有有效更新证据")
    row = next(r for _, r in read_jsonl(ROOT / config["validation_file"])
               if r["metadata"]["total_tokens"] <= run["max_example_tokens"])
    # 只给公开政策、工具和历史，绝不把当前 completion 喂给模型。
    prompt = tokenizer.apply_chat_template(row["prompt"], tools=row["tools"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
    model.eval()
    with torch.inference_mode():
        tokens = model.generate(prompt, attention_mask=torch.ones_like(prompt),
            do_sample=False, max_new_tokens=256, use_cache=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(tokens[0, prompt.shape[1]:], skip_special_tokens=False)
    text = text.removesuffix("<|im_end|>").strip()
    report = {"adapter_loaded": True, "finite_lora_parameters": True,
        "nonzero_lora_B": True, "dev_example_id": row["id"], "raw_text": text,
        "scope": "single_dev_generation_not_business_success_evaluation",
        "customer_api_calls": 0, "test_used": False}
    try:
        report.update(protocol_valid=True, parsed_action=parse_action(text))
    except (ValueError, json.JSONDecodeError) as error:
        report.update(protocol_valid=False, protocol_error=str(error))
    (directory / "adapter_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["protocol_valid"]:
        raise SystemExit("适配器已加载，但本回合输出协议检查失败；请查看报告")


if __name__ == "__main__":
    main()
