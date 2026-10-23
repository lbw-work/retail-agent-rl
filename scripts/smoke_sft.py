"""离线测试用随机小 Qwen3：真实 LoRA 更新与保存重载；不是 4B 正式训练。"""

import json
import sys
import tempfile
import tomllib
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, set_seed
from retail_agent.sft_data import TargetTurnCollator, encode_target
from retail_agent.train_sft import make_trainer


def main():
    set_seed(42)
    torch.set_num_threads(2)
    config = tomllib.loads((ROOT / "configs/sft.toml").read_text())
    config.update(gradient_accumulation_steps=1, per_device_train_batch_size=2,
                  learning_rate=0.001, save_steps=1)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
    rows = []
    for i, text in enumerate(("Hello! How can I help?", "Please share your order ID.")):
        row = {"id": f"infrastructure_test_{i}", "tools": [],
               "prompt": [{"role":"system", "content":"You are a retail assistant."},
                          {"role":"user", "content":"I need help."}],
               "completion":[{"role":"assistant", "content":text}]}
        rows.append(encode_target(row, tokenizer, 512))
    dataset = Dataset.from_list(rows)
    tiny = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
    tiny._attn_implementation = "eager"
    model = Qwen3ForCausalLM(tiny)
    model.config.use_cache = False
    pristine = deepcopy(model)
    with tempfile.TemporaryDirectory(prefix="retail-sft-smoke-") as temporary:
        trainer = make_trainer(model, tokenizer, dataset, dataset, config, Path(temporary), 2, cpu=True)
        before = {n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
        if not before or any("lora_" not in n for n in before):
            raise AssertionError("训练参数不只包含 LoRA")
        # 框架的数据加载器必须原样保留监督 token 数，验证 labels 没被默认 collator 重写。
        dataloader_batch = next(iter(trainer.get_train_dataloader()))
        expected = sum(len(r["input_ids"])-r["prompt_tokens"] for r in rows)
        if (dataloader_batch["labels"] != -100).sum().item() != expected:
            raise AssertionError("TRL 数据加载器改写了 labels")
        trainer.model.eval()
        batch = TargetTurnCollator(tokenizer.pad_token_id)(rows)
        with torch.no_grad():
            output = trainer.model(**batch)
            manual_loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].reshape(-1, tiny.vocab_size), batch["labels"][:, 1:].reshape(-1))
        torch.testing.assert_close(output.loss, manual_loss)
        result = trainer.train()
        changed = [n for n,p in trainer.model.named_parameters()
                   if n in before and not torch.equal(before[n], p.detach())]
        if not changed or not torch.isfinite(torch.tensor(result.training_loss)):
            raise AssertionError("LoRA 没有发生真实更新或损失不是有限值")
        for n,p in trainer.model.named_parameters():
            if ".lora_" in n:
                continue
            base_name = n.removeprefix("base_model.model.").replace(".base_layer.", ".")
            if not torch.equal(p.detach(), pristine.get_parameter(base_name).detach()):
                raise AssertionError("冻结底座被更新："+n)
        trainer.model.eval()
        with torch.no_grad():
            reference = trainer.model(**batch).logits.clone()
        trainer.save_model(str(Path(temporary) / "adapter"))
        resume_base = deepcopy(pristine)
        restored = PeftModel.from_pretrained(pristine, Path(temporary) / "adapter", local_files_only=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(**batch).logits, reference, rtol=1e-5, atol=1e-6)
        # 从第一步的完整状态恢复至相同总步数，必须与连续训练结果一致。
        resumed = make_trainer(resume_base, tokenizer, dataset, dataset,
                               config, Path(temporary), 2, cpu=True)
        resumed.train(resume_from_checkpoint=str(Path(temporary) / "checkpoint-1"))
        if resumed.state.global_step != 2:
            raise AssertionError("恢复步数不是原实验的总目标")
        resumed.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(resumed.model(**batch).logits, reference, rtol=1e-5, atol=1e-6)
        metrics = trainer.evaluate()
        if not torch.isfinite(torch.tensor(metrics["eval_loss"])):
            raise AssertionError("开发损失不是有限值")
    report = {"status":"passed", "scope":"random_tiny_qwen3_cpu_infrastructure_only",
        "real_qwen3_4b_training":False, "optimizer_steps":2,
        "tiny_model_parameters":sum(p.numel() for p in pristine.parameters()),
        "updated_lora_tensors":len(changed), "train_loss":result.training_loss,
        "eval_loss":metrics["eval_loss"], "verified":["TRL_preserves_target_labels", "shifted_cross_entropy",
        "finite_loss", "real_optimizer_updates", "frozen_base_unchanged", "adapter_save_reload_logits_match",
        "checkpoint_resume_matches_uninterrupted_training"],
        "dependencies":{n:version(n) for n in ("torch","transformers","trl","peft","accelerate","datasets")},
        "paid_cost_cny":0, "not_verified":["Qwen3-4B_CUDA_training", "16K_memory_throughput", "business_improvement"]}
    (ROOT / "outputs/reports/sft_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
