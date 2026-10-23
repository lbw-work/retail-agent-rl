import unittest
from importlib.util import find_spec
from pathlib import Path

from transformers import AutoTokenizer
from retail_agent.data import read_jsonl
from retail_agent.sft_data import TargetTurnCollator, encode_target, validate_encoded

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(find_spec("torch"), "SFT 测试使用独立 .venv-train 环境")
class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
        cls.examples = {}
        for _, row in read_jsonl(ROOT / "data/versions/retail_full_v2/processed/sft_train.jsonl"):
            cls.examples.setdefault("tool" if row["completion"][0].get("tool_calls") else "text", row)
            if row["metadata"]["total_tokens"] > 12288:
                cls.examples.setdefault("long", row)
            if len(cls.examples) == 3:
                break

    def test_actual_target_only_mask_and_terminator(self):
        for kind, row in self.examples.items():
            with self.subTest(kind=kind):
                encoded = encode_target(row, self.tokenizer, 16384)
                boundary = encoded["prompt_tokens"]
                self.assertEqual(encoded["labels"][:boundary], [-100]*boundary)
                self.assertEqual(encoded["labels"][boundary:], encoded["input_ids"][boundary:])
                self.assertEqual(len(encoded["input_ids"]), row["metadata"]["total_tokens"])
                target = self.tokenizer.decode(encoded["labels"][boundary:])
                self.assertIn("<|im_end|>", target)
                self.assertNotIn("<think>", target)
                if kind == "tool":
                    self.assertIn("<tool_call>", target)

    def test_full_history_over_limit_is_rejected(self):
        row = self.examples["long"]
        self.assertGreater(row["metadata"]["total_tokens"], 12288)
        with self.assertRaisesRegex(ValueError, "禁止截断"):
            encode_target(row, self.tokenizer, 8192)

    def test_collator_preserves_labels_and_masks_padding(self):
        rows = [encode_target(self.examples[k], self.tokenizer, 16384) for k in ("text", "tool")]
        self.assertNotEqual(len(rows[0]["input_ids"]), len(rows[1]["input_ids"]))
        batch = TargetTurnCollator(self.tokenizer.pad_token_id)(rows)
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            self.assertEqual(batch["labels"][i, :n].tolist(), row["labels"])
            self.assertTrue((batch["labels"][i, n:] == -100).all())
            self.assertTrue((batch["attention_mask"][i, :n] == 1).all())
            self.assertTrue((batch["attention_mask"][i, n:] == 0).all())

    def test_corrupted_mask_rejected(self):
        row = encode_target(self.examples["text"], self.tokenizer, 16384)
        row["labels"][0] = row["input_ids"][0]
        with self.assertRaisesRegex(ValueError, "掩码"):
            validate_encoded(row, 16384)

    def test_changed_source_counts_rejected(self):
        from copy import deepcopy
        row = deepcopy(self.examples["text"])
        row["metadata"]["total_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "计数"):
            encode_target(row, self.tokenizer, 16384)


@unittest.skipUnless(find_spec("torch"), "训练控制测试使用独立 .venv-train 环境")
class TrainingControlsTests(unittest.TestCase):
    def test_qlora_preparation_freezes_base_and_preserves_qwen_norm_dtype(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from retail_agent.train_sft import prepare_qlora
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8)).to(torch.bfloat16)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model = prepare_qlora(model)
        for n, p in model.named_parameters():
            self.assertFalse(p.requires_grad)
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        norm = model.model.layers[0].self_attn.q_norm
        self.assertEqual(norm(torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16)

    def test_cuda_wheel_suffix_does_not_allow_dependency_version_drift(self):
        from retail_agent.train_sft import dependency_matches
        self.assertTrue(dependency_matches("torch", "2.8.0+cu128", "2.8.0"))
        self.assertFalse(dependency_matches("torch", "2.9.0+cu128", "2.8.0"))
        self.assertFalse(dependency_matches("trl", "0.27.0", "0.26.2"))

    def test_time_limit_stops_and_requests_recoverable_checkpoint(self):
        from unittest.mock import patch
        from transformers import TrainerControl
        from retail_agent.train_sft import training_time_limit
        timer, control = training_time_limit(1), TrainerControl()
        with patch("retail_agent.train_sft.time.monotonic", side_effect=[100, 159, 160]):
            timer.on_train_begin(None, None, control)
            timer.on_step_end(None, None, control)
            self.assertFalse(control.should_training_stop)
            timer.on_step_end(None, None, control)
        self.assertTrue(timer.expired)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)


if __name__ == "__main__":
    unittest.main()
