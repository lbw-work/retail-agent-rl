"""开发集对照的权重转换与错误计量；不调用 API。"""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retail_agent.evaluate import select_dev_tasks, summarize


class EvaluationTests(unittest.TestCase):
    def test_selection_is_fixed_and_covers_businesses(self):
        businesses = [f"business_{i}" for i in range(7)]
        rows = [{"id": f"single_{i}", "business": b} for i, b in enumerate(businesses)]
        rows += [{"id": f"mixed_{i}", "business": "mixed"} for i in range(12)]
        selected = select_dev_tasks(rows, businesses, 42)
        self.assertEqual(selected, select_dev_tasks(rows, businesses, 42))
        self.assertEqual(len({r["id"] for r in selected}), 14)
        self.assertEqual([r["business"] for r in selected[:7]], businesses)
        self.assertEqual(select_dev_tasks(rows, businesses, 42, all_dev=True), rows)

    def test_system_errors_are_separate_and_pairs_require_both_arms(self):
        def result(checkpoint, task, reward):
            return {"checkpoint": checkpoint, "task_id": task, "business": "cancel",
                    "business_evaluation_eligible": reward is not None,
                    "reward": {"reward": reward} if reward is not None else None,
                    "termination": "completed" if reward is not None else "customer_service_error",
                    "assistant_actions": 5, "events": []}
        report = summarize([result("original", "a", 0), result("sft", "a", 1),
                            result("original", "b", None), result("sft", "b", 1)], ["a", "b", "c"])
        self.assertEqual(report["paired_eligible"], 1)
        self.assertEqual(report["improved"], 1)
        self.assertEqual(report["arms"]["original"]["system_errors"], 1)
        self.assertEqual(report["arms"]["original"]["eligible"], 1)
        self.assertEqual(report["arms"]["sft"]["success_rate_planned"], 2 / 3)
        self.assertFalse(report["comparison_complete"])
        self.assertIsNone(report["arms"]["sft"]["primary_success_rate"])

    def test_duplicate_or_unplanned_records_cannot_form_an_evaluation(self):
        row = {"checkpoint": "sft", "task_id": "a"}
        with self.assertRaisesRegex(ValueError, "重复"):
            summarize([row, row], ["a"])
        with self.assertRaisesRegex(ValueError, "非计划"):
            summarize([row], ["b"])

    def test_business_coverage_does_not_claim_atomic_business_success(self):
        rows = [{"checkpoint": arm, "task_id": "a", "business": "mixed",
            "businesses": ["return", "payment"], "business_evaluation_eligible": True,
            "reward": {"reward": 0}, "termination": "completed", "assistant_actions": 3,
            "events": []} for arm in ["original", "sft"]]
        report = summarize(rows, ["a"])
        self.assertTrue(report['comparison_complete'])
        self.assertEqual(report['arms']['sft']['primary_success_rate'], 0)
        self.assertEqual(report['arms']['sft']['business_coverage']['payment'],
            {'attempted': 1, 'eligible': 1, 'whole_task_successes': 0})


@unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "MLX 权重转换测试仅在本机推理环境执行")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        import mlx.nn as nn

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(self, name, nn.Linear(3, 4, bias=False))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [Block()]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

            @property
            def layers(self):
                return self.model.layers

        self.model = Model()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        (self.path / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none", "r": 16, "lora_alpha": 32}))
        rng = np.random.default_rng(12)
        self.a = rng.normal(size=(16, 3)).astype(np.float32)
        self.b = rng.normal(size=(4, 16)).astype(np.float32)
        self.weights = {}
        for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = f"base_model.model.model.layers.0.self_attn.{target}"
            self.weights[prefix + ".lora_A.weight"] = mx.array(self.a)
            self.weights[prefix + ".lora_B.weight"] = mx.array(self.b)
        mx.save_safetensors(str(self.path / "adapter_model.safetensors"), self.weights)

    def tearDown(self):
        self.temp.cleanup()

    def test_peft_delta_equivalence(self):
        import mlx.core as mx
        from retail_agent.adapter import load_peft_adapter
        base_weight = np.array(self.model.layers[0].self_attn.q_proj.weight)
        x = np.array([[0.4, -0.6, 1.2]], dtype=np.float32)
        # 独立 PEFT 定义：W + (alpha/r) * B @ A。
        expected = x @ (base_weight + 2.0 * self.b @ self.a).T
        info = load_peft_adapter(self.model, self.path)
        actual = np.array(self.model.layers[0].self_attn.q_proj(mx.array(x)))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(info["tensor_count"], 8)

    def test_partial_adapter_is_rejected(self):
        import mlx.core as mx
        from retail_agent.adapter import load_peft_adapter
        self.weights.pop(next(iter(self.weights)))
        mx.save_safetensors(str(self.path / "adapter_model.safetensors"), self.weights)
        with self.assertRaisesRegex(ValueError, "缺失"):
            load_peft_adapter(self.model, self.path)

    def test_zero_update_is_rejected(self):
        import mlx.core as mx
        from retail_agent.adapter import load_peft_adapter
        for name, value in self.weights.items():
            if ".lora_B." in name:
                self.weights[name] = mx.zeros_like(value)
        mx.save_safetensors(str(self.path / "adapter_model.safetensors"), self.weights)
        with self.assertRaisesRegex(ValueError, "全为零"):
            load_peft_adapter(self.model, self.path)


if __name__ == "__main__":
    unittest.main()
