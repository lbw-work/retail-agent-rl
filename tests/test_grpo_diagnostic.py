"""05C 诊断的同分/不完整组、实际温度概率与故障留痕；fixture 不是业务实测。"""

import importlib.util
import json
import math
import tempfile
import tomllib
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retail_agent.customer import CustomerError, GOAL_POLICY
from retail_agent.environment import ROOT
from retail_agent.grpo_diagnostic import MLXTokenSampler, resume_task_ids, summarize, verify_diagnostic
from retail_agent.evaluate import sha256
from retail_agent.grpo_rollout import RolloutError, TokenTape, run_token_episode


def fixture(trial, a, b=None, task="a"):
    return {"task_id": task, "trial": trial, "prompt_ids": [1], "completion_ids": [2, 3],
            "assistant_mask": [1, 0], "logprobs": [-1.0, 0.0], "trajectory": [],
            "termination": "completed", "scores": {"business_reward": a, "reward_a": a,
            "reward_b": a if b is None else b, "violations": []}}


class DiagnosticTests(unittest.TestCase):
    def test_normalized_business_signal_keeps_official_metrics_separate(self):
        rows = [fixture(i, int(i == 0), task=task) for task in ["a", "b"] for i in range(4)]
        for row in rows:
            row["scores"]["training_business_reward"] = row["scores"]["reward_a"]
            row["scores"]["business_reward"] = 0.
        report = summarize(rows, ["a", "b"], 4)
        self.assertEqual(report["business_successes"], 0)
        self.assertEqual(report["training_business_successes"], 2)
        self.assertEqual(report["business_signal_tasks"], ["a", "b"])
        self.assertTrue(report["local_reward_signal_supports_cuda_probe"])
        del rows[0]["scores"]["training_business_reward"]
        with self.assertRaisesRegex(RolloutError, "不同训练奖励版本"):
            summarize(rows, ["a", "b"], 4)

    def test_resume_skips_complete_groups_but_restarts_whole_partial_group(self):
        rows = [fixture(i, 1, task="a") for i in range(4)] + [fixture(i, 0, task="b") for i in range(2)]
        run, inherited = resume_task_ids(["a", "b", "c"], rows, [], 4, [])
        self.assertEqual(run, ["b", "c"])
        self.assertEqual(inherited, ["a"])
        self.assertEqual(resume_task_ids(["a", "b", "c"], rows, [], 4, ["a"]), (["a", "b", "c"], []))

    def test_resume_retains_ancestor_completions_and_rejects_bad_trials(self):
        rows = [fixture(i, 1, task="b") for i in range(4)]
        self.assertEqual(resume_task_ids(["a", "b", "c"], rows, ["a"], 4, []), (["c"], ["a", "b"]))
        for invalid in [[fixture(4, 0)], [fixture(0, 0), fixture(0, 0)]]:
            with self.assertRaises(RolloutError):
                resume_task_ids(["a"], invalid, [], 4, [])
        with self.assertRaises(RolloutError):
            resume_task_ids(["a"], [], [], 4, ["a"])

    def test_progress_links_actual_customer_api_record(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
        config = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
        row = next(r for r in map(json.loads, (ROOT / config["train_tasks"]).read_text().splitlines()) if r["id"] == "retail_264")
        ids = tokenizer.encode("Hello.<|im_end|>", add_special_tokens=False)
        sampler = Mock(generate=Mock(return_value={"token_ids": ids, "logprobs": [-1.] * len(ids)}))
        api = {"id": "actual-api-fixture", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        customer = Mock(reply=Mock(return_value={"text": "###STOP###", "api": api}))
        events = []
        run_token_episode(row, sampler, tokenizer, customer, "Hello", config, events.append)
        self.assertEqual(events[-1]["event"], "customer_feedback")
        self.assertEqual(events[-1]["api_record"], api)

    def test_incomplete_groups_do_not_create_variance_signal(self):
        report = summarize([fixture(0, 0), fixture(1, 1)], ["a", "b"], 4)
        self.assertEqual(report["completed_groups"], 0)
        self.assertEqual(report["completed_trajectories"], 2)
        self.assertEqual(report["group_statistics"]["reward_a"]["varying"], 0)
        self.assertFalse(report["local_reward_signal_supports_cuda_probe"])

    def test_same_zero_and_one_groups_have_zero_advantages(self):
        report = summarize([fixture(i, 0) for i in range(4)] + [fixture(i, 1, task="b") for i in range(4)], ["a", "b"], 4)
        self.assertEqual(report["group_statistics"]["reward_a"], {"all_zero": 1, "all_one": 1, "varying": 0})
        for group in report["groups"]:
            self.assertEqual(group["reward_a_advantages"], [0.0] * 4)

    def test_partial_rewards_do_not_imply_full_success_signal(self):
        report = summarize([fixture(i, 0, .15 if i == 0 else 0) for i in range(4)], ["a"], 4)
        self.assertEqual(report["extra_varying_groups_from_b"], ["a"])
        self.assertEqual(report["gated_successes"], 0)
        self.assertFalse(report["local_reward_signal_supports_cuda_probe"])

    def test_multiple_real_success_mixed_groups_support_probe(self):
        report = summarize([fixture(i, float(i == 0), task=t) for t in ["a", "b"] for i in range(4)], ["a", "b"], 4)
        self.assertTrue(report["local_reward_signal_supports_cuda_probe"])
        self.assertAlmostEqual(sum(report["groups"][0]["reward_a_advantages"]), 0)

    def test_gate_only_differences_do_not_prove_business_learning_signal(self):
        rows = [fixture(i, float(i == 0), task=t) for t in ["a", "b"] for i in range(4)]
        for row in rows:
            row["scores"]["business_reward"] = 1.
        report = summarize(rows, ["a", "b"], 4)
        self.assertEqual(report["reward_a_varying_tasks"], ["a", "b"])
        self.assertEqual(report["business_signal_tasks"], [])
        self.assertFalse(report["local_reward_signal_supports_cuda_probe"])

    def test_duplicate_trial_rejected(self):
        with self.assertRaises(RolloutError):
            summarize([fixture(0, 0), fixture(0, 1)], ["a"], 4)

    def test_service_error_preserves_sample_events_without_zero_score(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
        config = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
        row = next(r for r in map(json.loads, (ROOT / config["train_tasks"]).read_text().splitlines()) if r["id"] == "retail_264")
        ids = tokenizer.encode("How can I help?<|im_end|>", add_special_tokens=False)
        sampler = Mock(generate=Mock(return_value={"token_ids": ids, "logprobs": [-1.0] * len(ids)}))
        customer = Mock(reply=Mock(side_effect=CustomerError("fixture unavailable")))
        events = []
        with self.assertRaises(CustomerError):
            run_token_episode(row, sampler, tokenizer, customer, "Hello", config, events.append)
        self.assertEqual([e["event"] for e in events], ["episode_started", "assistant_sampled", "customer_request_pending"])
        self.assertEqual(events[1]["token_ids"], ids)
        self.assertFalse(any("scores" in e for e in events))

    def test_independent_verifier_rejects_journal_or_score_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.jsonl").write_text(json.dumps({"id": "a"}) + "\n")
            manifest = {"task_ids": ["a"], "config": {"train_tasks": "rows.jsonl"},
                        "immutable_sha256": {"rows.jsonl": sha256(root / "rows.jsonl")}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            original = fixture(0, 0)
            original["scores"]["completed_goals"] = [["modify_user_address", "user"]]
            original["turns"] = [{"start": 0, "end": 1}]
            events = [{"event": "episode_started", "task_id": "a", "trial": 0, "prompt_ids": [1]},
                      {"event": "sampled_token", "task_id": "a", "trial": 0, "token_id": 2, "logprob": -1.}]
            (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            def save(row):
                (root / "traces.jsonl").write_text(json.dumps(row) + "\n")
            save(original)
            replay_score = deepcopy(original["scores"])
            replay_score["completed_goals"] = [("modify_user_address", "user")]
            with patch("retail_agent.grpo_diagnostic.ROOT", root), patch("retail_agent.grpo_diagnostic.evaluate_training_task", return_value=replay_score):
                self.assertEqual(verify_diagnostic(root)["replayed_trajectories"], 1)
                for field, value in [("completion_ids", [4, 3]), ("logprobs", [-2., 0.]),
                                      ("assistant_mask", [0, 1]), ("prompt_ids", [4]), ("scores", {})]:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        changed = deepcopy(original)
                        changed[field] = value
                        save(changed)
                        verify_diagnostic(root)
                save(original)
                manifest["customer_config"] = {"goal_policy": GOAL_POLICY}
                (root / "manifest.json").write_text(json.dumps(manifest))
                budget = root / "outputs/budget"
                budget.mkdir(parents=True)
                api = {"event": "completed", "id": "fixture", "peak_cost_upper_cny": .01}
                (budget / "deepseek_requests.jsonl").write_text(json.dumps(api) + "\n")
                events.append({"event": "customer_feedback", "api_record": api})
                (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
                self.assertEqual(verify_diagnostic(root)["customer_api_records_verified"], 1)
                for corrupted in [None, {**api, "peak_cost_upper_cny": 0}]:
                    events[-1]["api_record"] = corrupted
                    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
                    with self.assertRaisesRegex(ValueError, "账本"):
                        verify_diagnostic(root)


@unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "MLX 诊断测试在本机推理环境运行")
class MLXSamplerTests(unittest.TestCase):
    def test_real_mlx_tokenizer_wrapper_supports_token_tape_validation(self):
        from transformers import AutoTokenizer
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
        wrapper = TokenizerWrapper(tokenizer)
        with self.assertRaises(TypeError):
            len(wrapper)
        tape = TokenTape(wrapper, [{"role": "user", "content": "Hello"}], [], 16384)
        ids = wrapper.encode("Hello.<|im_end|>", add_special_tokens=False)
        tape.sampled({"token_ids": ids, "logprobs": [-1.0] * len(ids)})
        self.assertEqual(tape.payload()["completion_ids"], ids)
        with self.assertRaises(RolloutError):
            tape.sampled({"token_ids": [len(tokenizer)], "logprobs": [-1.0]})

    def test_temperature_probabilities_and_actual_end_token_retained(self):
        import mlx.core as mx
        agent = SimpleNamespace(model=None, tokenizer=None, prompt_cache=None, max_context_tokens=100)
        events = []
        sampler = MLXTokenSampler(agent, .8, events.append)
        raw = mx.log(mx.array([.1, .2, .7]))
        responses = [SimpleNamespace(token=1, logprobs=raw), SimpleNamespace(token=2, logprobs=raw)]
        with patch("mlx_lm.stream_generate", return_value=iter(responses)) as stream, patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]):
            result = sampler.generate([4, 5], 2)
        self.assertEqual(result["token_ids"], [1, 2])
        expected = math.log(.7 ** (1 / .8) / sum(p ** (1 / .8) for p in [.1, .2, .7]))
        self.assertAlmostEqual(result["logprobs"][-1], expected, places=5)
        self.assertEqual([e["token_id"] for e in events], [1, 2])
        self.assertEqual(stream.call_args.args[2], [4, 5])

    def test_generator_interruption_keeps_already_sampled_tokens(self):
        import mlx.core as mx
        agent = SimpleNamespace(model=None, tokenizer=None, prompt_cache=None, max_context_tokens=100)
        events = []
        sampler = MLXTokenSampler(agent, .8, events.append)
        def interrupted():
            yield SimpleNamespace(token=1, logprobs=mx.array([-1., -1.]))
            raise RuntimeError("fixture inference failure")
        with patch("mlx_lm.stream_generate", return_value=interrupted()), patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]):
            with self.assertRaises(RuntimeError):
                sampler.generate([4, 5], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["token_id"], 1)


if __name__ == "__main__":
    unittest.main()
