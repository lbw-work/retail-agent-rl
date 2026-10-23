"""迁移接口、真实 Retail 环境与分块反向测试；不代表 Ray/vLLM/CUDA 验收。"""

import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import tomllib
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

from retail_agent.environment import ROOT, RetailSession
from retail_agent.grpo_rollout import RolloutError, run_token_episode
from retail_agent.observability import StageTimer
from retail_agent.verl_agent_loop import (
    ServerSampler, complete_group, probability_diagnostics, run_episode, selected_training_rows, shared_opening,
)

VERL = ROOT.parent / "verl-main"


def load_source_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("冻结源码接口不存在")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def runtime_namespace():
    """执行真实适配类；仅替换本机未安装的 Ray/TQ 进程边界。"""
    ns = dict(BaseModel=BaseModel, Optional=Optional, Any=Any, torch=torch)
    load_source_nodes(VERL / "verl/experimental/agent_loop/agent_loop.py",
                      {"AgentLoopMetrics", "AgentLoopOutput"}, ns)
    ns.update(ray=SimpleNamespace(remote=lambda cls: cls), AgentLoopBase=object,
        AgentLoopWorker=object, AgentLoopManagerTQ=object, PPOTrainerSync=object,
        Dataset=torch.utils.data.Dataset, tq=SimpleNamespace(async_kv_put=AsyncMock(), async_kv_batch_put=AsyncMock()),
        list_of_dict_to_tensordict=lambda fields: fields, complete_group=complete_group,
        shared_opening=AsyncMock(return_value="Please help."), json=json, math=math,
        Path=Path, contextmanager=contextmanager, time=time, RolloutError=RolloutError,
        RetailSession=RetailSession, selected_training_rows=selected_training_rows)
    load_source_nodes(ROOT / "src/retail_agent/verl_runtime.py", {"RetailDataset", "RetailAgentLoop",
        "RetailGroupWorker", "RetailAgentLoopManager", "RetailPPOTrainer", "RetailTaskRunner"}, ns)
    return ns


class ProbabilityTests(unittest.TestCase):
    def test_large_finite_difference_is_recorded_without_hard_stop(self):
        result = probability_diagnostics([-1., -2.], [-1.8, -2.01])
        self.assertAlmostEqual(result["retail/logprob_abs_max"], .8)
        self.assertEqual(result["retail/logprob_over_005"], 1)

    def test_invalid_probability_data_still_stops(self):
        for left, right in [([], []), ([-1.], []), ([-1.], [float("nan")]), ([-1.], [.1])]:
            with self.subTest(left=left, right=right), self.assertRaises(RolloutError):
                probability_diagnostics(left, right)


class GroupTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_keeps_shared_prompt_and_version(self):
        calls = []
        async def run_one(session_id, opening):
            calls.append((session_id, opening))
            return {"prompt_ids": [1, 2], "policy_version": 0}
        result = await complete_group({}, "shared", run_one, 4, 0)
        self.assertEqual(len(result), 4)
        self.assertEqual(calls, [(i, "shared") for i in range(4)])

    async def test_failed_session_does_not_return_partial_group_or_retry(self):
        calls = []
        async def run_one(session_id, opening):
            calls.append(session_id)
            if session_id == 1:
                raise RuntimeError("customer service unavailable")
            return {"prompt_ids": [1], "policy_version": 0}
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await complete_group({}, "shared", run_one, 4, 0)
        self.assertEqual(calls, [0, 1])

    async def test_mismatched_prompt_and_stale_weights_rejected(self):
        async def different(session_id, opening):
            return {"prompt_ids": [session_id], "policy_version": 0}
        with self.assertRaisesRegex(RolloutError, "prompt"):
            await complete_group({}, "shared", different, 4, 0)
        async def stale(session_id, opening):
            return {"prompt_ids": [1], "policy_version": 0}
        with self.assertRaisesRegex(RolloutError, "版本"):
            await complete_group({}, "shared", stale, 4, 1)

    async def test_sync_sampler_cannot_deadlock_main_loop(self):
        import asyncio
        sampler = ServerSampler(Mock(), asyncio.get_running_loop(), {}, 2, "request")
        with self.assertRaisesRegex(RuntimeError, "工作线程"):
            sampler.generate([1], 1)

    async def test_publish_uses_real_verl_output_schema_and_action_mask(self):
        ns = runtime_namespace()
        worker = object.__new__(ns["RetailGroupWorker"])
        worker.retail = {"reward_mode": "reward_a"}
        result = {"prompt_ids": [1, 2], "completion_ids": [3, 4, 5], "assistant_mask": [1, 0, 1],
                  "logprobs": [-1., 0., -2.], "scores": {"reward_a": 1}, "assistant_actions": 2,
                  "policy_version": 0}
        await worker._publish("group", 1, {"uid": "group", "global_steps": 1}, [result] * 4)
        call = ns["tq"].async_kv_batch_put.call_args.kwargs
        self.assertEqual(call["keys"], [f"group_{i}_0" for i in range(4)])
        field = call["fields"][0]
        self.assertEqual(field["input_ids"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(field["loss_mask"].tolist(), [1, 0, 1])
        self.assertEqual(field["attention_mask"].tolist(), [1] * 5)
        self.assertEqual(field["rollout_log_probs"].tolist(), [-1., 0., -2.])
        self.assertEqual(field["rm_scores"].sum().item(), 1)
        self.assertEqual(call["tags"][0]["min_global_steps"], 0)

    async def test_worker_failure_publishes_no_trajectory_and_propagates(self):
        ns = runtime_namespace()
        ns["RetailSession"] = lambda row: SimpleNamespace(add_user_message=lambda text: None,
                                                         agent_messages=lambda: [])
        worker = object.__new__(ns["RetailGroupWorker"])
        worker.rows = {"task": {"id": "task"}}
        worker.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            temperature=.8, top_p=1., top_k=-1, n=4)))
        worker.retail, worker.timing = {}, StageTimer()
        worker.agent = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("infrastructure")))
        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            await worker._run_group({"uid": "uid", "global_steps": 1, "extra_info": {"task_id": "task"}})
        ns["tq"].async_kv_batch_put.assert_not_called()
        self.assertEqual(ns["tq"].async_kv_put.call_args.kwargs["tag"]["status"], "failure")

    def test_calibration_skips_optimizer_and_step_end_weight_sync(self):
        ns = runtime_namespace()
        trainer = object.__new__(ns["RetailPPOTrainer"])
        trainer.config = SimpleNamespace(retail=SimpleNamespace(calibration_only=True))
        metrics, batch = {}, object()
        self.assertIs(trainer._update_actor(batch, metrics), batch)
        self.assertEqual(metrics["retail/calibration_only"], 1)
        trainer.on_step_end()


class RealRetailEpisodeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
        cls.config = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
        cls.row = next(r for r in map(json.loads, (ROOT / cls.config["train_tasks"]).read_text().splitlines())
                       if r["id"] == "retail_264")

    def texts(self):
        return ['<tool_call>' + json.dumps({"name": action["name"], "arguments": action["arguments"]},
                    separators=(",", ":")) + '</tool_call><|im_end|>'
                for action in self.row["evaluation_criteria"]["actions"]] + ['Done.<|im_end|>']

    async def test_bridge_matches_original_episode_tokens_masks_and_business_reward(self):
        encoded = [self.tokenizer.encode(text, add_special_tokens=False) for text in self.texts()]
        iterator = iter(encoded)
        requests = []
        async def generate(**kwargs):
            requests.append(kwargs)
            ids = next(iterator)
            return SimpleNamespace(token_ids=ids, log_probs=[-1.] * len(ids), stop_reason="completed",
                                   extra_fields={"global_steps": 0})
        customer = Mock(reply=Mock(return_value={"text": "###STOP###"}))
        result = await run_episode(self.row, SimpleNamespace(generate=generate), self.tokenizer,
            lambda scenario: customer, "Please help.", self.config,
            {"temperature": .8, "top_p": 1., "top_k": -1}, StageTimer(), "sticky")
        sync_iterator = iter(encoded)
        sampler = SimpleNamespace(generate=lambda prompt, max_new: (
            lambda ids: {"token_ids": ids, "logprobs": [-1.] * len(ids)})(next(sync_iterator)))
        reference = run_token_episode(self.row, sampler, self.tokenizer,
            Mock(reply=Mock(return_value={"text": "###STOP###"})), "Please help.", self.config)
        for key in ["prompt_ids", "completion_ids", "assistant_mask", "logprobs", "scores", "termination"]:
            self.assertEqual(result[key], reference[key], key)
        self.assertIn(0, result["assistant_mask"])
        self.assertEqual(result["scores"]["reward_a"], 1)
        self.assertTrue(all(req["request_id"] == "sticky" for req in requests))
        self.assertTrue(all(req["sampling_params"]["logprobs"] for req in requests))
        self.assertEqual(len(requests), len(encoded))

    async def test_shared_opening_is_one_request_and_client_closed(self):
        customer = Mock(reply=Mock(return_value={"text": "Please help."}))
        factory = Mock(return_value=customer)
        self.assertEqual(await shared_opening(self.row, factory, StageTimer()), "Please help.")
        customer.reply.assert_called_once()
        customer.http.close.assert_called_once()

    async def test_customer_failure_preserves_partial_token_journal(self):
        ids = self.tokenizer.encode("Hello.<|im_end|>", add_special_tokens=False)
        async def generate(**kwargs):
            return SimpleNamespace(token_ids=ids, log_probs=[-1.] * len(ids), stop_reason="completed",
                                   extra_fields={"global_steps": 0})
        customer = Mock(reply=Mock(side_effect=RuntimeError("customer unavailable")))
        with tempfile.TemporaryDirectory() as directory:
            timing = StageTimer(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "customer unavailable"):
                await run_episode(self.row, SimpleNamespace(generate=generate), self.tokenizer,
                    lambda scenario: customer, "Please help.", self.config,
                    {"temperature": .8, "top_p": 1., "top_k": -1}, timing)
            events = [json.loads(line) for path in (Path(directory) / "observations").glob("*.jsonl")
                      for line in path.read_text().splitlines()]
            sampled = [event for event in events if event["event"] == "episode_progress"
                       and event["detail"]["event"] == "assistant_sampled"]
            self.assertEqual(sampled[0]["detail"]["token_ids"], ids)
            self.assertEqual(sampled[0]["policy_versions"], [0])
        customer.http.close.assert_called_once()

    def test_selected_rows_reject_modified_target_and_duplicate_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.jsonl"
            path.write_text(json.dumps(self.row) + "\n")
            self.assertEqual(selected_training_rows(path), [self.row])
            row = deepcopy(self.row)
            row["user_scenario"]["instructions"] = "Changed private target"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                selected_training_rows(path)
            path.write_text((json.dumps(self.row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "重复"):
                selected_training_rows(path)

    def test_dataset_sends_only_public_prompt_and_task_id(self):
        ns = runtime_namespace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.jsonl"
            path.write_text(json.dumps(self.row) + "\n")
            item = ns["RetailDataset"](str(path), None, {})[0]
        self.assertEqual(item["raw_prompt"], RetailSession(self.row).agent_messages())
        self.assertEqual(item["extra_info"], {"task_id": self.row["id"]})
        self.assertNotIn("evaluation_criteria", item)
        self.assertNotIn("user_scenario", item)


class FusedNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        path = VERL / "verl/utils/experimental/torch_functional.py"
        spec = importlib.util.spec_from_file_location("retail_test_verl_fused", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_real_verl_chunked_forward_backward_matches_dense_with_mask(self):
        for dtype, atol, rtol in [(torch.float32, 2e-6, 2e-5), (torch.bfloat16, .03, .03)]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(7)
                h = torch.randn(2, 7, 16, dtype=dtype, requires_grad=True)
                w = torch.randn(31, 16, dtype=dtype, requires_grad=True)
                labels = torch.randint(0, 31, (2, 7))
                mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 1, 1]])
                advantages = torch.tensor([1., -1.])
                hd, wd = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
                dense_logits = (hd @ wd.t() / .8).float()
                reference = dense_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                fused, _ = self.module.FusedLinearForPPO(chunk_size=3)(h, w, labels, temperature=.8)
                torch.testing.assert_close(fused, reference, atol=atol, rtol=rtol)
                old = reference.detach()
                def objective(logps):
                    ratio = (logps - old).exp()
                    token_loss = -torch.minimum(ratio * advantages[:, None],
                                               ratio.clamp(.8, 1.2) * advantages[:, None])
                    return ((token_loss * mask).sum(-1) / mask.sum(-1)).mean()
                objective(reference).backward()
                objective(fused).backward()
                torch.testing.assert_close(h.grad, hd.grad, atol=atol, rtol=rtol)
                torch.testing.assert_close(w.grad, wd.grad, atol=atol, rtol=rtol)

    def test_lora_receives_nonzero_gradient_with_frozen_lm_head(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(3)
        model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, attention_dropout=0.)), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        dense = deepcopy(model)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        targets, mask = ids[:, 1:], torch.tensor([[1, 0, 0, 1, 1]])
        hidden = model.get_base_model().model(ids, use_cache=False).last_hidden_state[:, :-1]
        fused, _ = self.module.FusedLinearForPPO(chunk_size=2)(
            hidden, model.get_output_embeddings().weight, targets, temperature=.8)
        ref = (dense(ids, use_cache=False).logits[:, :-1] / .8).log_softmax(-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(fused, ref, atol=2e-6, rtol=2e-5)
        (-fused * mask).sum().backward()
        (-ref * mask).sum().backward()
        found = False
        for (name, param), (other_name, other) in zip(model.named_parameters(), dense.named_parameters(), strict=True):
            self.assertEqual(name, other_name)
            if param.requires_grad:
                torch.testing.assert_close(param.grad, other.grad, atol=2e-6, rtol=2e-5)
                found |= bool(torch.any(param.grad != 0))
            else:
                self.assertIsNone(param.grad)
        self.assertTrue(found)

    def test_memory_probe_dense_fused_shift_and_mask_alignment(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from retail_agent.profile_verl_memory import forward_scores
        model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, attention_dropout=0.)), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.eval()
        trace = {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5, 6]}
        dense = forward_scores(model, trace, .8)
        fused = forward_scores(model, trace, .8, self.module.FusedLinearForPPO(chunk_size=2))
        self.assertEqual(tuple(dense.shape), (1, 3))
        torch.testing.assert_close(dense, fused, atol=2e-6, rtol=2e-5)


class NativePolicyLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {"torch": torch, "Optional": Optional, "Any": Any,
            "ActorConfig": SimpleNamespace, "AlgoConfig": type("UnusedAlgoConfig", (), {}),
            "register_policy_loss": lambda name: lambda fn: fn,
            "verl_F": SimpleNamespace(masked_sum=lambda x, m: (x * m).sum(),
                masked_mean=lambda x, m: (x * m).sum() / m.sum())}
        load_source_nodes(VERL / "verl/trainer/ppo/core_algos.py",
                          {"agg_loss", "compute_policy_loss_vanilla"}, cls.ns)

    def config(self, global_size):
        class Config(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)
        return Config(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=1.e10,
                      global_batch_info={"dp_size": 1, "global_batch_size": global_size})

    def test_native_loss_and_gradient_invariant_to_micro_batch_split(self):
        current = torch.tensor([[-.7, -.9, -1.2, -1.], [-1.1, -.8, -1.2, -.6]], requires_grad=True)
        split = current.detach().clone().requires_grad_()
        old, adv = torch.full_like(current, -1), torch.tensor([[1.] * 4, [-1.] * 4])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        fn = self.ns["compute_policy_loss_vanilla"]
        full, _ = fn(old, current, adv, mask, "seq-mean-token-mean", self.config(2))
        full.backward()
        parts = []
        for i in range(2):
            part, _ = fn(old[i:i+1], split[i:i+1], adv[i:i+1], mask[i:i+1],
                         "seq-mean-token-mean", self.config(2))
            part.backward()
            parts.append(part.detach())
        torch.testing.assert_close(full.detach(), sum(parts))
        torch.testing.assert_close(current.grad, split.grad)

    def test_zero_advantage_native_loss_has_finite_zero_gradient(self):
        current = torch.full((1, 3), -1., requires_grad=True)
        loss, _ = self.ns["compute_policy_loss_vanilla"](current.detach(), current,
            torch.zeros_like(current), torch.ones_like(current), "seq-mean-token-mean", self.config(1))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))


class CheckpointAuditTests(unittest.TestCase):
    def test_export_rejects_cast_only_and_accepts_real_change(self):
        from safetensors.torch import save_file
        from retail_agent.audit_verl import export_adapter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, checkpoint = root / "original", root / "checkpoint"
            original.mkdir()
            checkpoint.mkdir()
            name = "base_model.model.layer.lora_A.weight"
            weight = torch.tensor([[1.0001, 2.0001]], dtype=torch.float32)
            save_file({name: weight}, str(original / "adapter_model.safetensors"))
            (original / "adapter_config.json").write_text('{"r":1}')
            torch.save({name: weight.to(torch.bfloat16)}, checkpoint / "model_world_size_1_rank_0.pt")
            with self.assertRaisesRegex(ValueError, "dtype"):
                export_adapter(checkpoint, original, root / "cast_only")
            torch.save({name: weight.to(torch.bfloat16).float()}, checkpoint / "model_world_size_1_rank_0.pt")
            with self.assertRaisesRegex(ValueError, "dtype"):
                export_adapter(checkpoint, original, root / "round_trip_only")
            changed = weight.to(torch.bfloat16) + .25
            torch.save({name: changed}, checkpoint / "model_world_size_1_rank_0.pt")
            torch.save({"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.ones(1)}}},
                       checkpoint / "optim_world_size_1_rank_0.pt")
            result = export_adapter(checkpoint, original, root / "changed")
            self.assertEqual(result["changed_tensors_after_dtype_alignment"], 1)
            self.assertFalse(result["cuda_reload_verified"])
            self.assertEqual(result["optimizer_calls"], 0)


if __name__ == "__main__":
    unittest.main()
