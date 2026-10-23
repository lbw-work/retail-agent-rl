"""Hydra 组合和费用记录检查；需要 hydra-core，不初始化 Ray/CUDA。"""

from importlib.util import find_spec
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from retail_agent.environment import ROOT
from retail_agent.train_verl import build_config, customer_costs, preflight


class CostTests(unittest.TestCase):
    def test_completed_and_unknown_requests_are_both_accounted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            self.assertEqual(customer_costs(path)["api_calls"], 0)
            events = [{"event": "started", "id": "a", "reserved_cny": .2},
                      {"event": "completed", "id": "a", "peak_cost_upper_cny": .1},
                      {"event": "started", "id": "b", "reserved_cny": .3}]
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            result = customer_costs(path)
            self.assertEqual(result["api_calls"], 2)
            self.assertAlmostEqual(result["api_cost_upper_cny"], .4)


@unittest.skipUnless(find_spec("hydra"), "Hydra 组合检查使用独立配置检查依赖")
class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verl = ROOT.parent / "verl-main"
        cls.rows, cls.baseline, cls.project, cls.report = preflight(cls.verl, ["retail_301", "retail_527"])

    def test_composed_config_keeps_training_contract_and_isolates_paths(self):
        config = build_config(self.verl, ROOT / "outputs/verl/config_test", self.baseline, self.project, 2)
        self.assertEqual(config.trainer.v1.trainer_mode, "sync")
        self.assertEqual(config.actor_rollout_ref.actor.loss_agg_mode, "seq-mean-token-mean")
        self.assertEqual(config.actor_rollout_ref.model.lora_rank, 16)
        self.assertEqual(config.actor_rollout_ref.actor.fsdp_config.model_dtype, "bf16")
        self.assertTrue(config.actor_rollout_ref.actor.checkpoint.save_lora_only)
        self.assertTrue(config.actor_rollout_ref.model.use_fused_kernels)
        self.assertEqual(config.actor_rollout_ref.rollout.max_num_seqs, 1)
        self.assertEqual(config.actor_rollout_ref.rollout.n, 4)
        self.assertEqual(config.actor_rollout_ref.rollout.gpu_memory_utilization, .55)
        self.assertFalse(config.trainer.val_before_train)
        self.assertFalse(config.algorithm.use_kl_in_reward)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertIsNone(config.algorithm.rollout_correction.rollout_is)
        self.assertNotIn("logprob_tolerance", config.retail)
        self.assertEqual(config.retail.train_tasks, str(ROOT / self.baseline["train_tasks"]))
        self.assertNotEqual(config.data.train_files[0], config.retail.train_tasks)
        self.assertEqual(self.report["api_calls"], 0)
        self.assertFalse(self.report["training_started"])

    def test_unfused_diagnostic_does_not_change_business_or_loss(self):
        from omegaconf import OmegaConf
        output = ROOT / "outputs/verl/config_test"
        fused = build_config(self.verl, output, self.baseline, self.project, 2)
        dense = build_config(self.verl, output, self.baseline, self.project, 2, fused=False)
        self.assertEqual(fused.retail, dense.retail)
        fused_actor = OmegaConf.to_container(fused.actor_rollout_ref.actor, resolve=True)
        dense_actor = OmegaConf.to_container(dense.actor_rollout_ref.actor, resolve=True)
        # actor 中该字段插值自 model，属于本次唯一受控的执行机制差异。
        fused_actor.pop("use_fused_kernels")
        dense_actor.pop("use_fused_kernels")
        self.assertEqual(fused_actor, dense_actor)
        self.assertFalse(dense.actor_rollout_ref.model.use_fused_kernels)

    def test_calibration_config_disables_checkpoint_and_learning_gate(self):
        config = build_config(self.verl, ROOT / "outputs/verl/calibration_test",
                              self.baseline, self.project, 1, calibration_only=True)
        self.assertTrue(config.retail.calibration_only)
        self.assertFalse(config.retail.require_nonzero_gradient)
        self.assertEqual(config.trainer.save_freq, -1)

    def test_native_profile_uses_new_base_identity_adapter_and_same_memory_limits(self):
        native = tomllib.loads((ROOT / 'configs/verl/mmopd.toml').read_text())
        config = build_config(self.verl, ROOT / 'outputs/verl/native_config_test',
                              {**self.baseline, **native}, self.project, 94)
        self.assertEqual(config.actor_rollout_ref.model.path, str(ROOT / native['model_path']))
        self.assertEqual(config.actor_rollout_ref.model.lora_adapter_path, str(ROOT / native['sft_adapter_dir']))
        self.assertNotEqual(config.actor_rollout_ref.model.lora_adapter_path,
                            str(ROOT / self.baseline['sft_adapter_dir']))
        self.assertTrue(config.retail.enable_thinking)
        self.assertEqual(config.retail.max_new_tokens_per_action, 4096)
        self.assertEqual(config.actor_rollout_ref.rollout.temperature, 1.0)
        self.assertEqual(config.actor_rollout_ref.rollout.top_p, 1.0)
        self.assertEqual(config.actor_rollout_ref.rollout.top_k, -1)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 16384)
        self.assertEqual(config.actor_rollout_ref.rollout.max_num_seqs, 1)
        self.assertEqual(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.param_offload)
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.optimizer_offload)
        self.assertEqual(config.trainer.total_training_steps, 94)


@unittest.skipUnless(find_spec("tensordict"), "真实 TensorDict 边界检查需要 tensordict==0.10.0")
class TensorDictBoundaryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT.parent / "verl-main/verl/utils/tensordict_utils.py"
        spec = importlib.util.spec_from_file_location("retail_test_real_td", path)
        cls.tu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tu)

    async def test_real_collated_metadata_reaches_worker(self):
        from unittest.mock import AsyncMock
        import numpy as np
        from tensordict import NonTensorData, NonTensorStack
        from test_verl import runtime_namespace
        ns = runtime_namespace()
        ns.update(NonTensorData=NonTensorData, NonTensorStack=NonTensorStack)
        worker = object.__new__(ns["RetailGroupWorker"])
        worker._run_group = AsyncMock()
        data = self.tu.get_tensordict({"uid": np.array(["uid"], dtype=object),
            "extra_info": np.array([{"task_id": "retail_301"}], dtype=object),
            "index": np.array([0], dtype=object)})
        self.tu.assign_non_tensor_data(data, "global_steps", 1)
        await worker.generate_sequences(data)
        prompt = worker._run_group.call_args.args[0]
        self.assertEqual(prompt["extra_info"], {"task_id": "retail_301"})
        self.assertEqual(prompt["global_steps"], 1)

    async def test_real_ragged_serialization_keeps_context_and_loss_masks(self):
        from test_verl import runtime_namespace
        ns = runtime_namespace()
        ns["list_of_dict_to_tensordict"] = self.tu.list_of_dict_to_tensordict
        worker = object.__new__(ns["RetailGroupWorker"])
        worker.retail = {"reward_mode": "reward_a"}
        results = [{"prompt_ids": [1, 2], "completion_ids": [3, 4, 5], "assistant_mask": [1, 0, 1],
                    "logprobs": [-1., 0., -2.], "scores": {"reward_a": 1}, "assistant_actions": 2,
                    "policy_version": 0},
                   {"prompt_ids": [1, 2], "completion_ids": [3, 4, 5, 6], "assistant_mask": [1, 0, 0, 1],
                    "logprobs": [-1., 0., 0., -2.], "scores": {"reward_a": 0}, "assistant_actions": 2,
                    "policy_version": 0}]
        await worker._publish("uid", 1, {"uid": "uid", "global_steps": 1}, results)
        fields = ns["tq"].async_kv_batch_put.call_args.kwargs["fields"]
        self.assertTrue(fields["input_ids"].is_nested)
        self.assertEqual(fields["input_ids"][1].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(fields["loss_mask"][1].tolist(), [1, 0, 0, 1])
        self.assertEqual(fields["attention_mask"][1].tolist(), [1] * 6)


if __name__ == "__main__":
    unittest.main()
