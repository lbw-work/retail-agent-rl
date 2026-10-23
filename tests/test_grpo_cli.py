"""GPU 启动前的参数与覆盖保护；不连接服务器或顾客 API。"""

import importlib.util
import tempfile
import json
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from retail_agent.environment import ROOT
from retail_agent.train_grpo import validate_run_args


class GRPOCLITests(unittest.TestCase):
    def arguments(self, **changes):
        fields = dict(smoke=True, max_steps=1, max_minutes=5., api_budget_cny=.15,
            output_dir='outputs/grpo/smoke', task_id='retail_174')
        return SimpleNamespace(**{**fields, **changes})

    def test_explicit_limits_and_smoke_task_are_required(self):
        config = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
        validate_run_args(self.arguments(), config)
        validate_run_args(self.arguments(api_budget_cny=None), config)
        for change in [dict(max_steps=None), dict(max_steps=config["max_steps"] + 1), dict(max_minutes=float('nan')),
                       dict(api_budget_cny=float('inf')), dict(api_budget_cny=0), dict(task_id=None),
                       dict(max_steps=2), dict(smoke=False)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_run_args(self.arguments(**change), config)
        validate_run_args(self.arguments(smoke=False, task_id=None, max_steps=94), config)
        with self.assertRaises(ValueError):
            validate_run_args(self.arguments(smoke=False, task_id=None, max_steps=95), config)

    @unittest.skipUnless(importlib.util.find_spec('torch'), '任务覆盖检查使用 TRL 训练环境')
    def test_first_94_updates_cover_all_training_tasks_once(self):
        from trl.trainer.utils import RepeatSampler
        config = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
        rows = [json.loads(line) for line in (ROOT / config['train_tasks']).read_text().splitlines()]
        self.assertEqual(config['max_steps'], len(rows))
        # 当前单卡：每批4行、每4微步生成一组并更新一次；其余重复批复用缓存。
        sampled = list(RepeatSampler(rows, mini_repeat_count=4, batch_size=1,
            repeat_count=4, shuffle=True, seed=config['seed']))
        batches = [sampled[i:i + 4] for i in range(0, len(sampled), 4)]
        generated = batches[::4][:config['max_steps']]
        self.assertTrue(all(len(set(group)) == 1 for group in generated))
        self.assertEqual(sorted(group[0] for group in generated), list(range(len(rows))))
        self.assertEqual(sum(map(len, generated)), 376)

    @unittest.skipUnless(importlib.util.find_spec('torch'), '启动边界测试使用训练环境')
    def test_existing_output_is_rejected_before_model_or_api(self):
        from retail_agent.train_grpo import run_cuda
        parent = ROOT / 'outputs/grpo'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as load, \
                 patch('retail_agent.customer.DeepSeekCustomer') as customer:
                with self.assertRaises(FileExistsError):
                    run_cuda(self.arguments(output_dir=temporary))
                load.assert_not_called()
                customer.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CUDA 检查使用训练环境')
    def test_no_cuda_is_rejected_without_model_or_api(self):
        from retail_agent.train_grpo import run_cuda
        parent = ROOT / 'outputs/grpo'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as load, \
                 patch('retail_agent.customer.DeepSeekCustomer') as customer:
                with self.assertRaisesRegex(ValueError, 'CUDA'):
                    run_cuda(self.arguments(output_dir=str(Path(temporary) / 'new')))
                load.assert_not_called()
                customer.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU 小模型启动编排测试需要训练环境')
    def test_entry_orchestration_with_real_cpu_update_and_mocked_gpu_api_boundaries(self):
        # 仅把设备/量化加载/API边界替换为 fixture；实际运行 TRL 反向、更新与保存。
        import torch
        from importlib.metadata import version
        from unittest.mock import Mock
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from retail_agent.train_grpo import run_cuda
        from retail_agent.grpo_trainer import make_grpo_args
        from retail_agent.experiment_protocol import check_protocol
        from test_grpo import GRPOFrameworkTests
        torch.set_num_threads(2)
        config = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=16384, pad_token_id=151643, eos_token_id=151645)
        config._attn_implementation = 'eager'
        base = Qwen3ForCausalLM(config)
        def adapter(model, directory, timing=None):
            return get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
                target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type='CAUSAL_LM'))
        def sampled(trainer, prompts):
            return GRPOFrameworkTests().fixture(trainer, [0., 1., 0., 1.])
        checked = {'model_path': str(ROOT / 'models/qwen3-4b'),
            'sft_adapter_dir': 'fixture_not_loaded', 'sft_adapter_sha256': 'fixture_not_real_sft',
            'protocol_sha256': check_protocol()['sha256']}
        probe = Mock(accounted_cny=Mock(return_value=11.))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs/grpo') as temporary:
            output = Path(temporary) / 'run'
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.device_count', return_value=1), \
                 patch('torch.cuda.is_bf16_supported', return_value=True), \
                 patch('torch.cuda.get_device_name', return_value='CPU_TEST_DEVICE_DOUBLE'), \
                 patch('torch.cuda.synchronize'), \
                 patch('torch.cuda.max_memory_allocated', return_value=0), \
                 patch('torch.cuda.max_memory_reserved', return_value=0), \
                 patch('torch.cuda.random.get_rng_state', return_value=torch.zeros(1, dtype=torch.uint8)), \
                 patch('retail_agent.train_grpo.version', side_effect=lambda n: '0.48.1' if n == 'bitsandbytes' else version(n)), \
                 patch('retail_agent.train_grpo.preflight', return_value=checked), \
                 patch('transformers.BitsAndBytesConfig'), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=base), \
                 patch('retail_agent.train_grpo.load_sft_adapter', side_effect=adapter), \
                 patch('retail_agent.customer.DeepSeekCustomer', return_value=probe), \
                 patch('retail_agent.grpo_trainer.make_grpo_args', side_effect=lambda c,o,s: make_grpo_args(c,o,s,cpu=True)), \
                 patch('retail_agent.grpo_trainer.RetailGRPOTrainer._generate', sampled):
                run_cuda(self.arguments(output_dir=str(output), api_budget_cny=None))
            run = json.loads((output / 'run.json').read_text())
            self.assertEqual((run['status'], run['global_step']), ('completed', 1))
            self.assertEqual(run['execution_precision'], 'float32')
            self.assertNotEqual(run['lora_before_sha256'], run['lora_after_sha256'])
            self.assertEqual(run['customer_config']['goal_policy'], 'scenario_locked_v1')
            self.assertIsNone(run['api_additional_limit_cny'])
            self.assertFalse(run['api_cost_limit_enforced'])
            self.assertTrue((output / 'adapter/adapter_model.safetensors').is_file())
            self.assertTrue((output / 'checkpoint-1/optimizer.pt').is_file())


if __name__ == '__main__':
    unittest.main()
