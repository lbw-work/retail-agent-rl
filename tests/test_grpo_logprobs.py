"""同权重同 token 的数值回归；小模型 CPU 结果不能替代真实 NF4 CUDA 验收。"""

import json
from importlib.util import find_spec
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if find_spec('torch'):
    import torch
    from accelerate.utils import convert_outputs_to_fp32
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from retail_agent.audit_grpo_logprobs import read_traces, run_matrix, validate_trace
    from retail_agent.grpo_rollout import RolloutError, TorchSampler
    from retail_agent.grpo_trainer import text_policy_logprobs


def tiny_bf16(vocab_size=128, pad_token_id=0, eos_token_id=127):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=512, pad_token_id=pad_token_id, eos_token_id=eos_token_id, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = get_peft_model(Qwen3ForCausalLM(config).to(torch.bfloat16),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.1)
    return model


@unittest.skipUnless(find_spec('torch'), '概率数值回归使用 .venv-train')
class LogProbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_peft_generation_bypasses_forward_autocast_and_explicit_context_repairs_it(self):
        model = tiny_bf16()
        # 精确复现 Accelerate.prepare_model 对外层 forward 的包装，不改变全局 Accelerator 状态。
        model.forward = convert_outputs_to_fp32(torch.autocast('cpu', dtype=torch.bfloat16)(model.forward))
        tokenizer = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda _: 127)
        prompt = [1, 5, 8, 7, 10, 13, 22, 39]
        states = []
        hook = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.register_forward_hook(
            lambda module, args, output: states.append(torch.is_autocast_enabled('cpu')))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        legacy = TorchSampler(model, tokenizer, .8).generate(prompt, 6)
        self.assertTrue(states and not any(states))
        states.clear()
        ids = torch.tensor([prompt + legacy['token_ids']])
        with torch.no_grad():
            recomputed, _ = text_policy_logprobs(model, ids, torch.ones_like(ids), len(legacy['token_ids']), .8)
        self.assertTrue(all(states))
        self.assertGreater((torch.tensor(legacy['logprobs']) - recomputed[0]).abs().max(), 1e-4)
        states.clear()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            fixed = TorchSampler(model, tokenizer, .8).generate(prompt, 6)
        self.assertTrue(all(states))
        ids = torch.tensor([prompt + fixed['token_ids']])
        with torch.no_grad():
            recomputed, _ = text_policy_logprobs(model, ids, torch.ones_like(ids), len(fixed['token_ids']), .8)
        torch.testing.assert_close(torch.tensor(fixed['logprobs']), recomputed[0], atol=1e-6, rtol=1e-6)
        hook.remove()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_fp32_temperature_and_causal_shift_with_gradient(self):
        # 使用低精度 logits，证明转 FP32 发生在温度缩放之前，而且不切断梯度。
        logits = torch.stack([torch.linspace(-8, 8, 128) + .2 * torch.sin(torch.arange(128.) * i)
                              for i in range(5)])[None].to(torch.bfloat16).requires_grad_()
        def forward(**kwargs):
            self.assertEqual(kwargs['logits_to_keep'], 4)
            self.assertFalse(kwargs['use_cache'])
            return SimpleNamespace(logits=logits[:, -4:])
        ids = torch.tensor([[1, 7, 19, 35, 70]])
        actual, entropy = text_policy_logprobs(forward, ids, torch.ones_like(ids), 3, .8, compute_entropy=True)
        expected = torch.log_softmax(logits[:, 1:4].float() / .8, -1).gather(-1, ids[:, 2:, None]).squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
        native_low_precision = torch.log_softmax(logits[:, 1:4] / .8, -1).gather(-1, ids[:, 2:, None]).squeeze(-1)
        self.assertGreater((actual - native_low_precision.float()).abs().max(), .01)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(entropy).all())
        actual.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.any(logits.grad != 0))

    def trace(self):
        return {'prompt_ids': [1, 5, 8], 'completion_ids': [9, 10, 40, 41, 12, 13],
                'assistant_mask': [1, 1, 0, 0, 1, 1],
                'logprobs': [-5., -5., 0., 0., -5., -5.],
                'turns': [{'start': 0, 'end': 2}, {'start': 4, 'end': 6}]}

    def test_trace_alignment_and_history_are_read_only(self):
        trace = self.trace()
        validate_trace(trace)
        for key, value in [('assistant_mask', [1] * 6), ('logprobs', [-5.] * 6),
                           ('turns', [{'start': 1, 'end': 9}]), ('completion_ids', [9])]:
            with self.subTest(key=key), self.assertRaises(RolloutError):
                validate_trace({**trace, key: value})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'history.jsonl'
            path.write_text('\n'.join(map(json.dumps, [
                {'event': 'run_manifest', 'torch_version': 'fixture'},
                {'event': 'rollout_trace', 'sampled_policy_step': 0, 'trace': trace}])))
            before = path.read_bytes()
            traces, _ = read_traces(path)
            self.assertEqual(traces, [trace])
            self.assertEqual(path.read_bytes(), before)

    def test_same_weights_tokens_cache_prefix_full_sdpa_eager_matrix(self):
        model = tiny_bf16()
        trace = self.trace()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        matrix = run_matrix(model, trace, 5, 9, .8)
        self.assertEqual(len(matrix), 24)
        self.assertTrue(all(torch.isfinite(torch.tensor(r['logprob'])) for r in matrix))
        for attention in ['sdpa', 'eager']:
            rows = [r for r in matrix if r['attention'] == attention and r['autocast_bf16']
                    and r['mode'] != 'full_trl_native']
            values = [r['logprob'] for r in rows]
            self.assertLess(max(values) - min(values), .02)
        self.assertEqual(model.config._attn_implementation, 'eager')
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            run_matrix(model, trace, 3, 9, .8)

    def test_production_generate_enters_accelerator_autocast(self):
        from test_grpo import GRPOFrameworkTests
        GRPOFrameworkTests.setUpClass()
        factory = GRPOFrameworkTests()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = factory.trainer(temporary)
            trace = {**self.trace(), 'task_id': 'retail_264', 'termination': 'fixture',
                     'scores': {'reward_a': 0.}}
            def generate(*args, **kwargs):
                self.assertTrue(torch.is_autocast_enabled('cpu'))
                return [trace] * 4
            with patch.object(trainer.accelerator, 'autocast',
                              side_effect=lambda: torch.autocast('cpu', dtype=torch.bfloat16)), \
                 patch('retail_agent.grpo_trainer.generate_groups', side_effect=generate):
                trainer._generate(['retail_264'] * 4)
            self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_offline_special_tokens_match_trainer_and_focused_four_paths(self):
        from transformers import Trainer
        model = tiny_bf16()
        model.config.pad_token_id = None
        model.config.bos_token_id = 1
        tokenizer = SimpleNamespace(pad_token_id=0, bos_token_id=None, eos_token_id=127)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        Trainer._align_special_tokens(SimpleNamespace(model=model, processing_class=tokenizer))
        self.assertEqual(model.config.pad_token_id, 0)
        self.assertIsNone(model.config.bos_token_id)
        events = []
        matrix = run_matrix(model, self.trace(), 5, 9, .8, record=events.append, focused=True)
        self.assertEqual(events, matrix)
        self.assertEqual([r['mode'] for r in matrix], ['cache', 'prefix', 'full_selected', 'full_training'])
        self.assertTrue(all(r['attention'] == 'sdpa' and r['autocast_bf16'] for r in matrix))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_fp16_preparation_preserves_fp32_lora_and_audits_same_precision(self):
        from retail_agent.train_grpo import prepare_grpo_qlora
        base = Qwen3ForCausalLM(tiny_bf16().config).to(torch.bfloat16)
        base = prepare_grpo_qlora(base, torch.float16)
        self.assertTrue(all(not p.requires_grad and p.dtype == torch.float16 for p in base.parameters()))
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=.1)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        matrix = run_matrix(model, self.trace(), 5, 9, .8, focused=True)
        self.assertTrue(all(r['autocast_enabled'] and not r['autocast_bf16']
                            and r['autocast_dtype'] == 'torch.float16' for r in matrix))
        values = [r['logprob'] for r in matrix]
        self.assertTrue(torch.isfinite(torch.tensor(values)).all())
        self.assertLess(max(values) - min(values), .02)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_fp32_execution_preserves_weights_and_has_finite_lora_backward(self):
        from retail_agent.train_grpo import prepare_grpo_qlora, grpo_execution_math
        base = prepare_grpo_qlora(Qwen3ForCausalLM(tiny_bf16().config).to(torch.bfloat16), torch.float32)
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with grpo_execution_math(model):
            matrix = run_matrix(model, self.trace(), 5, 9, .8, focused=True)
            self.assertFalse(any(row['autocast_enabled'] for row in matrix))
            self.assertLess(max(row['logprob'] for row in matrix) - min(row['logprob'] for row in matrix), 1e-5)
            ids = torch.tensor([self.trace()['prompt_ids'] + self.trace()['completion_ids']])
            logps, _ = text_policy_logprobs(model, ids, torch.ones_like(ids), 6, .8)
            (-logps[:, [0, 1, 4, 5]].mean()).backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.any(g != 0) for g in gradients))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_trainer_preserves_exact_sft_values_despite_trl_adapter_downcast(self):
        from trl import GRPOTrainer
        from test_grpo import GRPOFrameworkTests
        GRPOFrameworkTests.setUpClass()
        factory = GRPOFrameworkTests()
        original_init = GRPOTrainer.__init__
        captured = []
        def downcasting_init(trainer, *args, **kwargs):
            original_init(trainer, *args, **kwargs)
            for p in trainer.model.parameters():
                if p.requires_grad:
                    captured.append((p, p.detach().clone()))
                    p.data = p.data.to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary, patch.object(GRPOTrainer, '__init__', downcasting_init):
            trainer = factory.trainer(temporary)
            self.assertTrue(captured)
            for parameter, initial in captured:
                self.assertEqual(parameter.dtype, torch.float32)
                torch.testing.assert_close(parameter, initial, atol=0, rtol=0)

    @unittest.skipUnless(find_spec('torch') and torch.cuda.is_available(), '真实 NF4 分支检查需要 CUDA')
    def test_cuda_nf4_inference_training_parity_and_context_restoration(self):
        import bitsandbytes as bnb
        from transformers.integrations import sdpa_attention
        from retail_agent.train_grpo import grpo_execution_math
        linear = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float32,
                                  quant_type='nf4').cuda()
        model = SimpleNamespace(device=torch.device('cuda'),
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0, dtype=torch.float32)))
        matmul, gqa = bnb.matmul_4bit, sdpa_attention.use_gqa_in_sdpa
        x = torch.randn(1, 1, 64, device='cuda', requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'restore-context'):
            with grpo_execution_math(model):
                self.assertIsNot(bnb.matmul_4bit, matmul)
                with torch.no_grad():
                    sampled = linear(x.detach())
                trained = linear(x)
                torch.testing.assert_close(sampled, trained, atol=1e-6, rtol=1e-6)
                trained.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all() and torch.any(x.grad != 0))
                raise RuntimeError('restore-context')
        self.assertIs(bnb.matmul_4bit, matmul)
        self.assertIs(sdpa_attention.use_gqa_in_sdpa, gqa)

    def test_bf16_cpu_real_grpo_update_changes_lora_and_preserves_base(self):
        from datasets import Dataset
        from test_grpo import GRPOFrameworkTests
        from retail_agent.grpo_trainer import RetailGRPOTrainer, make_grpo_args
        GRPOFrameworkTests.setUpClass()
        factory = GRPOFrameworkTests()
        model = tiny_bf16(vocab_size=len(factory.tokenizer), pad_token_id=factory.tokenizer.pad_token_id,
                          eos_token_id=factory.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        model.forward = convert_outputs_to_fp32(torch.autocast('cpu', dtype=torch.bfloat16)(model.forward))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as temporary:
            trainer = RetailGRPOTrainer(model=model, processing_class=factory.tokenizer,
                args=make_grpo_args(factory.config, temporary, 1, cpu=True),
                rows=factory.rows, rollout_config=factory.config, customer_factory=lambda _: None,
                train_dataset=Dataset.from_list([{'prompt': key} for key in factory.rows]))
            fixture = factory.fixture(trainer, [0., 1., 0., 1.])
            with patch.object(trainer, '_generate', return_value=fixture):
                trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad))
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.isfinite(parameter).all())
                if not parameter.requires_grad:
                    torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
