"""新SFT原生思考、快照校验和零增量LoRA；CPU夹具不是GPU效果验证。"""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import AutoTokenizer

from retail_agent.environment import ROOT
from retail_agent.grpo_rollout import TokenTape
from retail_agent.mmopd import (
    NativeAgent, REPO, REVISION, ThinkingTokenizer, download, init_adapter, verify_snapshot, visible_action,
)
from retail_agent.model import parse_action
from retail_agent.observability import StageTimer
from retail_agent.verl_agent_loop import run_episode


class ThinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/raw/tokenizer', local_files_only=True)

    def test_only_closed_leading_thinking_is_removed_from_action_view(self):
        action = '<tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}</tool_call>'
        self.assertEqual(visible_action('<think>fixture</think>\n' + action), action)
        self.assertEqual(parse_action(visible_action('<think>fixture</think>\n' + action))['kind'], 'tool')
        for text in ['<think>unclosed' + action, '<think>a<think>b</think>' + action,
                     '<think>a</think><think>b</think>' + action, '</think>' + action]:
            with self.assertRaises(ValueError):
                parse_action(visible_action(text))
        with self.assertRaises(ValueError):
            parse_action(visible_action('<think>fixture</think>' + action + action))

    def test_native_template_full_tokens_and_feedback_masks_are_preserved(self):
        tokenizer = self.tokenizer
        wrapper = ThinkingTokenizer(tokenizer)
        messages = [{'role': 'user', 'content': 'Please help.'}]
        tape = TokenTape(wrapper, messages, [], 16384)
        expected = tokenizer.apply_chat_template(messages, tools=[], tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
        self.assertEqual(tape.prompt_ids, expected)
        self.assertNotEqual(expected, tokenizer.apply_chat_template(messages, tools=[], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        raw = '<think>fixture reasoning</think>\nHello.<|im_end|>'
        ids = tokenizer.encode(raw, add_special_tokens=False)
        tape.sampled({'token_ids': ids, 'logprobs': [-1.] * len(ids)})
        before = list(tape.ids)
        self.assertEqual(wrapper.decode(ids[:-1], skip_special_tokens=False), 'Hello.')
        tape.feedback({'role': 'user', 'content': 'Thanks'}, continue_generation=True)
        self.assertEqual(tape.ids[:len(before)], before)
        self.assertEqual(tape.mask[:len(ids)], [1] * len(ids))
        self.assertEqual(tape.mask[len(ids):], [0] * (len(tape.mask) - len(ids)))
        self.assertIn('<think>fixture reasoning</think>', tokenizer.decode(tape.ids))

    def test_native_inference_retains_previous_sampled_thinking_and_resets(self):
        tokenizer = self.tokenizer
        responses = [tokenizer.encode('<think>fixture</think>Hello.<|im_end|>', add_special_tokens=False),
                     tokenizer.encode('<think>again</think>Bye.<|im_end|>', add_special_tokens=False)]
        model = Mock(device=torch.device('cpu'))
        generated = iter(responses)
        model.generate.side_effect = lambda **kw: torch.cat(
            [kw['input_ids'], torch.tensor([next(generated)])], dim=1)
        agent = NativeAgent(model, tokenizer, {'max_trajectory_tokens': 16384,
            'max_new_tokens_per_action': 4096, 'temperature': 1.0})
        first = agent.generate([{'role': 'user', 'content': 'Hi'}], [])
        second = agent.generate([{'role': 'user', 'content': 'Thanks'}], [])
        prefix = first['prompt_token_ids'] + first['generated_token_ids']
        self.assertEqual(second['prompt_token_ids'][:len(prefix)], prefix)
        self.assertEqual(first['raw_text'], 'Hello.')
        self.assertIn('<think>', first['raw_model_text'])
        self.assertFalse('generated_logprobs' in first)
        agent.reset()
        self.assertIsNone(agent.ids)


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_environment_native_thinking_keeps_all_sampled_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/raw/tokenizer', local_files_only=True)
        config = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
        config.update(enable_thinking=True, max_new_tokens_per_action=4096)
        row = next(r for r in map(json.loads, (ROOT / config['train_tasks']).read_text().splitlines())
                   if r['id'] == 'retail_264')
        texts = ['<think>fixture reasoning</think><tool_call>' + json.dumps(
            {'name': a['name'], 'arguments': a['arguments']}) + '</tool_call><|im_end|>'
            for a in row['evaluation_criteria']['actions']] + ['<think>fixture</think>Done.<|im_end|>']
        encoded = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        iterator = iter(encoded)
        async def generate(**kwargs):
            ids = next(iterator)
            return SimpleNamespace(token_ids=ids, log_probs=[-1.] * len(ids),
                stop_reason='completed', extra_fields={'global_steps': 0})
        customer = Mock(reply=Mock(return_value={'text': '###STOP###'}))
        result = await run_episode(row, SimpleNamespace(generate=generate), tokenizer,
            lambda scenario: customer, 'Please help.', config,
            {'temperature': 1., 'top_p': 1., 'top_k': -1}, StageTimer(), 'native-test')
        self.assertEqual(result['scores']['reward_a'], 1.)
        self.assertTrue(result['enable_thinking'])
        self.assertEqual(len(result['raw_model_turns']), len(texts))
        for ids, turn in zip(encoded, result['turns'], strict=True):
            self.assertEqual(result['completion_ids'][turn['start']:turn['end']], ids)
            self.assertEqual(result['assistant_mask'][turn['start']:turn['end']], [1] * len(ids))


class SnapshotTests(unittest.TestCase):
    def fixture(self, folder):
        config = {'model_id': REPO, 'model_revision': REVISION, 'enable_thinking': True,
                  'model_path': 'models/new', 'sft_adapter_dir': 'models/init',
                  'experiment_profile': 'mmopd_native_think_v1', 'data_exposure': 'test', 'seed': 42}
        (folder / 'configs/verl').mkdir(parents=True)
        profile = folder / 'configs/verl/mmopd.toml'
        profile.write_text('fixture')
        payload = {'config.json': b'{}', 'tokenizer_config.json': b'{}', 'tokenizer.json': b'{}',
                   'chat_template.jinja': b'fixture',
                   'model.safetensors.index.json': json.dumps({'weight_map': {'w': 'model-1.safetensors'}}).encode(),
                   'model-1.safetensors': b'fixture weights, not a real model'}
        siblings = []
        for name, data in payload.items():
            sha = hashlib.sha256(data).hexdigest()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, blob_id=blob,
                lfs=SimpleNamespace(sha256=sha) if name.endswith('.safetensors') else None))
        def fetch(*args, **kwargs):
            target = Path(kwargs['local_dir'])
            target.mkdir(parents=True, exist_ok=True)
            for name, data in payload.items():
                (target / name).write_bytes(data)
        return config, profile, payload, SimpleNamespace(sha=REVISION, siblings=siblings), fetch

    def test_hub_hashes_are_verified_and_local_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, profile_path, payload, info, fetch = self.fixture(root)
            with patch('retail_agent.mmopd.ROOT', root), patch('retail_agent.mmopd.PROFILE', profile_path), \
                 patch('retail_agent.mmopd.profile', return_value=config), \
                 patch('huggingface_hub.HfApi') as api, \
                 patch('huggingface_hub.snapshot_download', side_effect=fetch):
                api.return_value.model_info.return_value = info
                report = download()
                self.assertEqual(report['model_revision'], REVISION)
                (root / 'models/new/model-1.safetensors').write_bytes(b'corrupted')
                with self.assertRaises(ValueError):
                    verify_snapshot()

    def test_corrupt_upstream_weight_never_gets_a_verified_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, profile_path, payload, info, fetch = self.fixture(root)
            payload['model-1.safetensors'] = b'corrupted download'
            with patch('retail_agent.mmopd.ROOT', root), patch('retail_agent.mmopd.PROFILE', profile_path), \
                 patch('retail_agent.mmopd.profile', return_value=config), \
                 patch('huggingface_hub.HfApi') as api, \
                 patch('huggingface_hub.snapshot_download', side_effect=fetch):
                api.return_value.model_info.return_value = info
                with self.assertRaisesRegex(ValueError, 'LFS'):
                    download()
                self.assertFalse((root / 'models/new/SOURCE_MANIFEST.json').exists())

    def test_fresh_lora_is_identity_and_records_real_initial_values(self):
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3ForCausalLM
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16))
        model.eval()
        inputs = torch.tensor([[1, 2]])
        with torch.no_grad():
            before = model(inputs).logits.clone()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, profile_path, _, _, _ = self.fixture(root)
            with patch('retail_agent.mmopd.ROOT', root), patch('retail_agent.mmopd.PROFILE', profile_path), \
                 patch('retail_agent.mmopd.profile', return_value=config), \
                 patch('retail_agent.mmopd.verify_snapshot', return_value={'model_manifest_sha256': 'fixture'}), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model):
                init_adapter()
            target = root / config['sft_adapter_dir']
            tensors = load_file(str(target / 'adapter_model.safetensors'))
            self.assertTrue(all(torch.count_nonzero(t) == 0 for k, t in tensors.items() if 'lora_B' in k))
            self.assertTrue(json.loads((target / 'initialization.json').read_text())['identity_adapter'])
            with torch.no_grad():
                self.assertTrue(torch.equal(before, model(inputs).logits))


if __name__ == '__main__':
    unittest.main()
