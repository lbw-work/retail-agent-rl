"""CUDA评选的完整性、续跑及选择规则；不代表真实模型/API评估。"""

import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from retail_agent.eval_verl import (
    CudaGreedyAgent, candidate_score, completed_tasks, digest, evaluate_candidate, prepare, select_best,
)
from retail_agent.environment import ROOT
from retail_agent.model import ContextLimit


def result(task, reward=0, eligible=True):
    return {'task_id': task, 'business_evaluation_eligible': eligible,
            'reward': {'reward': reward} if eligible else None, 'diagnostic': 'test_fault'}


class SelectionTests(unittest.TestCase):
    def test_business_failure_counts_in_denominator(self):
        score = candidate_score(1, [result('a', 1), result('b', 0)], ['a', 'b'])
        self.assertEqual(score['successes'], 1)
        self.assertEqual(score['success_rate'], .5)

    def test_incomplete_or_system_fault_cannot_win(self):
        for attempts in [[result('a', 1)], [result('a', 1), result('b', eligible=False)]]:
            with self.assertRaisesRegex(ValueError, '未完整'):
                candidate_score(1, attempts, ['a', 'b'])

    def test_no_business_failure_retry_or_unplanned_tasks(self):
        for attempts in [[result('a', 0), result('a', 1)], [result('test_task', 1)]]:
            with self.assertRaises(ValueError):
                completed_tasks(attempts, ['a'])

    def test_fault_retry_retains_fault_count(self):
        score = candidate_score(1, [result('a', eligible=False), result('a', 1)], ['a'])
        self.assertEqual(score['successes'], 1)
        self.assertEqual(score['system_fault_attempts'], 1)

    def test_missing_checkpoint_or_baseline_cannot_publish_best(self):
        for steps in [[0, 2], [1, 2], [0, 1, 1]]:
            with self.assertRaises(ValueError):
                select_best([{'step': s, 'planned': 16, 'successes': 9} for s in steps], 2)

    def test_tie_keeps_sft_and_earliest_checkpoint(self):
        scores = [{'step': s, 'planned': 16, 'successes': 9} for s in range(3)]
        self.assertEqual(select_best(scores, 2)['step'], 0)
        scores[1]['successes'] = scores[2]['successes'] = 10
        self.assertEqual(select_best(scores, 2)['step'], 1)

    def test_invalid_rewards_cannot_rank(self):
        for reward in [float('nan'), float('inf'), -1, 2, '1']:
            with self.assertRaises(ValueError):
                candidate_score(1, [result('a', reward)], ['a'])

    def test_predeclared_subset_requires_all_its_candidates_and_sft(self):
        scores = [{'step': s, 'planned': 16, 'successes': 9} for s in [0, 10, 20, 94]]
        self.assertEqual(select_best(scores, 94, [10, 20, 94])['step'], 0)
        with self.assertRaises(ValueError):
            select_best(scores, 94)
        with self.assertRaises(ValueError):
            select_best(scores[:-1], 94, [10, 20, 94])


class PreflightTests(unittest.TestCase):
    def test_full_pool_and_all_checkpoint_files_are_required(self):
        baseline = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
        original = ROOT / baseline['sft_adapter_dir'] / 'adapter_model.safetensors'
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'run.json').write_text(json.dumps({
                'status': 'training_returned_pending_reload_acceptance', 'max_steps': 94,
                'calibration_only': False,
                'sft_adapter_sha256': {'adapter_model.safetensors': digest(original),
                    'adapter_config.json': digest(original.parent / 'adapter_config.json')}}))
            (run / 'train_tasks.jsonl').write_bytes((ROOT / baseline['train_tasks']).read_bytes())
            for step in range(1, 95):
                folder = run / f'checkpoints/global_step_{step}/actor'
                folder.mkdir(parents=True)
                for name in ['model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt']:
                    (folder / name).write_bytes(b'test fixture, not a real checkpoint')
            protocol, rows, _, _ = prepare(run, 94)
            self.assertEqual(len(protocol['candidates']), 95)
            self.assertEqual(len(rows), 16)
            self.assertFalse(protocol['test_used'])
            self.assertNotIn('api_additional_limit_cny', protocol['evaluation_config'])
            (run / 'checkpoints/global_step_50/actor/optim_world_size_1_rank_0.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(run, 94)


class EvaluationLoopTests(unittest.TestCase):
    def test_resume_skips_business_results_but_retries_fault(self):
        protocol = {'task_ids': ['a', 'b'], 'seed': 42, 'customer': {},
                    'evaluation_config': {'max_assistant_actions': 24, 'max_consecutive_tool_errors': 3}}
        rows = [{'id': s, 'user_scenario': {}} for s in ['a', 'b']]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            trace = folder / 'traces.jsonl'
            trace.write_text('\n'.join(json.dumps(x) for x in
                [result('a', 0), result('b', eligible=False)]) + '\n')
            with patch('transformers.AutoTokenizer.from_pretrained'), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained'), \
                 patch('peft.PeftModel.from_pretrained'), \
                 patch('retail_agent.customer.DeepSeekCustomer') as customer, \
                 patch('retail_agent.rollout.run_episode', return_value=result('b', 1)) as episode:
                attempts = evaluate_candidate(folder, folder, rows, protocol, folder / 'ledger', 10)
            self.assertEqual(episode.call_count, 1)
            self.assertEqual(episode.call_args.args[0]['id'], 'b')
            self.assertEqual(customer.return_value.http.close.call_count, 1)
            score = candidate_score(1, attempts, ['a', 'b'])
            self.assertEqual(score['success_rate'], .5)
            self.assertEqual(score['system_fault_attempts'], 1)

    def test_fault_is_persisted_and_stops_batch(self):
        protocol = {'task_ids': ['a', 'b'], 'seed': 42, 'customer': {},
                    'evaluation_config': {'max_assistant_actions': 24, 'max_consecutive_tool_errors': 3}}
        rows = [{'id': s, 'user_scenario': {}} for s in ['a', 'b']]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch('transformers.AutoTokenizer.from_pretrained'), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained'), \
                 patch('peft.PeftModel.from_pretrained'), \
                 patch('retail_agent.customer.DeepSeekCustomer'), \
                 patch('retail_agent.rollout.run_episode', return_value=result('a', eligible=False)) as episode:
                with self.assertRaisesRegex(RuntimeError, '系统故障'):
                    evaluate_candidate(folder, folder, rows, protocol, folder / 'ledger', 10)
            self.assertEqual(episode.call_count, 1)
            self.assertFalse(json.loads((folder / 'traces.jsonl').read_text())['business_evaluation_eligible'])


class AgentTests(unittest.TestCase):
    def test_greedy_eos_decoding_preserves_tool_markers(self):
        import torch
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.convert_tokens_to_ids.return_value = 9
        tokenizer.pad_token_id = 0
        tokenizer.decode.return_value = '<tool_call>{}</tool_call>'
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 7, 8, 9]])
        agent = CudaGreedyAgent(model, tokenizer,
            {'max_new_tokens_per_action': 10, 'max_context_tokens': 20})
        tensor = torch.tensor
        with patch('torch.tensor', side_effect=lambda data, **kwargs: tensor(data, dtype=kwargs.get('dtype'))):
            generated = agent.generate([], [])
        self.assertEqual(generated['finish_reason'], 'stop')
        tokenizer.decode.assert_called_once_with([7, 8], skip_special_tokens=False)
        self.assertFalse(model.generate.call_args.kwargs['generation_config'].do_sample)
        self.assertEqual(model.generate.call_args.kwargs['logits_to_keep'], 1)

    def test_context_limit_never_truncates(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2]
        model = Mock()
        agent = CudaGreedyAgent(model, tokenizer,
            {'max_new_tokens_per_action': 10, 'max_context_tokens': 11})
        with self.assertRaises(ContextLimit):
            agent.generate([], [])
        model.generate.assert_not_called()
