"""离线协议夹具；不调用模型或顾客 API，不将脚本回复作为真实评测。"""

from copy import deepcopy
import json
import sys
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from retail_agent.community_protocol import parse_action, run_episode
from retail_agent.customer import CustomerError
from retail_agent.environment import ROOT, RetailSession
from retail_agent.reward import RewardEvaluationError
from retail_agent.run_community_local import MLXSampler
from retail_agent.training_reward import evaluate_training_task
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage


def block(name, arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return [1] * (len(text) // 4)


class ScriptedSampler:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.temperatures = []

    def generate(self, ids, count, *, temperature=None):
        self.temperatures.append(temperature)
        text = next(self.texts)
        return {'text': text, 'raw_model_text': text}


class ScriptedCustomer:
    def __init__(self):
        self.responses = iter(['james.jenkins8742@example.com', 'Yes, both orders to original payments.',
                               'Thanks. ###STOP###'])

    def reply(self, messages):
        return {'text': next(self.responses)}


class CommunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tomllib.loads((ROOT / 'configs/community_eval.toml').read_text())
        cls.config['review_manifest'] = 'data/versions/retail_rl_v3/manifests/review.json'
        cls.row = next(json.loads(x) for x in (ROOT / 'data/versions/retail_rl_v3/processed/rl_train.jsonl')
                       .read_text().splitlines() if json.loads(x)['id'] == 'retail_174')
        cls.calls = cls.row['evaluation_criteria']['actions']

    def success_samples(self):
        texts = [block('respond', {'content': 'What is your email?'})]
        texts += [block(c['name'], c['arguments']) for c in self.calls[:5]]
        texts += [block('respond', {'content': 'Confirm returning both orders to their original payments?'})]
        texts += [block(c['name'], c['arguments']) for c in self.calls[5:]]
        return texts + [block('respond', {'content': 'Both return requests succeeded.'})]

    def run_fixture(self, samples, **kwargs):
        return run_episode(self.row, ScriptedSampler(samples), Tokenizer(), ScriptedCustomer(),
                           'Return both delivered orders.', self.config, **kwargs)

    def test_original_environment_scoring_ids_and_state_isolation(self):
        untouched = RetailSession(self.row)
        initial = untouched.db_hash()
        result = self.run_fixture(self.success_samples())
        self.assertEqual(result['scores']['reward_a'], 1.)
        self.assertEqual(result['scores']['violations'], [])
        self.assertEqual(result['format_retries'], 0)
        self.assertFalse(result['training_ready'])
        calls = [c for m in result['trajectory'] for c in (m.get('tool_calls') or [])]
        self.assertEqual(len(calls), 7)
        self.assertEqual(len({c['id'] for c in calls}), 7)
        self.assertTrue(all(c['id'].startswith('call_') for c in calls))
        self.assertEqual(untouched.db_hash(), initial)
        self.assertEqual(RetailSession(self.row).db_hash(), initial)
        self.assertEqual(result['initial_db_hash'], initial)
        self.assertNotEqual(result['final_db_hash'], initial)
        for oid in ['#W2196851', '#W4202427']:
            self.assertEqual(result['final_db']['orders'][oid]['status'], 'return requested')
        self.assertTrue(any(m['content'].startswith('Tool result for ')
                            for m in result['native_messages'] if m['role'] == 'user'))

    def test_duplicate_call_ids_still_fail_the_original_scorer(self):
        result = self.run_fixture(self.success_samples())
        messages = deepcopy(result['trajectory'])
        old, replacement = None, None
        for m in messages:
            if m['role'] == 'assistant' and m.get('tool_calls'):
                if replacement is None:
                    replacement = m['tool_calls'][0]['id']
                else:
                    old = m['tool_calls'][0]['id']
                    m['tool_calls'][0]['id'] = replacement
            elif m['role'] == 'tool' and old and m['id'] == old:
                m['id'] = replacement
                break
        classes = {'assistant': AssistantMessage, 'tool': ToolMessage, 'user': UserMessage}
        history = [classes[m['role']].model_validate(m) for m in messages]
        with self.assertRaisesRegex(RewardEvaluationError, 'ID 重复'):
            evaluate_training_task(self.row, history, review_manifest=self.config['review_manifest'])

    def test_old_illegal_return_argument_is_never_removed(self):
        bad = deepcopy(self.calls[5])
        bad['arguments']['new_item_ids'] = bad['arguments']['item_ids']
        samples = [block(c['name'], c['arguments']) for c in self.calls[:5]]
        samples += [block(bad['name'], bad['arguments'])] * 3
        result = self.run_fixture(samples)
        self.assertEqual(result['termination'], 'too_many_errors')
        self.assertEqual(result['scores']['reward_a'], 0.)
        self.assertEqual(result['initial_db_hash'], result['final_db_hash'])
        self.assertIn('new_item_ids', result['parsed_actions'][-1]['arguments'])
        errors = [m for m in result['trajectory'] if m['role'] == 'tool' and m['error']]
        self.assertEqual(len(errors), 3)

    def test_multiple_calls_are_not_executed_and_retry_is_preserved(self):
        text = block(self.calls[0]['name'], self.calls[0]['arguments']) * 2
        sampler, events = ScriptedSampler([text, text]), []
        result = run_episode(self.row, sampler, Tokenizer(), ScriptedCustomer(), 'opening',
                             self.config, events.append)
        self.assertEqual(result['termination'], 'invalid_model_output')
        self.assertEqual(result['scores']['reward_a'], 0.)
        self.assertEqual(sampler.temperatures, [None, 0.])
        self.assertEqual(sum(e['event'] == 'assistant_sampled' for e in events), 2)
        self.assertFalse(any(m['role'] == 'tool' for m in result['trajectory']))

    def test_done_is_not_a_fabricated_business_success(self):
        result = self.run_fixture([block('done', {})])
        self.assertEqual(result['termination'], 'completed')
        self.assertEqual(result['scores']['reward_a'], 0.)
        self.assertEqual(result['scores']['progress'], 0.)

    def test_customer_service_failure_is_not_scored_as_zero(self):
        customer, events = Mock(), []
        customer.reply.side_effect = CustomerError('offline service failure fixture')
        with self.assertRaises(CustomerError):
            run_episode(self.row, ScriptedSampler([block('respond', {'content': 'Email?'})]),
                        Tokenizer(), customer, 'opening', self.config, events.append)
        self.assertEqual(events[-1]['event'], 'episode_failed')

    def test_context_limit_never_truncates_or_fabricates_generation(self):
        sampler = ScriptedSampler([])
        result = run_episode(self.row, sampler, Tokenizer(), ScriptedCustomer(), 'opening',
                             {**self.config, 'max_trajectory_tokens': 1})
        self.assertEqual(result['termination'], 'context_limit')
        self.assertEqual(result['generations'], 0)
        self.assertEqual(result['format_retries'], 0)
        self.assertEqual(result['scores']['reward_a'], 0.)

    def test_parse_keeps_arguments_and_rejects_extra_text(self):
        for arguments in [{'item_ids': ['a'], 'new_item_ids': ['b']}, {'content': 'hello'}]:
            self.assertEqual(parse_action(block('fixture', arguments))['arguments'], arguments)
        for raw in ['plain reply', '<think>x</think>' + block('done', {}), block('done', {}) + 'extra']:
            with self.assertRaises(ValueError):
                parse_action(raw)

    def test_mlx_stop_retains_raw_output_and_real_sample_probabilities(self):
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, **kw: ''.join({1: '<tool_call>{}',
                                                                 2: '</tool_call>tail'}[i] for i in ids)
        probabilities = Mock()
        probabilities.__getitem__ = Mock(return_value=SimpleNamespace(item=lambda: -0.5))
        responses = [SimpleNamespace(token=i, logprobs=probabilities, finish_reason=None) for i in [1, 2]]
        module = SimpleNamespace(stream_generate=Mock(return_value=iter(responses)))
        utils = SimpleNamespace(make_sampler=Mock(), make_logits_processors=Mock())
        with patch.dict(sys.modules, {'mlx_lm': module, 'mlx_lm.sample_utils': utils}):
            result = MLXSampler(None, tokenizer, self.config).generate([7], 16)
        self.assertTrue(result['text'].endswith('</tool_call>'))
        self.assertTrue(result['raw_model_text'].endswith('tail'))
        self.assertEqual(result['sampled_token_logprobs'], [-0.5, -0.5])
        self.assertEqual(result['token_ids'], [1, 2])


if __name__ == '__main__':
    unittest.main()
