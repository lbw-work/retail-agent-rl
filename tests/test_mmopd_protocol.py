"""多工具解析、顺序执行、完整反馈及原评分不变；金标准仅用于离线测试夹具。"""

import json
import tomllib
import unittest
from unittest.mock import Mock

import torch
from transformers import AutoTokenizer

from retail_agent.environment import ROOT
from retail_agent.grpo_rollout import run_token_episode
from retail_agent.mmopd import NativeAgent, ThinkingTokenizer
from retail_agent.mmopd_protocol import (
    PROTOCOL, NativeHistory, NativeRetailSession, NativeTokenTape, execute_batch, parse_native_action, parse_native_action_v1,
    run_native_token_episode,
)


def tool_text(calls):
    return ''.join('<tool_call>' + json.dumps({'name': a['name'], 'arguments': a['arguments']})
                   + '</tool_call>\n' for a in calls)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = AutoTokenizer.from_pretrained(ROOT / 'data/raw/tokenizer', local_files_only=True)
        cls.tokenizer = ThinkingTokenizer(cls.raw)
        cls.config = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
        cls.config.update(max_new_tokens_per_action=4096)
        cls.row = next(json.loads(s) for s in (ROOT / cls.config['train_tasks']).read_text().splitlines()
                       if json.loads(s)['id'] == 'retail_174')
        cls.calls = cls.row['evaluation_criteria']['actions']

    def test_parse_entire_batch_preserves_order_and_rejects_mixed_or_malformed_content(self):
        batch = tool_text(self.calls[2:5])
        parsed = parse_native_action_v1(batch)
        self.assertEqual([a['arguments'] for a in parsed['calls']],
                         [a['arguments'] for a in self.calls[2:5]])
        for text in ['Hello' + batch, batch + 'Hello', batch + '<tool_call>{broken}</tool_call>',
                     batch + '<tool_call>', '<think>hidden</think>' + batch]:
            with self.assertRaises(ValueError):
                parse_native_action_v1(text)

    def test_v2_separates_repeated_thinking_and_preserves_every_tool_call(self):
        text = '<think>fixture</think></think>' + tool_text(self.calls[5:6])
        text += '</think>' + tool_text(self.calls[6:])
        parsed = parse_native_action(text)
        self.assertEqual([a['arguments'] for a in parsed['calls']],
                         [a['arguments'] for a in self.calls[5:]])
        self.assertEqual(sum(r['kind'] == 'extra_think_close' for r in parsed['repairs']), 2)

    def test_v2_missing_closers_use_complete_json_boundaries_without_rewriting_arguments(self):
        text = tool_text(self.calls[2:5]).replace('</tool_call>', '﹅')
        parsed = parse_native_action(text)
        self.assertEqual([a['arguments'] for a in parsed['calls']],
                         [a['arguments'] for a in self.calls[2:5]])
        self.assertEqual(sum(r['kind'] == 'missing_tool_close' for r in parsed['repairs']), 3)

    def test_v2_keeps_customer_visible_text_in_its_original_order(self):
        text = 'Let me check the orders.\n' + tool_text(self.calls[2:3])
        text += 'Checking the other order.\n' + tool_text(self.calls[3:4])
        parsed = parse_native_action(text)
        self.assertEqual([s['kind'] for s in parsed['segments']], ['text', 'tool', 'text', 'tool'])
        self.assertEqual(parsed['segments'][0]['text'], 'Let me check the orders.')

    def test_v2_never_repairs_json_or_unclosed_reasoning(self):
        valid = tool_text(self.calls[:1])
        for text in [valid + '<tool_call>{broken}', '<think>unclosed' + valid,
                     '<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"a","arguments":{"x":NaN}}</tool_call>',
                     valid + '{"name":"b","arguments":{}}', valid + '<tool_call']:
            with self.assertRaises(ValueError):
                parse_native_action(text)

    def test_documented_empty_think_prefix_allows_complete_tool_call_only(self):
        parsed = parse_native_action('<think>\n' + tool_text(self.calls[:1]))
        self.assertEqual(parsed['calls'][0]['arguments'], self.calls[0]['arguments'])
        self.assertEqual(parsed['repairs'][0]['kind'], 'empty_think_before_tool')
        for raw in ['<think>', '<think>not empty ' + tool_text(self.calls[:1]),
                    '<think>\n<tool_call>{broken}']:
            with self.assertRaises(ValueError):
                parse_native_action(raw)

    def test_v2_does_not_interpret_tags_inside_json_argument_strings(self):
        call = {'name': 'fixture', 'arguments': {'text': '</think><tool_call>literal</tool_call>'}}
        parsed = parse_native_action(tool_text([call]).replace('</tool_call>\n', '\n'))
        self.assertEqual(parsed['calls'][0]['arguments'], call['arguments'])

    def test_native_history_removes_old_reasoning_after_customer_but_keeps_calls(self):
        history = NativeHistory(self.tokenizer, [{'role': 'user', 'content': 'opening'}], [], include_reasoning=True)
        raw = '<think>PRIVATE_OLD_REASONING</think>' + tool_text(self.calls[:1])
        raw += '</think>' + tool_text(self.calls[1:2])
        history.assistant(raw, parse_native_action(raw))
        history.feedback([{'role': 'tool', 'content': 'COMPLETE_RETURN'}])
        before = self.raw.decode(history.prompt_ids())
        self.assertIn('PRIVATE_OLD_REASONING', before)
        history.feedback([{'role': 'user', 'content': 'confirmed'}])
        after = self.raw.decode(history.prompt_ids())
        self.assertNotIn('PRIVATE_OLD_REASONING', after)
        self.assertIn('COMPLETE_RETURN', after)
        for call in self.calls[:2]:
            self.assertIn(call['name'], after)
        self.assertEqual(after.count('<tool_call>'), 2)

    def test_native_history_preserves_mixed_segment_order(self):
        history = NativeHistory(self.tokenizer, [{'role': 'user', 'content': 'opening'}], [])
        raw = 'FIRST_VISIBLE ' + tool_text(self.calls[:1]) + 'LAST_VISIBLE'
        history.assistant(raw, parse_native_action(raw))
        rendered = self.raw.decode(history.prompt_ids())
        self.assertLess(rendered.index('FIRST_VISIBLE'), rendered.index('<tool_call>'))
        self.assertLess(rendered.index('<tool_call>'), rendered.index('LAST_VISIBLE'))

    def test_official_history_does_not_replay_previous_thinking(self):
        history = NativeHistory(self.tokenizer, [{'role': 'user', 'content': 'opening'}], [])
        raw = '<think>PRIVATE_OLD_REASONING</think>' + tool_text(self.calls[:1])
        history.assistant(raw, parse_native_action(raw))
        history.feedback([{'role': 'tool', 'content': 'FULL_RETURN'}])
        rendered = self.raw.decode(history.prompt_ids())
        self.assertNotIn('PRIVATE_OLD_REASONING', rendered)
        self.assertIn(self.calls[0]['name'], rendered)
        self.assertIn('FULL_RETURN', rendered)

    def test_native_history_cannot_silently_enter_flat_training(self):
        with self.assertRaisesRegex(RuntimeError, '逐轮真实上下文'):
            run_native_token_episode(self.row, Mock(), self.tokenizer, Mock(), 'opening',
                {**self.config, 'history_mode': 'native'})

    def test_v2_mixed_turn_text_is_preserved_in_real_execution_transcript(self):
        texts = ['Checking your account. ' + tool_text(self.calls[:2]),
                 tool_text(self.calls[2:5]).replace('</tool_call>', ''),
                 tool_text(self.calls[5:6]) + '</think>' + tool_text(self.calls[6:]), 'Done.']
        result, _ = self.episode(texts, True)
        self.assertEqual(result['scores']['reward_a'], 1.)
        self.assertTrue(any(m.get('content') == 'Checking your account.' for m in result['trajectory']))
        self.assertEqual(sum(len(b['execution']) for b in result['tool_batches']), 7)

    def test_native_prompt_changes_tool_format_only(self):
        from retail_agent.environment import RetailSession
        old = RetailSession(self.row).agent_messages()[0]['content']
        new = NativeRetailSession(self.row).agent_messages()[0]['content']
        expected = old.replace('make exactly one tool call;',
            'make one or more tool calls, executed sequentially in the order given; ', 1)
        expected = expected.replace('You should at most make one tool call at a time,',
            'You may request one or more tool calls in a turn, executed sequentially,', 1)
        self.assertEqual(new, expected)

    def test_execution_preserves_business_error_and_continues_in_order(self):
        session = NativeRetailSession(self.row)
        calls = [{'kind': 'tool', 'name': 'unknown_fixture_tool', 'arguments': {}},
                 {'kind': 'tool', **self.calls[0]}]
        events = list(execute_batch(session, calls))
        self.assertEqual([e['batch_index'] for e in events], [0, 1])
        self.assertTrue(events[0]['response']['error'])
        self.assertFalse(events[1]['response']['error'])
        self.assertEqual(events[0]['message']['content'], events[0]['response']['content'])
        self.assertEqual(len(session.trajectory), 4)

    def test_batch_feedback_preserves_raw_prefix_and_all_returns_with_zero_loss(self):
        session = NativeRetailSession(self.row)
        tape = NativeTokenTape(self.tokenizer, session.agent_messages(), session.tool_schemas, 16384)
        sampled = self.raw.encode('<think>fixture</think>' + tool_text(self.calls[:2]) + '<|im_end|>',
                                  add_special_tokens=False)
        tape.sampled({'token_ids': sampled, 'logprobs': [-1.] * len(sampled)})
        before = list(tape.ids)
        feedback = [{'role': 'tool', 'content': 'FIRST_COMPLETE_RETURN'},
                    {'role': 'tool', 'content': 'SECOND_COMPLETE_RETURN'}]
        tape.feedback_batch(feedback, continue_generation=True)
        suffix = self.raw.decode(tape.ids[len(before):], skip_special_tokens=False)
        self.assertIn('FIRST_COMPLETE_RETURN', suffix)
        self.assertIn('SECOND_COMPLETE_RETURN', suffix)
        self.assertEqual(suffix.count('<|im_start|>assistant'), 1)
        self.assertEqual(suffix.count('<|im_start|>user'), 1)
        self.assertEqual(tape.ids[:len(before)], before)
        self.assertEqual(tape.mask[:len(sampled)], [1] * len(sampled))
        self.assertTrue(all(m == 0 for m in tape.mask[len(sampled):]))

    def episode(self, texts, native, *, native_history=False):
        encoded = [self.raw.encode('<think>fixture</think>' + t + '<|im_end|>',
                                  add_special_tokens=False) for t in texts]
        iterator = iter(encoded)
        sampler = Mock(generate=lambda *args: (lambda ids: {'token_ids': ids,
                       'logprobs': [-1.] * len(ids)})(next(iterator)))
        episode = run_native_token_episode if native else run_token_episode
        config = {**self.config, 'history_mode': 'native', 'native_history_evaluation_only': True} \
            if native_history else self.config
        result = episode(self.row, sampler, self.tokenizer,
            Mock(reply=Mock(return_value={'text': '###STOP###'})), 'fixture opening', config)
        return result, encoded

    def test_native_episode_keeps_real_turn_inputs_and_raw_samples_with_original_score(self):
        result, samples = self.episode([tool_text(self.calls[:2]), tool_text(self.calls[2:5]),
                                        tool_text(self.calls[5:]), 'Done.'], True, native_history=True)
        self.assertEqual(result['scores']['reward_a'], 1.)
        self.assertFalse(result['flat_training_ready'])
        self.assertEqual([turn['token_ids'] for turn in result['generation_turns']], samples)
        self.assertEqual(result['generation_turns'][0]['prompt_ids'], result['prompt_ids'])
        second = self.raw.decode(result['generation_turns'][1]['prompt_ids'])
        self.assertNotIn('<think>fixture</think>', second)
        self.assertEqual(second.count('<tool_response>'), 2)

    def test_gold_execution_scores_identically_under_unchanged_scorer(self):
        single, _ = self.episode([tool_text([a]) for a in self.calls] + ['Done.'], False)
        multi, samples = self.episode([tool_text(self.calls[:2]), tool_text(self.calls[2:5]),
                                      tool_text(self.calls[5:]), 'Done.'], True)
        self.assertEqual(single['scores'], multi['scores'])
        self.assertEqual(multi['scores']['reward_a'], 1.)
        self.assertEqual(multi['scores']['business_reward'], 1.)
        self.assertEqual([len(b['execution']) for b in multi['tool_batches']], [2, 3, 2])
        semantic = lambda history: [{k: v for k, v in m.items() if k != 'timestamp'} for m in history]
        self.assertEqual(semantic(single['trajectory']), semantic(multi['trajectory']))
        for ids, turn in zip(samples, multi['turns'], strict=True):
            self.assertEqual(multi['completion_ids'][turn['start']:turn['end']], ids)

    def test_multi_tool_does_not_bypass_authentication_scoring(self):
        result, _ = self.episode([tool_text(self.calls[2:]), 'Done.'], True)
        self.assertEqual(result['scores']['reward_a'], 0.)
        self.assertIn('private_access_before_authentication', result['scores']['violations'])

    def test_malformed_later_call_executes_none_of_the_batch(self):
        result, _ = self.episode([tool_text(self.calls[:1]) + '<tool_call>{broken}</tool_call>'], True)
        self.assertEqual(result['termination'], 'invalid_model_output')
        self.assertFalse(any(m['role'] == 'tool' for m in result['trajectory']))

    def test_native_cuda_controller_executes_every_call_and_keeps_every_response(self):
        from retail_agent.rollout import run_episode
        texts = [tool_text(self.calls[:2]), tool_text(self.calls[2:5]), tool_text(self.calls[5:]), 'Done.']
        agent = Mock(tool_protocol=PROTOCOL)
        agent.generate.side_effect = [{'raw_text': text, 'finish_reason': 'stop'} for text in texts]
        customer = Mock(reply=Mock(side_effect=[{'text': 'fixture opening'}, {'text': '###STOP###'}]))
        result = run_episode(self.row, agent, customer)
        self.assertEqual(result['reward']['reward'], 1.)
        events = [e for e in result['events'] if e['source'] == 'retail_tool']
        self.assertEqual([e['name'] for e in events], [a['name'] for a in self.calls])
        self.assertEqual(len(events), 7)
        self.assertTrue(all('content' in e['response'] for e in events))

    def test_cuda_native_agent_appends_all_new_tool_returns(self):
        model = Mock(device=torch.device('cpu'))
        eos = self.raw.convert_tokens_to_ids('<|im_end|>')
        model.generate.side_effect = lambda **kw: torch.cat([kw['input_ids'], torch.tensor([[eos]])], dim=1)
        config = {'max_trajectory_tokens': 16384, 'max_new_tokens_per_action': 4096,
                  'temperature': 1., 'tool_protocol': PROTOCOL}
        agent = NativeAgent(model, self.raw, config)
        messages = [{'role': 'user', 'content': 'fixture'}]
        first = agent.generate(messages, [])
        for content in ['first result', 'second result', 'third result']:
            messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': []},
                             {'role': 'tool', 'content': content}])
        second = agent.generate(messages, [])
        prefix = first['prompt_token_ids'] + first['generated_token_ids']
        added = self.raw.decode(second['prompt_token_ids'][len(prefix):])
        for content in ['first result', 'second result', 'third result']:
            self.assertIn(content, added)
        self.assertEqual(added.count('<|im_start|>assistant'), 1)
