import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from retail_agent.customer import BudgetExceeded, CustomerError, DeepSeekCustomer, GOAL_POLICY, customer_messages, peak_cost_cny
from retail_agent.model import ContextLimit, parse_action
from retail_agent.rollout import run_episode
from test_environment import example_task

CONFIG = {"model": "deepseek-flash", "base_url": "https://api.deepseek.com",
          "temperature": 0.0, "max_retries": 0, "timeout_seconds": 60}


class Phase3Tests(unittest.TestCase):
    def test_prompt_guard_keeps_full_history_and_reservation_with_explicit_limit(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            scenario = example_task()["user_scenario"]
            history = [{"role": "assistant", "content": "x" * 33_000}]
            default = DeepSeekCustomer(scenario, CONFIG, ledger_path=Path(directory) / "default.jsonl")
            default.http.post = Mock()
            with self.assertRaisesRegex(CustomerError, "输入保护"):
                default.reply(history)
            default.http.post.assert_not_called()
            fixed = DeepSeekCustomer(scenario, {**CONFIG, "max_prompt_bytes": 64_000}, ledger_path=Path(directory) / "fixed.jsonl")
            fixed.balance_cny = Mock(return_value=1.)
            data = {"usage": {"prompt_tokens": 10_000, "completion_tokens": 1}, "id": "long-test",
                    "model": "deepseek-flash", "choices": [{"finish_reason": "stop", "message": {"content": "Hello"}}]}
            fixed.http.post = Mock(return_value=Mock(status_code=200, json=lambda: data))
            fixed.reply(history)
            self.assertEqual(fixed.http.post.call_args.kwargs["json"]["messages"][-1]["content"], history[0]["content"])
            reserved = json.loads(fixed.ledger.read_text().splitlines()[0])["reserved_cny"]
            self.assertGreater(reserved, .07)
            with self.assertRaisesRegex(CustomerError, "输入保护"):
                fixed.reply([{"role": "assistant", "content": "x" * 65_000}])
            self.assertEqual(fixed.http.post.call_count, 1)
            limited = DeepSeekCustomer(scenario, {**CONFIG, "max_prompt_bytes": 64_000}, .001, Path(directory) / "limited.jsonl")
            limited.http.post = Mock()
            with self.assertRaises(BudgetExceeded):
                limited.reply(history)
            limited.http.post.assert_not_called()
            for invalid in [True, 0, 64_001]:
                with self.assertRaises(ValueError):
                    DeepSeekCustomer(scenario, {**CONFIG, "max_prompt_bytes": invalid}, ledger_path=Path(directory) / "invalid.jsonl")

    def test_goal_rule_is_customer_only_and_rejects_unknown_version(self):
        scenario = example_task()["user_scenario"]
        old = customer_messages(scenario, [])
        fixed = customer_messages(scenario, [], GOAL_POLICY)
        self.assertNotIn("fixed-goal evaluation", old[0]["content"])
        self.assertIn("fixed-goal evaluation", fixed[0]["content"])
        self.assertIn("unless the scenario explicitly permits", fixed[0]["content"])
        self.assertEqual(old[1:], fixed[1:])
        with self.assertRaises(ValueError):
            customer_messages(scenario, [], "unknown-policy")

    def test_balance_cache_subtracts_shared_ledger_reserves_and_expires(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            customer = DeepSeekCustomer(example_task()["user_scenario"], CONFIG, ledger_path=Path(directory) / "ledger.jsonl")
            customer.balance_cny = Mock(side_effect=[1., .8])
            self.assertEqual(customer.available_balance_cny(), 1.)
            customer._record({"event": "started", "id": "other-client", "reserved_cny": .4})
            self.assertAlmostEqual(customer.available_balance_cny(), .6)
            customer._record({"event": "completed", "id": "other-client", "peak_cost_upper_cny": .1})
            self.assertAlmostEqual(customer.available_balance_cny(), .9)
            self.assertEqual(customer.balance_cny.call_count, 1)
            customer._balance_checked_at -= 61
            self.assertEqual(customer.available_balance_cny(), .8)
            self.assertEqual(customer.balance_cny.call_count, 2)

    def test_successful_reply_needs_no_post_generation_balance_query(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            customer = DeepSeekCustomer(example_task()["user_scenario"], CONFIG, ledger_path=Path(directory) / "ledger.jsonl")
            customer.balance_cny = Mock(side_effect=[1., CustomerError("after-generation GET fails")])
            data = {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "id": "reply-test",
                    "model": "deepseek-flash", "choices": [{"finish_reason": "stop", "message": {"content": "Hello"}}]}
            customer.http.post = Mock(return_value=Mock(status_code=200, json=lambda: data))
            for _ in range(2):
                reply = customer.reply([])
                self.assertEqual(reply["text"], "Hello")
                self.assertIsNone(reply["api"]["balance_delta_cny"])
            self.assertEqual(customer.balance_cny.call_count, 1)
            self.assertEqual(customer.http.post.call_count, 2)
            self.assertAlmostEqual(customer.accounted_cny(), .00072)

    def test_invalid_cached_balance_does_not_start_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            customer = DeepSeekCustomer(example_task()["user_scenario"], CONFIG, ledger_path=Path(directory) / "ledger.jsonl")
            customer.balance_cny = Mock(return_value=float("nan"))
            customer.http.post = Mock()
            with self.assertRaises(CustomerError):
                customer.reply([])
            customer.http.post.assert_not_called()

    def test_balance_transient_failure_retries_without_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            customer = DeepSeekCustomer(example_task()["user_scenario"], {**CONFIG, "max_retries": 1},
                                        ledger_path=Path(directory) / "ledger.jsonl")
            response = Mock(status_code=200, json=lambda: {
                "is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "1.0"}]})
            customer.http.get = Mock(side_effect=[requests.ConnectionError(), response])
            customer.http.post = Mock()
            with patch("retail_agent.customer.time.sleep"):
                self.assertEqual(customer.balance_cny(), 1.0)
            self.assertEqual(customer.http.get.call_count, 2)
            customer.http.post.assert_not_called()
            self.assertFalse(customer.ledger.exists())

    def test_balance_failure_has_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            customer = DeepSeekCustomer(example_task()["user_scenario"], {**CONFIG, "max_retries": 1},
                                        ledger_path=Path(directory) / "ledger.jsonl")
            customer.http.get = Mock(side_effect=requests.ConnectionError())
            with patch("retail_agent.customer.time.sleep"), self.assertRaises(CustomerError):
                customer.balance_cny()
            self.assertEqual(customer.http.get.call_count, 2)

    def test_customer_view_excludes_private_tools_and_grader(self):
        scenario = example_task()["user_scenario"]
        history = [{"role": "system", "content": "PRIVATE_POLICY_MARKER"},
                   {"role": "user", "content": "My email is test@example.org"},
                   {"role": "assistant", "content": "", "tool_calls": [{"name": "get_user_details"}]},
                   {"role": "tool", "content": "PRIVATE_DB_MARKER"},
                   {"role": "assistant", "content": "Please confirm."}]
        messages = customer_messages(scenario, history)
        self.assertEqual([m["role"] for m in messages], ["system", "assistant", "user"])
        self.assertNotIn("PRIVATE_", json.dumps(messages))
        with self.assertRaises(ValueError):
            customer_messages({**scenario, "evaluation_criteria": {}}, [])

    def test_parser_accepts_one_action_and_rejects_mixed_or_reasoning(self):
        call = '<tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}</tool_call>'
        self.assertEqual(parse_action(call)["kind"], "tool")
        self.assertEqual(parse_action("Please give me your email.")["kind"], "message")
        for text in ["", "<think>hidden</think>reply", "Let me check. " + call, call + call,
                     '<tool_call>{"name":"x","arguments":"{}"}</tool_call>',
                     '<tool_call>{"name":"x","arguments":{},"id":"1"}</tool_call>']:
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_cost_cache_and_invalid_usage(self):
        self.assertAlmostEqual(peak_cost_cny({"prompt_tokens": 1000, "completion_tokens": 100,
                                             "prompt_cache_hit_tokens": 500}), 0.00182)
        with self.assertRaises(CustomerError):
            peak_cost_cny({"prompt_tokens": 2, "completion_tokens": 1, "prompt_cache_hit_tokens": 3})

    def test_successful_api_request_is_accounted_without_key_in_logs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            ledger = Path(directory) / "ledger.jsonl"
            customer = DeepSeekCustomer(example_task()["user_scenario"], CONFIG, ledger_path=ledger)
            customer.balance_cny = Mock(side_effect=[1.0, 0.9998])
            data = {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "id": "response-test",
                    "model": "deepseek-flash", "choices": [{"finish_reason": "stop",
                    "message": {"content": "Hello", "reasoning_content": None}}]}
            customer.http.post = Mock(return_value=Mock(status_code=200, json=lambda: data))
            self.assertEqual(customer.reply([])["text"], "Hello")
            payload = customer.http.post.call_args.kwargs["json"]
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertAlmostEqual(customer.accounted_cny(), 0.00036)
            self.assertNotIn("test-only-secret", ledger.read_text())

    def test_failed_request_keeps_reserve_across_restart_and_budget_blocks(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            ledger = Path(directory) / "ledger.jsonl"
            customer = DeepSeekCustomer(example_task()["user_scenario"], CONFIG, ledger_path=ledger)
            customer.balance_cny = Mock(return_value=1.0)
            customer.http.post = Mock(side_effect=requests.Timeout())
            with self.assertRaises(CustomerError):
                customer.reply([])
            self.assertGreater(customer.accounted_cny(), 0)
            failures = [json.loads(line) for line in ledger.read_text().splitlines() if json.loads(line)["event"] == "failed"]
            self.assertTrue(failures)
            self.assertTrue(all(record["elapsed_seconds"] >= 0 for record in failures))
            restarted = DeepSeekCustomer(customer.scenario, CONFIG, limit_cny=0.000001, ledger_path=ledger)
            restarted.http.post = Mock()
            with self.assertRaises(BudgetExceeded):
                restarted.reply([])
            restarted.http.post.assert_not_called()

    def test_unlimited_customer_records_cost_beyond_old_cap_and_still_checks_balance(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            ledger = Path(directory) / 'ledger.jsonl'
            ledger.write_text(json.dumps({'event': 'started', 'id': 'old', 'reserved_cny': 11.}) + '\n')
            customer = DeepSeekCustomer(example_task()['user_scenario'], CONFIG, limit_cny=None, ledger_path=ledger)
            customer.balance_cny = Mock(return_value=100.)
            data = {'usage': {'prompt_tokens': 100, 'completion_tokens': 20}, 'id': 'no-cap-test',
                'model': 'deepseek-flash', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Hello'}}]}
            customer.http.post = Mock(return_value=Mock(status_code=200, json=lambda: data))
            self.assertEqual(customer.reply([])['text'], 'Hello')
            self.assertGreater(customer.accounted_cny(), 11.)
            customer.available_balance_cny = Mock(return_value=0.)
            customer.http.post.reset_mock()
            with self.assertRaisesRegex(BudgetExceeded, '余额'):
                customer.reply([])
            customer.http.post.assert_not_called()

    def test_controller_routes_tools_without_calling_customer(self):
        row = example_task()
        sequence = [json.dumps({"name": a["name"], "arguments": a["arguments"]})
                    for a in row["evaluation_criteria"]["actions"]]
        agent = Mock()
        agent.generate.side_effect = [{"raw_text": "<tool_call>" + text + "</tool_call>",
                                      "finish_reason": "stop"} for text in sequence] + [
                                      {"raw_text": "Your order is cancelled.", "finish_reason": "stop"}]
        customer = Mock()
        customer.reply.side_effect = [{"text": "Please cancel my order", "api": {}},
                                      {"text": "###STOP###", "api": {}}]
        result = run_episode(row, agent, customer)
        self.assertEqual(customer.reply.call_count, 2)
        self.assertEqual(result["reward"]["reward"], 1.0)
        self.assertEqual(result["termination"], "completed")
        self.assertFalse(result["training_ready"])
        self.assertGreater(result["timing"]["stages"]["tool_execution"]["calls"], 0)
        self.assertEqual(result["timing"]["stages"]["customer_opening"]["calls"], 1)
        self.assertEqual(result["timing"]["stages"]["reward_evaluation"]["calls"], 1)

    def test_service_failure_context_limit_and_bad_output_are_distinct(self):
        row = example_task()
        for model_effect, customer_effect, reason, eligible in [
            (None, CustomerError("network"), "customer_service_error", False),
            (ContextLimit("too long"), {"text": "Hello"}, "context_limit", True),
            (RuntimeError("broken"), {"text": "Hello"}, "infrastructure_error", False),
            ({"raw_text": "<think>bad</think>", "finish_reason": "stop"}, {"text": "Hello"},
             "invalid_model_output", True),
            ({"raw_text": "incomplete", "finish_reason": "length"}, {"text": "Hello"},
             "generation_limit", True)]:
            with self.subTest(reason=reason):
                agent, customer = Mock(), Mock()
                if isinstance(model_effect, Exception): agent.generate.side_effect = model_effect
                else: agent.generate.return_value = model_effect
                if isinstance(customer_effect, Exception): customer.reply.side_effect = customer_effect
                else: customer.reply.return_value = customer_effect
                result = run_episode(row, agent, customer)
                self.assertEqual(result["termination"], reason)
                self.assertEqual(result["business_evaluation_eligible"], eligible)
                self.assertEqual(result["reward"]["reward"] if eligible else result["reward"], 0.0 if eligible else None)


if __name__ == "__main__":
    unittest.main()
