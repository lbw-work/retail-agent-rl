import unittest
from types import SimpleNamespace

from pydantic import create_model

from retail_agent.data import Groups, entity_keys, extract_policy, normalize_history, normalize_message, parse_task


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tools = {"get_order_details": SimpleNamespace(
            params=create_model("OrderArgs", order_id=(str, ...)))}

    def test_policy_uses_real_block(self):
        self.assertEqual(extract_policy("Use <policy> below. <policy>real rule</policy>"), "real rule")

    def test_target_removes_reasoning_and_keeps_tool_arguments(self):
        row = {"role": "assistant", "content": "", "thinking": "private",
               "tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "#W1"}}]}
        normalized = normalize_message(row, self.tools)
        self.assertNotIn("thinking", normalized)
        self.assertEqual(normalized['tool_calls'][0]['function']['arguments']['order_id'], '#W1')

    def test_unknown_or_extra_parameters_rejected(self):
        for name, arguments in [('unknown', {}), ('get_order_details', {'order_id': '#W1', 'extra': 1})]:
            with self.assertRaises(ValueError):
                normalize_message({'role': 'assistant', 'tool_calls': [{'name': name, 'arguments': arguments}]}, self.tools)

    def test_missing_required_parameter_rejected(self):
        with self.assertRaises(ValueError):
            normalize_message({'role': 'assistant', 'tool_calls': [{'name': 'get_order_details', 'arguments': {}}]}, self.tools)

    def test_shared_payment_or_product_is_not_customer_identity(self):
        keys = entity_keys({'user_id': 'alice_smith_12', 'payment_method_id': 'credit_card_001',
                            'product_id': '12345', 'order_id': '#W123'})
        self.assertEqual(keys, {'user:alice_smith_12', 'order:#W123'})

    def test_groups_are_transitive_and_do_not_join_distinct_users(self):
        g = Groups()
        g.add('a', {'user:alice_smith_12'})
        g.add('b', {'user:alice_smith_12', 'order:#W1'})
        g.add('c', {'order:#W1'})
        g.add('d', {'user:bob_smith_99'})
        self.assertEqual(g.find('a'), g.find('c'))
        self.assertNotEqual(g.find('a'), g.find('d'))

    def test_task_parses_criteria_without_mutating_source_and_rejects_traversal(self):
        row = {'evaluation_criteria': '{"actions": []}', 'db_path': 'tau2_rl_database/tau2_retail_new_db_1.json'}
        self.assertIsInstance(parse_task(row)['evaluation_criteria'], dict)
        self.assertIsInstance(row['evaluation_criteria'], str)
        with self.assertRaises(ValueError):
            parse_task({**row, 'db_path': '../tau2_retail_new_db_1.json'})

    def test_parallel_reads_pair_existing_responses(self):
        messages = [{'role': 'assistant', 'content': 'Checking orders', 'tool_calls': [
            {'name': 'get_order_details', 'arguments': {'order_id': '#W1'}},
            {'name': 'get_order_details', 'arguments': {'order_id': '#W2'}}]},
            {'role': 'tool', 'name': 'get_order_details', 'content': 'first'},
            {'role': 'tool', 'name': 'get_order_details', 'content': 'second'}]
        result, batches = normalize_history(messages, self.tools)
        self.assertEqual(batches, 1)
        self.assertEqual([m['role'] for m in result], ['assistant', 'tool', 'assistant', 'tool'])
        self.assertEqual(result[1]['content'], 'first')
        self.assertEqual(result[3]['content'], 'second')

    def test_parallel_writes_or_missing_responses_rejected(self):
        for name in ['cancel_pending_order', 'get_order_details']:
            with self.assertRaises(ValueError):
                normalize_history([{'role': 'assistant', 'tool_calls': [
                    {'name': name, 'arguments': {}}, {'name': name, 'arguments': {}}]}], self.tools)


if __name__ == '__main__':
    unittest.main()
