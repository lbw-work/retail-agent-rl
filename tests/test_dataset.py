import unittest

from retail_agent.dataset import assign_groups, confirmed, length_bucket, sft_policy_flags, task_business, task_spec_flags


def call(name):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": {}}}]}


class DatasetTests(unittest.TestCase):
    def test_write_requires_authentication_and_confirmation(self):
        target = call("cancel_pending_order")
        self.assertEqual(sft_policy_flags([], target),
                         ["missing_prior_authentication", "write_without_explicit_confirmation"])
        history = [call("find_user_id_by_email"),
                   {"role": "tool", "name": "find_user_id_by_email", "content": '"person_1"'},
                   {"role": "assistant", "content": "Confirm cancellation?"},
                   {"role": "user", "content": "Yes, please proceed."}]
        self.assertEqual(sft_policy_flags(history, target), [])
        history[-1]["content"] = "No, not yet."
        self.assertEqual(sft_policy_flags(history, target), ["write_without_explicit_confirmation"])

    def test_orphan_response_and_missing_response_are_flagged(self):
        response = {"role": "tool", "name": "get_user_details", "content": "{}"}
        self.assertIn("unpaired_tool_response", sft_policy_flags([response], {"role": "assistant", "content": "Hello"}))
        self.assertIn("missing_tool_response", sft_policy_flags([call("get_user_details")], call("get_order_details")))

    def test_buckets_and_business_combinations(self):
        self.assertEqual(length_bucket(8192), "up_to_8k")
        self.assertEqual(length_bucket(8193), "8k_to_12k")
        self.assertEqual(length_bucket(12289), "12k_to_16k")
        self.assertEqual(task_business([{"name": "get_user_details"}]), ([], "other"))
        self.assertEqual(task_business([{"name": "cancel_pending_order"}, {"name": "modify_user_address"}])[1], "pair")

    def test_ambiguous_reason_is_not_silently_fixed(self):
        task = {"id": "test", "user_scenario": {"goal": "Choose either no longer needed or ordered by mistake"},
                "evaluation_criteria": {"actions": [{"name": "cancel_pending_order", "arguments": {"reason": "ordered by mistake"}}]}}
        self.assertEqual(task_spec_flags(task), ["cancellation_reason_multiple_options_requires_review"])
        task["user_scenario"]["goal"] = "I ordered this by mistake"
        self.assertEqual(task_spec_flags(task), [])
        self.assertFalse(confirmed("Yes, but do not cancel yet"))

    def test_anchors_conflicts_and_reproducible_group_assignment(self):
        members = {str(i): [{"features": ["rl", "rl:cancel_pending_order"]}]*3 for i in range(30)}
        anchors = {"1": {"test"}, "2": {"dev"}, "3": {"dev", "test"}}
        first, conflicts = assign_groups(members, anchors)
        second, _ = assign_groups(dict(reversed(list(members.items()))), anchors)
        self.assertEqual(first, second)
        self.assertEqual(first["1"], "test")
        self.assertEqual(first["2"], "dev")
        self.assertNotIn("3", first)
        self.assertEqual(conflicts, {"3": ["dev", "test"]})
        self.assertEqual(set(first.values()), {"train", "dev", "test"})


if __name__ == "__main__":
    unittest.main()
