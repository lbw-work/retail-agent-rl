import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from retail_agent.data import read_jsonl
from retail_agent.environment import ROOT, RetailSession


def example_task(business="cancel_pending_order"):
    return next(row for _, row in read_jsonl(ROOT / "data/versions/retail_full_v2/processed/rl_dev.jsonl")
                if row["business"] == business)


def run_reference(session, row):
    for action in row["evaluation_criteria"]["actions"]:
        response = session.call_tool(action["name"], action["arguments"])
        if response.error:
            raise AssertionError(response.content)


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = example_task()
        cls.action = next(a for a in cls.row["evaluation_criteria"]["actions"]
                          if a["name"] == "cancel_pending_order")

    def test_independent_sessions_reset_and_source_unchanged(self):
        first, second = RetailSession(self.row), RetailSession(self.row)
        before = second.db_hash()
        path = ROOT / "data/databases" / self.row["db_path"].split("/")[-1]
        source = path.read_bytes()
        run_reference(first, self.row)
        self.assertNotEqual(first.db_hash(), before)
        self.assertEqual(second.db_hash(), before)
        self.assertEqual(path.read_bytes(), source)
        first.reset()
        self.assertEqual(first.db_hash(), before)
        self.assertEqual(first.trajectory, [])

    def test_unknown_missing_extra_and_wrong_type_are_feedback(self):
        session = RetailSession(self.row)
        before = session.db_hash()
        for name, args in [("nonexistent", {}), ("get_order_details", {}),
                           ("get_order_details", {"order_id": "#W1", "extra": 1}),
                           ("get_order_details", {"order_id": 123})]:
            with self.subTest(name=name, args=args):
                self.assertTrue(session.call_tool(name, args).error)
        self.assertEqual(session.db_hash(), before)
        self.assertEqual(len(session.trajectory), 8)

    def test_invalid_order_and_reason_do_not_change_state(self):
        session = RetailSession(self.row)
        before = session.db_hash()
        for args in [{**self.action["arguments"], "order_id": "#W_DOES_NOT_EXIST"},
                     {**self.action["arguments"], "reason": "invalid reason"}]:
            self.assertTrue(session.call_tool(self.action["name"], args).error)
        self.assertEqual(session.db_hash(), before)

    def test_repeat_cancel_and_exchange_are_rejected(self):
        for business in ["cancel_pending_order", "exchange_delivered_order_items"]:
            row = example_task(business)
            session = RetailSession(row)
            run_reference(session, row)
            before = session.db_hash()
            action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] == business)
            self.assertTrue(session.call_tool(action["name"], action["arguments"]).error)
            self.assertEqual(session.db_hash(), before)

    def test_cancel_delivered_order_and_bad_exchange_lengths_fail(self):
        row = example_task("exchange_delivered_order_items")
        action = next(a for a in row["evaluation_criteria"]["actions"]
                      if a["name"] == "exchange_delivered_order_items")
        session = RetailSession(row)
        before = session.db_hash()
        self.assertTrue(session.call_tool("cancel_pending_order", {
            "order_id": action["arguments"]["order_id"], "reason": "no longer needed"}).error)
        self.assertTrue(session.call_tool(action["name"], {
            **action["arguments"], "new_item_ids": []}).error)
        self.assertEqual(session.db_hash(), before)

    def test_public_messages_schema_and_copies(self):
        session = RetailSession(self.row)
        messages = session.agent_messages()
        expected = (ROOT / "data/versions/retail_full_v2/manifests/agent_system.txt").read_text().rstrip("\n")
        self.assertEqual(messages, [{"role": "system", "content": expected}])
        self.assertEqual(session.tool_schemas,
                         json.loads((ROOT / "data/versions/retail_full_v2/manifests/tool_schema.json").read_text()))
        for hidden in ["evaluation_criteria", "user_scenario", "db_path"]:
            self.assertNotIn(hidden, json.dumps(messages))
        session.add_user_message("Please check my order.")
        response = session.call_tool("get_order_details", {
            "order_id": self.action["arguments"]["order_id"]})
        self.assertFalse(response.error)
        response.content = "tampered"
        history = session.trajectory
        history[-1].content = "tampered"
        self.assertNotEqual(session.trajectory[-1].content, "tampered")
        self.assertEqual(session.agent_messages()[-1]["tool_call_id"], history[-1].id)

    def test_internal_error_is_not_business_feedback(self):
        session = RetailSession(self.row)
        with patch.object(session._environment.tools, "use_tool", side_effect=RuntimeError("broken")):
            with self.assertRaises(RuntimeError):
                session.call_tool("get_order_details", {"order_id": self.action["arguments"]["order_id"]})
        self.assertEqual(session.trajectory, [])

    def test_calculator_real_result_and_invalid_expressions(self):
        session = RetailSession(self.row)
        self.assertEqual(session.call_tool("calculate", {"expression": "(12 + 3) / 2"}).content, "7.5")
        for expression in ["1 / 0", "1 +", "__import__('os')"]:
            self.assertTrue(session.call_tool("calculate", {"expression": expression}).error)

    def test_initial_history_replayed_and_kept_on_reset(self):
        first = RetailSession(self.row)
        first.call_tool("get_order_details", {"order_id": self.action["arguments"]["order_id"]})
        first.add_user_message("Please continue.")
        row = deepcopy(self.row)
        row["initial_state"] = {"message_history": [m.model_dump(mode="json") for m in first.trajectory]}
        session = RetailSession(row)
        self.assertEqual(session.trajectory, first.trajectory)
        run_reference(session, row)
        session.reset()
        self.assertEqual(session.db_hash(), first.db_hash())
        self.assertEqual(session.trajectory, first.trajectory)


if __name__ == "__main__":
    unittest.main()
