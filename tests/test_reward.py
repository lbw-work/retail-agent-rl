import unittest
from copy import deepcopy

from retail_agent.data import read_jsonl
from retail_agent.environment import ROOT, RetailSession
from retail_agent.reward import RewardEvaluationError, evaluate_task
from test_environment import example_task, run_reference


class RewardTests(unittest.TestCase):
    def test_correct_reference_and_grading_does_not_modify_live_state(self):
        for business in ["cancel_pending_order", "exchange_delivered_order_items"]:
            row = example_task(business)
            session = RetailSession(row)
            run_reference(session, row)
            before = session.db_hash()
            result = evaluate_task(row, session.trajectory)
            self.assertEqual(result.reward, 1.0)
            self.assertTrue(result.db_check.db_match)
            self.assertEqual(result.reward_breakdown, {"DB": 1.0, "COMMUNICATE": 1.0})
            self.assertEqual(session.db_hash(), before)

    def test_saying_success_and_wrong_target_do_not_pass(self):
        row = example_task()
        session = RetailSession(row)
        session.add_assistant_message("Your order has been cancelled and refunded.")
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 0.0)
        for action in row["evaluation_criteria"]["actions"]:
            args = deepcopy(action["arguments"])
            if action["name"] == row["business"]:
                args["order_id"] = "#W_DOES_NOT_EXIST"
            session.call_tool(action["name"], args)
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 0.0)

    def test_db_reward_allows_different_read_sequence(self):
        row = example_task()
        session = RetailSession(row)
        # 只执行真实写操作；这验证结果判分，不代表符合身份验证/确认规则。
        for action in row["evaluation_criteria"]["actions"]:
            if action["name"] == row["business"]:
                session.call_tool(action["name"], action["arguments"])
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 1.0)

    def test_db_reward_does_not_enforce_authentication_or_confirmation(self):
        row = example_task()
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            if action["name"] == row["business"]:
                self.assertFalse(session.call_tool(action["name"], action["arguments"]).error)
        # No customer messages, identity lookup, or confirmation were supplied.
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 1.0)

    def test_final_state_reward_misses_profile_change_then_revert(self):
        row = next(row for _, row in read_jsonl(
            ROOT / "data/versions/retail_full_v2/processed/rl_train.jsonl") if row["id"] == "retail_124")
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            if action["name"] != "modify_user_address":
                self.assertFalse(session.call_tool(action["name"], action["arguments"]).error)
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 1.0)

    def test_valid_write_with_wrong_business_result_fails(self):
        row = example_task()
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            args = deepcopy(action["arguments"])
            if action["name"] == "cancel_pending_order":
                args["reason"] = ("no longer needed" if args["reason"] == "ordered by mistake"
                                  else "ordered by mistake")
            self.assertFalse(session.call_tool(action["name"], args).error)
        # 工具确实成功，但取消原因不符合任务期望，不能当成业务成功。
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 0.0)

    def test_required_communication_and_action_basis(self):
        row = deepcopy(example_task())
        row["evaluation_criteria"]["communicate_info"] = ["5-7 business days"]
        session = RetailSession(row)
        run_reference(session, row)
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 0.0)
        session.add_user_message("5-7 business days")
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 0.0)
        session.add_assistant_message("It takes 5-7 BUSINESS DAYS.")
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 1.0)
        row["evaluation_criteria"]["reward_basis"] = ["ACTION"]
        self.assertEqual(evaluate_task(row, session.trajectory).reward, 1.0)
        self.assertEqual(evaluate_task(row, []).reward, 0.0)

    def test_termination_and_unimplemented_judge(self):
        row = example_task()
        for reason in ["max_steps", "too_many_errors", "context_limit"]:
            self.assertEqual(evaluate_task(row, [], reason).reward, 0.0)
        for reason in ["infrastructure_error", "unknown"]:
            with self.assertRaises(RewardEvaluationError):
                evaluate_task(row, [], reason)
        row = deepcopy(row)
        row["evaluation_criteria"]["reward_basis"] = ["NL_ASSERTION"]
        with self.assertRaises(RewardEvaluationError):
            evaluate_task(row, [])

    def test_declared_environment_assertion_is_not_ignored(self):
        row = deepcopy(example_task())
        row["evaluation_criteria"]["reward_basis"] = ["ENV_ASSERTION"]
        row["evaluation_criteria"]["env_assertions"] = [{"env_type": "assistant",
            "func_name": "has_tool", "arguments": {"tool_name": "cancel_pending_order"},
            "assert_value": True}]
        self.assertEqual(evaluate_task(row, []).reward, 1.0)
        row["evaluation_criteria"]["env_assertions"][0]["assert_value"] = False
        result = evaluate_task(row, [])
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.env_assertions[0].met)

    def test_broken_reference_and_fabricated_tool_response_raise(self):
        row = example_task()
        session = RetailSession(row)
        run_reference(session, row)
        fake = session.trajectory
        fake[-1].content = "fake success"
        with self.assertRaises(RewardEvaluationError):
            evaluate_task(row, fake)
        broken = deepcopy(row)
        next(a for a in broken["evaluation_criteria"]["actions"]
             if a["name"] == broken["business"])["arguments"]["order_id"] = "#W_DOES_NOT_EXIST"
        with self.assertRaises(RewardEvaluationError):
            evaluate_task(broken, [])


if __name__ == "__main__":
    unittest.main()
