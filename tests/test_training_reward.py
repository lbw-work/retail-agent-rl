"""真实 Retail 工具的离线反例；脚本轨迹不代表模型 rollout 或训练。"""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from retail_agent.dataset import WRITES
from retail_agent.environment import ROOT, RetailSession
from retail_agent.reward import RewardEvaluationError
from retail_agent.training_reward import (
    LEGACY_VERSION, VERSION, audit_scenario_identity, compile_business_goals, evaluate_training_task,
    ready_training_rows, reference_subset, training_db_hash,
)


class TrainingRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "data/versions/retail_full_v2/reviewed/rl_train_verified.jsonl"
        cls.rows = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}

    def row(self, number=264):
        return self.rows[f"retail_{number}"]

    def full(self, row):
        return reference_subset(row, set(compile_business_goals(row)["goals"]))

    def authenticate(self, session, row):
        action = next(a for a in row["evaluation_criteria"]["actions"]
                      if a["name"].startswith("find_user_id_"))
        self.assertFalse(session.call_tool(action["name"], action["arguments"]).error)

    def address_history(self, row, **overrides):
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            args = deepcopy(action["arguments"])
            if action["name"] in {"modify_user_address", "modify_pending_order_address"}:
                args.update(overrides)
            self.assertFalse(session.call_tool(action["name"], args).error)
        return session

    def test_country_alias_success_keeps_official_score_and_live_state(self):
        row = self.row(540)
        for country in ["USA", "US", "United States"]:
            with self.subTest(country=country):
                session = self.address_history(row, country=country)
                before = session.db_hash()
                score = evaluate_training_task(row, session.trajectory)
                self.assertEqual(score["business_reward"], float(country == "USA"))
                self.assertEqual((score["reward_a"], score["reward_b"], score["progress"]), (1, 1, 1))
                self.assertEqual(session.db_hash(), before)

    def test_country_equivalence_does_not_relax_other_address_fields(self):
        row = self.row(540)
        invalid = [{"country": c} for c in ["Canada", "usa", "USA ", "United States of America"]]
        invalid += [{"country": "US", key: value} for key, value in
                    [("address1", "742"), ("address2", ""), ("city", "Boston"),
                     ("state", "MA"), ("zip", "00000")]]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                score = evaluate_training_task(row, self.address_history(row, **overrides).trajectory)
                self.assertEqual((score["reward_a"], score["reward_b"], score["progress"]), (0, 0, 0))

    def test_default_address_country_alias_and_atomic_partial_progress(self):
        score = evaluate_training_task(self.row(131), self.address_history(self.row(131), country="US").trajectory)
        self.assertEqual((score["business_reward"], score["reward_a"]), (0, 1))
        row = self.row(222)
        session = RetailSession(row)
        self.authenticate(session, row)
        action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] == "modify_pending_order_address")
        self.assertFalse(session.call_tool(action["name"], {**action["arguments"], "country": "United States"}).error)
        score = evaluate_training_task(row, session.trajectory)
        self.assertEqual((score["business_reward"], score["reward_a"], score["progress"], score["reward_b"]), (0, 0, .5, .15))

    def test_country_alias_cannot_bypass_authentication(self):
        row = self.row(540)
        session = RetailSession(row)
        action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] == "modify_pending_order_address")
        self.assertFalse(session.call_tool(action["name"], {**action["arguments"], "country": "US"}).error)
        score = evaluate_training_task(row, session.trajectory)
        self.assertEqual((score["business_reward"], score["gate"], score["reward_a"], score["reward_b"]), (0, 0, 0, 0))

    def test_country_alias_legacy_replay_and_unknown_version(self):
        row = self.row(540)
        history = self.address_history(row, country="United States").trajectory
        legacy = evaluate_training_task(row, history, version=LEGACY_VERSION)
        current = evaluate_training_task(row, history)
        self.assertEqual((legacy["version"], legacy["reward_a"], legacy["progress"]), (LEGACY_VERSION, 0, 0))
        self.assertNotIn("training_business_reward", legacy)
        self.assertEqual((current["version"], current["business_reward"], current["training_business_reward"]), (VERSION, 0, 1))
        with self.assertRaisesRegex(RewardEvaluationError, "未知训练奖励版本"):
            evaluate_training_task(row, history, version="unknown")
        for reason in ["context_limit", "generation_limit", "invalid_model_output"]:
            score = evaluate_training_task(row, history, reason)
            self.assertEqual((score["training_business_reward"], score["reward_a"], score["reward_b"]), (0, 0, 0))

    def test_country_normalization_preserves_money_and_unrelated_database_state(self):
        session = self.full(self.row(196))
        before = session._environment.tools.db.model_dump()
        expected = training_db_hash(session)
        self.assertEqual(session._environment.tools.db.model_dump(), before)
        db = session._environment.tools.db
        card = next(p for u in db.users.values() for p in u.payment_methods.values() if p.source == "gift_card")
        card.balance += .01
        self.assertNotEqual(training_db_hash(session), expected)
        card.balance -= .01
        other = next(u for u in db.users.values() if u.user_id != compile_business_goals(self.row(196))["owner"])
        other.email = "changed@example.invalid"
        self.assertNotEqual(training_db_hash(session), expected)

    def test_country_alias_does_not_allow_non_target_address_write(self):
        row = self.row(540)
        session = self.address_history(row, country="US")
        owner = compile_business_goals(row)["owner"]
        original = session._environment.tools.db.users[owner].address.model_dump()
        replacement = "US" if original["country"] != "US" else "USA"
        self.assertFalse(session.call_tool("modify_user_address", {"user_id": owner, **original, "country": replacement}).error)
        score = evaluate_training_task(row, session.trajectory)
        self.assertIn("successful_non_target_write", score["violations"])
        self.assertEqual((score["gate"], score["reward_a"], score["reward_b"]), (0, 0, 0))

    def test_actual_customer_email_conflict_is_not_fixed_by_valid_reference(self):
        row = self.row(494)
        result = audit_scenario_identity(row)
        self.assertEqual(result["owner"], "bob_harris_382")
        self.assertIn("known_email_does_not_authenticate_target", result["issues"])
        email_route = next(r for r in result["routes"] if r["tool"] == "find_user_id_by_email")
        self.assertEqual(email_route["returned_identity"], "bob_ellis_1507")
        self.assertEqual(evaluate_training_task(row, self.full(row).trajectory)["reward_a"], 1.)
        ready = ready_training_rows(ROOT / "data/versions/retail_full_v2/reviewed/rl_train_ready.jsonl")
        self.assertEqual(len(ready), 29)
        self.assertNotIn("retail_494", {r["id"] for r in ready})
        self.assertEqual({b for r in ready for b in r["businesses"]}, set(WRITES))
        self.assertTrue(all(r == self.rows[r["id"]] for r in ready))
        with self.assertRaisesRegex(RewardEvaluationError, "训练池"):
            ready_training_rows(ROOT / "data/versions/retail_full_v2/reviewed/rl_train_verified.jsonl")
        with patch("retail_agent.training_reward.audit_scenario_identity", return_value={"issues": ["changed_identity"]}):
            with self.assertRaisesRegex(RewardEvaluationError, "身份矛盾"):
                ready_training_rows(ROOT / "data/versions/retail_full_v2/reviewed/rl_train_ready.jsonl")

    def test_pool_exactly_projects_train_and_covers_all_businesses(self):
        path = ROOT / "data/versions/retail_full_v2/processed/rl_train.jsonl"
        source = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
        self.assertEqual(len(self.rows), 30)
        self.assertEqual({b for r in self.rows.values() for b in r["businesses"]}, set(WRITES))
        self.assertNotIn("retail_7", self.rows)
        self.assertNotIn("retail_10", self.rows)
        for task_id, row in self.rows.items():
            self.assertEqual(row, source[task_id])
        manifest = json.loads((ROOT / "outputs/review/pre_grpo/review_manifest.json").read_text())
        status = {e["id"]: e["status"] for e in manifest["entries"]}
        self.assertEqual(status["retail_10"], "quarantined_for_review")
        self.assertEqual(status["retail_7"], "quarantined_for_review")

    def test_full_reward_and_no_live_state_mutation(self):
        row = self.row()
        session = self.full(row)
        before = session.db_hash()
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual((result["business_reward"], result["reward_a"], result["reward_b"]), (1, 1, 1))
        self.assertEqual(session.db_hash(), before)

    def test_half_business_reward_and_repeated_queries(self):
        row = self.row(213)
        goal = next(g for g in compile_business_goals(row)["goals"] if g[0] == "modify_pending_order_payment")
        session = reference_subset(row, {goal})
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual((result["reward_a"], result["progress"], result["reward_b"]), (0, .5, .15))
        for _ in range(3):
            session.call_tool("get_order_details", {"order_id": goal[1]})
        session.add_assistant_message("Everything is successfully completed.")
        self.assertEqual(evaluate_training_task(row, session.trajectory)["reward_b"], .15)

    def test_queries_and_speech_do_not_receive_progress(self):
        row = self.row()
        session = reference_subset(row, set())
        session.add_assistant_message("Your order has been successfully changed.")
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual((result["progress"], result["reward_a"], result["reward_b"]), (0, 0, 0))

    def test_atomic_item_batch_not_split_into_two_rewards(self):
        row = self.row(410)
        session = RetailSession(row)
        self.authenticate(session, row)
        action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] in WRITES)
        args = deepcopy(action["arguments"])
        args["item_ids"], args["new_item_ids"] = args["item_ids"][:1], args["new_item_ids"][:1]
        self.assertFalse(session.call_tool(action["name"], args).error)
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual((result["goal_count"], result["progress"], result["reward_b"]), (1, 0, 0))

    def test_wrong_remaining_state_does_not_pass_strict_subset(self):
        row = self.row(213)
        goal = next(g for g in compile_business_goals(row)["goals"] if g[0] == "modify_pending_order_payment")
        session = reference_subset(row, {goal})
        action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] == "modify_pending_order_address")
        self.assertFalse(session.call_tool(action["name"], {**action["arguments"], "address1": "Wrong street"}).error)
        self.assertEqual(evaluate_training_task(row, session.trajectory)["reward_b"], 0)

    def test_unauthenticated_or_failed_authentication_writes_gated(self):
        row = self.row()
        for failed_auth in [False, True]:
            session = RetailSession(row)
            if failed_auth:
                self.assertTrue(session.call_tool("find_user_id_by_email", {"email": "missing@example.invalid"}).error)
            for action in row["evaluation_criteria"]["actions"]:
                if action["name"] in WRITES:
                    session.call_tool(action["name"], action["arguments"])
            result = evaluate_training_task(row, session.trajectory)
            self.assertEqual((result["business_reward"], result["gate"], result["reward_b"]), (1, 0, 0))

    def test_private_read_before_authentication_not_repaired_by_later_auth(self):
        row = self.row()
        session = RetailSession(row)
        action = next(a for a in row["evaluation_criteria"]["actions"] if a["name"] in WRITES)
        session.call_tool("get_order_details", {"order_id": action["arguments"]["order_id"]})
        for a in row["evaluation_criteria"]["actions"]:
            session.call_tool(a["name"], a["arguments"])
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual(result["business_reward"], 1)
        self.assertIn("private_access_before_authentication", result["violations"])
        self.assertEqual(result["reward_a"], 0)

    def test_other_users_private_read_and_identity_switch_gated(self):
        row = self.row()
        session = self.full(row)
        owner = compile_business_goals(row)["owner"]
        other = next(u for u in session._environment.tools.db.users.values() if u.user_id != owner and u.orders)
        session.call_tool("get_order_details", {"order_id": other.orders[0]})
        session.call_tool("get_user_details", {"user_id": other.user_id})
        session.call_tool("find_user_id_by_email", {"email": other.email})
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual(result["business_reward"], 1)
        self.assertIn("order_not_owned_by_authenticated_user", result["violations"])
        self.assertIn("profile_not_owned_by_authenticated_user", result["violations"])
        self.assertIn("identity_switch", result["violations"])
        self.assertEqual(result["reward_b"], 0)

    def test_non_target_write_then_revert_still_gated(self):
        row = self.row()
        session = RetailSession(row)
        self.authenticate(session, row)
        owner = compile_business_goals(row)["owner"]
        original = session._environment.tools.db.users[owner].address.model_dump()
        session.call_tool("modify_user_address", {"user_id": owner, **original, "address1": "Unrequested street"})
        session.call_tool("modify_user_address", {"user_id": owner, **original})
        for a in row["evaluation_criteria"]["actions"]:
            session.call_tool(a["name"], a["arguments"])
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual(result["business_reward"], 1)
        self.assertIn("successful_non_target_write", result["violations"])
        self.assertEqual(result["reward_b"], 0)

    def test_reverted_target_goal_loses_its_progress(self):
        row = self.row(222)
        session = self.full(row)
        goal = compile_business_goals(row)["goals"][0]
        original = RetailSession(row)._environment.tools.db.orders[goal[1]].address.model_dump()
        session.call_tool(goal[0], {"order_id": goal[1], **original})
        result = evaluate_training_task(row, session.trajectory)
        self.assertEqual((result["business_reward"], result["progress"], result["reward_b"]), (0, .5, .15))

    def test_different_authentication_reads_and_valid_write_order_allowed(self):
        row = self.row(16)
        session = RetailSession(row)
        user = session._environment.tools.db.users[compile_business_goals(row)["owner"]]
        session.call_tool("find_user_id_by_email", {"email": user.email})
        session.call_tool("get_user_details", {"user_id": user.user_id})
        for a in reversed(row["evaluation_criteria"]["actions"]):
            if a["name"] in WRITES:
                session.call_tool(a["name"], a["arguments"])
        # 该第三笔 pending 地址已正确；显式设置同一地址也应被接受。
        order = session._environment.tools.db.orders[user.orders[0]]
        session.call_tool("modify_pending_order_address", {"order_id": order.order_id, **order.address.model_dump()})
        self.assertEqual(evaluate_training_task(row, session.trajectory)["reward_b"], 1)

    def test_invalid_parameter_feedback_can_be_corrected(self):
        row = self.row()
        session = RetailSession(row)
        self.authenticate(session, row)
        self.assertTrue(session.call_tool("get_order_details", {"order_id": []}).error)
        self.assertTrue(session.call_tool("modify_pending_order_items", {}).error)
        for a in row["evaluation_criteria"]["actions"]:
            session.call_tool(a["name"], a["arguments"])
        self.assertEqual(evaluate_training_task(row, session.trajectory)["reward_b"], 1)

    def test_fabricated_tool_content_and_error_flag_raise(self):
        row = self.row()
        for field, value in [("content", "fake success"), ("error", True)]:
            history = self.full(row).trajectory
            setattr(history[-1], field, value)
            with self.assertRaises(RewardEvaluationError):
                evaluate_training_task(row, history)

    def test_missing_reply_and_duplicate_call_id_raise(self):
        row = self.row()
        history = self.full(row).trajectory
        with self.assertRaises(RewardEvaluationError):
            evaluate_training_task(row, history[:-1])
        history[2].tool_calls[0].id = history[0].tool_calls[0].id
        history[3].id = history[0].tool_calls[0].id
        with self.assertRaisesRegex(RewardEvaluationError, "重复"):
            evaluate_training_task(row, history)

    def test_cutoffs_zero_and_service_faults_raise(self):
        row = self.row(213)
        goal = compile_business_goals(row)["goals"][0]
        history = reference_subset(row, {goal}).trajectory
        for reason in ["max_steps", "context_limit", "generation_limit", "invalid_model_output"]:
            result = evaluate_training_task(row, history, reason)
            self.assertEqual((result["reward_a"], result["reward_b"]), (0, 0))
        for reason in ["infrastructure_error", "customer_service_error", "unknown"]:
            with self.assertRaises(RewardEvaluationError):
                evaluate_training_task(row, history, reason)

    def test_unreviewed_or_changed_source_not_used_for_training(self):
        changed = deepcopy(self.row())
        changed["user_scenario"]["instructions"]["task_instructions"] += " changed"
        with self.assertRaises(RewardEvaluationError):
            evaluate_training_task(changed, [])
        raw = ROOT / "data/versions/retail_full_v2/processed/rl_train.jsonl"
        row = next(r for r in map(json.loads, raw.read_text().splitlines()) if r["id"] == "retail_10")
        with self.assertRaises(RewardEvaluationError):
            evaluate_training_task(row, [])

    def test_gift_card_dependency_excludes_impossible_subset(self):
        row = self.row(196)
        spec = compile_business_goals(row)
        payment = next(g for g in spec["goals"] if g[0] == "modify_pending_order_payment")
        self.assertTrue(spec["rejected_subsets"])
        with self.assertRaises(RewardEvaluationError):
            reference_subset(row, {payment})
        session = self.full(row)
        self.assertEqual(evaluate_training_task(row, session.trajectory)["reward_a"], 1)


if __name__ == "__main__":
    unittest.main()
