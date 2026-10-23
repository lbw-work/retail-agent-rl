"""05A：输出逐条语义核对的原始 TRAIN 子集与真实离线奖励验收证据。"""

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from finalize_rl_review import MANUAL_QUARANTINE, REVIEWED_SMOKE, main as finalize_review
from review_rl_data import ROOT, SOURCE, digest, save
from retail_agent.dataset import WRITES
from retail_agent.environment import RetailSession
from retail_agent.reward import evaluate_task
from retail_agent.training_reward import (
    MANIFEST, VERSION, audit_scenario_identity, compile_business_goals, evaluate_training_task,
    reference_subset, row_fingerprint,
)

OUT = ROOT / "outputs/review/phase05a"
REVIEWED = {
    **REVIEWED_SMOKE,
    "retail_264": "只把pending蓝瑜伽垫改粉色，不改水壶；唯一在库粉色变体，唯一原信用卡",
    "retail_204": "拒绝已送达订单改址；两笔 pending 全取消，理由 no longer needed，原付款退款",
    "retail_317": "只改默认地址；新 Seattle 地址完整来自 known_info，不更改已有订单",
    "retail_381": "初始 Rehob 缺州；原文明确由顾客补充 Rehoboth Beach/Delaware/无房号",
    "retail_449": "顾客补齐 Rehoboth Beach/DE 后改默认地址；订单只是背景，无订单写目标",
    "retail_457": "T-Shirt 改 Running Shoes 应拒绝；仅默认地址改 Seattle，地址信息可获得",
    "retail_222": "两笔 pending 改同一 Austin 地址；第三笔 delivered 保持不变",
    "retail_453": "两笔 pending 改同一 Maple Drive 地址，顾客新地址完整",
    "retail_256": "只改指定 pending 至 Visa9876；礼品卡150不足340.71，另一 delivered 不改",
    "retail_19": "两笔 pending 分别改 PayPal 和礼品卡；礼品卡2500可覆盖583.17",
    "retail_66": "拒绝 delivered 改址；pending Grill 改礼品卡，1200足够1031",
    "retail_84": "单 pending 三种商品匹配；Visa1234改9876，跑鞋颜色只咨询不改商品",
    "retail_158": "拒绝另一 delivered 改址；仅 Grill 退现有礼品卡，Patio Umbrella 不退",
    "retail_401": "拒绝 processed 改址；两笔 delivered 全退，明确退原礼品卡/原信用卡",
    "retail_278": "两笔 delivered 全退，原文指定原支付方式；分别信用卡与礼品卡",
    "retail_16": "三笔 pending 中一笔已有DC地址；另两笔及默认地址更新，原邮编85032可核验",
    "retail_148": "三笔 pending 中一笔已有DC地址；另两笔及默认地址更新，原邮编78205可核验",
    "retail_180": "DC完整地址在原订单；另两笔pending和默认地址改至该地址，78205核验一致",
    "retail_197": "DC地址从原订单读出；另两笔pending和默认地址更新，85001核验一致",
    "retail_208": "DC地址来自一笔已有pending；另两笔和默认地址更新，78205核验一致",
    "retail_113": "两笔 pending 礼品卡改信用卡，另一笔取消；无效原因须澄清为 no longer needed",
    "retail_196": "两笔 Smart Watch 取消后礼品卡余额足够，第三笔98.69付款改礼品卡，有顺序依赖",
    "retail_213": "Backpack订单改礼品卡，650覆盖231.67；Office Chair订单改完整Austin地址",
}


def counterexamples(rows):
    evidence = {}
    row = rows["retail_10"]
    puzzles = []
    for variant_id in ["1096508426", "4068787148"]:
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            args = deepcopy(action["arguments"])
            if action["name"] == "modify_pending_order_items":
                args["new_item_ids"] = [variant_id]
            assert not session.call_tool(action["name"], args).error
        order = session._environment.tools.db.orders["#W9686481"]
        item = next(i for i in order.items if i.item_id == variant_id)
        puzzles.append({"variant_id": variant_id, "options": item.options,
                        "reward": evaluate_task(row, session.trajectory).reward,
                        "trajectory": [m.model_dump(mode="json") for m in session.trajectory]})
    assert [p["reward"] for p in puzzles] == [1, 0]
    evidence["retail_10"] = {"reason": MANUAL_QUARANTINE["retail_10"], "alternatives": puzzles}
    row = rows["retail_7"]
    results = []
    for variant_id in ["4273929280", "7609274509"]:
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            args = deepcopy(action["arguments"])
            if action["name"] == "exchange_delivered_order_items":
                args["new_item_ids"][1] = variant_id
            assert not session.call_tool(action["name"], args).error
        product = session._environment.tools.db.products["3801771308"]
        variant = product.variants[variant_id]
        results.append({"variant_id": variant_id, "available": variant.available,
                        "options": variant.options, "reward": evaluate_task(row, session.trajectory).reward,
                        "trajectory": [m.model_dump(mode="json") for m in session.trajectory]})
    assert [r["reward"] for r in results] == [1, 0]
    evidence["retail_7"] = {"reason": MANUAL_QUARANTINE["retail_7"], "alternatives": results}
    for task_id in ["retail_104", "retail_223"]:
        row = rows[task_id]
        assert not audit_scenario_identity(row)["issues"], task_id
        session = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            assert not session.call_tool(action["name"], action["arguments"]).error
        order_id = "#W2734180" if task_id == "retail_104" else "#W7269489"
        order = session._environment.tools.db.orders[order_id]
        user = session._environment.tools.db.users[order.user_id]
        gift = next(p for p in user.payment_methods.values() if p.source == "gift_card")
        balance = gift.balance
        before_reward = evaluate_task(row, session.trajectory).reward
        response = session.call_tool("modify_pending_order_payment", {
            "order_id": order_id, "payment_method_id": gift.id})
        after_reward = evaluate_task(row, session.trajectory).reward
        assert not response.error and before_reward == 1 and after_reward == 0
        evidence[task_id] = {"reason": MANUAL_QUARANTINE[task_id],
            "balance_after_refunds": balance, "order_id": order_id,
            "reference_reward": before_reward, "all_requested_writes_reward": after_reward,
            "trajectory": [m.model_dump(mode="json") for m in session.trajectory]}
    return evidence


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    assert len(REVIEWED) == 30 and not set(REVIEWED) & set(MANUAL_QUARANTINE)
    version = MANIFEST.parent
    immutable = [p for p in (version.parent / "processed").glob("*.jsonl")]
    immutable += list((ROOT / "data/databases").glob("*.json"))
    immutable += [ROOT / "outputs/budget/deepseek_requests.jsonl",
                  ROOT / "outputs/evaluation/dev_original_sft_20261012/summary.json"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in immutable}
    evidence = counterexamples(rows)
    old_manifest = json.loads((ROOT / "outputs/review/pre_grpo/review_manifest.json").read_text())
    old_candidates = {e["id"] for e in old_manifest["entries"] if e["status"] != "quarantined_for_review"}
    selected, entries = [], []
    for task_id, note in REVIEWED.items():
        assert task_id in old_candidates
        row = rows[task_id]
        spec = compile_business_goals(row)
        session = RetailSession(row)
        db = session._environment.tools.db
        known = json.loads(row["user_scenario"]["instructions"]["known_info"])
        user = db.users[spec["owner"]]
        if "user_id" in known:
            assert known["user_id"] == user.user_id, task_id
        auth = [a for a in row["evaluation_criteria"]["actions"] if a["name"].startswith("find_user_id_")]
        assert auth and all(a["arguments"].get("email", user.email) == user.email for a in auth)
        for a in auth:
            if a["name"] == "find_user_id_by_name_zip":
                assert known.get("zip") == a["arguments"]["zip"] == user.address.zip, task_id
        selected.append(row)
        entries.append({"id": task_id, "row_sha256": row_fingerprint(row),
                        "businesses": row["businesses"], "difficulty": row["difficulty"],
                        "group": row["group"], "manual_note": note,
                        "goals": spec["goals"], "rejected_subsets": spec["rejected_subsets"]})
        print(f"reviewed {len(selected)}/30 {task_id}", flush=True)
    pool = version / "rl_train_verified.jsonl"
    pool.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    save(MANIFEST, {"version": VERSION, "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE), "pool_sha256": digest(pool),
        "immutable_sha256": {k: v for k, v in before.items() if k.startswith("data/")},
        "review_method": "assistant_source_read_and_database_crosscheck_plus_offline_replay",
        "count": len(entries), "entries": entries, "paid_cost_cny": 0})
    matrix = []
    for row in selected:
        spec = compile_business_goals(row)
        for variant in spec["variants"]:
            session = reference_subset(row, set(variant["goals"]))
            result = evaluate_training_task(row, session.trajectory)
            fraction = len(variant["goals"]) / len(spec["goals"])
            assert result["gate"] == 1 and result["progress"] == fraction, result
            assert result["reward_a"] == float(fraction == 1), result
            assert result["reward_b"] == (1.0 if fraction == 1 else 0.3 * fraction), result
            matrix.append(result)
        writes_only = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            if action["name"] in WRITES:
                assert not writes_only.call_tool(action["name"], action["arguments"]).error
        result = evaluate_training_task(row, writes_only.trajectory)
        assert result["business_reward"] == 1 and result["reward_a"] == result["reward_b"] == 0
        matrix.append(result)
        print(f"reward checks {row['id']}", flush=True)
    finalize_review()
    after = {str(p.relative_to(ROOT)): digest(p) for p in immutable}
    assert before == after, "原始数据、快照、旧开发成绩、费用或起步集被改动"
    for task_id in MANUAL_QUARANTINE:
        evidence.setdefault(task_id, {"reason": MANUAL_QUARANTINE[task_id],
            "row_sha256": row_fingerprint(rows[task_id]),
            "instructions": rows[task_id]["user_scenario"]["instructions"],
            "reference_actions": rows[task_id]["evaluation_criteria"]["actions"],
            "note": "原文/快照语义疑点；其余条目没有额外对照轨迹，不是模型试跑"})
    save(OUT / "counterexamples.json", {"kind": "scripted_environment_replay_not_model_rollout",
                                       "source_sha256": digest(SOURCE), "cases": evidence})
    save(OUT / "reward_acceptance.json", {"version": VERSION, "status": "offline_passed",
        "kind": "scripted_reference_subsets_not_model_rollout", "paid_cost_cny": 0,
        "task_count": len(selected), "case_count": len(matrix), "cases": matrix,
        "difficulty_counts": dict(Counter(r["difficulty"] for r in selected)),
        "business_coverage": dict(Counter(b for r in selected for b in r["businesses"])),
        "immutable_sha256": before, "source_unchanged": before == after,
        "previous_review_counts": old_manifest["counts"],
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT / "src/retail_agent/training_reward.py", ROOT / "scripts/finalize_rl_review.py"]}})
    print(json.dumps({"verified_tasks": len(selected), "reward_cases": len(matrix), "paid_cost_cny": 0}))


if __name__ == "__main__":
    main()
