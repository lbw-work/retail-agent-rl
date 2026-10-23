"""将离线审计结果转换为可追溯的 GRPO 训练清单。"""
import json
import re
from collections import Counter
from pathlib import Path

from review_rl_data import OUT, ROOT, SOURCE, digest, save
from retail_agent.training_reward import row_fingerprint

REVIEWED_SMOKE = {
    "retail_131": "默认地址目标完整，目标用户与数据库相符",
    "retail_174": "两笔已送达订单及各自全部退货商品相符",
    "retail_301": "头盔白色 M 码是现货，订单及支付方式相符",
    "retail_410": "外套和跑鞋目标选项均有现货，单次合并修改相符",
    "retail_444": "两笔待处理订单可取消，已送达订单的地址请求应拒绝",
    "retail_494": "原礼品卡、目标信用卡及订单状态相符",
    "retail_540": "新地址和目标待处理订单相符",
}

# 05A 对照原文与快照发现的疑点；复现证据见 outputs/review/phase05a/。
MANUAL_QUARANTINE = {
    "retail_10": "single_gold_rejects_another_easier_fewer_piece_puzzle",
    "retail_7": "single_gold_rejects_explicitly_allowed_variant",
    "retail_104": "refund_funded_payment_goal_omitted",
    "retail_223": "refund_funded_payment_goal_omitted",
    "retail_134": "available_exchange_goal_omitted",
    "retail_163": "claimed_failed_payment_not_in_snapshot",
    "retail_274": "described_delivered_damage_items_not_in_snapshot",
    "retail_207": "known_authentication_zip_conflicts_with_profile",
    "retail_488": "payment_revert_goal_omitted",
    "retail_563": "payment_revert_goal_omitted",
    "retail_24": "return_quantity_and_order_description_conflict_with_snapshot",
    "retail_25": "exchange_request_but_reference_is_return",
    "retail_550": "non_delivered_shoes_request_but_no_shoes_in_snapshot",
    "retail_185": "four_order_requests_but_only_three_orders",
    "retail_506": "known_user_id_conflicts_with_authenticated_identity",
    "retail_551": "known_user_id_conflicts_with_authenticated_identity",
}


def mentioned_missing_products(instructions, owned_names, catalog_names):
    text = instructions.get("task_instructions", "").lower()
    missing = []
    for name in catalog_names:
        phrase = re.escape(name.lower())
        if name == "Bookshelf":
            phrase = r"bookshel(?:f|ves)"
        elif name == "Smart Watch":
            phrase = r"smart\s?watch"
        elif name == "T-Shirt":
            phrase = r"t[- ]?shirts?"
        if re.search(r"\b" + phrase + r"s?\b", text) and name not in owned_names:
            missing.append(name)
    return sorted(missing)


def address_without_source(instructions, writes, addresses):
    text = json.dumps(instructions, ensure_ascii=False).lower()
    fields = ("address1", "address2", "city", "state", "zip")
    result = []
    for write in writes:
        if write["tool"] not in {"modify_user_address", "modify_pending_order_address"}:
            continue
        args = write["arguments"]
        if any(all(args[k] == addr[k] for k in fields) for addr in addresses):
            continue
        missing = [k for k in fields if args[k] and str(args[k]).lower() not in text]
        if missing:
            result.append({"tool_index": write["index"], "missing_fields": missing})
    return result


def main():
    audit = json.loads((OUT / "train_audit.json").read_text())
    assert audit["sha256"] == digest(SOURCE)
    cases = {case["id"]: case for case in audit["cases"]}
    assert len(cases) == audit["count"] == 364
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    verified_path = ROOT / "data/versions/retail_full_v2/reviewed/phase05a_manifest.json"
    verified = {}
    if verified_path.exists():
        manifest = json.loads(verified_path.read_text())
        assert manifest["source_sha256"] == digest(SOURCE)
        verified = {entry["id"]: entry for entry in manifest["entries"]}
    databases = {path.name: json.loads(path.read_text()) for path in (ROOT / "data/databases").glob("*.json")}
    entries, screened, smoke = [], [], []
    for row in rows:
        case = cases[row["id"]]
        database = databases[Path(row["db_path"]).name]
        order_ids = {a["arguments"]["order_id"] for a in row["evaluation_criteria"]["actions"]
                     if "order_id" in a["arguments"]}
        user_ids = {database["orders"][oid]["user_id"] for oid in order_ids}
        user_ids.update(a["arguments"]["user_id"] for a in row["evaluation_criteria"]["actions"]
                        if a["name"] == "get_user_details")
        users = [database["users"][uid] for uid in user_ids]
        orders = [database["orders"][oid] for user in users for oid in user["orders"]]
        names = {item["name"] for order in orders for item in order["items"]}
        addresses = [user["address"] for user in users] + [order["address"] for order in orders]
        reasons = [flag["type"] for flag in case["flags"]]
        if row["id"] in MANUAL_QUARANTINE:
            reasons.append(MANUAL_QUARANTINE[row["id"]])
        missing_products = mentioned_missing_products(case["instructions"], names,
                                                     {p["name"] for p in database["products"].values()})
        if missing_products:
            reasons.append("mentioned_product_not_owned_requires_review")
        missing_addresses = address_without_source(case["instructions"], case["writes"], addresses)
        if missing_addresses:
            reasons.append("gold_address_not_fully_grounded_requires_review")
        if case["omit_all_profile_writes_reward"] == 1:
            reasons.append("profile_write_omission_still_full_reward")
        if row["id"] == "retail_500":
            # The written scenario asks for phone address/storage changes; gold only edits the keyboard.
            reasons.append("gold_omits_explicit_phone_goals")
        if case["reference_checks"]["issues"]:
            reasons.append("reference_execution_or_reward_failure")
        reasons = sorted(set(reasons))
        if row["id"] in REVIEWED_SMOKE:
            assert not reasons, (row["id"], reasons)
            status = "manually_reviewed_smoke"
            smoke.append(row)
        elif row["id"] in verified:
            assert not reasons and verified[row["id"]]["row_sha256"] == row_fingerprint(row), row["id"]
            status = "manually_reviewed_verified"
        elif reasons:
            status = "quarantined_for_review"
        else:
            status = "screened_candidate"
        if status != "quarantined_for_review":
            screened.append(row)
        entries.append({"id": row["id"], "status": status, "reasons": reasons,
                        "missing_product_mentions": missing_products,
                        "ungrounded_address_fields": missing_addresses,
                        "manual_note": REVIEWED_SMOKE.get(row["id"]) or
                                       verified.get(row["id"], {}).get("manual_note")})
    assert len(entries) == len(rows) == len({e["id"] for e in entries})
    assert {r["id"] for r in smoke} == set(REVIEWED_SMOKE)
    version = ROOT / "data/versions/retail_full_v2/reviewed"
    version.mkdir(parents=True, exist_ok=True)
    for name, subset in (("rl_train_screened.jsonl", screened), ("rl_train_smoke.jsonl", smoke)):
        (version / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset))
    save(OUT / "review_manifest.json", {"source": str(SOURCE.relative_to(ROOT)),
         "source_sha256": digest(SOURCE), "audit_sha256": digest(OUT / "train_audit.json"),
         "script_sha256": digest(Path(__file__)), "paid_cost_cny": 0,
         "counts": dict(Counter(e["status"] for e in entries)),
         "reason_counts": dict(Counter(reason for e in entries for reason in e["reasons"])),
         "screened_sha256": digest(version / "rl_train_screened.jsonl"),
         "smoke_sha256": digest(version / "rl_train_smoke.jsonl"), "entries": entries})
    print(json.dumps({"counts": dict(Counter(e["status"] for e in entries)),
                      "reason_counts": dict(Counter(reason for e in entries for reason in e["reasons"]))}))


if __name__ == "__main__":
    main()
