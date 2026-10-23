"""离线复核 train 与已公开的 dev 疑点；不读取 test，不调用模型。"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import digest, reference_probe, save
from retail_agent.dataset import WRITES
from retail_agent.environment import RetailSession
from retail_agent.reward import evaluate_task

OUT = ROOT / "outputs/review/pre_grpo"
SOURCE = ROOT / "data/versions/retail_full_v2/processed/rl_train.jsonl"


def inspect_task(row):
    session = RetailSession(row)
    db = session._environment.tools.db
    instructions = row["user_scenario"]["instructions"]
    text = json.dumps(instructions, ensure_ascii=False)
    flags, changes, writes = [], [], []
    known = instructions.get("known_info")
    try:
        known = json.loads(known) if isinstance(known, str) else known
    except json.JSONDecodeError:
        known = None
    if isinstance(known, dict):
        for item in known.get("order_items", []):
            order = db.orders.get(item.get("order_id"))
            actual = next((x for x in order.items if x.item_id == item.get("item_id")), None) if order else None
            if actual is None:
                flags.append({"type": "known_item_missing", "claimed": item})
            else:
                actual = actual.model_dump()
                mismatch = {k: {"claimed": item[k], "actual": actual.get(k)}
                            for k in ("name", "options") if k in item and item[k] != actual.get(k)}
                if mismatch:
                    flags.append({"type": "known_item_mismatch", "item_id": item["item_id"], "details": mismatch})
    actions = row["evaluation_criteria"]["actions"]
    for i, action in enumerate(actions):
        name, args = action["name"], action["arguments"]
        if name in WRITES:
            writes.append({"index": i, "tool": name, "arguments": args})
        if name in {"modify_pending_order_items", "exchange_delivered_order_items"}:
            order = db.orders[args["order_id"]]
            for old, new in zip(args["item_ids"], args["new_item_ids"], strict=True):
                item = next(x for x in order.items if x.item_id == old)
                product = db.products[item.product_id]
                variant = product.variants[new]
                changes.append({"order_id": order.order_id, "product": product.name,
                                "old_id": old, "new_id": new,
                                "old": item.options, "new": variant.options,
                                "old_price": item.price, "new_price": variant.price})
        response = session.call_tool(name, args)
        if response.error:
            flags.append({"type": "reference_execution_error", "index": i, "error": response.content})
            break
    # Candidate flags require a human decision; they are not claims of invalidity.
    reason = instructions.get("reason_for_call", "")
    if changes and re.search(r"\b(size|sizes|bigger|larger|smaller)\b", reason, re.I):
        if not any("size" in c["old"] and c["old"].get("size") != c["new"].get("size") for c in changes):
            flags.append({"type": "size_goal_without_size_change"})
    if changes and re.search(r"\b(cheaper|fewer pieces|easier)\b", reason, re.I):
        for c in changes:
            if "cheaper" in reason.lower() and c["new_price"] >= c["old_price"]:
                flags.append({"type": "cheaper_goal_not_cheaper", "change": c})
            if "fewer pieces" in reason.lower() and "pieces" in c["old"] and int(c["new"]["pieces"]) >= int(c["old"]["pieces"]):
                flags.append({"type": "fewer_pieces_goal_not_fewer", "change": c})
    if re.search(r"\b(vague|mismatch|wrong|incorrect|size|sizes|cheaper|conditional|conditions)\b", reason, re.I):
        priority = True
    else:
        priority = False
    # Remove all profile writes: catches the update-then-revert blind spot of DB-only rewards.
    profile = [w for w in writes if w["tool"] == "modify_user_address"]
    omission = None
    if len(profile) > 1:
        omitted = RetailSession(row)
        for action in actions:
            if action["name"] != "modify_user_address":
                omitted.call_tool(action["name"], action["arguments"])
        omission = evaluate_task(row, omitted.trajectory).reward
    return {"id": row["id"], "flags": flags, "priority_read": priority,
            "instructions": instructions, "changes": changes, "writes": writes,
            "omit_all_profile_writes_reward": omission,
            "reference_checks": reference_probe(row)}


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    source_hash = digest(SOURCE)
    cases = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect_task, rows):
            cases.append(result)
            if len(cases) % 25 == 0:
                print(f"reviewed {len(cases)}/{len(rows)}", flush=True)
    assert digest(SOURCE) == source_hash
    save(OUT / "train_audit.json", {"source": str(SOURCE.relative_to(ROOT)), "sha256": source_hash,
        "script_sha256": digest(Path(__file__)), "paid_cost_cny": 0,
        "count": len(cases), "flags": dict(Counter(f["type"] for c in cases for f in c["flags"])),
        "reference_failures": [c["id"] for c in cases if c["reference_checks"]["issues"]],
        "cases": cases})
    # Previously disclosed dev case only; it is never exported into training.
    dev = ROOT / "data/versions/retail_full_v2/processed/rl_dev.jsonl"
    case = next(json.loads(line) for line in dev.read_text().splitlines() if json.loads(line)["id"] == "retail_501")
    save(OUT / "dev_501_audit.json", inspect_task(case))
    print(json.dumps({"cases": len(cases), "flagged": sum(bool(c["flags"]) for c in cases),
                      "reference_failures": sum(bool(c["reference_checks"]["issues"]) for c in cases)}))


if __name__ == "__main__":
    main()
