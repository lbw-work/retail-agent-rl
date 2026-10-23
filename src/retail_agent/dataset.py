"""新版数据的离线筛查与整组划分；不生成顾客回复或业务答案。"""

import json
import random
import re
from collections import Counter


WRITES = (
    "cancel_pending_order", "exchange_delivered_order_items",
    "return_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
)
RATIOS = {"train": .70, "dev": .15, "test": .15}


def task_business(actions):
    mutations = sorted({a["name"] for a in actions if a["name"] in WRITES})
    level = "single" if len(mutations) == 1 else "pair" if len(mutations) == 2 else "complex" if mutations else "other"
    return mutations, level


def length_bucket(tokens):
    if tokens <= 8192:
        return "up_to_8k"
    if tokens <= 12288:
        return "8k_to_12k"
    return "12k_to_16k"


def confirmed(text):
    # Conservative lexical gate, not a semantic proof of transaction-specific consent.
    if re.search(r"\b(no|not yet|do not|don't|wait|hold on)\b", text, re.I):
        return False
    return bool(re.search(r"\b(yes|yeah|yep|confirm(?:ed)?|proceed|go ahead|please do|do it|sounds good|that's correct|that is correct|sure)\b", text, re.I))


def sft_policy_flags(history, target):
    """检查调用返回配对、身份查询和写操作前明确确认；不伪造缺失回合。"""
    flags, pending, authenticated, latest_user = set(), None, False, ""
    for message in [*history, target]:
        role = message["role"]
        if pending and role != "tool":
            flags.add("missing_tool_response")
        if role == "user":
            latest_user = message["content"]
        elif role == "tool":
            if pending != message.get("name"):
                flags.add("unpaired_tool_response")
            if message.get("name") in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
                response = message["content"].lower()
                if pending == message["name"] and response.strip() and not any(x in response for x in ("error", "not found", "null")):
                    authenticated = True
            pending = None
        elif role == "assistant" and message.get("tool_calls"):
            name = message["tool_calls"][0]["function"]["name"]
            if name in WRITES or name in {"get_user_details", "get_order_details", "get_product_details"}:
                if not authenticated:
                    flags.add("missing_prior_authentication")
            if name in WRITES and not confirmed(latest_user):
                flags.add("write_without_explicit_confirmation")
            pending = name
    # Target calls intentionally have no response yet.
    if pending and not target.get("tool_calls"):
        flags.add("missing_tool_response")
    return sorted(flags)


def task_spec_flags(task):
    """隔离可明确定位的原因歧义；未命中不代表任务语义完全正确。"""
    text = json.dumps(task["user_scenario"], ensure_ascii=False).lower()
    reasons = {a["arguments"]["reason"] for a in task["evaluation_criteria"]["actions"]
               if a["name"] == "cancel_pending_order"}
    flags = []
    for reason in sorted(reasons):
        other = "no longer needed" if reason == "ordered by mistake" else "ordered by mistake"
        if other in text and reason in text:
            flags.append("cancellation_reason_multiple_options_requires_review")
        elif reason not in text:
            synonyms = (r"\b(mistake|mistakenly|accidentally|accidental)\b" if reason == "ordered by mistake"
                        else r"\b(no longer|don't need|do not need|not need|unneeded|unnecessary)\b")
            if not re.search(synonyms, text):
                flags.append("cancellation_reason_not_grounded_requires_review")
    if task["id"] == "retail_130":
        flags.append("reviewed_refund_paypal_goal_but_credit_card_reference")
    return sorted(set(flags))


def assign_groups(members, anchors, seed=42):
    """固定旧留出侧，联合平衡 SFT/RL 数量与业务覆盖；从不拆关联组。"""
    conflicts = {g: sorted(sides) for g, sides in anchors.items() if len(sides) > 1}
    assignments = {g: next(iter(sides)) for g, sides in anchors.items() if len(sides) == 1}
    totals = Counter(feature for rows in members.values() for row in rows for feature in row["features"])
    counts = {side: Counter() for side in RATIOS}
    for group, side in assignments.items():
        for row in members.get(group, []):
            counts[side].update(row["features"])
    rng = random.Random(seed)
    tie_break = {g: rng.random() for g in sorted(members)}
    order = sorted((g for g in members if g not in assignments and g not in conflicts),
                   key=lambda g: (-len(members[g]), tie_break[g], g))
    for group in order:
        additions = Counter(f for row in members[group] for f in row["features"])
        def change(side):
            # Normalize each feature so common text turns do not erase rare business coverage.
            return sum(((counts[side][f] + n - totals[f]*RATIOS[side])**2
                        - (counts[side][f] - totals[f]*RATIOS[side])**2) / max(totals[f], 1)
                       for f, n in additions.items())
        side = min(RATIOS, key=lambda s: (change(s), list(RATIOS).index(s)))
        assignments[group] = side
        counts[side].update(additions)
    return assignments, conflicts
