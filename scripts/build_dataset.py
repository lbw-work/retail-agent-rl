"""全量重建 Retail v2；只使用固定本地输入，不调用模型、API 或训练。"""

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from difflib import SequenceMatcher
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_agent.data import (Groups, entity_keys, make_system_prompt, normalize_history,
                               normalize_message, normalized_reason, parse_task, read_jsonl)
from retail_agent.dataset import (RATIOS, WRITES, assign_groups, length_bucket,
                                  sft_policy_flags, task_business, task_spec_flags)
from retail_agent.environment import RetailSession, UPSTREAM
from retail_agent.reward import evaluate_task
from tau2.data_model.tasks import Task
from transformers import AutoTokenizer

VERSION = "retail_full_v2"
DEST = ROOT / "data/versions" / VERSION
PROVENANCE = ROOT / "data/provenance"
MAX_LENGTH = 16384


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: r["id"]):
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")


def reference_probe(task):
    """参考/口头宣称/错误目标、状态隔离、重置与评分非修改的真实离线检查。"""
    evidence = {"id": task["id"], "issues": [], "policy_flags": task_spec_flags(task)}
    try:
        gold, wrong = RetailSession(task), RetailSession(task)
        initial = wrong.db_hash()
        user_ids = {a["arguments"]["user_id"] for a in task["evaluation_criteria"]["actions"]
                    if a["name"] == "get_user_details"}
        for action in task["evaluation_criteria"]["actions"]:
            name, args = action["name"], action["arguments"]
            db = gold._environment.tools.db
            if name in WRITES:
                if "order_id" in args and args["order_id"] in db.orders and user_ids:
                    if db.orders[args["order_id"]].user_id not in user_ids:
                        evidence["policy_flags"].append("reference_write_other_user_order")
                if name == "modify_user_address" and user_ids and args["user_id"] not in user_ids:
                    evidence["policy_flags"].append("reference_write_other_user_profile")
                if name in {"exchange_delivered_order_items", "modify_pending_order_items"}:
                    if any(a == b for a, b in zip(args["item_ids"], args["new_item_ids"], strict=True)):
                        evidence["policy_flags"].append("reference_same_variant_change")
            response = gold.call_tool(name, args)
            if response.error:
                evidence["issues"].append({"type": "reference_tool_error", "tool": name,
                                            "arguments": args, "error": response.content})
                break
        if evidence["issues"]:
            return evidence
        assert wrong.db_hash() == initial, "state isolation failed"
        changed = gold.db_hash()
        result = evaluate_task(task, gold.trajectory)
        assert gold.db_hash() == changed, "grader modified live state"
        evidence["reference_reward"] = result.reward
        evidence["reward_basis"] = [str(x.value) for x in result.reward_basis]
        if result.reward != 1:
            evidence["issues"].append({"type": "reference_reward_not_one"})
        wrong.add_assistant_message("Your request is completed successfully.")
        evidence["claim_only_reward"] = evaluate_task(task, wrong.trajectory).reward
        if evidence["claim_only_reward"] != 0:
            evidence["issues"].append({"type": "uninformative_database_reward"})
        else:
            for action in task["evaluation_criteria"]["actions"]:
                args = deepcopy(action["arguments"])
                is_write = action["name"] in WRITES
                if is_write:
                    key = "user_id" if action["name"] == "modify_user_address" else "order_id"
                    args[key] = "USER_DOES_NOT_EXIST" if key == "user_id" else "#W_DOES_NOT_EXIST"
                response = wrong.call_tool(action["name"], args)
                assert response.error == is_write, "wrong-target feedback differs from expectation"
            evidence["wrong_target_reward"] = evaluate_task(task, wrong.trajectory).reward
            assert evidence["wrong_target_reward"] == 0, "wrong writes passed reward"
        gold.reset()
        assert gold.db_hash() == initial and not gold.trajectory, "reset failed"
        evidence["isolation_reset_and_grader_nonmutation"] = True
    except Exception as error:
        evidence["issues"].append({"type": type(error).__name__, "error": str(error)})
    evidence["policy_flags"] = sorted(set(evidence["policy_flags"]))
    return evidence


def main():
    started = time.monotonic()
    old = json.loads((PROVENANCE / "legacy_holdout_nodes.json").read_text())
    sources = json.loads((PROVENANCE / "raw_sources.json").read_text())
    input_paths = [ROOT / x["path"] for x in sources["files"]]
    input_paths += [ROOT / x["path"] for x in sources["environment"]["source_files"]]
    input_paths += sorted((ROOT / "data/databases").glob("*.json"))
    input_paths += [UPSTREAM / "data/tau2/domains/retail/policy.md",
                    ROOT / "data/raw/tokenizer/tokenizer.json", ROOT / "data/raw/tokenizer/tokenizer_config.json"]
    input_paths += [ROOT / "src/retail_agent" / f for f in ("data.py", "dataset.py", "environment.py", "reward.py")]
    input_paths += [Path(__file__), PROVENANCE / "legacy_holdout_nodes.json",
                    PROVENANCE / "raw_sources.json", ROOT / "requirements/environment.lock.txt"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    dependencies = {name: version(name) for name in sources["environment_dependencies"]}
    for item in sources["files"] + sources["environment"]["source_files"]:
        if item["path"] in hashes:
            assert hashes[item["path"]] == item["sha256"], item["path"]
    policy = (UPSTREAM / "data/tau2/domains/retail/policy.md").read_text()
    system = make_system_prompt(policy)
    groups, reasons, entries, payloads = Groups(), {}, [], {}
    schemas, tools = None, None
    cache_path = DEST / "reports/reference_checks.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cached = {r["id"]: r for r in cache.get("cases", [])} if (
        cache.get("input_sha256") == hashes and cache.get("dependencies") == dependencies) else {}
    checks = []
    raw_tasks = [(number, parse_task(raw)) for number, raw in read_jsonl(ROOT / "data/raw/tau2_rl_train.jsonl")
                 if raw["id"].startswith("retail_")]
    # Independent snapshots are replayed in four local CPU processes; no model or API.
    executor = ProcessPoolExecutor(max_workers=4)
    pending_checks = {task["id"]: executor.submit(reference_probe, task) for _, task in raw_tasks
                      if task["id"] not in cached}
    for number, task in raw_tasks:
        Task.model_validate(task)
        if tools is None:
            session = RetailSession(task)
            tools, schemas = session._environment.tool_defs, session.tool_schemas
        node = "rl:" + task["id"]
        entities = sorted(entity_keys([task["user_scenario"], task["evaluation_criteria"]["actions"]]))
        groups.add(node, entities)
        reasons[node] = normalized_reason(task["user_scenario"]["instructions"].get("reason_for_call") or "")
        business, difficulty = task_business(task["evaluation_criteria"]["actions"])
        check = cached.get(task["id"])
        if check is None:
            check = pending_checks[task["id"]].result()
        checks.append(check)
        entry = {"id": task["id"], "kind": "rl", "source_line": number, "node": node,
                 "entities": entities, "businesses": business, "difficulty": difficulty,
                 "issues": check["issues"], "policy_flags": check["policy_flags"],
                 "split": "excluded", "reason": "reference_or_reward_failure" if check["issues"]
                    else "task_spec_requires_review" if check["policy_flags"] else None}
        entries.append(entry)
        payloads[task["id"]] = {"id": task["id"], "business": business[0] if len(business)==1 else "mixed" if business else "other",
                               "businesses": business, "difficulty": difficulty,
                               "prompt": [{"role": "system", "content": system}],
                               "user_scenario": task["user_scenario"], "evaluation_criteria": task["evaluation_criteria"],
                               "db_path": task["db_path"], "dataset_version": VERSION}
        if len(checks) % 20 == 0:
            save(cache_path, {"input_sha256": hashes, "dependencies": dependencies, "cases": checks})
            print(f"RL 离线执行核查 {len(checks)}/563；运行失败 {sum(bool(c['issues']) for c in checks)}；需语义复查 {sum(bool(c['policy_flags']) for c in checks)}", flush=True)
    executor.shutdown()
    save(cache_path, {"input_sha256": hashes, "dependencies": dependencies, "cases": checks})

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
    seen, sft_count = set(), 0
    for number, raw in read_jsonl(ROOT / "data/raw/tau2_sft_train.jsonl"):
        meta = raw["metadata"]
        dialog = str(meta["source_dialog_id"])
        if not dialog.startswith("retail_"):
            continue
        sft_count += 1
        identifier, node = f"{dialog}:{meta['turn_index']}", "sft:"+dialog
        entities = entity_keys([*raw["messages"][1:], raw["answer"]])
        if meta.get("scenario_id"):
            entities.add("scenario:"+str(meta["scenario_id"]))
        groups.add(node, entities)
        reasons[node] = normalized_reason(meta.get("reason_for_call") or "")
        entry = {"id": identifier, "kind": "sft", "node": node, "dialog_id": dialog,
                 "source_line": number, "entities": sorted(entities), "split": "excluded",
                 "reason": None, "policy_flags": []}
        entries.append(entry)
        try:
            history, repacked = normalize_history(raw["messages"][1:], tools)
            target = normalize_message(raw["answer"], tools)
            if target["role"] != "assistant" or not (target["content"].strip() or target.get("tool_calls")):
                raise ValueError("empty or non-assistant target")
            calls = target.get("tool_calls") or []
            entry["target_tool"] = calls[0]["function"]["name"] if calls else "text"
            flags = sft_policy_flags(history, target)
            entry["policy_flags"] = flags
            if meta.get("correct") != 1 or meta.get("reward", 0) <= 0:
                entry["reason"] = "source_failure"
            elif flags:
                entry["reason"] = "policy_or_history_requires_review"
            elif identifier in seen:
                entry["reason"] = "duplicate_target_id"
            else:
                messages = [{"role": "system", "content": system}] + history
                prompt = tokenizer.apply_chat_template(messages, tools=schemas,
                            enable_thinking=False, add_generation_prompt=True, tokenize=True)
                full = tokenizer.apply_chat_template(messages+[target], tools=schemas,
                            enable_thinking=False, tokenize=True)
                assert full[:len(prompt)] == prompt, "target boundary is not exact prefix"
                assert len(full) > len(prompt), "empty supervised token sequence"
                entry.update(total_tokens=len(full), prompt_tokens=len(prompt), target_tokens=len(full)-len(prompt),
                             length_bucket=length_bucket(len(full)))
                if len(full) > MAX_LENGTH:
                    entry["reason"] = "over_16384_no_truncation"
                else:
                    seen.add(identifier)
                    payloads[identifier] = {"id": identifier, "prompt": messages, "completion": [target], "tools": schemas,
                        "metadata": {"source_dialog_id": dialog, "turn_index": meta["turn_index"],
                            "scenario_id": meta.get("scenario_id"), "source_line": number,
                            "total_tokens": len(full), "target_tokens": len(full)-len(prompt),
                            "length_bucket": entry["length_bucket"], "repacked_read_batches": repacked,
                            "quality_checks": "source_success_schema_history_auth_and_lexical_confirmation",
                            "dataset_version": VERSION}}
        except (ValueError, TypeError, KeyError, AssertionError) as error:
            entry["reason"] = "invalid_message: "+str(error)
        if sft_count % 1000 == 0:
            print(f"SFT 全量审查 {sft_count}/11395", flush=True)

    reason_nodes = defaultdict(list)
    for node, reason in reasons.items():
        if len(reason.split()) >= 8:
            reason_nodes[reason].append(node)
    for nodes in reason_nodes.values():
        for node in nodes[1:]:
            groups.union(nodes[0], node)
    descriptions, near = sorted(reason_nodes), []
    words = {r: set(r.split()) for r in descriptions}
    for i, first in enumerate(descriptions):
        if len(first.split()) < 12:
            continue
        for second in descriptions[i+1:]:
            if len(second.split()) < 12 or len(words[first]&words[second])/len(words[first]|words[second]) < .75:
                continue
            ratio = SequenceMatcher(None, first, second).ratio()
            if ratio >= .9:
                groups.union(reason_nodes[first][0], reason_nodes[second][0])
                near.append({"first": reason_nodes[first][0], "second": reason_nodes[second][0], "ratio": ratio})

    anchors = defaultdict(set)
    for entry in old["nodes"]:
        anchors[groups.find(entry["node"])].add(entry["side"])
    members = defaultdict(list)
    for entry in entries:
        entry["group"] = groups.find(entry["node"])
        if entry["reason"] is None:
            features = [entry["kind"]]
            if entry["kind"] == "rl":
                features += ["rl:"+b for b in entry["businesses"]]+["difficulty:"+entry["difficulty"]]
            else:
                features += ["sft:"+entry["target_tool"], "length:"+entry["length_bucket"]]
            members[entry["group"]].append({"features": features})
    assignments, conflicts = assign_groups(members, anchors)
    rows = {name: [] for name in ("sft_train", "sft_dev", "sft_test", "rl_train", "rl_dev", "rl_test")}
    for entry in entries:
        group = entry["group"]
        entry["side"] = assignments.get(group, "quarantine") if group not in conflicts else "quarantine"
        if entry["reason"] is None:
            if group in conflicts:
                entry["reason"] = "legacy_holdout_side_conflict"
                continue
            side = assignments[group]
            entry.update(split=entry["kind"]+"_"+side, reason="selected")
            payload = payloads[entry["id"]]
            if entry["kind"] == "sft":
                payload["metadata"]["group"] = group
            else:
                payload["group"] = group
            rows[entry["split"]].append(payload)
    overlap = []
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        ea = [e for e in entries if e["reason"]=="selected" and e["side"]==a]
        eb = [e for e in entries if e["reason"]=="selected" and e["side"]==b]
        shared = {x for e in ea for x in e["entities"]} & {x for e in eb for x in e["entities"]}
        shared_groups = {e["group"] for e in ea}&{e["group"] for e in eb}
        assert not shared and not shared_groups, (a, b, shared, shared_groups)
        overlap.append({"sides": [a,b], "shared_entities": 0, "shared_groups": 0})
    for name, rs in rows.items():
        save_rows(DEST / "processed" / (name+".jsonl"), rs)
    save(DEST / "manifests/splits.json", {"version": VERSION, "seed": 42, "target_ratios": RATIOS,
        "method": "connected_entities_scenarios_lexical_templates_joint_sft_rl_sides",
        "legacy_anchor_sides": dict((g, sorted(v)) for g,v in anchors.items()),
        "anchor_conflicts": conflicts, "near_reason_matches": near,
        "sft": [e for e in entries if e["kind"]=="sft"], "rl": [e for e in entries if e["kind"]=="rl"]})
    save(DEST / "manifests/tool_schema.json", schemas)
    (DEST / "manifests/agent_system.txt").write_text(system+"\n")
    stats = {"version": VERSION, "scope": "offline_dataset_and_environment_not_model_or_training",
        "max_sequence_length": MAX_LENGTH, "raw_sft_rows": sft_count, "raw_rl_tasks": len(checks),
        "sft_exclusion_reasons": dict(Counter(e["reason"] for e in entries if e["kind"]=="sft" and e["reason"]!="selected")),
        "sft_policy_flags": dict(Counter(f for e in entries if e["kind"]=="sft" for f in e["policy_flags"])),
        "rl_exclusion_reasons": dict(Counter(e["reason"] for e in entries if e["kind"]=="rl" and e["reason"]!="selected")),
        "rl_policy_flags": dict(Counter(f for e in entries if e["kind"]=="rl" for f in e["policy_flags"])),
        "splits": {}, "cross_side_overlap": overlap, "legacy_anchor_conflicts": conflicts,
        "paid_cost_cny": 0, "elapsed_seconds": round(time.monotonic()-started,2),
        "limitations": ["Lexical consent and task-specification checks are conservative, not a semantic proof.",
            "Full DB reward does not ensure authentication or transaction-specific consent.",
            "SFT tool outputs cannot all be replayed against uniquely identified source databases.",
            "Unrecognized semantic paraphrases or policy contradictions may remain.",
            "All data is a candidate pool; actual paid training volume remains unbenchmarked."]}
    for name, rs in rows.items():
        info = {"rows": len(rs), "groups": len({e["group"] for e in entries if e["split"]==name})}
        if name.startswith("sft"):
            info.update(dialogues=len({r["metadata"]["source_dialog_id"] for r in rs}),
                length_buckets=dict(Counter(r["metadata"]["length_bucket"] for r in rs)),
                total_tokens=sum(r["metadata"]["total_tokens"] for r in rs),
                supervised_tokens=sum(r["metadata"]["target_tokens"] for r in rs),
                target_tools=dict(Counter(c["function"]["name"] for r in rs for c in r["completion"][0].get("tool_calls",[]))))
        else:
            info.update(business_incidence=dict(Counter(b for r in rs for b in r["businesses"])),
                        difficulty=dict(Counter(r["difficulty"] for r in rs)))
        stats["splits"][name] = info
    save(DEST / "reports/data_audit.json", stats)
    files = [p for directory in (DEST/"processed", DEST/"manifests", DEST/"reports")
             for p in sorted(directory.glob("*")) if p.is_file()
             and p.name not in {"sources.json", "verification.json", "reproducibility.json"}]
    save(DEST / "manifests/sources.json", {"version": VERSION,
        "dataset_id": sources["dataset_id"], "dataset_revision": sources["dataset_revision"],
        "dataset_license": sources["dataset_license"], "model_id": sources["model_id"],
        "model_license": sources["model_license"], "environment_license": sources["environment"]["license"],
        "environment_commit": sources["environment"]["commit"], "model_revision": sources["model_revision"],
        "dependencies": dependencies, "input_sha256": hashes,
        "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                                            "sha256": digest(p)} for p in files]})
    assert {str(p.relative_to(ROOT)): digest(p) for p in input_paths} == hashes, "inputs changed during build"
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
