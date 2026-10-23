"""独立读取 v2 产物，核对来源、全部 SFT token、RL 执行证据和留出边界。"""

import hashlib
import json
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from retail_agent.data import entity_keys, normalize_history, normalize_message, parse_task, read_jsonl
from retail_agent.dataset import length_bucket, sft_policy_flags, task_business
from retail_agent.environment import RetailSession

DEST = ROOT / "data/versions/retail_full_v2"
PROVENANCE = ROOT / "data/provenance"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    sources = json.loads((DEST / "manifests/sources.json").read_text())
    require({name: version(name) for name in sources["dependencies"]} == sources["dependencies"], "依赖版本改变")
    for path, expected in sources["input_sha256"].items():
        require(digest(ROOT / path) == expected, "输入文件改变：" + path)
    for item in sources["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "产物改变：" + item["path"])
    splits = json.loads((DEST / "manifests/splits.json").read_text())
    stats = json.loads((DEST / "reports/data_audit.json").read_text())
    references = json.loads((DEST / "reports/reference_checks.json").read_text())
    require(references["input_sha256"] == sources["input_sha256"], "执行证据不是本次输入")
    require(references["dependencies"] == sources["dependencies"], "执行证据依赖版本不一致")
    checks = {c["id"]: c for c in references["cases"]}
    require(len(checks) == len(references["cases"]) == 563, "不是全部 563 个任务的执行证据")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
    system = (DEST / "manifests/agent_system.txt").read_text().rstrip("\n")
    schema = json.loads((DEST / "manifests/tool_schema.json").read_text())
    outputs, rows_by_id = {}, {}
    for name in stats["splits"]:
        rows = [r for _, r in read_jsonl(DEST / "processed" / (name + ".jsonl"))]
        require(len(rows) == stats["splits"][name]["rows"], name + " 数量错误")
        outputs[name] = rows
        for row in rows:
            require(row["id"] not in rows_by_id, "重复样本：" + row["id"])
            rows_by_id[row["id"]] = (name, row)
    entries = splits["sft"] + splits["rl"]
    require(len(splits["sft"]) == 11395 and len(splits["rl"]) == 563, "原始行没有全量入账")
    selected = {e["id"]: e for e in entries if e["reason"] == "selected"}
    require(set(selected) == set(rows_by_id), "输出和清单不一致")
    for entry in entries:
        require(bool(entry["reason"]), "缺少纳入/排除原因")
        if entry["reason"] == "selected":
            name, row = rows_by_id[entry["id"]]
            require(entry["split"] == name, "划分不同")
            if entry["kind"] == "rl":
                require(not entry["issues"], "参考执行失败")
            require(not entry["policy_flags"], "待复查样本进入输出")
            require(row["prompt"][0] == {"role": "system", "content": system}, "公开政策不一致")
            require(row.get("group", row.get("metadata", {}).get("group")) == entry["group"], "分组元数据不同")

    side_groups, side_entities = defaultdict(set), defaultdict(set)
    for e in entries:
        if e["reason"] == "selected":
            side_groups[e["side"]].add(e["group"])
            side_entities[e["side"]].update(e["entities"])
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        require(not side_groups[a] & side_groups[b], "跨侧分组重叠")
        require(not side_entities[a] & side_entities[b], "跨侧实体重叠")
    legacy = json.loads((PROVENANCE / "legacy_holdout_nodes.json").read_text())
    new_by_node = {e["node"]: e for e in entries}
    for e in legacy["nodes"]:
        current = new_by_node[e["node"]]
        require(current["side"] in {e["side"], "quarantine"}, "旧留出侧移动：" + e["node"])

    rl_seen, tools = set(), None
    for line, raw in read_jsonl(ROOT / "data/raw/tau2_rl_train.jsonl"):
        if not raw["id"].startswith("retail_"):
            continue
        task = parse_task(raw)
        rl_seen.add(task["id"])
        require(task["id"] in checks, "任务缺少执行证据")
        if task["id"] not in selected:
            continue
        entry = selected[task["id"]]
        row = rows_by_id[task["id"]][1]
        check = checks[task["id"]]
        require(entry["source_line"] == line, "RL 来源行错误")
        for field in ("user_scenario", "evaluation_criteria", "db_path"):
            require(row[field] == task[field], "隐藏目标/评估答案/快照被改写：" + task["id"])
        require(row["prompt"] == [{"role": "system", "content": system}], "RL 初始提示包含非公开目标")
        require(task_business(task["evaluation_criteria"]["actions"]) == (row["businesses"], row["difficulty"]), "RL 分层错误")
        require(not check["issues"] and not check["policy_flags"], "待复查参考任务被纳入")
        require(check["reference_reward"] == 1 and check["claim_only_reward"] == 0
                and check["wrong_target_reward"] == 0
                and check["isolation_reset_and_grader_nonmutation"], "任务离线验收证据不完整")
        if tools is None:
            session = RetailSession(task)
            tools = session._environment.tool_defs
            require(session.tool_schemas == schema, "环境工具与 SFT schema 不一致")
    require(rl_seen == set(checks), "执行证据不是原始 RL 任务全集")

    sft_seen, sft_verified, token_total = set(), 0, 0
    for line, raw in read_jsonl(ROOT / "data/raw/tau2_sft_train.jsonl"):
        meta = raw["metadata"]
        dialog = str(meta["source_dialog_id"])
        if not dialog.startswith("retail_"):
            continue
        identifier = f"{dialog}:{meta['turn_index']}"
        require(identifier not in sft_seen, "原始 SFT ID 重复")
        sft_seen.add(identifier)
        if identifier not in selected:
            continue
        row, entry = rows_by_id[identifier][1], selected[identifier]
        require(meta.get("correct") == 1 and meta.get("reward", 0) > 0, "失败源示范被纳入")
        history, repacked = normalize_history(raw["messages"][1:], tools)
        target = normalize_message(raw["answer"], tools)
        require(row["prompt"][1:] == history and row["completion"] == [target], "SFT 来源被改写")
        require(not sft_policy_flags(history, target), "SFT 未通过声明的筛查规则")
        require(row["tools"] == schema, "工具 schema 不一致")
        require(entry["source_line"] == row["metadata"]["source_line"] == line, "SFT 来源行不同")
        require(row["metadata"]["repacked_read_batches"] == repacked, "历史改排记录不同")
        prompt = tokenizer.apply_chat_template(row["prompt"], tools=schema,
            enable_thinking=False, add_generation_prompt=True, tokenize=True)
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tools=schema,
            enable_thinking=False, tokenize=True)
        require(full[:len(prompt)] == prompt and 0 < len(full) - len(prompt), "监督边界错误")
        require(len(full) <= 16384 and len(full) == row["metadata"]["total_tokens"], "长度超限或被截断")
        require(len(full) - len(prompt) == row["metadata"]["target_tokens"], "目标 token 统计错误")
        require(length_bucket(len(full)) == row["metadata"]["length_bucket"], "长度桶错误")
        entities = entity_keys([*raw["messages"][1:], raw["answer"]])
        if meta.get("scenario_id"):
            entities.add("scenario:" + str(meta["scenario_id"]))
        require(sorted(entities) == entry["entities"], "SFT 实体来源不同")
        require(row["metadata"]["dataset_version"] == "retail_full_v2", "数据版本缺失")
        sft_verified += 1
        token_total += len(full)
        if sft_verified % 1000 == 0:
            print(f"独立核对 SFT 来源与完整 token {sft_verified}", flush=True)
    require(sft_seen == {e["id"] for e in splits["sft"]}, "SFT 来源清单不完整")

    addresses = {}
    address_fields = {"address1", "address2", "city", "state", "country", "zip"}
    for path in sorted((ROOT / "data/databases").glob("*.json")):
        database = json.loads(path.read_text())
        for kind in ("users", "orders"):
            require(all(address_fields <= set(r.get("address", {})) for r in database[kind].values()),
                    "数据库缺少嵌套地址：" + path.name + kind)
        addresses[path.name] = {k: len(database[k]) for k in ("users", "orders")}
    require(len(addresses) == 4, "快照数量不同")
    report = {"status": "passed", "dataset_version": "retail_full_v2", "paid_cost_cny": 0,
        "raw_sft_rows_accounted": len(sft_seen), "raw_rl_tasks_accounted": len(rl_seen),
        "all_selected_sft_source_and_tokens_verified": sft_verified, "sft_total_tokens": token_total,
        "selected_rl_current_input_execution_evidence_verified": sum(len(v) for k, v in outputs.items() if k.startswith("rl")),
        "legacy_holdout_nodes_verified": len(legacy["nodes"]), "nested_addresses_verified": addresses,
        "cross_side_shared_groups": 0, "cross_side_shared_entities": 0,
        "processed_sha256": {k: digest(DEST / "processed" / (k + ".jsonl")) for k in outputs},
        "not_verified": ["new_model_rollouts", "SFT_or_RL_training", "complete_policy_semantic_compliance",
                         "semantic_template_similarity_beyond_declared_lexical_rules"],
        "rl_evidence_note": "All raw tasks were actually replayed by build_dataset; this verifier checks frozen evidence, not a second full replay."}
    (DEST / "reports/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
