"""离线验证有限国家等价奖励，另存真实轨迹新旧分数；不调用模型或 API。"""

import json
from copy import deepcopy

from pydantic import TypeAdapter

from retail_agent.dataset import WRITES
from retail_agent.environment import ROOT, RetailSession
from retail_agent.evaluate import sha256
from retail_agent.grpo_diagnostic import summarize, verify_diagnostic
from retail_agent.training_reward import (
    COUNTRY_ALIASES, IDENTITY_MANIFEST, LEGACY_VERSION, VERSION,
    compile_business_goals, evaluate_training_task, ready_training_rows,
    reference_subset, row_fingerprint,
)
from tau2.data_model.message import Message


def main():
    directory = ROOT / "outputs/review/phase05c"
    selection_path = directory / "ready_sampling_review_20261013.json"
    selection = json.loads(selection_path.read_text())["selection"]
    pool = ROOT / "data/versions/retail_full_v2/reviewed/rl_train_ready.jsonl"
    rows = {r["id"]: r for r in ready_training_rows(pool)}
    immutable = {str(selection_path.relative_to(ROOT)): sha256(selection_path),
                 str(IDENTITY_MANIFEST.relative_to(ROOT)): sha256(IDENTITY_MANIFEST),
                 "outputs/budget/deepseek_requests.jsonl": sha256(ROOT / "outputs/budget/deepseek_requests.jsonl")}
    verifications, batches = {}, {}
    for entry in selection.values():
        source = entry["source"]
        if source in batches:
            continue
        path = ROOT / source
        for name in ["manifest", "traces"]:
            suffix = ".json" if name == "manifest" else ".jsonl"
            assert sha256(path / (name + suffix)) == entry[name + "_sha256"], source
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest.get("training_reward_version", LEGACY_VERSION) == LEGACY_VERSION
        immutable.update(manifest["immutable_sha256"])
        for name in ["manifest.json", "traces.jsonl", "events.jsonl", "summary.json"]:
            immutable[str((path / name).relative_to(ROOT))] = sha256(path / name)
        verifications[source] = verify_diagnostic(path)
        batches[source] = [json.loads(s) for s in (path / "traces.jsonl").read_text().splitlines()]
    original, corrected, comparisons = [], [], []
    for task_id, entry in selection.items():
        group = sorted([r for r in batches[entry["source"]] if r["task_id"] == task_id], key=lambda r: r["trial"])
        assert [r["trial"] for r in group] == [0, 1, 2, 3], task_id
        for trace in group:
            history = TypeAdapter(list[Message]).validate_python(trace["trajectory"])
            old = evaluate_training_task(rows[task_id], history, trace["termination"], version=LEGACY_VERSION)
            assert json.dumps(old, sort_keys=True) == json.dumps(trace["scores"], sort_keys=True)
            new = evaluate_training_task(rows[task_id], history, trace["termination"])
            assert new["business_reward"] == old["business_reward"]
            assert new["gate"] == old["gate"] and new["violations"] == old["violations"]
            assert new["reward_a"] >= old["reward_a"] and new["reward_b"] >= old["reward_b"]
            original.append(trace)
            updated = deepcopy(trace)
            updated["scores"] = new
            corrected.append(updated)
            comparisons.append({"task_id": task_id, "trial": trace["trial"], "source": entry["source"],
                                "trace_sha256": row_fingerprint(trace), "row_sha256": row_fingerprint(rows[task_id]),
                                "old_scores": old, "new_scores": new,
                                "training_reward_changed": any(old[k] != new[k] for k in ["reward_a", "reward_b", "progress"])})
    cases = []
    for row in rows.values():
        spec = compile_business_goals(row)
        for variant in spec["variants"]:
            session = reference_subset(row, set(variant["goals"]))
            result = evaluate_training_task(row, session.trajectory)
            fraction = len(variant["goals"]) / len(spec["goals"])
            assert result["gate"] == 1 and result["progress"] == fraction
            assert result["reward_a"] == result["training_business_reward"] == float(fraction == 1)
            assert result["reward_b"] == (1.0 if fraction == 1 else .3 * fraction)
            cases.append(result)
        writes_only = RetailSession(row)
        for action in row["evaluation_criteria"]["actions"]:
            if action["name"] in WRITES:
                assert not writes_only.call_tool(action["name"], action["arguments"]).error
        result = evaluate_training_task(row, writes_only.trajectory)
        assert result["business_reward"] == result["training_business_reward"] == 1
        assert result["gate"] == result["reward_a"] == result["reward_b"] == 0
        cases.append(result)
        print("reward checks " + row["id"], flush=True)
    assert len(cases) == 145 and len(corrected) == 32
    assert all(sha256(ROOT / name) == digest for name, digest in immutable.items()), "来源或账本变化"
    report = {"decision": "user_approved_limited_country_equivalence", "training_reward_version": VERSION,
              "country_aliases": COUNTRY_ALIASES, "official_evaluation_unchanged": True,
              "scope": "Posthoc reward correction of existing TRAIN diagnostics; no model improvement or CUDA training.",
              "original_summary": summarize(original, list(selection), 4),
              "corrected_summary": summarize(corrected, list(selection), 4),
              "comparisons": comparisons, "source_verifications": verifications,
              "ready_pool_acceptance": {"task_count": len(rows), "case_count": len(cases), "all_passed": True, "cases": cases},
              "immutable_sha256": immutable, "original_inputs_and_ledger_unchanged": True,
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
                  ROOT / "src/retail_agent/training_reward.py", ROOT / "src/retail_agent/grpo_diagnostic.py",
                  ROOT / "src/retail_agent/reward.py", ROOT / "scripts/review_country_rewards.py"]},
              "paid_cost_cny": 0, "optimizer_steps": 0, "cuda_training_verified": False}
    path = directory / "country_reward_review_20261013.json"
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("已有国家奖励复核结果不同，禁止覆盖；需另存新的实验记录")
    path.write_text(content)
    print(json.dumps({"output": str(path.relative_to(ROOT)), "trajectories": 32, "reward_cases": len(cases),
                      "official_successes": report["corrected_summary"]["business_successes"],
                      "training_successes": report["corrected_summary"]["gated_successes"], "paid_cost_cny": 0}))


if __name__ == "__main__":
    main()
