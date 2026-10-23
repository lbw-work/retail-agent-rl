"""开发集真实本机 Qwen + DeepSeek 顾客试跑；不是正式留出集评估。"""

import argparse
import json
import sys
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_agent.customer import DeepSeekCustomer
from retail_agent.data import read_jsonl
from retail_agent.model import QwenAgent
from retail_agent.rollout import run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", action="append", help="只能选择 rl_dev 中的现有 ID")
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    config = tomllib.loads((ROOT / "configs/pilot.toml").read_text())
    if not 0 < config["api_limit_cny"] <= min(project["budget"]["api_limit"], 1.0):
        raise ValueError("首次试跑 API 上限须在 1 元以内")
    local_model = json.loads((ROOT / "models/manifest.json").read_text())
    if (local_model["model_id"], local_model["model_revision"]) != (project["model_id"], project["model_revision"]):
        raise ValueError("本机模型与固定底座/revision 不符")
    if config.get('review_manifest'):
        from retail_agent.rl_dataset import reviewed_rl_rows
        rows = reviewed_rl_rows(config['tasks'], 'dev', config['review_manifest'])
    elif config["tasks"] == "data/versions/retail_full_v2/processed/rl_dev.jsonl":
        rows = [row for _, row in read_jsonl(ROOT / config["tasks"])]
    else:
        raise ValueError("阶段三只允许开发集试跑，不使用最终测试集选方案")
    available = {row['id']: row for row in rows}
    if args.task_id:
        selected = [available[identifier] for identifier in args.task_id]
    else:
        selected = [next(row for row in available.values() if row["business"] == business)
                    for business in ["cancel_pending_order", "exchange_delivered_order_items"]]
    agent = QwenAgent(ROOT / config["model_path"], max_context_tokens=config["max_context_tokens"],
                      max_new_tokens=config["max_new_tokens_per_action"], seed=config["seed"])
    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id, results = uuid.uuid4().hex, []

    def progress(task_id, step, text):
        print(f"{task_id} 客服动作 {step + 1}: {text[:160]}", flush=True)

    for row in selected:
        customer = DeepSeekCustomer(row["user_scenario"], project["customer"], config["api_limit_cny"])
        result = run_episode(row, agent, customer, config["max_assistant_actions"],
                             config["max_consecutive_tool_errors"], progress=progress)
        result.update(run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
                      model_id=project["model_id"], model_revision=project["model_revision"],
                      backend=config["backend"], seed=config["seed"], split="rl_dev")
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        summary = {key: result[key] for key in ["task_id", "business", "termination", "assistant_actions", "diagnostic"]}
        summary["reward"] = result["reward"]["reward"] if result["reward"] is not None else None
        summary["tool_calls"] = sum(e["source"] == "retail_tool" for e in result["events"])
        summary["api_accounted_peak_upper_cny"] = customer.accounted_cny()
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if result["termination"] in {"budget_limit", "customer_service_error", "infrastructure_error", "grading_error"}:
            break
    report_name = 'pilot_rl_v3_summary.json' if config.get('review_manifest') else 'pilot_v2_summary.json'
    (ROOT / 'outputs/reports' / report_name).write_text(json.dumps({
        "run_id": run_id, "scope": "local_quantized_original_model_dev_pilot_not_final_evaluation",
        "model_id": project["model_id"], "model_revision": project["model_revision"],
        "config": config, "results": results}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
