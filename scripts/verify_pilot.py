"""离线重放真实开发轨迹，验证来源、采样记录、工具路由和最终奖励。"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from retail_agent.data import read_jsonl, parse_task
from retail_agent.environment import RetailSession
from retail_agent.model import parse_action
from retail_agent.reward import evaluate_task


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    summary = json.loads((ROOT / "outputs/reports/history/phase3_pilot_summary.json").read_text())
    traces = [r for _, r in read_jsonl(ROOT / "outputs/pilot/original_traces.jsonl")
              if r["run_id"] == summary["run_id"]]
    require(len(traces) == len(summary["results"]) and bool(traces), "试跑与轨迹数量不符")
    tasks = {r["id"]: parse_task(r) for _, r in read_jsonl(ROOT / "data/raw/tau2_rl_train.jsonl")
             if r["id"] in {t["task_id"] for t in traces}}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen3-4b", local_files_only=True)
    ledger = [r for _, r in read_jsonl(ROOT / "outputs/budget/deepseek_requests.jsonl")]
    started = {e["id"]: e for e in ledger if e["event"] == "started"}
    completed = {e["id"]: e for e in ledger if e["event"] == "completed"}
    counts, results = Counter(), []
    for trace in traces:
        require(trace["task_id"] in tasks and trace["split"] == "rl_dev", "不是开发任务")
        require(trace["model_id"] == summary["model_id"] and
                trace["model_revision"] == summary["model_revision"], "底座/revision 不一致")
        session = RetailSession(tasks[trace["task_id"]])
        pending = None
        for event in trace["events"]:
            source = event["source"]
            counts[source] += 1
            if source == "deepseek_customer":
                identifier = event["api"]["id"]
                require(identifier in completed and started[identifier]["thinking"] == "disabled", "顾客请求无凭据记录")
                require(event["api"] == completed[identifier], "顾客请求用量记录不一致")
                session.add_user_message(event["text"])
            elif source == "qwen_agent":
                expected = tokenizer.apply_chat_template(session.agent_messages(), tools=session.tool_schemas,
                    add_generation_prompt=True, enable_thinking=False, tokenize=True)
                require(event["prompt_token_ids"] == expected, "客服输入不是当前公开对话")
                tokens, probabilities = event["generated_token_ids"], event["generated_logprobs"]
                require(len(tokens) == len(probabilities) and bool(tokens), "采样 token/分数记录缺失")
                require(all(math.isfinite(v) for v in probabilities), "模型 token 分数不是有限值")
                require(tokenizer.decode(tokens, skip_special_tokens=True) == event["raw_text"], "原采样 token 与保存文本不一致")
                if event["finish_reason"] == "stop":
                    try:
                        action = parse_action(event["raw_text"])
                    except ValueError:
                        require(trace["termination"] == "invalid_model_output", "非法输出未正确标记")
                        continue
                    if action["kind"] == "message":
                        session.add_assistant_message(action["text"])
                    else:
                        pending = action
            elif source == "retail_tool":
                require(pending is not None and pending["name"] == event["name"] and
                        pending["arguments"] == event["arguments"], "工具动作并非来自上一条模型输出")
                response = session.call_tool(event["name"], event["arguments"])
                require(response.content == event["response"]["content"] and
                        response.error == event["response"]["error"], "保存的工具返回与真实重放不同")
                counts["tool_errors" if response.error else "successful_tool_calls"] += 1
                pending = None
            else:
                raise AssertionError("未知轨迹来源")
        require(pending is None, "模型工具调用没有真实执行")
        require(session.agent_messages() == trace["messages"], "最终公开消息被改写")
        if trace["business_evaluation_eligible"]:
            expected_reward = evaluate_task(tasks[trace["task_id"]], session.trajectory,
                                           trace["termination"]).model_dump(mode="json")
            require(expected_reward == trace["reward"], "最终评分不可重现")
        else:
            require(trace["reward"] is None, "服务故障被伪装为业务零奖励")
        results.append({"task_id": trace["task_id"], "business": trace["business"],
                        "termination": trace["termination"],
                        "reward": trace["reward"]["reward"] if trace["reward"] else None})
    require(counts["deepseek_customer"] >= 2 and counts["qwen_agent"] >= 2 and
            counts["successful_tool_calls"] >= 1, "尚未完成真实多轮工具执行链")
    reserved = {e["id"]: e["reserved_cny"] for e in started.values()}
    for identifier, event in completed.items():
        reserved[identifier] = event["peak_cost_upper_cny"]
    require(sum(reserved.values()) <= 1.0, "试跑上界记账超 1 元")
    files = ["src/retail_agent/customer.py", "src/retail_agent/model.py", "src/retail_agent/rollout.py",
             "requirements/phase3_local.lock.txt",
             "third_party/tau2-bench/data/tau2/user_simulator/simulation_guidelines.md"]
    report = {"status": "passed", "run_id": summary["run_id"], "scope": summary["scope"],
              "model_id": summary["model_id"], "model_revision": summary["model_revision"],
              "counts": dict(counts), "results": results,
              "api_accounted_peak_upper_cny": sum(reserved.values()),
              "observed_balance_delta_cny": (sum(e["balance_delta_cny"] for e in completed.values())
                  if all(e.get("balance_delta_cny") is not None for e in completed.values()) else None),
              "balance_delta_note": "缺少任一余额差值时总差值为未知；余额显示精度/其它调用影响差额，不将差额当作逐请求账单",
              "verified": ["real_model_and_api_multiturn", "public_prompt_token_ids", "sampled_tokens_match_text",
                           "model_selected_tools_replayed", "reproducible_rewards", "api_usage_and_budget_ledger"],
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
              "not_verified": ["SFT_or_GRPO_training", "GRPO_policy_logprobs_and_env_mask",
                               "formal_three_model_comparison", "complete_policy_compliance"]}
    (ROOT / "outputs/reports/history/phase3_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
