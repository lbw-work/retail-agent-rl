"""05C：真实 SFT 组内随机采样、持久化中断记录与独立奖励复核；没有优化器。"""

import argparse
import json
import math
import shutil
import statistics
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

from retail_agent.customer import BudgetExceeded, DeepSeekCustomer, GOAL_POLICY
from retail_agent.environment import ROOT, RetailSession
from retail_agent.evaluate import audit_artifact, sha256
from retail_agent.grpo_rollout import RolloutError, run_token_episode
from retail_agent.model import ContextLimit, QwenAgent
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import IDENTITY_MANIFEST, LEGACY_VERSION, VERSION, evaluate_training_task, ready_training_rows, row_fingerprint


class MLXTokenSampler:
    def __init__(self, agent, temperature, on_token=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("随机采样温度必须为有限正数")
        self.agent, self.temperature, self.on_token = agent, temperature, on_token

    def generate(self, prompt_ids, max_new_tokens):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        if max_new_tokens < 1 or len(prompt_ids) + max_new_tokens > self.agent.max_context_tokens:
            raise ContextLimit("采样没有空间或超过完整轨迹上限，禁止截断")
        cache, rest = (self.agent.prompt_cache.fetch_nearest_cache("agent", prompt_ids)
                       if self.agent.prompt_cache is not None else (None, prompt_ids))
        if cache is None:
            cache = make_prompt_cache(self.agent.model)
        ids, logps = [], []
        for response in stream_generate(self.agent.model, self.agent.tokenizer, rest,
                max_tokens=max_new_tokens, sampler=make_sampler(temp=self.temperature,
                top_p=1.0, top_k=0, min_p=0.0), prefill_step_size=256, prompt_cache=cache):
            # stream_generate 返回未调温概率；须按实际 categorical 的温度重新归一化。
            scaled = response.logprobs / self.temperature
            logp = float((scaled[response.token] - mx.logsumexp(scaled)).item())
            ids.append(int(response.token))
            logps.append(logp)
            if self.on_token is not None:
                self.on_token({"event": "sampled_token", "token_id": int(response.token), "logprob": logp})
        if self.agent.prompt_cache is not None:
            self.agent.prompt_cache.insert_cache("agent", list(prompt_ids) + ids, cache)
        mx.clear_cache()
        return {"token_ids": ids, "logprobs": logps}


def summarize(results, task_ids, size):
    modes = ["business_reward", "reward_a", "reward_b"]
    if results and all("training_business_reward" in r["scores"] for r in results):
        modes.append("training_business_reward")
    elif any("training_business_reward" in r["scores"] for r in results):
        raise RolloutError("不同训练奖励版本不得混合统计")
    business_mode = "training_business_reward" if "training_business_reward" in modes else "business_reward"
    groups = []
    for task_id in task_ids:
        trials = sorted([r for r in results if r["task_id"] == task_id], key=lambda r: r["trial"])
        if len(trials) > size or len({r["trial"] for r in trials}) != len(trials):
            raise RolloutError("轨迹重复或任务组过大")
        group = {"task_id": task_id, "completed_trials": len(trials), "complete": len(trials) == size}
        for mode in modes:
            values = [r["scores"][mode] for r in trials]
            group[mode] = values
            if group["complete"]:
                std = statistics.stdev(values)
                group[mode + "_sample_std"] = std
                group[mode + "_advantages"] = [(v - statistics.mean(values)) / (std + 1e-4) for v in values]
        groups.append(group)
    complete = [g for g in groups if g["complete"]]
    varying = [g["task_id"] for g in complete if g["reward_a_sample_std"] > 0]
    business_signal = [g["task_id"] for g in complete
                       if g["reward_a_sample_std"] > 0 and g[business_mode + "_sample_std"] > 0]
    counters = {}
    for mode in modes:
        counters[mode] = {"all_zero": sum(all(v == 0 for v in g[mode]) for g in complete),
                          "all_one": sum(all(v == 1 for v in g[mode]) for g in complete),
                          "varying": sum(g[mode + "_sample_std"] > 0 for g in complete)}
    lengths = [len(r["prompt_ids"]) + len(r["completion_ids"]) for r in results]
    assistant_lengths = [sum(r["assistant_mask"]) for r in results]
    tools = [m for r in results for m in r["trajectory"] if m["role"] == "tool"]
    report = {"planned_groups": len(task_ids), "planned_trajectories": len(task_ids) * size,
            "completed_groups": len(complete), "completed_trajectories": len(results), "groups": groups,
            "group_statistics": counters, "reward_a_varying_tasks": varying,
            "business_signal_tasks": business_signal,
            "extra_varying_groups_from_b": [g["task_id"] for g in complete
                if g["reward_a_sample_std"] == 0 and g["reward_b_sample_std"] > 0],
            "business_successes": sum(r["scores"]["business_reward"] == 1 for r in results),
            "gated_successes": sum(r["scores"]["reward_a"] == 1 for r in results),
            "terminations": dict(Counter(r["termination"] for r in results)),
            "violations": dict(Counter(v for r in results for v in r["scores"]["violations"])),
            "tool_calls": len(tools), "tool_errors": sum(bool(m["error"]) for m in tools),
            "full_token_lengths": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)} if lengths else {},
            "assistant_token_lengths": {"min": min(assistant_lengths), "median": statistics.median(assistant_lengths), "max": max(assistant_lengths)} if lengths else {},
            "local_reward_signal_supports_cuda_probe": len(business_signal) >= 2,
            "optimizer_steps": 0, "cuda_training_verified": False}
    if business_mode == "training_business_reward":
        report["training_business_successes"] = sum(r["scores"][business_mode] == 1 for r in results)
    return report


def resume_task_ids(task_ids, results, inherited, size, recheck):
    if not set(inherited) <= set(task_ids) or not set(recheck) <= set(task_ids):
        raise RolloutError("恢复任务不在原定清单")
    done = set(inherited)
    for row in results:
        if row["task_id"] not in task_ids or type(row["trial"]) is not int or not 0 <= row["trial"] < size:
            raise RolloutError("恢复轨迹任务或组内序号错误")
    for task_id in task_ids:
        group = [r for r in results if r["task_id"] == task_id]
        trials = [r["trial"] for r in group]
        if len(set(trials)) != len(trials):
            raise RolloutError("恢复来源包含重复尝试")
        if len(group) == size:
            if any(r["prompt_ids"] != group[0]["prompt_ids"] for r in group):
                raise RolloutError("完整组初始提示不一致")
            done.add(task_id)
    if not set(recheck) <= done:
        raise RolloutError("显式复查只能针对已完成任务")
    done -= set(recheck)
    return [t for t in task_ids if t not in done], [t for t in task_ids if t in done]


def verify_diagnostic(output):
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest.get("runtime_code_sha256", {}).items():
        if sha256(output / "code_snapshot" / name) != expected:
            raise ValueError("实验源码快照变化：" + name)
    if manifest.get("resume"):
        resume = manifest["resume"]
        for name in ["manifest", "traces", "events"]:
            suffix = ".json" if name == "manifest" else ".jsonl"
            if sha256(ROOT / resume["source"] / (name + suffix)) != resume[name + "_sha256"]:
                raise ValueError("恢复来源记录变化：" + name)
    for path, expected in manifest["immutable_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError("实验原始输入变化：" + path)
    rows = {r["id"]: r for r in map(json.loads, (ROOT / manifest["config"]["train_tasks"]).read_text().splitlines())}
    results = [json.loads(s) for s in (output / "traces.jsonl").read_text().splitlines()]
    events = [json.loads(s) for s in (output / "events.jsonl").read_text().splitlines()]
    for result in results:
        if result["task_id"] not in manifest["task_ids"]:
            raise ValueError("轨迹不在事先固定清单")
        if type(result["trial"]) is not int or not 0 <= result["trial"] < manifest["config"].get("num_generations", 4):
            raise ValueError("组内序号不符")
        if "seed" in manifest["config"]:
            tasks = manifest.get("original_task_ids", manifest["task_ids"])
            seed = manifest["config"]["seed"] + tasks.index(result["task_id"]) * manifest["config"]["num_generations"] + result["trial"]
            if result.get("seed") != seed:
                raise ValueError("轨迹随机种子不符")
        messages = TypeAdapter(list[Message]).validate_python(result["trajectory"])
        score = evaluate_training_task(rows[result["task_id"]], messages, result["termination"],
                                       version=manifest.get("training_reward_version", LEGACY_VERSION))
        # JSON 将子目标 tuple 保存成 list；比较序列化值，仍逐字段拒绝评分变化。
        if json.dumps(score, sort_keys=True) != json.dumps(result["scores"], sort_keys=True):
            raise ValueError("真实轨迹重放评分不一致")
        n = len(result["completion_ids"])
        if n != len(result["assistant_mask"]) or n != len(result["logprobs"]):
            raise ValueError("token 与掩码/概率长度不一致")
        mask = [0] * n
        for turn in result["turns"]:
            start, end = turn["start"], turn["end"]
            if not 0 <= start < end <= n or any(mask[start:end]):
                raise ValueError("生成回合边界错误或重复")
            mask[start:end] = [1] * (end - start)
        if mask != result["assistant_mask"] or not any(mask):
            raise ValueError("损失掩码与真实客服回合不符")
        recorded = [e for e in events if e.get("task_id") == result["task_id"] and e.get("trial") == result["trial"]]
        sampled = [e for e in recorded if e["event"] == "sampled_token"]
        actual_ids = [t for t, active in zip(result["completion_ids"], mask, strict=True) if active]
        actual_logps = [p for p, active in zip(result["logprobs"], mask, strict=True) if active]
        if actual_ids != [e["token_id"] for e in sampled] or actual_logps != [e["logprob"] for e in sampled]:
            raise ValueError("轨迹客服 token/概率与实时逐 token 记录不一致")
        starts = [e for e in recorded if e["event"] == "episode_started"]
        if len(starts) != 1 or starts[0]["prompt_ids"] != result["prompt_ids"]:
            raise ValueError("初始提示与实时记录不一致")
        for active, logp in zip(mask, result["logprobs"], strict=True):
            if not math.isfinite(logp) or logp > 1e-6 or (not active and logp != 0):
                raise ValueError("采样概率或外部反馈概率错误")
    for task in manifest["task_ids"]:
        group = [r for r in results if r["task_id"] == task]
        if group and any(r["prompt_ids"] != group[0]["prompt_ids"] for r in group):
            raise ValueError("任务组的初始提示不同")
    summarize(results, manifest["task_ids"], manifest["config"].get("num_generations", 4))
    api_checked = 0
    if manifest.get("customer_config", {}).get("goal_policy") == GOAL_POLICY:
        ledger = {e["id"]: e for e in map(json.loads, (ROOT / "outputs/budget/deepseek_requests.jsonl").read_text().splitlines()) if e["event"] == "completed"}
        for event in events:
            if event["event"] in {"opening_received", "customer_feedback"}:
                record = event.get("api_record")
                if not record or record != ledger.get(record["id"]):
                    raise ValueError("顾客事件与真实 API 用量账本未关联")
                api_checked += 1
    return {"replayed_trajectories": len(results), "scores_match": True,
            "token_masks_and_logprobs_checked": True, "immutable_inputs_unchanged": True,
            "sampled_tokens_match_live_journal": True,
            "customer_api_records_verified": api_checked,
            "traces_sha256": sha256(output / "traces.jsonl"), "optimizer_steps": 0,
            "probability_scope": "MLX_temperature_policy_not_CUDA_on_policy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="新诊断目录，禁止覆盖")
    parser.add_argument("--preflight", action="store_true", help="只读检查，不加载模型或调用 API")
    parser.add_argument("--verify", action="store_true", help="独立重放已保存的真实轨迹，不调用 API")
    parser.add_argument("--resume-from", help="读取上一批记录；仅重新运行未完成的整个任务组")
    parser.add_argument("--recheck-task", action="append", default=[], help="显式重复一组已完成任务，作为单独的新条件复查")
    args = parser.parse_args()
    output = ROOT / args.output
    if args.verify:
        verification = verify_diagnostic(output)
        (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise ValueError("诊断目录已存在，禁止覆盖")
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    diagnostic = tomllib.loads((ROOT / "configs/grpo_diagnostic.toml").read_text())
    config = tomllib.loads((ROOT / diagnostic.get('recipe_config', 'configs/grpo.toml')).read_text())
    ids = diagnostic["task_ids"]
    if not 8 <= len(ids) <= 12 or len(ids) != len(set(ids)) or config["num_generations"] != 4:
        raise ValueError("诊断须事先固定 8～12 个不同任务，每组 4 条")
    if diagnostic["model_path"] != "models/qwen3-4b-mlx4bit" or not 0 < diagnostic["api_additional_limit_cny"] <= 1:
        raise ValueError("诊断底座或新增预算不符")
    if diagnostic.get("goal_policy") != GOAL_POLICY:
        raise ValueError("本批诊断须使用已验收的固定目标顾客规则")
    if diagnostic.get("max_customer_prompt_bytes") != 64_000:
        raise ValueError("本批诊断的顾客输入保护须为 64000 字节")
    frozen = json.loads((ROOT / "data/versions/retail_full_v2/reviewed/phase05a_manifest.json").read_text())
    pool = ROOT / config["train_tasks"]
    rows = {r["id"]: r for r in ready_training_rows(pool)}
    identity_review = json.loads(IDENTITY_MANIFEST.read_text())
    excluded = diagnostic.get("excluded_task_ids", [])
    if set(excluded) != set(identity_review["excluded_task_ids"]) & set(ids):
        raise ValueError("诊断隔离清单与身份复核不一致")
    eligible = [i for i in ids if i not in excluded]
    if len(eligible) < 8:
        raise ValueError("身份复核后原计划不足八条可用任务，须重新制定清单")
    approved = {r["id"]: r["row_sha256"] for r in frozen["entries"]}
    if any(i not in rows or approved.get(i) != row_fingerprint(rows[i]) for i in eligible):
        raise ValueError("所选任务没有复核或来源行已变化")
    immutable = dict(frozen["immutable_sha256"])
    immutable[identity_review["source"]] = identity_review["source_sha256"]
    immutable[str(IDENTITY_MANIFEST.relative_to(ROOT))] = sha256(IDENTITY_MANIFEST)
    immutable[config["train_tasks"]] = sha256(pool)
    immutable["outputs/evaluation/dev_original_sft_20261012/summary.json"] = sha256(ROOT / "outputs/evaluation/dev_original_sft_20261012/summary.json")
    for path, expected in immutable.items():
        if sha256(ROOT / path) != expected:
            raise ValueError("冻结输入变化：" + path)
    model_path, adapter = ROOT / diagnostic["model_path"], ROOT / config["sft_adapter_dir"]
    artifact = audit_artifact(adapter.parent, project, model_path)
    previous = json.loads((ROOT / "outputs/evaluation/dev_original_sft_20261012/manifest.json").read_text())["artifact"]
    if any(artifact[k] != previous[k] for k in ["run_sha256", "adapter_sha256"]):
        raise ValueError("SFT 起点与已评测成果不一致")
    immutable[str(adapter.relative_to(ROOT) / "adapter_model.safetensors")] = artifact["adapter_sha256"]
    immutable[str(adapter.parent.relative_to(ROOT) / "run.json")] = artifact["run_sha256"]
    for item in json.loads((ROOT / "models/manifest.json").read_text())["converted_files"]:
        immutable[item["path"]] = item["sha256"]
    schema = project["data"]["manifest_dir"] + "/tool_schema.json"
    immutable[schema] = sha256(ROOT / schema)
    customer_config = {**project["customer"], "goal_policy": diagnostic["goal_policy"],
                       "max_prompt_bytes": diagnostic["max_customer_prompt_bytes"]}
    run_ids, inherited, resume = list(eligible), [], None
    if args.resume_from:
        source = ROOT / args.resume_from
        prior = json.loads((source / "manifest.json").read_text())
        if prior.get("training_reward_version", LEGACY_VERSION) != VERSION:
            raise ValueError("奖励版本不同，不能恢复混合采样；请独立重评分或建立新批次")
        prior_sampling = {k: v for k, v in prior["config"].items() if k != "train_tasks"}
        sampling = {k: v for k, v in config.items() if k != "train_tasks"}
        if (prior.get("original_task_ids", prior["task_ids"]) != ids or prior_sampling != sampling
                or prior["artifact"]["adapter_sha256"] != artifact["adapter_sha256"]
                or any(immutable.get(p) != digest for p, digest in prior["immutable_sha256"].items())):
            raise ValueError("恢复的任务、采样条件或原始输入不一致")
        verify_diagnostic(source)
        prior_results = [json.loads(s) for s in (source / "traces.jsonl").read_text().splitlines()]
        run_ids, inherited = resume_task_ids(ids, prior_results,
            prior.get("inherited_complete_task_ids", []), config["num_generations"], args.recheck_task)
        if set(args.recheck_task) & set(excluded):
            raise ValueError("不得重新采样身份矛盾任务")
        run_ids = [i for i in run_ids if i not in excluded]
        inherited = [i for i in inherited if i not in excluded]
        resume = {"source": str(source.relative_to(ROOT)), "manifest_sha256": sha256(source / "manifest.json"),
                  "traces_sha256": sha256(source / "traces.jsonl"), "events_sha256": sha256(source / "events.jsonl"),
                  "source_goal_policy": prior["customer_config"].get("goal_policy", "upstream"),
                  "source_customer_config": prior["customer_config"],
                  "explicit_recheck_tasks": args.recheck_task}
    elif args.recheck_task:
        raise ValueError("复查参数需要恢复来源")
    if not run_ids:
        raise ValueError("任务组已全部完成，无需重复采样")
    customer = DeepSeekCustomer(rows[eligible[0]]["user_scenario"], customer_config)
    start_cost = customer.accounted_cny()
    customer.http.close()
    budget = project["budget"]
    limit = min(start_cost + diagnostic["api_additional_limit_cny"], diagnostic["api_ledger_limit_cny"], budget["api_limit"],
                budget["stop_starting_new_work_at"] - budget["gpu_limit"] - budget["reserve_limit"])
    if args.resume_from:
        limit = min(limit, prior["api_ledger_limit_cny"])
    if limit <= start_cost:
        raise BudgetExceeded("本轮预算没有新增空间")
    manifest = {"stage": "05C", "scope": "TRAIN_SFT_sampling_only", "utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "training_reward_version": VERSION, "diagnostic_config": diagnostic, "task_ids": run_ids,
                "original_task_ids": ids, "inherited_complete_task_ids": inherited, "resume": resume,
                "excluded_task_ids": excluded, "identity_review_sha256": sha256(IDENTITY_MANIFEST),
                "selection": "predeclared_7_businesses_plus_multi_order_and_pairs",
                "api_ledger_start_cny": start_cost, "api_ledger_limit_cny": limit,
                "artifact": artifact, "immutable_sha256": immutable,
                "customer_config": customer_config, "backend": "mlx_local_4bit",
                "dependencies": {name: version(name) for name in ["mlx", "mlx-lm", "transformers"]},
                "runtime_code_sha256": {name: sha256(ROOT / "src/retail_agent" / name) for name in
                    ["grpo_diagnostic.py", "grpo_rollout.py", "model.py", "adapter.py", "customer.py", "reward.py", "training_reward.py", "environment.py"]},
                "seed_rule": "seed + group_index * num_generations + trial_index_zero_based",
                "model_id": project["model_id"], "model_revision": project["model_revision"],
                "sampling_temperature": config["temperature"], "top_p": 1.0, "top_k": 0,
                "enable_thinking": False, "optimizer_steps": 0, "test_used": False}
    if args.preflight:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name, expected in manifest["runtime_code_sha256"].items():
        shutil.copyfile(ROOT / "src/retail_agent" / name, snapshot / name)
        if sha256(snapshot / name) != expected:
            raise ValueError("执行源码快照不一致")
    (output / "traces.jsonl").touch()
    results, status, failure = [], "running", None
    started = time.monotonic()
    def journal(record):
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **record}, ensure_ascii=False) + "\n")
    def save():
        report = {"status": status, **summarize(results, run_ids, config["num_generations"]), "failure": failure,
                  "inherited_complete_task_ids": inherited, "goal_policy": diagnostic["goal_policy"],
                  "excluded_task_ids": excluded,
                  "api_additional_accounted_upper_cny": customer.accounted_cny() - start_cost,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report
    save()
    active = {}
    try:
        agent = QwenAgent(model_path, max_context_tokens=config["max_trajectory_tokens"], adapter_path=adapter)
        sampler = MLXTokenSampler(agent, config["temperature"])
        import mlx.core as mx
        for task_id in run_ids:
            group_index = ids.index(task_id)
            row = rows[task_id]
            active = {"task_id": task_id, "trial": None}
            journal({"event": "opening_pending", **active})
            customer = DeepSeekCustomer(row["user_scenario"], customer_config, limit)
            try:
                reply = customer.reply(RetailSession(row).agent_messages())
            finally:
                customer.http.close()
            opening = reply["text"]
            if not isinstance(opening, str) or not opening.strip() or any(m in opening for m in STOP_MARKERS):
                raise RolloutError("顾客开场无效或已终止")
            journal({"event": "opening_received", **active, "text": opening, "api_record": reply.get("api")})
            for trial in range(config["num_generations"]):
                seed = config["seed"] + group_index * config["num_generations"] + trial
                mx.random.seed(seed)
                active = {"task_id": task_id, "trial": trial, "seed": seed}
                journal({"event": "trace_started", **active})
                print(f"{group_index + 1}/{len(ids)} {task_id} trial {trial + 1}/4 seed={seed}", flush=True)
                def progress(event):
                    journal({**event, **active})
                    if event["event"] != "sampled_token":
                        print(f"  action {event.get('action', '?')}: {event['event']}", flush=True)
                sampler.on_token = progress
                customer = DeepSeekCustomer(row["user_scenario"], customer_config, limit)
                try:
                    result = run_token_episode(row, sampler, agent.tokenizer, customer, opening, config, progress)
                finally:
                    customer.http.close()
                result.update(trial=trial, seed=seed, business=row["business"], businesses=row["businesses"],
                              difficulty=row["difficulty"], split="rl_train", backend="mlx_local_4bit")
                with (output / "traces.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                journal({"event": "trace_completed", **active, "termination": result["termination"], "scores": result["scores"]})
                save()
                print(f"  {result['termination']} A={result['scores']['reward_a']} B={result['scores']['reward_b']}", flush=True)
        status = "completed"
    except (Exception, KeyboardInterrupt) as error:
        status = "stopped_budget_limit" if isinstance(error, BudgetExceeded) else "stopped_system_error"
        if isinstance(error, KeyboardInterrupt):
            status = "interrupted"
        failure = {**active, "type": type(error).__name__, "message": str(error), "counted_as_business_failure": False}
        journal({"event": "run_stopped", **failure})
    report = save()
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, ensure_ascii=False, indent=2), flush=True)
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
