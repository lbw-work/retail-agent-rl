"""同一固定开发任务上比较 original 与 SFT；最终 test 保持封存。"""

import argparse
import gc
import hashlib
import json
import random
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from retail_agent.customer import DeepSeekCustomer
from retail_agent.data import read_jsonl
from retail_agent.environment import ROOT
from retail_agent.model import QwenAgent
from retail_agent.rollout import run_episode
from retail_agent.observability import StageTimer
from retail_agent.experiment_protocol import check_protocol


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_dev_tasks(rows, businesses, seed, all_dev=False):
    if all_dev:
        return rows
    # 在查看模型结果前固定：每类一条单业务 + 七条复合任务。
    rng = random.Random(seed)
    selected = []
    for business in businesses:
        candidates = [row for row in rows if row["business"] == business]
        if not candidates:
            raise ValueError(f"开发集缺少单业务任务：{business}")
        selected.append(rng.choice(candidates))
    selected.extend(rng.sample([row for row in rows if row["business"] == "mixed"], 7))
    return selected


def summarize(results, task_ids):
    keys = [(r["checkpoint"], r["task_id"]) for r in results]
    if (len(keys) != len(set(keys)) or len(task_ids) != len(set(task_ids))
            or any(arm not in {"original", "sft"} or task not in task_ids for arm, task in keys)):
        raise ValueError("评测包含重复记录、非计划任务或未实现的模型组")
    arms = {}
    for checkpoint in ("original", "sft"):
        arm = [result for result in results if result["checkpoint"] == checkpoint]
        eligible = [r for r in arm if r["business_evaluation_eligible"]]
        successful = [r for r in eligible if r["reward"]["reward"] == 1.0]
        tools = [event for r in arm for event in r["events"] if event["source"] == "retail_tool"]
        errors = sum(bool(event["response"].get("error")) for event in tools)
        arms[checkpoint] = {
            "planned": len(task_ids), "attempted": len(arm), "eligible": len(eligible),
            "successes": len(successful), "system_errors": len(arm) - len(eligible),
            "not_attempted": len(task_ids) - len(arm),
            "complete": len(arm) == len(task_ids) and len(eligible) == len(task_ids),
            "primary_success_rate": len(successful) / len(task_ids)
                if task_ids and len(eligible) == len(task_ids) else None,
            "success_rate_eligible": len(successful) / len(eligible) if eligible else None,
            "success_rate_planned": len(successful) / len(task_ids),
            "terminations": dict(Counter(r["termination"] for r in arm)),
            "tool_calls": len(tools), "tool_errors": errors,
            "tool_execution_success_rate": (len(tools) - errors) / len(tools) if tools else None,
            "mean_actions_successful": (sum(r["assistant_actions"] for r in successful)
                                        / len(successful) if successful else None),
            "episode_wall_seconds": sum(r.get("timing", {}).get("wall_seconds", 0) for r in arm)
                if arm and all("timing" in r for r in arm) else None,
            "business_coverage": {b: {
                "attempted": sum(b in r.get("businesses", [r["business"]]) for r in arm),
                "eligible": sum(b in r.get("businesses", [r["business"]]) for r in eligible),
                "whole_task_successes": sum(b in r.get("businesses", [r["business"]]) for r in successful)}
                for b in sorted({b for r in arm for b in r.get("businesses", [r["business"]])})},
        }
    paired = []
    by_key = {(r["checkpoint"], r["task_id"]): r for r in results}
    for task_id in task_ids:
        pair = [by_key.get((checkpoint, task_id)) for checkpoint in ("original", "sft")]
        if all(r and r["business_evaluation_eligible"] for r in pair):
            paired.append({"task_id": task_id, "business": pair[0]["business"],
                           "original": pair[0]["reward"]["reward"],
                           "sft": pair[1]["reward"]["reward"]})
    return {"comparison_complete": all(a["complete"] for a in arms.values()),
            "arms": arms, "paired_eligible": len(paired),
            "improved": sum(r["sft"] > r["original"] for r in paired),
            "regressed": sum(r["sft"] < r["original"] for r in paired), "pairs": paired}


def audit_artifact(run_dir, project, model_dir):
    run = json.loads((run_dir / "run.json").read_text())
    if (run["model_id"], run["model_revision"]) != (project["model_id"], project["model_revision"]):
        raise ValueError("SFT 底座或 revision 与项目不符")
    if run["status"] not in {"completed", "stopped_time_limit"} or run["global_step"] < 1:
        raise ValueError("SFT 成果缺少真实完成步骤")
    if run["data_manifest_sha256"] != sha256(ROOT / "data/prepared/retail_full_v2_sft/manifest.json"):
        raise ValueError("训练数据清单与本机不一致")
    train_ids = {row["id"] for _, row in read_jsonl(ROOT / project["data"]["processed_dir"] / "sft_train.jsonl")}
    if not run["train_ids"] or len(set(run["train_ids"])) != len(run["train_ids"]) or not set(run["train_ids"]) <= train_ids:
        raise ValueError("训练样本不是当前 train 的合法子集")
    from transformers import AutoTokenizer
    original = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    trained = AutoTokenizer.from_pretrained(run_dir / "adapter", local_files_only=True)
    if (original.chat_template != trained.chat_template
            or original.get_vocab() != trained.get_vocab()
            or original.special_tokens_map != trained.special_tokens_map):
        raise ValueError("训练与本机推理 tokenizer/模板不一致")
    local = json.loads((ROOT / "models/manifest.json").read_text())
    if (local["model_id"], local["model_revision"]) != (project["model_id"], project["model_revision"]):
        raise ValueError("本机量化底座版本不符")
    for item in local["converted_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("本机量化底座文件哈希不符")
    return {"run_sha256": sha256(run_dir / "run.json"),
            "adapter_sha256": sha256(run_dir / "adapter/adapter_model.safetensors"),
            "data_manifest_matches": True, "train_ids_in_current_train": True,
            "tokenizer_and_template_match": True, "base_hashes_match": True,
            "training_code_matches_local": run["training_code_sha256"] == sha256(ROOT / "src/retail_agent/train_sft.py"),
            "training_code_sha256_recorded": run["training_code_sha256"],
            "global_step": run["global_step"], "selected_train_rows": len(run["train_ids"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_dev.toml")
    parser.add_argument("--output", required=True, help="新目录，保存固定任务清单、轨迹和汇总")
    parser.add_argument("--all-dev", action="store_true", help="使用配置的全部 dev，仍不接触 test")
    args = parser.parse_args()
    timing = StageTimer()
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    config = tomllib.loads((ROOT / args.config).read_text())
    protocol = check_protocol() if config.get('review_manifest') else None
    if protocol is not None and config != protocol['protocol']['evaluation_config']:
        raise ValueError('评测配置与冻结协议不同，须另立实验协议')
    if config.get('review_manifest'):
        from retail_agent.rl_dataset import reviewed_rl_rows
        reviewed_rows = reviewed_rl_rows(config['tasks'], 'dev', config['review_manifest'])
    elif config["tasks"] == "data/versions/retail_full_v2/processed/rl_dev.jsonl":
        reviewed_rows = None
    else:
        raise ValueError("本入口只允许已复核 rl_dev，最终 test 不做模型评测")
    if not 0 < config["api_additional_limit_cny"] <= 1.0:
        raise ValueError("本轮新增 API 费用上限须在 1 元以内")
    output = ROOT / args.output
    if output.exists():
        raise ValueError("评测目录已存在，禁止覆盖实验记录")
    model_dir, run_dir = ROOT / config["model_path"], ROOT / config["sft_run_dir"]
    if model_dir.resolve() != (ROOT / "models/qwen3-4b-mlx4bit").resolve():
        raise ValueError("当前对照固定使用已验收的 MLX 量化底座")
    with timing.measure("artifact_audit"):
        artifact = audit_artifact(run_dir, project, model_dir)
    rows = reviewed_rows if reviewed_rows is not None else [row for _, row in read_jsonl(ROOT / config["tasks"])]
    selected = select_dev_tasks(rows, project["environment"]["rl_businesses"], config["seed"],
                                args.all_dev or config.get('all_reviewed_dev', False))
    customer = DeepSeekCustomer(selected[0]["user_scenario"], project["customer"])
    start_cost = customer.accounted_cny()
    customer.http.close()
    api_limit = min(project["budget"]["api_limit"], start_cost + config["api_additional_limit_cny"])
    if api_limit <= start_cost:
        raise ValueError("顾客 API 总预算已耗尽")
    import importlib.metadata
    manifest = {"scope": "dev_comparison_not_final_test", "utc": datetime.now(timezone.utc).isoformat(),
                "model_id": project["model_id"], "model_revision": project["model_revision"],
                "config": config, "customer_config": project["customer"], "artifact": artifact,
                "task_ids": [row["id"] for row in selected], "task_file_sha256": sha256(ROOT / config["tasks"]),
                "selection": "all_configured_dev" if args.all_dev or config.get('all_reviewed_dev', False) else "seeded_7_single_business_plus_7_mixed",
                "rl_dataset_version": project['data'].get('rl_dataset_version') if config.get('review_manifest') else 'retail_full_v2',
                "review_manifest_sha256": sha256(ROOT / config['review_manifest']) if config.get('review_manifest') else None,
                "api_ledger_start_cny": start_cost, "api_ledger_limit_cny": api_limit,
                "evaluation_code_sha256": sha256(__file__),
                "runtime_code_sha256": {name: sha256(ROOT / "src/retail_agent" / name)
                    for name in ["evaluate.py", "model.py", "adapter.py", "customer.py", "rollout.py", "reward.py", "environment.py"]},
                "environment_commit": project["environment"]["commit"],
                "tool_schema_sha256": sha256(ROOT / project["data"]["manifest_dir"] / "tool_schema.json"),
                "prompt_cache": "mlx_lm_LRUPromptCache_one_entry_exact_token_prefix",
                "dependencies": {name: importlib.metadata.version(name) for name in ["mlx", "mlx-lm", "transformers"]},
                "test_used": False, "num_trials": 1, "temperature": 0.0, "enable_thinking": False}
    output.mkdir(parents=True)
    timing.directory = output / "observations"
    timing.directory.mkdir()
    manifest["observation_run_id"] = timing.run_id
    manifest["protocol_sha256"] = protocol["sha256"] if protocol else None
    manifest["runtime_code_sha256"]["observability.py"] = sha256(ROOT / "src/retail_agent/observability.py")
    if protocol is not None:
        (output / "protocol.json").write_text(json.dumps(protocol["protocol"], ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results, stop = [], False
    def save_summary(status):
        report = {"status": status, **summarize(results, manifest["task_ids"]),
                  "api_additional_accounted_upper_cny": customer.accounted_cny() - start_cost,
                  "timing": timing.summary(), "protocol_sha256": manifest["protocol_sha256"]}
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report

    for checkpoint in ("original", "sft"):
        with timing.measure("model_and_adapter_load", checkpoint=checkpoint):
            agent = QwenAgent(model_dir, max_context_tokens=config["max_context_tokens"],
                              max_new_tokens=config["max_new_tokens_per_action"], seed=config["seed"],
                              adapter_path=run_dir / "adapter" if checkpoint == "sft" else None)
        for index, row in enumerate(selected):
            print(f"{checkpoint} {index + 1}/{len(selected)} {row['id']} {row['business']}", flush=True)
            customer = DeepSeekCustomer(row["user_scenario"], project["customer"], api_limit)
            with timing.measure("evaluation_episode", checkpoint=checkpoint, task_id=row["id"]):
                result = run_episode(row, agent, customer, config["max_assistant_actions"],
                                     config["max_consecutive_tool_errors"],
                                     progress=lambda task, step, text: print(f"  {task} #{step + 1}: {text[:100]}", flush=True))
            customer.http.close()
            result.update(checkpoint=checkpoint, businesses=row["businesses"], difficulty=row["difficulty"],
                          split="rl_dev", adapter_info=agent.adapter_info)
            results.append(result)
            with (output / "traces.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            save_summary("running")
            print(json.dumps({"termination": result["termination"], "reward": result["reward"],
                              "diagnostic": result["diagnostic"]}, ensure_ascii=False), flush=True)
            if not result["business_evaluation_eligible"]:
                stop = True
                break
        del agent
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        if stop:
            break
    timing.save()
    report = save_summary("stopped_system_error" if stop else "completed")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
