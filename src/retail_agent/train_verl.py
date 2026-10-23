"""独立 verl V1 入口；预检/配置检查不启动模型、API、Ray 或优化器。"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib
import traceback

from retail_agent.environment import ROOT
from retail_agent.experiment_protocol import check_protocol
from retail_agent.training_reward import ready_training_rows


def source_fingerprint(directory):
    paths = sorted(p for p in (directory / "verl").rglob("*") if p.suffix in {".py", ".yaml"})
    paths += [directory / "pyproject.toml", directory / "uv.lock"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(directory)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "file_count": len(paths)}


def preflight(verl_root, task_ids=None, profile="legacy"):
    baseline = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    protocol = check_protocol()
    lock = json.loads((ROOT / "configs/verl/source_lock.json").read_text())
    source = source_fingerprint(verl_root)
    if source != lock["verl_source"]:
        raise ValueError("verl 源码快照已改变；需审阅后重新冻结，不能静默换版本")
    extra = {}
    if profile == "mmopd":
        from retail_agent.mmopd import PROFILE, digest, profile as load_profile, verify_snapshot
        native = load_profile()
        extra = verify_snapshot(require_adapter=True)
        extra['profile_sha256'] = digest(PROFILE)
        extra['native_adapter_code_sha256'] = digest(ROOT / 'src/retail_agent/mmopd.py')
        extra['history_token_policy'] = 'append_only_including_all_reasoning'
        from retail_agent.mmopd_protocol import PROTOCOL
        extra['tool_protocol'] = PROTOCOL
        extra['tool_protocol_code_sha256'] = digest(ROOT / 'src/retail_agent/mmopd_protocol.py')
        baseline.update(native)
        project.update(model_id=native['model_id'], model_revision=native['model_revision'])
    elif profile != "legacy":
        raise ValueError("未知实验profile")
    rows = ready_training_rows(ROOT / baseline["train_tasks"], baseline["review_manifest"])
    if task_ids:
        requested = set(task_ids)
        if len(requested) != len(task_ids) or requested - {r["id"] for r in rows}:
            raise ValueError("任务必须是互不重复的已复核 TRAIN ID")
        by_id = {r["id"]: r for r in rows}
        rows = [by_id[key] for key in task_ids]
    adapter = ROOT / baseline["sft_adapter_dir"]
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if ((profile == "legacy" and project["model_id"] != "Qwen/Qwen3-4B") or adapter_config["r"] != 16
            or adapter_config["lora_alpha"] != 32
            or set(adapter_config["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj"}
            or adapter_config.get("modules_to_save") or adapter_config.get("lora_dropout") != 0):
        raise ValueError("模型或 SFT LoRA 与本次迁移协议不一致")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()
              if p.name in {"adapter_config.json", "adapter_model.safetensors"}}
    if "adapter_model.safetensors" not in hashes:
        raise FileNotFoundError("SFT LoRA 权重缺失")
    if profile == "legacy":
        artifact = json.loads((ROOT / "outputs/evaluation/dev_original_sft_20261012/manifest.json").read_text())["artifact"]
        if hashes["adapter_model.safetensors"] != artifact["adapter_sha256"]:
            raise ValueError("SFT 适配器与已验收的起点不一致")
    return rows, baseline, project, {"verl_source": source, "sft_adapter_sha256": hashes,
        "baseline_protocol_sha256": protocol["sha256"], "task_ids": [r["id"] for r in rows],
        "training_started": False, "api_calls": 0,
        "precision": "bf16_new_baseline", "logprob_hard_stop": None, **extra}


def build_config(verl_root, output, baseline, project, max_steps, fused=True, calibration_only=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(verl_root / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(ROOT / "configs/verl/retail_grpo.yaml"))
    selected = output / "train_tasks.jsonl"
    config.data.train_files = [str(selected)]
    # 不自动启动 dev/test；保留非空 val dataset 仅满足框架初始化。
    config.data.val_files = [str(selected)]
    config.actor_rollout_ref.model.path = str(ROOT / baseline.get("model_path", "models/qwen3-4b"))
    config.actor_rollout_ref.model.lora_adapter_path = str(ROOT / baseline["sft_adapter_dir"])
    config.actor_rollout_ref.model.use_fused_kernels = fused
    if baseline.get('enable_thinking'):
        config.actor_rollout_ref.actor.optim.lr = baseline['learning_rate']
        config.actor_rollout_ref.rollout.temperature = baseline['temperature']
        config.actor_rollout_ref.rollout.top_p = 1.0
        config.actor_rollout_ref.rollout.top_k = -1
    config.trainer.total_training_steps = max_steps
    config.trainer.experiment_name = output.name
    config.trainer.default_local_dir = str(output / "checkpoints")
    config.trainer.rollout_data_dir = str(output / "rollouts")
    config.retail = {key: baseline[key] for key in ["max_trajectory_tokens", "max_assistant_actions",
        "max_new_tokens_per_action", "max_consecutive_tool_errors", "reward_mode", "review_manifest"]}
    config.retail.review_manifest = str(ROOT / baseline["review_manifest"])
    config.retail.train_tasks = str(ROOT / baseline["train_tasks"])
    config.retail.output_dir = str(output)
    config.retail.customer = project["customer"]
    config.retail.require_nonzero_gradient = not calibration_only
    config.retail.calibration_only = calibration_only
    if baseline.get('enable_thinking'):
        config.retail.enable_thinking = True
        config.retail.experiment_profile = baseline['experiment_profile']
        from retail_agent.mmopd_protocol import PROTOCOL
        config.retail.tool_protocol = PROTOCOL
    if calibration_only:
        config.trainer.save_freq = -1
    config.ray_kwargs.ray_init.runtime_env.py_executable = sys.executable
    # 从项目目录启动时，Ray worker 也必须能导入同一份 Retail 源码。
    config.ray_kwargs.ray_init.runtime_env.env_vars = {"PYTHONPATH": str(ROOT / "src")}
    OmegaConf.resolve(config)
    return config


@contextmanager
def gpu_monitor(output):
    stop = threading.Event()

    def poll():
        with (output / "gpu_memory.jsonl").open("a") as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    event = {"unix_time": time.time(), "returncode": result.returncode,
                             "gpu_rows": result.stdout.splitlines()}
                except (OSError, subprocess.TimeoutExpired) as error:
                    event = {"unix_time": time.time(), "monitor_error": type(error).__name__}
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                stop.wait(1)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=12)


def customer_costs(path):
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    reservations = {event["id"]: event["reserved_cny"] for event in events if event["event"] == "started"}
    for event in events:
        if event["event"] == "completed":
            reservations[event["id"]] = event["peak_cost_upper_cny"]
    return {"api_calls": sum(event["event"] == "started" for event in events),
            "api_completed": sum(event["event"] == "completed" for event in events),
            "api_cost_upper_cny": sum(reservations.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--calibrate", action="store_true", help="真实采样和概率重算，不更新参数")
    parser.add_argument("--verl-root", type=Path, default=ROOT.parent / "verl-main")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--unfused", action="store_true", help="训练链路消融；固定轨迹对照用 profile_verl_memory")
    parser.add_argument("--profile", choices=["legacy", "mmopd"], default="legacy")
    args = parser.parse_args()
    max_steps = args.max_steps if args.max_steps is not None else (1 if args.calibrate else 2)
    verl_root = args.verl_root.resolve()
    rows, baseline, project, report = preflight(verl_root, args.task_id, args.profile)
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.calibrate and max_steps != 1:
        parser.error("校准只允许 max-steps=1，不更新参数或跨组同步")
    if not 1 <= max_steps <= len(rows):
        parser.error("max-steps 必须在 1 到所选 TRAIN 任务数之间，不自动循环补采")
    output = (args.output_dir or ROOT / "outputs/verl" / datetime.now().strftime("migration_%Y%m%d_%H%M%S")).resolve()
    if not output.is_relative_to(ROOT / "outputs/verl"):
        parser.error("输出必须位于项目 outputs/verl 下")
    config = build_config(verl_root, output, baseline, project, max_steps, fused=not args.unfused,
                          calibration_only=args.calibrate)
    from omegaconf import OmegaConf
    if args.dry_run:
        print(OmegaConf.to_yaml(config))
        return
    import torch
    import verl
    if Path(verl.__file__).resolve().parents[1] != verl_root:
        raise RuntimeError("实际导入的 verl 与冻结源码路径不同")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("第一版真实运行要求单张可见 CUDA GPU")
    if args.profile == "legacy":
        from retail_agent.train_sft import verify_model
        verify_model({"model_path": "models/qwen3-4b"}, project)
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from retail_agent.verl_runtime import RetailTaskRunner
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    output.mkdir(parents=True, exist_ok=False)
    (output / "train_tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "config.yaml").write_text(OmegaConf.to_yaml(config))
    if args.profile == 'mmopd':
        from retail_agent.mmopd import profile as load_profile
        (output / 'protocol.json').write_text(json.dumps({
            **load_profile(), 'parent_protocol_sha256': report['baseline_protocol_sha256'],
            'model_manifest_sha256': report['model_manifest_sha256'],
            'initial_adapter_sha256': report['initial_adapter_sha256'],
            'task_ids': report['task_ids'], 'group_size': 4,
            'history_token_policy': report['history_token_policy'],
            'reasoning_loss_mask': 1, 'external_feedback_loss_mask': 0,
            'old_log_prob_source': 'same_batch_actor_forward_before_optimizer',
            'rollout_log_prob_source': 'actual_vllm_sampled_tokens',
            'logprob_hard_stop': None, 'test_used': False,
            'tool_protocol': report['tool_protocol'],
            'tool_protocol_code_sha256': report['tool_protocol_code_sha256'],
        }, ensure_ascii=False, indent=2) + '\n')
    report.update(status="running", utc=datetime.now(timezone.utc).isoformat(),
                  training_started=not args.calibrate, calibration_only=args.calibrate,
                  max_steps=max_steps, fused=not args.unfused,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  total_gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  cuda_runtime=torch.version.cuda)
    report["dependencies"] = {name: version(name) for name in ["torch", "transformers", "peft", "vllm", "ray"]}
    report["entry_sha256"] = {name: hashlib.sha256((ROOT / "src/retail_agent" / name).read_bytes()).hexdigest()
                              for name in ["train_verl.py", "verl_runtime.py", "verl_agent_loop.py"]}
    report["config_sha256"] = hashlib.sha256((output / "config.yaml").read_bytes()).hexdigest()
    (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with gpu_monitor(output):
            run_ppo(config, task_runner_class=RetailTaskRunner)
        report["status"] = ("calibration_recorded_no_optimizer_update" if args.calibrate
                            else "training_returned_pending_reload_acceptance")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(customer_costs(output / "customer_costs.jsonl"))
        report["wall_seconds"] = time.perf_counter() - started
        (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
