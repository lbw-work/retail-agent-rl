"""GRPO 只读检查或显式启动单卡 CUDA 运行；先做一次真实更新 smoke。"""

import argparse
from contextlib import contextmanager
import json
import tomllib
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

from retail_agent.environment import ROOT
from retail_agent.observability import StageTimer
from retail_agent.experiment_protocol import check_protocol
from retail_agent.sft_data import file_sha256
from retail_agent.train_sft import dependency_matches, verify_model
from retail_agent.training_reward import VERSION, row_fingerprint, ready_training_rows


def load_sft_adapter(base, directory, timing=None):
    from peft import PeftModel
    # 不传新 peft_config，不 merge/unload；直接继续更新已训练的 SFT LoRA 副本。
    timing = timing if timing is not None else StageTimer()
    with timing.measure("sft_adapter_load", gpu=True):
        model = PeftModel.from_pretrained(base, directory, is_trainable=True, local_files_only=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n for n in names):
        raise ValueError("SFT 适配器没有可训练 LoRA，或底座未冻结")
    return model


def prepare_grpo_qlora(model, dtype):
    from retail_agent.train_sft import prepare_qlora
    model = prepare_qlora(model)
    # 只改 GRPO 执行精度，量化权重和 SFT 入口保持原样。
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.__class__.__name__ != "Params4bit":
            parameter.data = parameter.data.to(dtype)
    return model


@contextmanager
def grpo_execution_math(model):
    """单进程 CUDA FP32：采样/训练共用 NF4 矩阵路径，避免低精度形状漂移。"""
    import torch
    if model.device.type != "cuda" or model.get_input_embeddings().weight.dtype != torch.float32:
        yield
        return
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from transformers.integrations import sdpa_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    original_matmul, original_gqa = bnb.matmul_4bit, sdpa_attention.use_gqa_in_sdpa
    def matrix_matmul(a, b, quant_state, out=None, bias=None):
        return MatMul4Bit.apply(a, b, out, bias, quant_state)
    try:
        bnb.matmul_4bit = matrix_matmul
        # Torch 2.8 Efficient SDPA 需要显式展开 GQA，仍由 Transformers 完成。
        sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    finally:
        bnb.matmul_4bit, sdpa_attention.use_gqa_in_sdpa = original_matmul, original_gqa


def preflight():
    protocol = check_protocol()
    config = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    if (config["model_id"] != project["model_id"] or config["enable_thinking"]
            or config["num_generations"] != 4 or config["num_iterations"] != 1
            or config["use_vllm"] or config["loss_type"] != "grpo"
            or config["scale_rewards"] != "group" or config["beta"] != 0
            or config["reward_mode"] != "reward_a" or not 0 < config["temperature"] <= 1
            or config["loss_scope"] != "sampled_assistant_tokens_only"
            or not 0 < config["max_trajectory_tokens"] <= 16384
            or config["max_assistant_actions"] < 1 or config["max_new_tokens_per_action"] < 1
            or config["max_new_tokens_per_action"] > config["max_trajectory_tokens"]
            or config["max_consecutive_tool_errors"] < 1
            or not 0 < config["logprob_tolerance"] <= 0.05):
        raise ValueError("GRPO 配方超出当前验收范围")
    dependencies = {}
    for file in ["train.txt", "grpo.txt"]:
        for line in (ROOT / "requirements" / file).read_text().splitlines():
            if not line or line.startswith(("#", "-")):
                continue
            name, expected = line.split("==")
            actual = version(name)
            if not dependency_matches(name, actual, expected):
                raise ValueError(f"训练依赖版本漂移：{name}")
            dependencies[name] = actual
    manifest_path = ROOT / config.get('review_manifest',
        'data/versions/retail_full_v2/reviewed/phase05a_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    pool = ROOT / config["train_tasks"]
    rows = ready_training_rows(pool, config.get('review_manifest'))
    if config.get('review_manifest'):
        from retail_agent.rl_dataset import reviewed_rl_rows
        reviewed_rl_rows(config['development_tasks'], 'dev', config['review_manifest'])
    for relative, expected in manifest["immutable_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("原始划分或数据库已变化：" + relative)
    approved = {e["id"]: e["row_sha256"] for e in manifest["entries"]
                if not config.get('review_manifest') or e['split'] == 'train' and e['status'] == 'accepted'}
    if len(rows) != len({r["id"] for r in rows}) or any(
            approved.get(r["id"]) != row_fingerprint(r) for r in rows):
        raise ValueError("任务池重复、未复核或来源行被修改")
    directory = ROOT / config["sft_adapter_dir"]
    run = json.loads((directory.parent / "run.json").read_text())
    artifact = json.loads((ROOT / "outputs/evaluation/dev_original_sft_20261012/manifest.json").read_text())["artifact"]
    if (file_sha256(directory / "adapter_model.safetensors") != artifact["adapter_sha256"]
            or file_sha256(directory.parent / "run.json") != artifact["run_sha256"]
            or (run["model_id"], run["model_revision"]) != (project["model_id"], project["model_revision"])
            or run["data_manifest_sha256"] != file_sha256(ROOT / "data/prepared/retail_full_v2_sft/manifest.json")):
        raise ValueError("SFT 起点与已评测的成果不一致")
    adapter = json.loads((directory / "adapter_config.json").read_text())
    if (adapter["peft_type"] != "LORA" or adapter["r"] != 16 or adapter["lora_alpha"] != 32
            or adapter["lora_dropout"] != 0 or adapter["bias"] != "none"
            or set(adapter["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj"}
            or any(adapter.get(k) for k in ["use_dora", "use_rslora", "use_qalora", "fan_in_fan_out",
                "lora_bias", "rank_pattern", "alpha_pattern", "modules_to_save", "layers_to_transform",
                "layer_replication", "trainable_token_indices", "alora_invocation_tokens"])):
        raise ValueError("SFT LoRA 配置不符")
    model_path = verify_model({"model_path": "models/qwen3-4b"}, run)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data/raw/tokenizer", local_files_only=True)
    if tokenizer.convert_tokens_to_ids("<|im_end|>") == tokenizer.unk_token_id:
        raise ValueError("缺少固定结束 token")
    return {"status": "offline_preflight_passed_not_cuda_training",
        "protocol_sha256": protocol["sha256"],
        "training_reward_version": VERSION,
        "config": config, "dependencies": dependencies, "task_count": len(rows),
        "pool_sha256": file_sha256(pool), "sft_adapter_sha256": artifact["adapter_sha256"],
        "base_model_files_sha256_verified": True, "model_path": str(model_path),
        "sft_adapter_dir": str(directory), "customer_api_called": False,
        "optimizer_steps": 0, "paid_cost_cny": 0}


def parameter_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not torch.isfinite(parameter).all():
                raise ValueError("LoRA 参数非有限：" + name)
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_run_args(args, config):
    if (args.max_steps is None or not 1 <= args.max_steps <= config["max_steps"]
            or args.max_minutes is None or not math.isfinite(args.max_minutes) or args.max_minutes <= 0
            or args.api_budget_cny is not None and (
                not math.isfinite(args.api_budget_cny) or args.api_budget_cny <= 0)
            or not args.output_dir):
        raise ValueError("训练须指定正数步数、时限和新输出目录；可选API限额必须为正数，步数不得超过配置上限")
    if args.smoke and (args.max_steps != 1 or not args.task_id):
        raise ValueError("smoke 固定一次更新，并显式指定一个已复核 TRAIN task-id")
    if not args.smoke and args.task_id:
        raise ValueError("正式训练使用完整可用 TRAIN 池，不能用 task-id 筛选")


def run_cuda(args):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from transformers.trainer_utils import SaveStrategy
    from retail_agent.customer import DeepSeekCustomer, GOAL_POLICY
    from retail_agent.grpo_trainer import RetailGRPOTrainer, make_grpo_args
    from retail_agent.train_sft import training_time_limit

    config = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    validate_run_args(args, config)
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "outputs/grpo").resolve()):
        raise ValueError("训练输出必须位于项目 outputs/grpo 下")
    if output.exists():
        raise FileExistsError("输出目录已存在，不能覆盖或静默恢复旧实验")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("真实 GRPO 需要单张 CUDA GPU；用 CUDA_VISIBLE_DEVICES=0 指定")
    if version("bitsandbytes") != "0.48.1":
        raise ValueError("GPU 量化依赖须为 bitsandbytes==0.48.1")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    timing = StageTimer(output, synchronize=lambda: torch.cuda.synchronize())
    with timing.measure("offline_preflight"):
        checked = preflight()
    rows = ready_training_rows(ROOT / config["train_tasks"], config["review_manifest"])
    if args.smoke:
        rows = [row for row in rows if row["id"] == args.task_id]
        if not rows:
            raise ValueError("smoke task-id 不属于当前复核 TRAIN 池")
    customer_config = {**project["customer"], "goal_policy": GOAL_POLICY}
    probe = DeepSeekCustomer(rows[0]["user_scenario"], customer_config, limit_cny=None)
    start_api = probe.accounted_cny()
    probe.http.close()
    api_limit = start_api + args.api_budget_cny if args.api_budget_cny is not None else None
    if api_limit is not None and api_limit > project["budget"]["api_limit"]:
        raise ValueError("本次 API 费用预留超过项目 API 上限")
    run = {"status": "initializing", "kind": "cuda_smoke" if args.smoke else "cuda_baseline",
        "utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": checked["protocol_sha256"],
        "config": config, "max_steps": args.max_steps, "max_minutes": args.max_minutes,
        "train_task_ids": [row["id"] for row in rows], "customer_config": customer_config,
        "api_ledger_start_cny": start_api, "api_additional_limit_cny": args.api_budget_cny,
        "api_cost_limit_enforced": api_limit is not None,
        "gpu": torch.cuda.get_device_name(), "observation_run_id": timing.run_id,
        "sft_adapter_sha256": checked["sft_adapter_sha256"], "global_step": 0,
        "execution_precision": "float32",
        "base_quantization_input_precision": "bfloat16",
        "fp16_reduced_precision_reduction": False, "bf16_reduced_precision_reduction": False,
        "entry_code_sha256": file_sha256(Path(__file__)),
        "note": "步结束后检查时限；一次长组可超过时限。服务器计费须另行核对。"}
    (output / "protocol.json").write_bytes((ROOT / "configs/grpo_baseline_protocol.json").read_bytes())
    trainer, customers = None, []
    def customer_factory(scenario):
        customer = DeepSeekCustomer(scenario, customer_config, limit_cny=api_limit)
        customers.append(customer)
        return customer
    try:
        with timing.measure("cuda_run", gpu=True):
            set_seed(config["seed"])
            with timing.measure("base_model_load", gpu=True):
                tokenizer = AutoTokenizer.from_pretrained(checked["model_path"], local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(checked["model_path"], local_files_only=True,
                    # 保留 SFT 的 BF16→NF4 量化起点，随后只切换执行精度。
                    dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0},
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float32))
                model = prepare_grpo_qlora(model, torch.float32)
            model = load_sft_adapter(model, checked["sft_adapter_dir"], timing=timing)
            model.config.use_cache = False
            training_args = make_grpo_args(config, output, args.max_steps)
            training_args.save_strategy = SaveStrategy.STEPS
            training_args.save_steps = 1 if args.smoke or args.max_steps < 5 else 5
            training_args.save_total_limit = 2
            trainer = RetailGRPOTrainer(model=model, processing_class=tokenizer,
                args=training_args, rows={row["id"]: row for row in rows}, rollout_config=config,
                customer_factory=customer_factory, timing=timing,
                require_nonzero_gradient=args.smoke,
                train_dataset=Dataset.from_list([{"prompt": row["id"]} for row in rows]))
            with timing.measure("adapter_fingerprint", gpu=True):
                before = parameter_digest(model)
            limit = training_time_limit(args.max_minutes)
            trainer.add_callback(limit)
            run.update(status="running", lora_before_sha256=before,
                dependencies={name: version(name) for name in
                    ["torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes"]})
            (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
            with grpo_execution_math(model):
                result = trainer.train()
            with timing.measure("adapter_fingerprint", gpu=True):
                after = parameter_digest(model)
            if before == after:
                raise ValueError("没有可验证的 LoRA 更新，不标记训练完成")
            trainer.save_model(str(output / "adapter"))
            with timing.measure("tokenizer_save"):
                tokenizer.save_pretrained(output / "adapter")
            run.update(status="stopped_time_limit" if limit.expired and trainer.state.global_step < args.max_steps
                else "completed", metrics=result.metrics, lora_after_sha256=after,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        run.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        for customer in customers:
            customer.http.close()
        run.update(global_step=trainer.state.global_step if trainer is not None else 0,
            api_additional_accounted_upper_cny=probe.accounted_cny() - start_api,
            timing=timing.summary())
        (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        timing.save()
    print(json.dumps(run, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="只读检查，无模型采样/API/更新")
    mode.add_argument("--smoke", action="store_true", help="一次真实 CUDA 更新，会调用顾客 API")
    mode.add_argument("--train", action="store_true", help="显式正式 CUDA 训练，会调用顾客 API")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--api-budget-cny", type=float,
                        help="可选：新增API费用上限；默认不设费用上限，仍记录费用并检查余额")
    parser.add_argument("--output-dir")
    parser.add_argument("--task-id", help="smoke 使用的已复核 TRAIN ID")
    args = parser.parse_args()
    if not args.preflight:
        run_cuda(args)
        return
    if any(getattr(args, key) is not None for key in
           ["max_steps", "max_minutes", "api_budget_cny", "output_dir", "task_id"]):
        parser.error("preflight 不接受训练参数")
    timing = StageTimer(ROOT / "outputs/preflight")
    with timing.measure("offline_preflight"):
        result = preflight()
    result["observation_run_id"] = timing.run_id
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
