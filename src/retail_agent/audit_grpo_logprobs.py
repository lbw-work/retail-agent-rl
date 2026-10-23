"""只读复算首步已保存轨迹：不生成新顾客、不训练、不访问 API。"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace
import time
import tomllib

import torch
from trl import GRPOTrainer

from retail_agent.environment import ROOT
from retail_agent.grpo_rollout import RolloutError
from retail_agent.grpo_trainer import text_policy_logprobs


def validate_trace(trace):
    prompt, completion = trace['prompt_ids'], trace['completion_ids']
    mask, logps = trace['assistant_mask'], trace['logprobs']
    if (not prompt or not completion or len(completion) != len(mask) or len(mask) != len(logps)
            or any(type(t) is not int or t < 0 for t in prompt + completion)
            or any(type(m) is not int or m not in (0, 1) for m in mask)):
        raise RolloutError('保存轨迹的 token/mask/logprob 长度或值非法')
    covered = [0] * len(completion)
    previous = 0
    for turn in trace['turns']:
        start, end = turn['start'], turn['end']
        if not previous <= start < end <= len(completion):
            raise RolloutError('保存轨迹的回合边界重叠或越界')
        covered[start:end] = [1] * (end - start)
        previous = end
    if covered != mask or not any(mask):
        raise RolloutError('回合边界与 assistant mask 不对应')
    if not torch.isfinite(torch.tensor(logps)).all() or any(
            p > 1e-6 if m else p != 0 for m, p in zip(mask, logps, strict=True)):
        raise RolloutError('保存轨迹的行为概率无效')


def read_traces(path):
    traces, manifest = [], None
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get('event') == 'run_manifest':
            manifest = event
        if event.get('event') == 'rollout_trace' and event.get('sampled_policy_step') == 0:
            validate_trace(event['trace'])
            traces.append(event['trace'])
    if manifest is None or not traces:
        raise ValueError('需要同次运行的 run_manifest 和第0步完整 rollout_trace，不能用 MLX 轨迹替代')
    return traces, manifest


def forward_case(model, trace, token, padded_completion_length, temperature, mode):
    """所有路径使用同一 token；cache 从该动作的完整前缀开始，匹配实际逐轮重建。"""
    prompt, completion = trace['prompt_ids'], trace['completion_ids']
    absolute = len(prompt) + token
    ids = torch.tensor([prompt + completion], device=model.device)
    attention = torch.ones_like(ids)
    if mode == 'prefix':
        logits = model(input_ids=ids[:, :absolute], attention_mask=attention[:, :absolute],
                       use_cache=False, logits_to_keep=1).logits[0, -1]
    elif mode == 'cache':
        turn = next(t for t in trace['turns'] if t['start'] <= token < t['end'])
        prefix_end = len(prompt) + turn['start']
        output = model(input_ids=ids[:, :prefix_end], attention_mask=attention[:, :prefix_end],
                       position_ids=torch.arange(prefix_end, device=ids.device)[None],
                       cache_position=torch.arange(prefix_end, device=ids.device),
                       use_cache=True, logits_to_keep=1)
        for position in range(prefix_end, absolute):
            output = model(input_ids=ids[:, position:position + 1],
                           attention_mask=attention[:, :position + 1],
                           position_ids=torch.tensor([[position]], device=ids.device),
                           cache_position=torch.tensor([position], device=ids.device),
                           past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        logits = output.logits[0, -1]
    elif mode == 'full_selected':
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False,
                       logits_to_keep=torch.tensor([absolute - 1], device=ids.device)).logits[0, 0]
    else:
        pad = padded_completion_length - len(completion)
        ids = torch.nn.functional.pad(ids, (0, pad), value=model.config.pad_token_id)
        attention = torch.nn.functional.pad(attention, (0, pad), value=0)
        if mode == 'full_training':
            probabilities, _ = text_policy_logprobs(model, ids, attention, padded_completion_length,
                                                    temperature, batch_size=1)
        elif mode in ('full_trl_native', 'full_trl_prepared'):
            # 原版 0.26.2 源码，保留其 dtype/temperature 行为供对照，不用于生产 Loss。
            native = SimpleNamespace(model_kwarg_keys={'logits_to_keep'}, temperature=temperature,
                                     args=SimpleNamespace(report_to=[]))
            def prepared_forward(**kwargs):
                output = model(**kwargs)
                output.logits = output.logits.float()
                return output
            probabilities, _ = GRPOTrainer._get_per_token_logps_and_entropies(
                native, prepared_forward if mode == 'full_trl_prepared' else model,
                ids, attention, padded_completion_length, batch_size=1)
        else:
            raise ValueError('未知 forward 路径')
        return float(probabilities[0, token]), str(probabilities.dtype)
    probability = torch.log_softmax(logits.float() / temperature, -1)[completion[token]]
    return float(probability), str(logits.dtype)


def run_matrix(model, trace, token, padded_completion_length, temperature, record=None, *, focused=False):
    validate_trace(trace)
    if not 0 <= token < len(trace['completion_ids']) or trace['assistant_mask'][token] != 1:
        raise ValueError('目标必须是 completion 中真实采样的客服 token，位置从0开始')
    device_type = model.device.type
    execution_dtype = model.get_input_embeddings().weight.dtype
    rows = []
    model.eval()
    original_attention = model.config._attn_implementation
    try:
        for attention in ['sdpa'] if focused else ['sdpa', 'eager']:
            model.get_base_model().set_attn_implementation(attention)
            for autocast in [True] if focused else [False, True]:
                modes = ['cache', 'prefix', 'full_selected', 'full_training'] if focused else [
                    'cache', 'prefix', 'full_selected', 'full_trl_native', 'full_trl_prepared', 'full_training']
                for mode in modes:
                    autocast_enabled = autocast and execution_dtype != torch.float32
                    if device_type == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.no_grad(), torch.autocast(device_type, dtype=execution_dtype, enabled=autocast_enabled):
                        logp, dtype = forward_case(model, trace, token, padded_completion_length, temperature, mode)
                    if device_type == 'cuda':
                        torch.cuda.synchronize()
                    row = {'attention': attention, 'autocast_bf16': autocast and execution_dtype == torch.bfloat16,
                           'autocast_enabled': autocast_enabled, 'autocast_dtype': str(execution_dtype), 'mode': mode,
                           'logprob': logp, 'dtype': dtype, 'seconds': time.perf_counter() - started,
                           'delta_from_saved_behavior': logp - trace['logprobs'][token]}
                    if device_type == 'cuda':
                        row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    rows.append(row)
                    if record is not None:
                        record(row)
    finally:
        model.get_base_model().set_attn_implementation(original_attention)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-file', required=True, type=Path)
    parser.add_argument('--run-json', type=Path, help='默认取 trace-file 的 ../../run.json')
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--completion-token', type=int, required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--focused', action='store_true',
                        help='先只复算SDPA及该运行训练精度下的四条路径，避免重复完整矩阵')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('诊断目录已存在；不覆盖历史证据')
    traces, manifest = read_traces(args.trace_file)
    if not 0 <= args.row < len(traces):
        raise ValueError('轨迹行号越界')
    trace = traces[args.row]
    if not 0 <= args.completion_token < len(trace['completion_ids']) or not trace['assistant_mask'][args.completion_token]:
        raise ValueError('诊断位置不是客服采样 token')
    run_path = args.run_json or args.trace_file.parent.parent / 'run.json'
    run = json.loads(run_path.read_text())
    if run['global_step'] != 0 or not run.get('lora_before_sha256'):
        raise ValueError('本工具只复核参数尚未更新的首步；缺少起点摘要不能声称相同权重')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('真实 NF4 复算需要现有 CUDA BF16 环境；不会转用 CPU/MLX 冒充')
    for name, expected in {'trl': '0.26.2', 'transformers': '4.57.6', 'peft': '0.18.0',
                           'accelerate': '1.12.0', 'bitsandbytes': '0.48.1'}.items():
        if version(name) != expected:
            raise ValueError('诊断依赖版本漂移：' + name)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
    from retail_agent.train_grpo import load_sft_adapter, parameter_digest, prepare_grpo_qlora, grpo_execution_math
    from retail_agent.train_sft import verify_model
    from retail_agent.sft_data import file_sha256
    project = tomllib.loads((ROOT / 'configs/project.toml').read_text())
    config = run['config']
    precision = run.get('execution_precision', 'bfloat16')
    if precision not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('未验收的历史运行精度')
    execution_dtype = getattr(torch, precision)
    if execution_dtype in (torch.float16, torch.float32):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if config['use_vllm'] or config['enable_thinking'] or config['model_id'] != 'Qwen/Qwen3-4B':
        raise ValueError('输入不属于当前 Transformers Qwen3 非思考路径')
    adapter = ROOT / config['sft_adapter_dir']
    if file_sha256(adapter / 'adapter_model.safetensors') != run['sft_adapter_sha256']:
        raise ValueError('SFT adapter 与失败运行不同')
    artifact = json.loads((adapter.parent / 'run.json').read_text())
    if artifact['model_revision'] != project['model_revision']:
        raise ValueError('底座 revision 不一致')
    model_path = verify_model({'model_path': 'models/qwen3-4b'}, artifact)
    args.output_dir.mkdir(parents=True)
    report = {'status': 'initializing', 'api_calls': 0, 'optimizer_steps': 0,
              'source_trace_sha256': file_sha256(args.trace_file), 'source_run_sha256': file_sha256(run_path),
              'historical_protocol_sha256': run['protocol_sha256'], 'historical_runtime': manifest,
              'audit_entry_sha256': file_sha256(Path(__file__)),
              'math_code_sha256': file_sha256(ROOT / 'src/retail_agent/grpo_trainer.py'),
              'normalization_note': 'native=unwrapped TRL; prepared=TRL after FP32 output conversion; '
                                    'autocast true reproduces training precision, false tests legacy generate',
              'target': {'row': args.row, 'completion_token': args.completion_token,
                         'absolute_token': len(trace['prompt_ids']) + args.completion_token,
                         'token_id': trace['completion_ids'][args.completion_token],
                         'saved_behavior_logprob': trace['logprobs'][args.completion_token]},
              'dependencies': {name: version(name) for name in
                               ['torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes']},
              'gpu': torch.cuda.get_device_name(), 'torch_cuda': torch.version.cuda,
              'execution_precision': precision}
    report['focused'] = args.focused
    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=execution_dtype))
        model = load_sft_adapter(prepare_grpo_qlora(model, execution_dtype), adapter)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # 与实际 Trainer 完全复用同一个对齐方法；底座原始 pad=None，不能用它补齐离线轨迹。
        Trainer._align_special_tokens(SimpleNamespace(model=model, processing_class=tokenizer))
        report['aligned_special_tokens'] = {name: getattr(model.config, name)
                                            for name in ['pad_token_id', 'bos_token_id', 'eos_token_id']}
        before = parameter_digest(model)
        if before != run['lora_before_sha256']:
            raise ValueError('实际可训练参数摘要与失败时不同，拒绝无效比较')
        if max(trace['prompt_ids'] + trace['completion_ids']) >= model.config.vocab_size:
            raise ValueError('token 超出当前模型词表')
        report['weights_before_sha256'] = before
        report['matrix'] = []
        def record(row):
            report['matrix'].append(row)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
        with grpo_execution_math(model):
            run_matrix(model, trace, args.completion_token,
                max(len(t['completion_ids']) for t in traces), config['temperature'], record=record,
                focused=args.focused)
        report['weights_after_sha256'] = parameter_digest(model)
        if report['weights_after_sha256'] != before:
            raise ValueError('离线复算意外改变参数')
        report['status'] = 'offline_forward_matrix_completed_not_smoke_acceptance'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(report_path)


if __name__ == '__main__':
    main()
