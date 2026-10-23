"""CUDA BF16 下逐检查点评估完整 DEV；不以 TEST 或训练奖励选择模型。"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tomllib

from retail_agent.environment import ROOT
from retail_agent.model import ContextLimit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def completed_tasks(attempts, task_ids):
    """只允许显式续跑系统故障；业务失败也是已完成任务，不重新抽奖。"""
    latest = {}
    for result in attempts:
        task = result['task_id']
        if task not in task_ids or (task in latest and latest[task]['business_evaluation_eligible']):
            raise ValueError('非计划任务或重复业务评估：' + task)
        latest[task] = result
    return {key: value for key, value in latest.items() if value['business_evaluation_eligible']}


def candidate_score(step, attempts, task_ids):
    completed = completed_tasks(attempts, task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids) or len(completed) != len(task_ids):
        raise ValueError('开发评估未完整完成，不能参与模型选择')
    rewards = [completed[key]['reward']['reward'] for key in task_ids]
    if any(not isinstance(r, (int, float)) or not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError('开发奖励无效')
    successes = sum(r == 1 for r in rewards)
    return {'step': step, 'planned': len(task_ids), 'successes': successes,
            'success_rate': successes / len(task_ids),
            'system_fault_attempts': sum(not r['business_evaluation_eligible'] for r in attempts)}


def select_best(scores, expected_steps, candidate_steps=None):
    wanted = list(range(expected_steps + 1)) if candidate_steps is None else [0, *candidate_steps]
    if sorted(s['step'] for s in scores) != sorted(wanted) or len(set(wanted)) != len(wanted):
        raise ValueError('缺少 SFT 对照或训练检查点，不能声称全检查点最优')
    if len({s['planned'] for s in scores}) != 1:
        raise ValueError('候选模型的开发集不一致')
    # 同分优先更早的步骤；SFT 为 step 0，避免无收益时强行替换。
    return min(scores, key=lambda s: (-s['successes'], s['step']))


class CudaGreedyAgent:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    def generate(self, messages, tools):
        import torch
        from transformers import GenerationConfig

        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        limit = self.config['max_new_tokens_per_action']
        if len(prompt) + limit > self.config['max_context_tokens']:
            raise ContextLimit('评估上下文超限；不截断')
        inputs = torch.tensor([prompt], device='cuda', dtype=torch.long)
        eos = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        generation = GenerationConfig(do_sample=False, max_new_tokens=limit,
            eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=generation, use_model_defaults=False, disable_compile=True,
                logits_to_keep=1)
        tokens = output[0, len(prompt):].tolist()
        stopped = bool(tokens) and tokens[-1] == eos
        return {'raw_text': self.tokenizer.decode(tokens[:-1] if stopped else tokens,
                    skip_special_tokens=False), 'finish_reason': 'stop' if stopped else 'length',
                'prompt_token_ids': prompt, 'generated_token_ids': tokens,
                'backend': 'cuda_bf16_sdpa_greedy', 'sampling_temperature': 0.0}


def prepare(run_dir, steps, candidate_steps=None):
    from retail_agent.experiment_protocol import check_protocol
    from retail_agent.rl_dataset import reviewed_rl_rows
    from retail_agent.verl_agent_loop import selected_training_rows

    run = json.loads((run_dir / 'run.json').read_text())
    if (run.get('status') != 'training_returned_pending_reload_acceptance'
            or run.get('calibration_only') or run.get('max_steps') != steps):
        raise ValueError('需要已完整返回的对应步数 verl 训练')
    parent = check_protocol()
    config = {key: parent['protocol']['evaluation_config'][key] for key in
        ['tasks', 'review_manifest', 'seed', 'max_assistant_actions',
         'max_new_tokens_per_action', 'max_context_tokens', 'max_consecutive_tool_errors']}
    project = tomllib.loads((ROOT / 'configs/project.toml').read_text())
    baseline = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
    native = None
    if run.get('experiment_profile') == 'mmopd_native_think_v1':
        from retail_agent.mmopd import PROFILE, profile, verify_snapshot
        native = profile()
        assets = verify_snapshot(require_adapter=True)
        if (run.get('model_manifest_sha256') != assets['model_manifest_sha256']
                or run.get('profile_sha256') != digest(PROFILE)):
            raise ValueError('MMOPD训练快照或profile发生变化')
        baseline.update(native)
        project.update(model_id=native['model_id'], model_revision=native['model_revision'])
        config.update(enable_thinking=True, max_context_tokens=native['max_trajectory_tokens'],
            max_new_tokens_per_action=native['max_new_tokens_per_action'],
            max_trajectory_tokens=native['max_trajectory_tokens'], temperature=native['temperature'])
        if run.get('tool_protocol'):
            from retail_agent.mmopd_protocol import PROTOCOL, LEGACY_PROTOCOL
            if run['tool_protocol'] not in {PROTOCOL, LEGACY_PROTOCOL}:
                raise ValueError('未知的训练多工具协议')
            config['tool_protocol'] = run['tool_protocol']
    rows = reviewed_rl_rows(config['tasks'], 'dev', config['review_manifest'])
    if len(rows) != 16 or [r['id'] for r in rows] != parent['protocol']['development_task_ids']:
        raise ValueError('必须使用冻结的完整 DEV16')
    original = ROOT / baseline['sft_adapter_dir']
    for name in ['adapter_model.safetensors', 'adapter_config.json']:
        if run['sft_adapter_sha256'].get(name) != digest(original / name):
            raise ValueError('训练起点与当前 SFT 不一致：' + name)
    selected = selected_training_rows(run_dir / 'train_tasks.jsonl')
    if len(selected) < steps or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('训练任务数量不足或重复')
    if candidate_steps is None:
        candidate_steps = ([s for s in native['evaluation_steps'] if s <= steps] if native
                           else list(range(1, steps + 1)))
        if steps not in candidate_steps:
            candidate_steps.append(steps)
    if (not candidate_steps or len(set(candidate_steps)) != len(candidate_steps)
            or any(type(s) is not int or not 1 <= s <= steps for s in candidate_steps)):
        raise ValueError('候选检查点步数非法')
    candidate_steps = sorted(candidate_steps)
    candidates = [{'step': 0, 'adapter_sha256': digest(original / 'adapter_model.safetensors'),
                   'adapter_config_sha256': digest(original / 'adapter_config.json')}]
    for step in candidate_steps:
        folder = run_dir / f'checkpoints/global_step_{step}/actor'
        candidates.append({'step': step,
            'model_sha256': digest(folder / 'model_world_size_1_rank_0.pt'),
            'optimizer_sha256': digest(folder / 'optim_world_size_1_rank_0.pt')})
    protocol = {'version': 'verl_cuda_dev_selection_v1', 'parent_protocol_sha256': parent['sha256'],
        'training_run_sha256': digest(run_dir / 'run.json'),
        'training_tasks_sha256': digest(run_dir / 'train_tasks.jsonl'), 'candidates': candidates,
        'task_ids': [r['id'] for r in rows], 'dev_sha256': digest(ROOT / config['tasks']),
        'backend': 'cuda_bf16_sdpa_greedy', 'evaluation_config': config,
        'customer': {**project['customer'],
                     'goal_policy': parent['protocol']['evaluation_customer_goal_policy']},
        'seed': config['seed'], 'trials_per_task': 1,
        'selection': 'strict_dev_success_count_then_earliest_step_including_sft', 'test_used': False,
        'model_path': native['model_path'] if native else 'models/qwen3-4b',
        'experiment_profile': native['experiment_profile'] if native else 'legacy',
        'data_exposure': native['data_exposure'] if native else 'original_project_split',
        'candidate_steps': candidate_steps, 'total_training_steps': steps,
        'checkpoint_selection_scope': ('all_checkpoints' if len(candidate_steps) == steps
                                       else 'predeclared_checkpoint_subset'),
        'code_sha256': {name: digest(ROOT / 'src/retail_agent' / name) for name in
            ['eval_verl.py', 'audit_verl.py', 'rollout.py', 'customer.py', 'model.py',
             'reward.py', 'environment.py', 'rl_dataset.py']}}
    if native:
        protocol['code_sha256']['mmopd.py'] = digest(ROOT / 'src/retail_agent/mmopd.py')
        protocol['backend'] = 'cuda_bf16_sdpa_native_thinking_full_vocab_sampling'
        if config.get('tool_protocol'):
            protocol['code_sha256']['mmopd_protocol.py'] = digest(ROOT / 'src/retail_agent/mmopd_protocol.py')
    return protocol, rows, original, project


def evaluate_candidate(adapter, folder, rows, protocol, ledger, api_limit):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from retail_agent.customer import DeepSeekCustomer
    from retail_agent.rollout import run_episode

    folder.mkdir(parents=True, exist_ok=True)
    trace = folder / 'traces.jsonl'
    attempts = [json.loads(s) for s in trace.read_text().splitlines() if s.strip()] if trace.exists() else []
    done = completed_tasks(attempts, protocol['task_ids'])
    if len(done) == len(rows):
        return attempts
    config = protocol['evaluation_config']
    model_path = ROOT / protocol.get('model_path', 'models/qwen3-4b')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    if config.get('enable_thinking'):
        from retail_agent.mmopd import NativeAgent
        agent = NativeAgent(model, tokenizer, config)
    else:
        agent = CudaGreedyAgent(model, tokenizer, config)
    try:
        for index, row in enumerate(rows):
            if row['id'] in done:
                continue
            torch.manual_seed(protocol['seed'])
            if config.get('enable_thinking'):
                agent.reset()
            customer = DeepSeekCustomer(row['user_scenario'], protocol['customer'],
                limit_cny=api_limit, ledger_path=ledger)
            print(f'{folder.name} {index + 1}/{len(rows)} {row["id"]}', flush=True)
            try:
                result = run_episode(row, agent, customer, config['max_assistant_actions'],
                                     config['max_consecutive_tool_errors'])
            finally:
                customer.http.close()
            with trace.open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            attempts.append(result)
            if not result['business_evaluation_eligible']:
                raise RuntimeError('评估系统故障，保留轨迹并停止：' + str(result['diagnostic']))
    finally:
        del agent, model, base
        gc.collect()
        torch.cuda.empty_cache()
    return attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, default=94)
    parser.add_argument('--api-limit-cny', type=float, default=10.0,
                        help='历史账本累计上限；MMOPD可传0不限额，仍完整记账')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true', help='仅复用完整业务评估，系统故障任务可续跑')
    parser.add_argument('--candidate-steps', type=int, nargs='+', help='预先固定的评估步骤；MMOPD默认每10步及末步')
    parser.add_argument('--all-checkpoints', action='store_true', help='评估全部步骤；会增加顾客API调用')
    args = parser.parse_args()
    if not 1 <= args.expected_steps <= 94 or not math.isfinite(args.api_limit_cny) or args.api_limit_cny < 0:
        parser.error('步数或API上限无效')
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    if not run_dir.is_relative_to(ROOT / 'outputs/verl') or not output.is_relative_to(ROOT / 'outputs/verl'):
        parser.error('训练与评估目录必须在本项目outputs/verl下')
    if args.candidate_steps and args.all_checkpoints:
        parser.error('不能同时指定候选列表和全部检查点')
    wanted = list(range(1, args.expected_steps + 1)) if args.all_checkpoints else args.candidate_steps
    protocol, rows, original, project = prepare(run_dir, args.expected_steps, wanted)
    if args.api_limit_cny == 0 and protocol['experiment_profile'] == 'legacy':
        parser.error('旧实验仍须保留原评估预算；不限额仅适用独立MMOPD实验')
    if args.preflight:
        print(json.dumps({'candidates': len(protocol['candidates']), 'dev_tasks': len(rows),
            'planned_episodes': len(rows) * len(protocol['candidates']), 'test_used': False,
            'candidate_steps': protocol['candidate_steps'],
            'checkpoint_selection_scope': protocol['checkpoint_selection_scope'],
            'data_exposure': protocol['data_exposure'],
            'training_started': False, 'api_calls': 0}, indent=2))
        return
    import torch
    from retail_agent.audit_verl import export_adapter, reload_cuda
    from retail_agent.train_sft import verify_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('只支持一张可见CUDA GPU')
    if protocol['experiment_profile'] == 'legacy':
        verify_model({'model_path': 'models/qwen3-4b'}, project)
    if output.exists():
        if not args.resume or json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('输出已存在；仅允许相同协议的显式resume')
        if (output / 'best.json').exists():
            raise ValueError('评选已完成，禁止覆盖')
    else:
        output.mkdir(parents=True)
        (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    ledger = ROOT / 'outputs/budget/deepseek_requests.jsonl'
    with (output / 'execution.jsonl').open('a') as stream:
        stream.write(json.dumps({'api_limit_cny': args.api_limit_cny, 'resume': args.resume,
            'ledger': str(ledger), 'torch_version': torch.__version__,
            'gpu': torch.cuda.get_device_name(0)}) + '\n')
    scores = []
    for candidate in protocol['candidates']:
        step = candidate['step']
        folder = output / f'candidate_{step:03d}'
        adapter = original if step == 0 else folder / 'adapter'
        if step and not adapter.exists():
            report = export_adapter(run_dir / f'checkpoints/global_step_{step}/actor', original, adapter)
            (adapter / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
        attempts = evaluate_candidate(adapter, folder, rows, protocol, ledger, args.api_limit_cny or None)
        score = candidate_score(step, attempts, protocol['task_ids'])
        score['adapter'] = str(adapter)
        scores.append(score)
        (folder / 'summary.json').write_text(json.dumps(score, indent=2) + '\n')
        (output / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    best = select_best(scores, args.expected_steps, protocol['candidate_steps'])
    target = output / 'best_adapter'
    if target.exists():
        if not args.resume or any(digest(target / name) != digest(Path(best['adapter']) / name)
            for name in ['adapter_model.safetensors', 'adapter_config.json']):
            raise ValueError('最佳adapter目录已存在且不能确认一致，禁止覆盖')
    else:
        shutil.copytree(best['adapter'], target)
    reload_cuda(target, ROOT / protocol['model_path'])
    report = {**best, 'status': 'all_candidates_evaluated_best_cuda_reload_verified',
        'best_adapter': str(target), 'candidate_count': len(scores), 'test_used': False,
        'candidate_steps': protocol['candidate_steps'],
        'checkpoint_selection_scope': protocol['checkpoint_selection_scope'],
        'data_exposure': protocol['data_exposure'],
        'adapter_sha256': digest(target / 'adapter_model.safetensors'),
        'best_rl_step': min(scores[1:], key=lambda s: (-s['successes'], s['step']))['step'],
        'limitation': 'DEV16单次评估的经验最优；顾客API非严格可复现，不代表TEST或统计显著提升'}
    (output / 'best.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
