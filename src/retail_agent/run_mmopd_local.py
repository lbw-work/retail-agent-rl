"""MLX新模型四条真实Retail轨迹诊断；复用训练采样循环，不更新参数。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import tomllib
import traceback

from retail_agent.customer import DeepSeekCustomer, GOAL_POLICY
from retail_agent.environment import ROOT, RetailSession
from retail_agent.experiment_protocol import check_protocol
from retail_agent.grpo_rollout import RolloutError
from retail_agent.mmopd import PROFILE, ThinkingTokenizer, digest, profile, verify_snapshot
from retail_agent.mmopd_protocol import PROTOCOL, run_native_token_episode
from retail_agent.observability import StageTimer
from retail_agent.rollout import STOP_MARKERS
from retail_agent.train_verl import customer_costs
from retail_agent.training_reward import audit_scenario_identity, ready_training_rows


class MLXSampler:
    def __init__(self, model, tokenizer, temperature):
        self.model, self.tokenizer, self.temperature = model, tokenizer, temperature

    def generate(self, prompt_ids, max_new_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        ids, logps = [], []
        for response in stream_generate(self.model, self.tokenizer, list(prompt_ids),
                max_tokens=max_new_tokens, prefill_step_size=256,
                sampler=make_sampler(temp=self.temperature, top_p=1.0, top_k=0)):
            # MLX最后一个response包含实际EOS及其概率；不补造结束符。
            ids.append(int(response.token))
            logps.append(float(response.logprobs[response.token].item()))
        return {'token_ids': ids, 'logprobs': logps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', default='retail_174')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--opening-from', type=Path, help='复用此前真实开场，保持协议对照条件一致')
    parser.add_argument('--history-mode', choices=['append', 'native'], default='native')
    args = parser.parse_args()
    parent = check_protocol()
    config = {**tomllib.loads((ROOT / 'configs/grpo.toml').read_text()), **profile()}
    config['tool_protocol'] = PROTOCOL
    config.update(history_mode=args.history_mode, native_history_evaluation_only=True)
    config['history_reasoning_policy'] = 'official_tau2_content_calls_only' if args.history_mode == 'native' else 'raw_append'
    rows = ready_training_rows(ROOT / config['train_tasks'], config['review_manifest'])
    row = next((r for r in rows if r['id'] == args.task_id), None)
    if row is None or audit_scenario_identity(row)['issues']:
        parser.error('必须是通过身份复核的冻结TRAIN任务')
    assets = verify_snapshot()
    quant = ROOT / (config['model_path'] + '-mlx4bit')
    conversion = json.loads((quant / 'CONVERSION_MANIFEST.json').read_text())
    if conversion['source_manifest_sha256'] != assets['model_manifest_sha256']:
        raise ValueError('本地量化版不属于当前固定SFT')
    for name, expected in conversion['files'].items():
        if Path(name).name != name or digest(quant / name) != expected['sha256']:
            raise ValueError('本地量化文件校验失败：' + name)
    output = (args.output_dir or ROOT / 'outputs/mmopd_local' /
              datetime.now().strftime('group_%Y%m%d_%H%M%S')).resolve()
    if not output.is_relative_to(ROOT / 'outputs/mmopd_local'):
        parser.error('输出必须在outputs/mmopd_local内')
    output.mkdir(parents=True, exist_ok=False)
    report = {**assets, 'status': 'running', 'task_id': row['id'], 'backend': 'mlx_4bit',
        'quantization': conversion['quantization'], 'conversion_sha256': digest(quant / 'CONVERSION_MANIFEST.json'),
        'parent_protocol_sha256': parent['sha256'], 'profile_sha256': digest(PROFILE),
        'entry_sha256': digest(Path(__file__)), 'shared_opening': True,
        'tool_protocol': PROTOCOL,
        'history_mode': args.history_mode, 'flat_training_ready': args.history_mode == 'append',
        'protocol_code_sha256': digest(ROOT / 'src/retail_agent/mmopd_protocol.py'),
        'scorer_sha256': {n: digest(ROOT / 'src/retail_agent' / n)
                          for n in ['training_reward.py', 'reward.py']},
        'group_size': 4, 'seeds': [config['seed'] + i for i in range(4)],
        'training_started': False, 'optimizer_calls': 0, 'test_used': False,
        'config': config, 'data_exposure': config['data_exposure']}
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    started = time.perf_counter()
    timing = StageTimer(output)
    results = []
    ledger = output / 'customer_costs.jsonl'
    project = tomllib.loads((ROOT / 'configs/project.toml').read_text())
    customer_config = {**project['customer'], 'goal_policy': GOAL_POLICY}
    report['customer'] = customer_config
    try:
        import mlx.core as mx
        from mlx_lm import load

        model, tokenizer = load(str(quant))
        sampler = MLXSampler(model, tokenizer, config['temperature'])
        if args.opening_from:
            original = args.opening_from.resolve()
            previous = json.loads((original.parent / 'run.json').read_text())
            if previous['task_id'] != row['id'] or previous['model_id'] != assets['model_id']:
                raise ValueError('复用开场必须来自同任务、同模型的真实记录')
            opening_reply = json.loads(original.read_text())
            report.update(opening_source=str(original), opening_source_sha256=digest(original))
        else:
            opening_customer = DeepSeekCustomer(row['user_scenario'], customer_config,
                                                limit_cny=None, ledger_path=ledger)
            try:
                opening_reply = opening_customer.reply(RetailSession(row).agent_messages())
            finally:
                opening_customer.http.close()
        (output / 'opening.json').write_text(json.dumps(opening_reply, ensure_ascii=False, indent=2) + '\n')
        opening = opening_reply['text']
        if not opening.strip() or any(marker in opening for marker in STOP_MARKERS):
            raise RolloutError('共享开场无效，停止整组')
        print('OUTPUT:', output, '\nOPENING:', opening, flush=True)
        for index in range(4):
            mx.random.seed(config['seed'] + index)
            folder = output / f'trajectory_{index}'
            folder.mkdir()
            customer = DeepSeekCustomer(row['user_scenario'], customer_config,
                                        limit_cny=None, ledger_path=ledger)

            def progress(event):
                entry = dict(event)
                if entry['event'] == 'assistant_sampled':
                    entry['raw_model_text'] = tokenizer.decode(entry['token_ids'], skip_special_tokens=False)
                    print(f"trajectory={index} action={entry['action']} assistant: "
                          + ThinkingTokenizer(tokenizer).decode(entry['token_ids'][:-1],
                              skip_special_tokens=False)[:350], flush=True)
                elif entry['event'] in {'tool_feedback', 'customer_feedback'}:
                    print(f"trajectory={index} {entry['event']}: "
                          + json.dumps(entry['message'], ensure_ascii=False)[:450], flush=True)
                with (folder / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + '\n')

            try:
                result = run_native_token_episode(row, sampler, ThinkingTokenizer(tokenizer), customer,
                                           opening, config, progress=progress, timing=timing)
            finally:
                customer.http.close()
            result.update(session_id=index, seed=config['seed'] + index)
            (folder / 'trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            results.append(result)
            print('SCORES:', index, result['termination'], result['scores'], flush=True)
        if any(r['prompt_ids'] != results[0]['prompt_ids'] for r in results):
            raise RolloutError('同组初始prompt不一致')
        timing.observe_group(results, config['reward_mode'], task_businesses=row['businesses'])
        summary = {'task_id': row['id'], 'trajectories': [{
            'session_id': r['session_id'], 'seed': r['seed'], 'termination': r['termination'],
            'diagnostic': r['diagnostic'], 'scores': r['scores'],
            'assistant_actions': r['assistant_actions'],
            'assistant_tokens': sum(r['assistant_mask']),
            'total_tokens': len(r['prompt_ids']) + len(r['completion_ids']),
            'tool_calls': sum(m['role'] == 'tool' for m in r['trajectory']),
            'tool_errors': sum(m['role'] == 'tool' and bool(m.get('error')) for m in r['trajectory']),
        } for r in results], 'reward_a': [r['scores']['reward_a'] for r in results],
            'all_zero': all(r['scores']['reward_a'] == 0 for r in results),
            'effective_group': len({r['scores']['reward_a'] for r in results}) > 1,
            'peak_memory_gb': mx.get_peak_memory() / 1e9,
            'limitation': 'One TRAIN task, MLX4bit native thinking; not CUDA BF16 or an improvement estimate'}
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        report['status'] = 'four_real_trajectories_scored'
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        report.update(status='failed_preserved_partial_evidence', error_type=type(error).__name__,
                      completed_trajectories=len(results))
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        report.update(customer_costs(ledger), wall_seconds=time.perf_counter() - started)
        (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
