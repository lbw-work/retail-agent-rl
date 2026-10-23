"""MMOPD原生思考的一条真实Retail轨迹；不启动Ray或优化器。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import tomllib
import traceback

from retail_agent.environment import ROOT
from retail_agent.mmopd import NativeAgent, PROFILE, digest, profile, verify_snapshot
from retail_agent.training_reward import ready_training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--task-id', default='retail_174')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = profile()
    from retail_agent.mmopd_protocol import PROTOCOL
    config['tool_protocol'] = PROTOCOL
    model_report = verify_snapshot()
    old = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
    project = tomllib.loads((ROOT / 'configs/project.toml').read_text())
    rows = ready_training_rows(ROOT / old['train_tasks'], old['review_manifest'])
    selected = next((r for r in rows if r['id'] == args.task_id), None)
    if selected is None:
        parser.error('单用例必须来自冻结TRAIN池')
    report = {**model_report, 'status': 'preflight_passed', 'task_id': args.task_id,
        'profile_sha256': digest(PROFILE), 'config': config, 'training_started': False,
        'optimizer_calls': 0, 'api_calls': 0,
        'tool_protocol': PROTOCOL,
        'tool_protocol_code_sha256': digest(ROOT / 'src/retail_agent/mmopd_protocol.py'),
        'review_manifest_sha256': digest(ROOT / old['review_manifest']),
        'code_sha256': {name: digest(ROOT / 'src/retail_agent' / name) for name in
            ['run_mmopd_case.py', 'mmopd.py', 'rollout.py', 'training_reward.py', 'reward.py']}}
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output = (args.output_dir or ROOT / 'outputs/mmopd_case' /
              datetime.now().strftime('case_%Y%m%d_%H%M%S')).resolve()
    if not output.is_relative_to(ROOT / 'outputs/mmopd_case'):
        parser.error('输出必须在outputs/mmopd_case内')
    import torch
    from pydantic import TypeAdapter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from retail_agent.customer import DeepSeekCustomer, GOAL_POLICY
    from retail_agent.rollout import run_episode
    from retail_agent.train_verl import customer_costs, gpu_monitor
    from retail_agent.training_reward import evaluate_training_task
    from tau2.data_model.message import Message

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('单用例要求单张可见CUDA GPU')
    output.mkdir(parents=True, exist_ok=False)
    report.update(status='running', utc=datetime.now(timezone.utc).isoformat(),
        gpu=torch.cuda.get_device_name(0), dependencies={k: version(k) for k in ['torch', 'transformers']},
        customer={**project['customer'], 'goal_policy': GOAL_POLICY})
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    started = time.perf_counter()
    customer = None
    try:
        with gpu_monitor(output):
            torch.manual_seed(config['seed'])
            tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model_path'], local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(ROOT / config['model_path'],
                torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').to('cuda')
            model.eval()
            agent = NativeAgent(model, tokenizer, config, output / 'assistant_turns.jsonl')
            customer = DeepSeekCustomer(selected['user_scenario'], report['customer'],
                limit_cny=None, ledger_path=output / 'customer_costs.jsonl')
            def progress(task, step, text):
                print(f'{task} #{step + 1}: {text[:240]}', flush=True)
            result = run_episode(selected, agent, customer, config['max_assistant_actions'],
                                 config['max_consecutive_tool_errors'], progress=progress)
            (output / 'trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            if not result['business_evaluation_eligible']:
                raise RuntimeError('用例基础设施或评分失败：' + str(result['diagnostic']))
            messages = TypeAdapter(list[Message]).validate_python(result['trajectory'])
            score = evaluate_training_task(selected, messages, result['termination'],
                                          review_manifest=old['review_manifest'])
            turns = [e for e in result['events'] if e['source'] == 'qwen_agent']
            tools = [e for e in result['events'] if e['source'] == 'retail_tool']
            summary = {'task_id': args.task_id, 'termination': result['termination'],
                'diagnostic': result['diagnostic'], 'scores': score,
                'assistant_actions': result['assistant_actions'], 'tool_calls': len(tools),
                'tool_errors': sum(bool(e['response'].get('error')) for e in tools),
                'generated_tokens': sum(len(e['generated_token_ids']) for e in turns),
                'turns_with_thinking': sum(e['raw_model_text'].lstrip().startswith('<think>') for e in turns),
                'training_started': False, 'test_used': False,
                'note': '一个TRAIN用例的可运行性/业务结果，不代表总体效果提升'}
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            report['status'] = 'case_completed_business_result_recorded'
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if customer is not None:
            customer.http.close()
        report.update(customer_costs(output / 'customer_costs.jsonl'))
        report['wall_seconds'] = time.perf_counter() - started
        (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
