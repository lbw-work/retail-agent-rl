"""复用失败轨迹的相同真实对话，对比两种历史；不调用顾客或执行工具。"""

import argparse
import json
from pathlib import Path

from retail_agent.environment import ROOT
from retail_agent.mmopd import ThinkingTokenizer, digest
from retail_agent.mmopd_protocol import NativeHistory, NativeRetailSession, NativeTokenTape, parse_native_action
from retail_agent.run_mmopd_local import MLXSampler


def saved_prefix(folder, action_number, tokenizer, row, config, *, include_reasoning=False):
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    session = NativeRetailSession(row)
    opening = json.loads((folder.parent / 'opening.json').read_text())['text']
    session.add_user_message(opening)
    tape = NativeTokenTape(ThinkingTokenizer(tokenizer), session.agent_messages(), session.tool_schemas,
                           config['max_trajectory_tokens'])
    history = NativeHistory(tokenizer, session.agent_messages(), session.tool_schemas,
                            include_reasoning=include_reasoning)
    raw, calls, successful = '', [], []
    for event in events:
        if event['event'] == 'assistant_sampled':
            if event['action'] == action_number:
                trace = json.loads((folder / 'trace.json').read_text())
                boundary = trace['turns'][action_number - 1]['start']
                if tape.ids != trace['prompt_ids'] + trace['completion_ids'][:boundary]:
                    raise ValueError('追加式输入与原始采样记录不一致，不能作对照')
                return {'append': list(tape.ids), 'native': history.prompt_ids()}, history.messages, successful
            tape.sampled(event)
            raw = tokenizer.decode(event['token_ids'][:-1], skip_special_tokens=False)
        elif event['event'] == 'assistant_parsed':
            history.assistant(raw, event['parsed'])
        elif event['event'] == 'tool_feedback':
            calls.append(event['message'])
            if not event['response']['error']:
                successful.append(event['call'])
        elif event['event'] == 'tool_batch_completed':
            tape.feedback_batch(calls, continue_generation=True)
            history.feedback(calls)
            calls = []
        elif event['event'] == 'customer_feedback':
            tape.feedback(event['message'], continue_generation=True)
            history.feedback([event['message']])
    raise ValueError('记录中没有指定回合')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='复用已完成的追加式及保留思考对照，避免重复推理')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.source / 'run.json').read_text())
    config = report['config']
    row = next(json.loads(line) for line in (ROOT / config['train_tasks']).read_text().splitlines()
               if json.loads(line)['id'] == report['task_id'])
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(str(ROOT / (config['model_path'] + '-mlx4bit')))
    sampler = MLXSampler(model, tokenizer, config['temperature'])
    results = {'source_sha256': digest(args.source / 'run.json'), 'customer_calls': 0,
               'tool_executions': 0, 'max_new_tokens': 2048, 'cases': [],
               'native_reasoning_policy': 'official_tau2_content_and_calls_only'}
    for trajectory, action in [(0, 3), (3, 6), (3, 8)]:
        prompts, messages, successful = saved_prefix(args.source / f'trajectory_{trajectory}', action,
                                                     tokenizer, row, config)
        case = {'trajectory': trajectory, 'action': action, 'seed': 42 + trajectory,
                'native_messages': messages, 'outputs': {}}
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            if baseline['source_sha256'] != results['source_sha256']:
                raise ValueError('对照来源不一致')
            old = next(c for c in baseline['cases'] if c['trajectory'] == trajectory and c['action'] == action)
            if old['outputs']['append']['prompt_ids'] != prompts['append']:
                raise ValueError('追加式对照上下文不一致')
            case['outputs'].update(append=old['outputs']['append'],
                                   native_keep_reasoning=old['outputs']['native'])
            del prompts['append']
            results['baseline_sha256'] = digest(args.baseline)
            if prompts['native'] == old['outputs']['native']['prompt_ids']:
                case['outputs']['native'] = {**old['outputs']['native'],
                    'reused_identical_prompt_seed_and_budget': True}
                del prompts['native']
        for mode, ids in prompts.items():
            mx.random.seed(case['seed'])
            sampled = sampler.generate(ids, 2048)
            raw = tokenizer.decode(sampled['token_ids'][:-1], skip_special_tokens=False)
            output = {'prompt_ids': ids, 'prompt_text': tokenizer.decode(ids, skip_special_tokens=False),
                      **sampled, 'raw_model_text': tokenizer.decode(sampled['token_ids'], skip_special_tokens=False),
                      'eos': sampled['token_ids'][-1] == tokenizer.convert_tokens_to_ids('<|im_end|>')}
            try:
                if not output['eos']:
                    raise ValueError('generation_limit: 本对照预算内未结束，不作为完整动作')
                parsed = parse_native_action(raw)
                output['parsed'] = parsed
                output['repeated_successful_calls'] = [call for call in parsed.get('calls', [])
                    if any(call['name'] == old['name'] and call['arguments'] == old['arguments']
                           for old in successful)]
            except ValueError as error:
                output['parse_error'] = str(error)
            case['outputs'][mode] = output
            print(json.dumps({'trajectory': trajectory, 'action': action, 'mode': mode,
                'prompt_tokens': len(ids), 'eos': output['eos'], 'parse_error': output.get('parse_error'),
                'raw_tail': raw[-600:]}, ensure_ascii=False), flush=True)
        results['cases'].append(case)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
