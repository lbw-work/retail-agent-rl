"""社区 GRPO 检查点在原 Retail 环境中的四轨迹真实评测，不进入训练。"""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import tomllib
import traceback

from retail_agent.community_protocol import PROTOCOL, run_episode
from retail_agent.customer import DeepSeekCustomer, GOAL_POLICY
from retail_agent.environment import ROOT, RetailSession
from retail_agent.experiment_protocol import check_protocol
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import (
    audit_scenario_identity, compile_business_goals, evaluate_training_task,
    ready_training_rows, reference_subset,
)

PROFILE = ROOT / 'configs/community_eval.toml'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify_model(config):
    model = ROOT / config['model_path']
    manifest_path = model / 'CONVERSION_MANIFEST.json'
    if digest(manifest_path) != config['conversion_manifest_sha256']:
        raise ValueError('量化来源清单不匹配固定社区检查点')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['source_model'], manifest['source_revision']) != (
            config['model_id'], config['model_revision']):
        raise ValueError('模型身份不匹配')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(model / name) != expected:
            raise ValueError('量化文件校验失败：' + name)
    return model


class MLXSampler:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    def generate(self, prompt_ids, max_new_tokens, *, temperature=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        start = time.perf_counter()
        temp = self.config['temperature'] if temperature is None else temperature
        ids, selected_logps, finish = [], [], 'length'
        for response in stream_generate(self.model, self.tokenizer, list(prompt_ids),
                max_tokens=max_new_tokens, prefill_step_size=256,
                sampler=make_sampler(temp=temp, top_p=self.config['top_p'], top_k=self.config['top_k']),
                logits_processors=make_logits_processors(
                    repetition_penalty=self.config['repetition_penalty'])):
            ids.append(int(response.token))
            selected_logps.append(float(response.logprobs[response.token].item()))
            if '</tool_call>' in self.tokenizer.decode(ids):
                finish = 'stop_string'
                break
            if response.finish_reason == 'stop':
                finish = 'eos'
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        end = raw.find('</tool_call>')
        text = raw[:end + len('</tool_call>')] if end >= 0 else self.tokenizer.decode(
            ids, skip_special_tokens=True)
        return {'text': text, 'raw_model_text': raw, 'token_ids': ids,
                'sampled_token_logprobs': selected_logps, 'temperature': temp,
                'finish_reason': finish, 'seconds': time.perf_counter() - start}


def progress_writer(folder):
    def progress(event):
        with (folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        if event['event'] == 'assistant_sampled':
            print(folder.name, 'MODEL:', event['raw_model_text'][:500], flush=True)
        elif event['event'] in {'tool_feedback', 'customer_feedback'}:
            print(folder.name, event['event'], json.dumps(event['message'], ensure_ascii=False)[:600], flush=True)
    return progress


def recording_customer(row, config, ledger, progress):
    customer = DeepSeekCustomer(row['user_scenario'], config, limit_cny=50., ledger_path=ledger)

    def record_response(response, *args, **kwargs):
        if response.request.method == 'POST' and response.request.url.endswith('/chat/completions'):
            # 仅保存 JSON 请求体与返回，不保存含 Authorization 的 HTTP 头。
            try:
                body = response.json()
            except ValueError:
                body = response.text
            progress({'event': 'customer_http_response', 'status_code': response.status_code,
                      'request': json.loads(response.request.body), 'response': body})
        return response

    customer.http.hooks['response'].append(record_response)
    return customer


def write_transcript(folder, result=None):
    text = []
    if (folder / 'events.jsonl').exists():
        for line in (folder / 'events.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e['event'] == 'assistant_sampled':
                text.append('模型原始输出：\n' + e['raw_model_text'])
            elif e['event'] == 'tool_feedback':
                text.append('工具调用与唯一ID：\n' + json.dumps(e['call'], ensure_ascii=False)
                            + '\n' + json.dumps(e['response'], ensure_ascii=False, indent=2))
            elif e['event'] == 'customer_feedback':
                text.append('顾客回复：\n' + e['message']['content'])
    if result:
        text.insert(0, '顾客开场：\n' + result['native_messages'][1]['content'])
        text.append('最终评分：\n' + json.dumps(result['scores'], ensure_ascii=False, indent=2))
    (folder / '完整对话与工具调用.txt').write_text('\n\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', default='retail_174')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    parent = check_protocol()
    config = tomllib.loads(PROFILE.read_text())
    baseline = tomllib.loads((ROOT / 'configs/grpo.toml').read_text())
    config['review_manifest'] = baseline['review_manifest']
    row = next(r for r in ready_training_rows(ROOT / baseline['train_tasks'], baseline['review_manifest'])
               if r['id'] == args.task_id)
    if audit_scenario_identity(row)['issues']:
        raise ValueError('任务身份复核失败')
    model_path = verify_model(config)
    output = (args.output_dir or ROOT / 'outputs/community_local' /
              datetime.now().strftime('group_%Y%m%d_%H%M%S')).resolve()
    if not output.is_relative_to(ROOT / 'outputs/community_local'):
        parser.error('输出必须在 outputs/community_local 内')
    output.mkdir(parents=True, exist_ok=False)
    frozen = ['src/retail_agent/environment.py', 'src/retail_agent/reward.py',
              'src/retail_agent/training_reward.py', 'src/retail_agent/customer.py',
              'configs/grpo.toml', 'configs/grpo_baseline_protocol.json', baseline['review_manifest'],
              baseline['train_tasks'], 'third_party/tau2-bench/src/tau2/domains/retail/tools.py']
    report = {'status': 'preflight', 'config': config, 'task_id': row['id'], 'backend': 'mlx_4bit',
              'tool_protocol': PROTOCOL, 'parent_protocol_sha256': parent['sha256'],
              'frozen_sha256': {n: digest(ROOT / n) for n in frozen},
              'new_code_sha256': {n: digest(ROOT / n) for n in [str(PROFILE.relative_to(ROOT)),
                  'src/retail_agent/community_protocol.py', 'src/retail_agent/run_community_local.py']},
              'dependencies': {p: importlib.metadata.version(p) for p in ['mlx-lm', 'mlx', 'transformers', 'pydantic']},
              'group_size': 4, 'seeds': [config['seed'] + i for i in range(4)], 'shared_opening': True,
              'training_started': False, 'optimizer_calls': 0, 'training_ready': False,
              'data_exposure': 'One previously inspected TRAIN task; not a held-out benchmark',
              'completed_trajectories': 0}
    save(output / 'run.json', report)
    save(output / 'task.json', row)
    started, results = time.perf_counter(), []
    ledger = output / 'customer_costs.jsonl'
    try:
        spec = compile_business_goals(row)
        positive = evaluate_training_task(row, reference_subset(row, set(spec['goals'])).trajectory,
                                         review_manifest=config['review_manifest'])
        negative = evaluate_training_task(row, [], review_manifest=config['review_manifest'])
        if positive['reward_a'] != 1. or negative['reward_a'] != 0.:
            raise ValueError('原评分器正反对照失败')
        save(output / 'score_preflight.json', {'positive': positive, 'negative': negative})
        if args.preflight_only:
            report['status'] = 'offline_preflight_passed_no_api_no_inference'
            return
        from dotenv import dotenv_values
        import mlx.core as mx
        from mlx_lm import load

        # 每次启动从本机 .env 读取，避免继承旧密钥；不输出密钥。
        key = dotenv_values(ROOT / '.env').get('DEEPSEEK_API_KEY')
        if not key:
            raise ValueError('请在原项目 .env 中设置顾客密钥')
        os.environ['DEEPSEEK_API_KEY'] = key
        customer_config = {**tomllib.loads((ROOT / 'configs/project.toml').read_text())['customer'],
                           'goal_policy': GOAL_POLICY}
        report.update(status='running', customer=customer_config)
        save(output / 'run.json', report)
        model, tokenizer = load(str(model_path))
        sampler = MLXSampler(model, tokenizer, config)
        customer = recording_customer(row, customer_config, ledger, progress_writer(output))
        try:
            opening = customer.reply(RetailSession(row).agent_messages())
        finally:
            customer.http.close()
        save(output / 'opening.json', opening)
        if any(marker in opening['text'] for marker in STOP_MARKERS):
            raise ValueError('顾客开场已终止，停止整组而非伪造业务测试')
        for index, seed in enumerate(report['seeds']):
            mx.random.seed(seed)
            folder = output / f'trajectory_{index}'
            folder.mkdir()
            customer = recording_customer(row, customer_config, ledger, progress_writer(folder))
            result = None
            try:
                result = run_episode(row, sampler, tokenizer, customer, opening['text'], config,
                                     progress_writer(folder))
                result.update(session_id=index, seed=seed)
                save(folder / 'trace.json', {k: v for k, v in result.items() if k != 'final_db'})
                save(folder / 'final_db.json', result['final_db'])
                results.append(result)
                print('SCORE:', index, result['termination'], result['scores'], flush=True)
            finally:
                customer.http.close()
                write_transcript(folder, result)
            report['completed_trajectories'] = len(results)
            save(output / 'run.json', report)
        if len({r['initial_db_hash'] for r in results}) != 1:
            raise ValueError('组内初始数据库不一致')
        first_prompts = []
        for index in range(4):
            events = [json.loads(line) for line in (output / f'trajectory_{index}/events.jsonl').read_text().splitlines()]
            first_prompts.append(next(e['prompt_ids'] for e in events if e['event'] == 'assistant_prompt'))
        if any(p != first_prompts[0] for p in first_prompts):
            raise ValueError('同组初始客服 prompt 不一致')
        summary = {'task_id': row['id'], 'reward_a': [r['scores']['reward_a'] for r in results],
                   'trajectories': [{k: r[k] for k in ['session_id', 'seed', 'termination', 'diagnostic',
                       'assistant_actions', 'generations', 'format_retries', 'scores']} for r in results],
                   'peak_memory_gb': mx.get_peak_memory() / 1e9,
                   'shared_initial_prompt_verified': True, 'initial_database_isolation_verified': True,
                   'training_started': False, 'optimizer_calls': 0}
        save(output / 'summary.json', summary)
        report['status'] = 'four_real_trajectories_scored'
    except BaseException as error:
        report.update(status='failed_partial_evidence_preserved', error_type=type(error).__name__)
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['completed_trajectories'] = len(results)
        report['frozen_sources_unchanged'] = all(digest(ROOT / n) == h
                                                for n, h in report['frozen_sha256'].items())
        save(output / 'run.json', report)


if __name__ == '__main__':
    main()
