"""原生多工具协议：完整解析、顺序执行、整批反馈；评分器仍重放真实执行事件。"""

from copy import deepcopy
import json
import re

from retail_agent.environment import RetailSession
from retail_agent.grpo_rollout import RolloutError, TokenTape
from retail_agent.model import ContextLimit, parse_action
from retail_agent.observability import StageTimer
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import evaluate_training_task

LEGACY_PROTOCOL = 'mmopd_native_multi_tool_v1'
PROTOCOL = 'mmopd_native_multi_tool_v2'


class NativeRetailSession(RetailSession):
    def agent_messages(self):
        messages = super().agent_messages()
        old = 'make exactly one tool call;'
        if old not in messages[0]['content']:
            raise RolloutError('原系统提示已变化，须重新审查多工具协议')
        messages[0]['content'] = messages[0]['content'].replace(
            old, 'make one or more tool calls, executed sequentially in the order given; ', 1)
        messages[0]['content'] = messages[0]['content'].replace(
            'You should at most make one tool call at a time,',
            'You may request one or more tool calls in a turn, executed sequentially,', 1)
        return messages


def parse_native_action_v1(text):
    """每个调用沿用原严格JSON解析；整批检查完才允许执行第一个调用。"""
    text = text.strip()
    if '<tool_call>' not in text and '</tool_call>' not in text:
        return parse_action(text)
    calls, end = [], 0
    for match in re.finditer(r'<tool_call>.*?</tool_call>', text, flags=re.S):
        if text[end:match.start()].strip():
            raise ValueError('工具批次不能混入顾客文字、思考或未闭合标签')
        calls.append(parse_action(match.group()))
        end = match.end()
    if not calls or text[end:].strip():
        raise ValueError('工具批次包含不完整调用或混合内容')
    return {'kind': 'tools', 'calls': calls}


def parse_native_action(text):
    """只恢复格式边界；JSON、调用顺序和参数均取自原文，修复写入审计记录。"""
    segments, repairs = [], []

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('调用JSON含重复键，拒绝歧义参数')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('调用JSON包含非标准数值：' + value)

    decoder = json.JSONDecoder(object_pairs_hook=unique_keys, parse_constant=invalid_constant)
    tags = ['<think>', '</think>', '<tool_call>', '</tool_call>']
    position = 0
    while position < len(text):
        if text.startswith('<think>', position):
            start, depth = position, 1
            position += len('<think>')
            # 模型说明记录的原生空思考特例；含任何思考正文时仍要求闭合。
            if text[position:].lstrip().startswith('<tool_call>'):
                repairs.append({'kind': 'empty_think_before_tool', 'start': start, 'end': position})
                continue
            while depth:
                match = re.search(r'</?think>', text[position:])
                if match is None:
                    raise ValueError('思考块未闭合，禁止从其中提取调用')
                token = match.group()
                position += match.end()
                depth += 1 if token == '<think>' else -1
            repairs.append({'kind': 'separate_reasoning', 'start': start, 'end': position})
        elif text.startswith('</think>', position):
            repairs.append({'kind': 'extra_think_close', 'start': position,
                            'end': position + len('</think>')})
            position += len('</think>')
        elif text.startswith('<tool_call>', position):
            start = position
            position += len('<tool_call>')
            while position < len(text) and text[position].isspace():
                position += 1
            value, end = decoder.raw_decode(text, position)
            if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
                raise ValueError('工具调用必须是仅包含name和arguments的完整JSON对象')
            if not isinstance(value['name'], str) or not isinstance(value['arguments'], dict):
                raise ValueError('工具名称或参数类型错误')
            call = {'kind': 'tool', **value}
            after = end
            while after < len(text) and text[after].isspace():
                after += 1
            if text.startswith('</tool_call>', after):
                position = after + len('</tool_call>')
            else:
                repairs.append({'kind': 'missing_tool_close', 'start': end, 'end': end})
                position = end
                if text.startswith('﹅', after):
                    repairs.append({'kind': 'tool_close_glyph', 'start': after, 'end': after + 1})
                    position = after + 1
            segments.append({'kind': 'tool', 'call': call, 'start': start, 'end': position})
        elif text.startswith('</tool_call>', position):
            raise ValueError('出现无法归属的工具闭合标签')
        else:
            end = min([i for tag in tags if (i := text.find(tag, position)) >= 0] or [len(text)])
            content = text[position:end].strip()
            if content:
                if any(token in content for token in ['<tool', '</tool', '<|im_', '<|endoftext|>']):
                    raise ValueError('输出含损坏标签或非法角色分隔符')
                if content.startswith(('{', '[')):
                    raise ValueError('标签外出现JSON，不能猜测是否遗漏了调用')
                segments.append({'kind': 'text', 'text': content, 'start': position, 'end': end})
            position = end
    calls = [s['call'] for s in segments if s['kind'] == 'tool']
    if not calls:
        action = parse_action('\n\n'.join(s['text'] for s in segments))
        return {**action, 'segments': segments, 'repairs': repairs}
    if any(s['kind'] == 'text' for s in segments):
        repairs.append({'kind': 'split_mixed_text_and_tools', 'start': 0, 'end': len(text)})
    return {'kind': 'tools', 'calls': calls, 'segments': segments, 'repairs': repairs}


def execute_batch(session, calls):
    """保留业务错误返回并继续下一调用；内部程序异常向上传播，不伪造返回。"""
    for index, call in enumerate(calls):
        response = session.call_tool(call['name'], call['arguments'])
        yield {'batch_index': index, 'call': deepcopy(call),
                       'message': session.agent_messages()[-1],
                       'response': response.model_dump(mode='json')}


class NativeTokenTape(TokenTape):
    def feedback_batch(self, messages, *, continue_generation):
        if not messages or any(m['role'] != 'tool' for m in messages):
            raise RolloutError('批反馈只接受完整工具消息列表')
        text = '\n' + self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=continue_generation, enable_thinking=True)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(self.ids) + len(ids) > self.max_length:
            raise ContextLimit('完整工具批反馈超出上下文；不裁剪或丢弃返回')
        self.ids.extend(ids)
        self.mask.extend([0] * len(ids))
        self.logprobs.extend([0.] * len(ids))


class NativeHistory:
    """按原生模板重建推理上下文；原始采样另外保存，不能冒充连续训练轨迹。"""

    def __init__(self, tokenizer, messages, tools, *, include_reasoning=False):
        self.tokenizer = getattr(tokenizer, 'original', tokenizer)
        self.messages, self.tools = deepcopy(messages), tools
        self.include_reasoning = include_reasoning

    def prompt_ids(self):
        return self.tokenizer.apply_chat_template(self.messages, tools=self.tools,
            tokenize=True, add_generation_prompt=True, enable_thinking=True)

    def assistant(self, raw, parsed):
        reasoning = []
        for repair in parsed.get('repairs', []):
            if repair['kind'] == 'separate_reasoning':
                reasoning.append(re.sub(r'</?think>', '', raw[repair['start']:repair['end']]).strip())
        segments = parsed.get('segments') or [{'kind': 'text', 'text': parsed['text']}]
        # 显式 reasoning_content 避免原生模板按最后一个 </think> 截掉前面的工具调用。
        # 官方 tau2 历史不回传 reasoning_content；思考原文保留在独立采样记录。
        message = {'role': 'assistant', 'reasoning_content':
                   '\n'.join(reasoning) if self.include_reasoning else '', 'content': ''}
        if all(s['kind'] == 'tool' for s in segments):
            message['tool_calls'] = [{'type': 'function', 'function': {
                'name': s['call']['name'], 'arguments': deepcopy(s['call']['arguments'])}}
                for s in segments]
        else:
            message['content'] = '\n'.join(s['text'] if s['kind'] == 'text' else
                '<tool_call>\n' + json.dumps({k: s['call'][k] for k in ['name', 'arguments']},
                    ensure_ascii=False) + '\n</tool_call>' for s in segments)
        self.messages.append(message)

    def feedback(self, messages):
        self.messages.extend(deepcopy(messages))


def run_native_token_episode(row, sampler, tokenizer, customer, initial_text, config,
                             progress=None, timing=None):
    timing = timing if timing is not None else StageTimer()
    session = NativeRetailSession(row)
    session.add_user_message(initial_text)
    tape = NativeTokenTape(tokenizer, session.agent_messages(), session.tool_schemas,
                           config['max_trajectory_tokens'])
    native_history = config.get('history_mode', 'append') == 'native'
    if native_history and not config.get('native_history_evaluation_only'):
        raise RolloutError('原生重建历史仅用于推理诊断；训练须先适配逐轮真实上下文')
    history = NativeHistory(tokenizer, session.agent_messages(), session.tool_schemas) if native_history else None
    generation_turns = []
    reason, diagnostic, actions, errors = 'max_steps', None, 0, 0
    batches = []
    parsed_actions = []
    protocol = config.get('tool_protocol', PROTOCOL)
    parser = parse_native_action_v1 if protocol == LEGACY_PROTOCOL else parse_native_action

    def emit(event, **fields):
        if progress is not None:
            progress({'event': event, 'action': actions, **fields})

    emit('episode_started', prompt_ids=list(tape.prompt_ids), tool_protocol=protocol)
    try:
        for _ in range(config['max_assistant_actions']):
            prompt_ids = history.prompt_ids() if history else list(tape.ids)
            remaining = min(tape.max_length - len(tape.ids), tape.max_length - len(prompt_ids))
            if remaining < 1:
                raise ContextLimit('没有下一回合生成空间')
            emit('assistant_prompt', prompt_ids=prompt_ids, history_mode='native' if history else 'append')
            with timing.measure('assistant_generation', gpu=True, task_id=row['id']):
                sampled = sampler.generate(prompt_ids,
                    min(config['max_new_tokens_per_action'], remaining))
            generation_turns.append({'prompt_ids': prompt_ids, **deepcopy(sampled)})
            tape.sampled(sampled)
            actions += 1
            emit('assistant_sampled', **sampled)
            if sampled['token_ids'][-1] != tape.eos_id:
                reason = 'generation_limit'
                break
            try:
                if tokenizer.convert_tokens_to_ids('<|im_start|>') in sampled['token_ids']:
                    raise ValueError('客服采样包含非法角色分隔符')
                decoder = tokenizer if protocol == LEGACY_PROTOCOL else getattr(tokenizer, 'original', tokenizer)
                raw = decoder.decode(sampled['token_ids'][:-1], skip_special_tokens=False)
                action = parser(raw)
            except ValueError as error:
                reason, diagnostic = 'invalid_model_output', str(error)
                break
            parsed_actions.append({'action': actions, **deepcopy(action)})
            emit('assistant_parsed', parsed=action)
            if history:
                history.assistant(raw, action)
            if action['kind'] == 'tools':
                batch = {'action': actions, 'calls': deepcopy(action['calls']),
                         'segments': deepcopy(action.get('segments', [])),
                         'repairs': deepcopy(action.get('repairs', [])), 'execution': []}
                batches.append(batch)
                stopped = False
                with timing.measure('tool_execution', task_id=row['id']):
                    segments = action.get('segments') or [{'kind': 'tool', 'call': c} for c in action['calls']]
                    for segment in segments:
                        if segment['kind'] == 'text':
                            session.add_assistant_message(segment['text'])
                            emit('assistant_text_in_tool_turn', text=segment['text'])
                            continue
                        event = next(execute_batch(session, [segment['call']]))
                        event['batch_index'] = len(batch['execution'])
                        batch['execution'].append(event)
                        emit('tool_feedback', **event)
                        errors = errors + 1 if event['response']['error'] else 0
                        stopped = stopped or errors >= config['max_consecutive_tool_errors']
                emit('tool_batch_completed', batch=batch)
                if history:
                    history.feedback([e['message'] for e in batch['execution']])
                tape.feedback_batch([e['message'] for e in batch['execution']],
                    continue_generation=not stopped and actions < config['max_assistant_actions'])
                if stopped:
                    reason = 'too_many_errors'
                    break
            else:
                errors = 0
                session.add_assistant_message(action['text'])
                emit('customer_request_pending', assistant_text=action['text'])
                with timing.measure('customer_reply', task_id=row['id']):
                    reply = customer.reply(session.agent_messages())
                timing.record('customer_api', task_id=row['id'], api=reply.get('api'))
                if not isinstance(reply.get('text'), str) or not reply['text'].strip():
                    raise RolloutError('顾客返回为空或类型错误')
                session.add_user_message(reply['text'])
                emit('customer_feedback', message=session.agent_messages()[-1], api_record=reply.get('api'))
                if history:
                    history.feedback([session.agent_messages()[-1]])
                stopped = any(marker in reply['text'] for marker in STOP_MARKERS)
                tape.feedback(session.agent_messages()[-1],
                    continue_generation=not stopped and actions < config['max_assistant_actions'])
                if stopped:
                    reason = 'completed'
                    break
    except ContextLimit as error:
        reason, diagnostic = 'context_limit', str(error)
    with timing.measure('reward_evaluation', task_id=row['id']):
        score = evaluate_training_task(row, session.trajectory, reason,
                                       review_manifest=config.get('review_manifest'))
    return {**tape.payload(), 'task_id': row['id'], 'termination': reason,
        'diagnostic': diagnostic, 'assistant_actions': actions, 'scores': score,
        'trajectory': [m.model_dump(mode='json') for m in session.trajectory],
        'tool_batches': batches, 'tool_protocol': protocol, 'parsed_actions': parsed_actions,
        'scoring_transcript': 'actual_sequential_single_call_and_response_pairs',
        'history_mode': 'native' if history else 'append',
        'flat_training_ready': not native_history,
        'generation_turns': generation_turns,
        'native_messages': history.messages if history else None,
        'timing_run_id': timing.run_id}
