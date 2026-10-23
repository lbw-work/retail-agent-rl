"""社区单动作协议，仅评测；业务状态、调用 ID 和评分由原 Retail 代码负责。

提示词与动作格式取自 Apache-2.0 的 jbarnes850/Tau2-RL-Pipeline，
固定版本 c0d434b453719e149aac70452325f6e04ac086cf 的 prompting.py/actions.py。
"""

from copy import deepcopy
import json

from retail_agent.environment import RetailSession
from retail_agent.model import ContextLimit
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import evaluate_training_task

PROTOCOL = 'community_single_action_eval_v1'
FORMAT_ERROR = ('FORMAT ERROR. Re-output EXACTLY in the required <tool_call> format: '
                '<tool_call>{"name": "...", "arguments": {...}}</tool_call>. One action only.')


def done() -> str:
    """Call this function when you are done with the task."""
    return '###STOP###'


def build_system_prompt(policy, tools):
    from tau2.environment.tool import as_tool

    schemas = []
    for tool in [*tools, as_tool(done).openai_schema]:
        fn = tool.get('function', tool)
        schemas.append({'name': fn.get('name', 'unknown'),
                        'description': fn.get('description', ''),
                        'parameters': fn.get('parameters', {'type': 'object', 'properties': {}})})
    tools_text = json.dumps(schemas, indent=2)
    return f"""## Output Format
Every turn: exactly ONE tool call in this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Special actions:
- respond: <tool_call>{{"name": "respond", "arguments": {{"content": "message"}}}}</tool_call>
- done: <tool_call>{{"name": "done", "arguments": {{}}}}</tool_call>

---

You are a retail customer support agent. Complete the user's task following the policy below.

{policy}

## Available Tools
{tools_text}

## Rules
- One tool call per turn (no plain text responses)
- Authenticate user before state changes
- Confirm before modifications
- Communicate all relevant details to the user
- Call done immediately when task is complete
"""


def parse_action(text):
    text = text.strip()
    start, end = '<tool_call>', '</tool_call>'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('必须是一个完整的工具调用')
    if not text.startswith(start) or not text.endswith(end):
        raise ValueError('工具标签外不能有文字或思考')
    value = json.loads(text[len(start):-len(end)].strip())
    name, args = value.get('name'), value.get('arguments') or {}
    if not isinstance(name, str) or not name:
        raise ValueError('工具名不能为空')
    if isinstance(args, str):
        args = json.loads(args) if args.strip() else {}
    if not isinstance(args, dict):
        raise ValueError('工具参数必须是对象')
    return {'name': name, 'arguments': args}


def functional_call(action):
    args = ', '.join(f'{key}={value!r}' for key, value in action['arguments'].items())
    return f"{action['name']}({args})"


def run_episode(row, sampler, tokenizer, customer, opening, config, progress=None):
    """模型侧保留社区文本历史；评分侧只保存原 RetailSession 的真实执行消息。"""
    session = RetailSession(row)
    initial_hash = session.db_hash()
    session.add_user_message(opening)
    messages = [{'role': 'system', 'content': build_system_prompt(
        session._environment.get_policy(), session.tool_schemas)},
        {'role': 'user', 'content': opening}]
    reason, diagnostic, errors = 'max_steps', None, 0
    generation_count, actions, retries, parsed = 0, 0, 0, []

    def emit(event, **fields):
        if progress:
            progress({'event': event, 'action': actions, **deepcopy(fields)})

    def sample(temperature=None):
        nonlocal generation_count
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        remaining = config['max_trajectory_tokens'] - len(ids)
        if remaining < 1:
            raise ContextLimit('完整历史超出上下文，未截断消息')
        emit('assistant_prompt', prompt=prompt, prompt_ids=ids,
             temperature=config['temperature'] if temperature is None else temperature)
        output = sampler.generate(ids, min(config['max_new_tokens_per_action'], remaining),
                                  temperature=temperature)
        generation_count += 1
        emit('assistant_sampled', **output)
        return output['text'].strip()

    emit('episode_started', initial_db_hash=initial_hash, tool_protocol=PROTOCOL)
    try:
        for _ in range(config['max_assistant_actions']):
            actions += 1
            text = sample()
            try:
                action = parse_action(text)
            except (ValueError, TypeError, AttributeError) as error:
                retries += 1
                emit('format_retry', error=str(error))
                messages.extend([{'role': 'assistant', 'content': text},
                                 {'role': 'user', 'content': FORMAT_ERROR}])
                text = sample(temperature=0.0)
                try:
                    action = parse_action(text)
                except (ValueError, TypeError, AttributeError) as failure:
                    reason, diagnostic = 'invalid_model_output', str(failure)
                    break
            parsed.append(action)
            messages.append({'role': 'assistant', 'content': text})
            emit('assistant_parsed', parsed=action)
            name, args = action['name'], action['arguments']
            if name == 'respond':
                content = args.get('content')
                if not isinstance(content, str) or not content.strip():
                    reason, diagnostic = 'invalid_model_output', 'respond 缺少非空 content'
                    break
                errors = 0
                session.add_assistant_message(content)
                emit('customer_request_pending', assistant_text=content)
                reply = customer.reply(session.agent_messages())
                session.add_user_message(reply['text'])
                messages.append({'role': 'user', 'content': reply['text']})
                emit('customer_feedback', message=session.agent_messages()[-1], api=reply.get('api'))
                if any(marker in reply['text'] for marker in STOP_MARKERS):
                    reason = 'completed'
                    break
            elif name == 'done':
                # 控制动作沿用社区 STOP 语义，不伪装成 Retail 业务工具。
                session.add_assistant_message(done())
                emit('agent_done', arguments=args)
                reason = 'completed'
                break
            else:
                response = session.call_tool(name, args)
                emit('tool_feedback', call=action, message=session.agent_messages()[-1],
                     response=response.model_dump(mode='json'))
                messages.append({'role': 'user', 'content':
                    f'Tool result for {functional_call(action)}:\n{response.content}'})
                errors = errors + 1 if response.error else 0
                if errors >= config['max_consecutive_tool_errors']:
                    reason = 'too_many_errors'
                    break
    except ContextLimit as error:
        reason, diagnostic = 'context_limit', str(error)
    except BaseException as error:
        emit('episode_failed', error_type=type(error).__name__,
             trajectory=[m.model_dump(mode='json') for m in session.trajectory],
             final_db=session._environment.tools.db.model_dump(mode='json'))
        raise
    scores = evaluate_training_task(row, session.trajectory, reason,
                                    review_manifest=config['review_manifest'])
    return {'task_id': row['id'], 'termination': reason, 'diagnostic': diagnostic,
            'scores': scores, 'assistant_actions': actions, 'generations': generation_count,
            'format_retries': retries,
            'trajectory': [m.model_dump(mode='json') for m in session.trajectory],
            'native_messages': messages, 'parsed_actions': parsed,
            'initial_db_hash': initial_hash, 'final_db_hash': session.db_hash(),
            'final_db': session._environment.tools.db.model_dump(mode='json'),
            'training_ready': False, 'tool_protocol': PROTOCOL}
