"""客服 -> 工具或顾客 -> 客服，多轮真实交互；训练接口留到阶段五。"""

from retail_agent.customer import BudgetExceeded, CustomerError
from retail_agent.environment import RetailSession
from retail_agent.model import ContextLimit, parse_action
from retail_agent.reward import evaluate_task
from retail_agent.observability import StageTimer

STOP_MARKERS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def run_episode(row, agent, customer, max_actions=24, max_errors=3, progress=None):
    timing = StageTimer()
    native_multi = getattr(agent, 'tool_protocol', None) in {
        'mmopd_native_multi_tool_v1', 'mmopd_native_multi_tool_v2'}
    if native_multi:
        from retail_agent.mmopd_protocol import NativeRetailSession, parse_native_action, parse_native_action_v1
        native_parser = parse_native_action if agent.tool_protocol.endswith('_v2') else parse_native_action_v1
    with timing.measure("episode_setup"):
        session = NativeRetailSession(row) if native_multi else RetailSession(row)
    events, consecutive_errors, actions = [], 0, 0
    reason, diagnostic = "max_steps", None
    try:
        with timing.measure("customer_opening"):
            reply = customer.reply(session.agent_messages())
        session.add_user_message(reply["text"])
        events.append({"source": "deepseek_customer", **reply})
        if any(marker in reply["text"] for marker in STOP_MARKERS):
            reason = "completed"
        else:
            for step in range(max_actions):
                with timing.measure("assistant_generation"):
                    generated = agent.generate(session.agent_messages(), session.tool_schemas)
                actions += 1
                events.append({"source": "qwen_agent", "step": step, **generated})
                if progress:
                    progress(row["id"], step, generated["raw_text"])
                if generated["finish_reason"] != "stop":
                    reason = "generation_limit"
                    break
                try:
                    text = (generated.get('raw_model_text', generated['raw_text'])
                            if native_multi and agent.tool_protocol.endswith('_v2') else generated['raw_text'])
                    action = (native_parser if native_multi else parse_action)(text)
                except ValueError as error:
                    reason, diagnostic = "invalid_model_output", str(error)
                    break
                if native_multi:
                    events.append({'source': 'protocol_parse', 'assistant_step': step, 'parsed': action})
                if action["kind"] in {"tool", "tools"}:
                    calls = action['calls'] if action['kind'] == 'tools' else [action]
                    stopped = False
                    segments = action.get('segments') or [{'kind': 'tool', 'call': c} for c in calls]
                    index = 0
                    for segment in segments:
                        if segment['kind'] == 'text':
                            session.add_assistant_message(segment['text'])
                            events.append({'source': 'assistant_text_in_tool_turn', 'assistant_step': step,
                                           'text': segment['text']})
                            continue
                        call = segment['call']
                        with timing.measure("tool_execution"):
                            response = session.call_tool(call["name"], call["arguments"])
                        event = {"source": "retail_tool", "name": call["name"],
                                 "arguments": call["arguments"],
                                 "response": response.model_dump(mode="json")}
                        if native_multi:
                            event.update(assistant_step=step, batch_index=index)
                        events.append(event)
                        index += 1
                        consecutive_errors = consecutive_errors + 1 if response.error else 0
                        stopped = stopped or consecutive_errors >= max_errors
                    if stopped:
                        reason = "too_many_errors"
                        break
                else:
                    consecutive_errors = 0
                    session.add_assistant_message(action["text"])
                    with timing.measure("customer_reply"):
                        reply = customer.reply(session.agent_messages())
                    session.add_user_message(reply["text"])
                    events.append({"source": "deepseek_customer", **reply})
                    if any(marker in reply["text"] for marker in STOP_MARKERS):
                        reason = "completed"
                        break
    except ContextLimit as error:
        reason, diagnostic = "context_limit", str(error)
    except BudgetExceeded as error:
        reason, diagnostic = "budget_limit", str(error)
    except CustomerError as error:
        reason, diagnostic = "customer_service_error", str(error)
    except Exception as error:
        reason, diagnostic = "infrastructure_error", type(error).__name__

    eligible = reason not in {"budget_limit", "customer_service_error", "infrastructure_error"}
    reward = None
    if eligible:
        try:
            with timing.measure("reward_evaluation"):
                reward = evaluate_task(row, session.trajectory, termination=reason).model_dump(mode="json")
        except Exception as error:
            reason, diagnostic, eligible = "grading_error", str(error), False
    return {"task_id": row["id"], "business": row["business"], "termination": reason,
            "diagnostic": diagnostic, "business_evaluation_eligible": eligible, "reward": reward,
            "assistant_actions": actions, "events": events, "messages": session.agent_messages(),
            "trajectory": [m.model_dump(mode="json") for m in session.trajectory],
            "training_ready": False, "timing": timing.summary()}
