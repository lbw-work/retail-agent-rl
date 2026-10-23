"""追加式真实 token 轨迹；外部反馈有注意力但没有策略损失。"""

import math

from retail_agent.environment import RetailSession
from retail_agent.model import ContextLimit, parse_action
from retail_agent.observability import StageTimer
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import audit_scenario_identity, evaluate_training_task


class RolloutError(RuntimeError):
    """中止整个任务组；不能补造 token、概率或故障零分。"""


class TokenTape:
    def __init__(self, tokenizer, messages, tools, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length
        # MLX 的 TokenizerWrapper 代理 get_vocab，但 Python 的 len 不会被代理。
        self.vocab_size = len(tokenizer.get_vocab())
        self.prompt_ids = tokenizer.apply_chat_template(messages, tools=tools,
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not self.prompt_ids or len(self.prompt_ids) >= max_length:
            raise ContextLimit("初始提示没有生成空间，禁止截断")
        self.ids = list(self.prompt_ids)
        self.mask, self.logprobs, self.turns = [], [], []
        self.eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def sampled(self, result):
        ids, logps = result["token_ids"], result["logprobs"]
        if not ids or len(ids) != len(logps):
            raise RolloutError("采样 token 与概率长度错误或为空")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids):
            raise RolloutError("采样 token ID 无效")
        if any(not math.isfinite(p) or p > 1e-6 for p in logps):
            raise RolloutError("采样概率非有限或大于零")
        if self.eos_id in ids[:-1]:
            raise RolloutError("采样在结束符后仍有 token")
        if len(self.ids) + len(ids) > self.max_length:
            raise RolloutError("采样超出完整轨迹上限")
        start = len(self.mask)
        self.ids.extend(ids)
        self.mask.extend([1] * len(ids))
        self.logprobs.extend(logps)
        self.turns.append({"start": start, "end": len(self.mask)})

    def feedback(self, message, *, continue_generation):
        if message["role"] not in {"user", "tool"}:
            raise RolloutError("只能追加顾客或工具反馈")
        # 只编码新增片段；绝不规范化或重编码已经采样的客服历史。
        text = "\n" + self.tokenizer.apply_chat_template([message], tokenize=False,
            add_generation_prompt=continue_generation, enable_thinking=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(self.ids) + len(ids) > self.max_length:
            raise ContextLimit("外部反馈超出完整轨迹上限，禁止截断")
        self.ids.extend(ids)
        self.mask.extend([0] * len(ids))
        self.logprobs.extend([0.0] * len(ids))

    def payload(self):
        completion = self.ids[len(self.prompt_ids):]
        if not completion or not any(self.mask):
            raise RolloutError("轨迹没有客服采样 token，不能进入 GRPO")
        if len(completion) != len(self.mask) or len(completion) != len(self.logprobs):
            raise RolloutError("轨迹 token、概率与损失掩码不一致")
        return {"prompt_ids": list(self.prompt_ids), "completion_ids": completion,
                "assistant_mask": list(self.mask), "logprobs": list(self.logprobs),
                "turns": list(self.turns)}


class TorchSampler:
    """无词表裁剪的 temperature softmax，与固定 TRL 的更新概率相同。"""

    def __init__(self, model, tokenizer, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("训练采样温度必须为有限正数")
        self.model, self.tokenizer, self.temperature = model, tokenizer, temperature

    def generate(self, prompt_ids, max_new_tokens):
        import torch
        from transformers import GenerationConfig

        if max_new_tokens < 1:
            raise ContextLimit("轨迹没有生成空间")
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
        config = GenerationConfig(do_sample=True, temperature=self.temperature,
            top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0,
            max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                result = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    generation_config=config, return_dict_in_generate=True, output_scores=True,
                    # 禁止模型默认 top_p 回填，保持与 TRL 的完整词表 softmax 一致。
                    use_model_defaults=False, disable_compile=True)
        finally:
            self.model.train(was_training)
        if not torch.equal(result.sequences[:, :inputs.shape[1]], inputs):
            raise RolloutError("生成器改写了采样提示 token")
        ids = result.sequences[0, inputs.shape[1]:].tolist()
        if len(ids) != len(result.scores):
            raise RolloutError("生成器未返回每个真实采样 token 的概率")
        logps = [float(torch.log_softmax(score[0].float(), -1)[token].item())
                 for token, score in zip(ids, result.scores, strict=True)]
        return {"token_ids": ids, "logprobs": logps}


def run_token_episode(row, sampler, tokenizer, customer, initial_text, config, progress=None, timing=None):
    timing = timing if timing is not None else StageTimer()
    with timing.measure("episode_setup"):
        session = RetailSession(row)
        session.add_user_message(initial_text)
        tape = TokenTape(tokenizer, session.agent_messages(), session.tool_schemas,
                         config["max_trajectory_tokens"])
    reason, errors, actions, diagnostic = "max_steps", 0, 0, None
    def emit(event, **fields):
        if progress is not None:
            progress({"event": event, "action": actions, **fields})
    emit("episode_started", prompt_ids=list(tape.prompt_ids))
    try:
        for _ in range(config["max_assistant_actions"]):
            remaining = tape.max_length - len(tape.ids)
            if remaining < 1:
                raise ContextLimit("没有下一回合生成空间")
            with timing.measure("assistant_generation", gpu=True, task_id=row["id"]):
                result = sampler.generate(list(tape.ids), min(config["max_new_tokens_per_action"], remaining))
            tape.sampled(result)
            actions += 1
            emit("assistant_sampled", **result)
            if result["token_ids"][-1] != tape.eos_id:
                reason = "generation_limit"
                break
            text = tokenizer.decode(result["token_ids"][:-1], skip_special_tokens=False)
            try:
                if tokenizer.convert_tokens_to_ids("<|im_start|>") in result["token_ids"]:
                    raise ValueError("客服采样包含非法角色分隔符")
                action = parse_action(text)
            except ValueError as error:
                reason, diagnostic = "invalid_model_output", str(error)
                break
            if action["kind"] == "tool":
                with timing.measure("tool_execution", task_id=row["id"]):
                    response = session.call_tool(action["name"], action["arguments"])
                emit("tool_feedback", message=session.agent_messages()[-1])
                errors = errors + 1 if response.error else 0
                if errors >= config["max_consecutive_tool_errors"]:
                    reason = "too_many_errors"
                    tape.feedback(session.agent_messages()[-1], continue_generation=False)
                    break
                tape.feedback(session.agent_messages()[-1],
                              continue_generation=actions < config["max_assistant_actions"])
            else:
                errors = 0
                session.add_assistant_message(action["text"])
                emit("customer_request_pending", assistant_text=action["text"])
                with timing.measure("customer_reply", task_id=row["id"]):
                    reply = customer.reply(session.agent_messages())
                timing.record("customer_api", task_id=row["id"], api=reply.get("api"))
                if not isinstance(reply.get("text"), str) or not reply["text"].strip():
                    raise RolloutError("顾客返回为空或类型错误")
                session.add_user_message(reply["text"])
                emit("customer_feedback", message=session.agent_messages()[-1],
                     api_record=reply.get("api"))
                stopped = any(marker in reply["text"] for marker in STOP_MARKERS)
                tape.feedback(session.agent_messages()[-1],
                              continue_generation=not stopped and actions < config["max_assistant_actions"])
                if stopped:
                    reason = "completed"
                    break
    except ContextLimit as error:
        reason, diagnostic = "context_limit", str(error)
    # 服务/预算/工具内部/评分异常直接传播，整组中止；不变成业务失败。
    with timing.measure("reward_evaluation", task_id=row["id"]):
        score = evaluate_training_task(row, session.trajectory, reason,
                                      review_manifest=config.get('review_manifest'))
    return {**tape.payload(), "task_id": row["id"], "termination": reason,
            "diagnostic": diagnostic, "assistant_actions": actions, "scores": score,
            "trajectory": [m.model_dump(mode="json") for m in session.trajectory],
            "timing_run_id": timing.run_id}


def validate_task_groups(task_ids, group_size):
    if group_size < 2 or not task_ids or len(task_ids) % group_size:
        raise RolloutError("批次必须包含完整任务组")
    for start in range(0, len(task_ids), group_size):
        if len(set(task_ids[start:start + group_size])) != 1:
            raise RolloutError("任务组混入了不同任务，不能计算组内优势")


def generate_groups(task_ids, rows, sampler, tokenizer, customer_factory, config, timing=None):
    timing = timing if timing is not None else StageTimer()
    size = config["num_generations"]
    validate_task_groups(task_ids, size)
    for task_id in set(task_ids):
        with timing.measure("task_identity_validation", task_id=task_id):
            if audit_scenario_identity(rows[task_id])["issues"]:
                raise RolloutError("顾客认证信息与业务归属矛盾：" + task_id)
    results = []
    for start in range(0, len(task_ids), size):
        row = rows[task_ids[start]]
        opening_customer = customer_factory(row["user_scenario"])
        with timing.measure("customer_opening", task_id=row["id"]):
            opening_reply = opening_customer.reply(RetailSession(row).agent_messages())
            opening = opening_reply["text"]
        timing.record("customer_api", task_id=row["id"], shared_opening=True, api=opening_reply.get("api"))
        if not isinstance(opening, str) or not opening.strip() or any(m in opening for m in STOP_MARKERS):
            raise RolloutError("顾客开场无效或已终止，没有可比较的客服采样")
        # 同组首条顾客消息固定；每次尝试重新建立独立数据库与顾客实例。
        for _ in range(size):
            with timing.measure("episode", task_id=row["id"]):
                results.append(run_token_episode(row, sampler, tokenizer,
                    customer_factory(row["user_scenario"]), opening, config, timing=timing))
        timing.observe_group(results[-size:], config["reward_mode"], task_businesses=row["businesses"])
        prompts = [r["prompt_ids"] for r in results[-size:]]
        if any(p != prompts[0] for p in prompts):
            raise RolloutError("同一任务组初始提示 token 不一致")
    return results
