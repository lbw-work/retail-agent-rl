"""verl 的异步推理桥接；复用已经验收的 Retail 交互和 TokenTape。"""

import asyncio
import json
import math
from pathlib import Path
import tomllib
import uuid

from retail_agent.environment import ROOT, RetailSession
from retail_agent.grpo_rollout import RolloutError, run_token_episode
from retail_agent.rollout import STOP_MARKERS
from retail_agent.training_reward import audit_scenario_identity, ready_training_rows


def selected_training_rows(data_files, max_samples=-1):
    baseline = tomllib.loads((ROOT / "configs/grpo.toml").read_text())
    approved = {row["id"]: row for row in ready_training_rows(
        ROOT / baseline["train_tasks"], baseline["review_manifest"])}
    paths = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
    rows = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Retail 任务为空或重复")
    if any(approved.get(row["id"]) != row for row in rows):
        raise ValueError("迁移数据必须逐字来自已复核的 TRAIN 池，不能混入 dev/test 或修改目标")
    return rows


class ServerSampler:
    """只在工作线程调用；将同步 episode 的生成请求交给主事件循环。"""

    def __init__(self, client, loop, sampling_params, eos_id, request_id):
        self.client, self.loop = client, loop
        self.params = dict(sampling_params)
        self.eos_id, self.request_id = eos_id, request_id
        self.versions = []

    def generate(self, prompt_ids, max_new_tokens):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self.loop:
            raise RuntimeError("ServerSampler 必须在工作线程执行，避免阻塞事件循环")
        params = dict(self.params, max_tokens=max_new_tokens, logprobs=True,
                      stop_token_ids=[self.eos_id], ignore_eos=False)
        future = asyncio.run_coroutine_threadsafe(self.client.generate(
            request_id=self.request_id, prompt_ids=list(prompt_ids), sampling_params=params), self.loop)
        output = future.result()
        if output.stop_reason == "aborted":
            raise RolloutError("vLLM 生成被中断，禁止使用不完整采样")
        if output.log_probs is None:
            raise RolloutError("vLLM 未返回真实采样概率")
        version = output.extra_fields.get("global_steps")
        if version is None:
            raise RolloutError("采样输出缺少权重版本，无法验收权重同步")
        self.versions.append(int(version))
        return {"token_ids": list(output.token_ids), "logprobs": list(output.log_probs)}


def close_customer(customer):
    if getattr(customer, "http", None) is not None:
        customer.http.close()


async def shared_opening(row, customer_factory, timing):
    if audit_scenario_identity(row)["issues"]:
        raise RolloutError("顾客认证信息与业务归属矛盾：" + row["id"])

    def fetch():
        customer = customer_factory(row["user_scenario"])
        try:
            with timing.measure("customer_opening", task_id=row["id"]):
                reply = customer.reply(RetailSession(row).agent_messages())
            timing.record("customer_api", task_id=row["id"], shared_opening=True, api=reply.get("api"))
            text = reply.get("text")
            if not isinstance(text, str) or not text.strip() or any(m in text for m in STOP_MARKERS):
                raise RolloutError("顾客开场无效或已终止")
            return text
        finally:
            close_customer(customer)

    return await asyncio.to_thread(fetch)


async def run_episode(row, client, tokenizer, customer_factory, initial_text, config,
                      sampling_params, timing, request_id=None):
    if config.get('enable_thinking'):
        from retail_agent.mmopd import ThinkingTokenizer
        tokenizer = ThinkingTokenizer(tokenizer)
    sampler = ServerSampler(client, asyncio.get_running_loop(), sampling_params,
                            tokenizer.convert_tokens_to_ids("<|im_end|>"), request_id or uuid.uuid4().hex)

    def execute():
        customer = customer_factory(row["user_scenario"])
        try:
            episode = run_token_episode
            if config.get('tool_protocol') in {'mmopd_native_multi_tool_v1', 'mmopd_native_multi_tool_v2'}:
                from retail_agent.mmopd_protocol import run_native_token_episode
                episode = run_native_token_episode
            return episode(row, sampler, tokenizer, customer, initial_text, config, timing=timing,
                progress=lambda detail: timing.record("episode_progress", request_id=sampler.request_id,
                    policy_versions=list(sampler.versions), detail=detail))
        finally:
            close_customer(customer)

    result = await asyncio.to_thread(execute)
    if not sampler.versions or len(set(sampler.versions)) != 1:
        raise RolloutError("同步训练的一条轨迹跨越了多个权重版本")
    result["policy_version"] = sampler.versions[0]
    if config.get('enable_thinking'):
        result['enable_thinking'] = True
        result['raw_model_turns'] = [tokenizer.original.decode(
            result['completion_ids'][turn['start']:turn['end']], skip_special_tokens=False)
            for turn in result['turns']]
    return result


async def complete_group(row, opening, run_one, size, expected_version):
    """第一版低并发；所有轨迹通过后才由调用方发布整组。"""
    results = []
    for session_id in range(size):
        result = await run_one(session_id, opening)
        results.append(result)
    if not results or any(r["prompt_ids"] != results[0]["prompt_ids"] for r in results):
        raise RolloutError("同组初始 prompt token 不一致")
    if any(r["policy_version"] != expected_version for r in results):
        raise RolloutError("采样端权重版本与训练步不一致")
    return results


def probability_diagnostics(sampled, recomputed):
    """只统计已采样动作的偏差，不设 0.05 硬停止门槛。"""
    if not sampled or len(sampled) != len(recomputed):
        raise RolloutError("有效采样与重算概率为空或未对齐")
    if any(not math.isfinite(v) or v > 1e-6 for v in sampled + recomputed):
        raise RolloutError("采样或重算概率非有限或大于零")
    deltas = sorted(abs(a - b) for a, b in zip(sampled, recomputed, strict=True))
    return {"retail/logprob_abs_max": deltas[-1],
            "retail/logprob_abs_mean": sum(deltas) / len(deltas),
            "retail/logprob_abs_p95": deltas[min(len(deltas) - 1, int(.95 * len(deltas)))],
            "retail/logprob_over_005": sum(d > .05 for d in deltas),
            "retail/logprob_checked_tokens": len(deltas)}
