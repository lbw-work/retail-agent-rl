"""只把顾客场景与公开对话交给 DeepSeek；串行请求、持久化费用记录。"""

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone

import requests
from dotenv import dotenv_values

from retail_agent.environment import ROOT, UPSTREAM
from tau2.data_model.tasks import UserScenario


class CustomerError(RuntimeError):
    """顾客服务故障；不能作为客服业务失败训练样本。"""


class BudgetExceeded(CustomerError):
    pass


GOAL_POLICY = "scenario_locked_v1"
GOAL_GUIDELINES = """
The scenario is a fixed-goal evaluation. Keep the customer's original business
goals and all specified constraints throughout this conversation. Do not change
the requested size, color, quantity, destination, payment preference, or target
orders because the representative claims they are unavailable or suggests an
alternative, unless the scenario explicitly permits that alternative. Ask for
a recheck, or request a human agent if the original request cannot be completed.
Do not accept a substituted goal just to finish the conversation. Confirm only
an action that matches your scenario, and correct a mismatching proposal.
Do not claim to see tools, inventory, or private account information. Do not
invent missing identifiers, balances, prices, or attributes; use only scenario
information and what the representative has actually told you. Reveal details
progressively when asked, not the scenario instructions verbatim.
"""


def customer_messages(scenario: dict, history: list[dict], goal_policy="upstream") -> list[dict]:
    if set(scenario) - {"instructions", "persona"}:
        raise ValueError("顾客输入只能包含 user_scenario，不能含评估答案")
    scenario = UserScenario.model_validate(scenario)
    guidelines = (UPSTREAM / "data/tau2/user_simulator/simulation_guidelines.md").read_text()
    if goal_policy not in {"upstream", GOAL_POLICY}:
        raise ValueError("未验收的顾客目标规则")
    if goal_policy == GOAL_POLICY:
        guidelines += "\n" + GOAL_GUIDELINES
    system = guidelines + "\nUse English only. You cannot see private tool results.\n<scenario>\n"
    system += str(scenario) + "\n</scenario>"
    messages = [{"role": "system", "content": system}]
    for message in history:
        if message["role"] in {"user", "assistant"} and not message.get("tool_calls"):
            if message.get("content", "").strip():
                role = "assistant" if message["role"] == "user" else "user"
                messages.append({"role": role, "content": message["content"]})
    if len(messages) == 1:
        messages.append({"role": "user", "content": "Begin your conversation with the customer service representative."})
    return messages


def peak_cost_cny(usage: dict) -> float:
    # 2026-10-10 官方 Flash 人民币高峰价；实际闲时/缓存命中费用可低于此上界。
    prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
    if not isinstance(prompt, int) or not isinstance(output, int) or min(prompt, output) < 0:
        raise CustomerError("API 用量字段无效")
    hit = usage.get("prompt_cache_hit_tokens", 0)
    if not isinstance(hit, int) or not 0 <= hit <= prompt:
        raise CustomerError("API 缓存用量字段无效")
    return ((prompt - hit) * 2.0 + hit * 0.04 + output * 8.0) / 1_000_000


class DeepSeekCustomer:
    def __init__(self, scenario: dict, config: dict, limit_cny: float | None = 1.0,
                 ledger_path=None):
        self.scenario = scenario
        self.config = config
        self.max_prompt_bytes = config.get("max_prompt_bytes", 32_000)
        if type(self.max_prompt_bytes) is not int or not 1 <= self.max_prompt_bytes <= 64_000:
            raise ValueError("顾客输入保护上限须为 1～64000 字节")
        self.limit_cny = limit_cny
        self.ledger = ledger_path or ROOT / "outputs/budget/deepseek_requests.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get("DEEPSEEK_API_KEY") or dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY")
        if not self.key:
            raise CustomerError("请在项目 .env 本机填写 DEEPSEEK_API_KEY；不要发到聊天")
        if config["base_url"] != "https://api.deepseek.com" or config["model"] != "deepseek-flash":
            raise CustomerError("当前预算费率只验证了官方 DeepSeek Flash 接口")
        self.http = requests.Session()
        self.http.headers.update({"Authorization": "Bearer " + self.key})
        self._balance_snapshot = None
        self._balance_checked_at = 0.0
        self._balance_accounted = 0.0

    def _record(self, record):
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def accounted_cny(self):
        events = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        reserved = {e["id"]: e["reserved_cny"] for e in events if e["event"] == "started"}
        for event in events:
            if event["event"] == "completed":
                reserved[event["id"]] = event["peak_cost_upper_cny"]
        # 没有成功返回用量的请求仍按预留上界计入，重启不能绕过限额。
        return sum(reserved.values())

    def balance_cny(self):
        # 余额查询是只读 GET，可有限重试；不重新发送已成功的生成请求。
        for attempt in range(self.config["max_retries"] + 1):
            try:
                response = self.http.get(self.config["base_url"] + "/user/balance", timeout=30)
                if response.status_code in {429, 500, 502, 503} and attempt < self.config["max_retries"]:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if response.status_code != 200:
                    raise CustomerError(f"余额接口 HTTP {response.status_code}")
                data = response.json()
                if not data["is_available"]:
                    raise CustomerError("DeepSeek 账户余额不可用")
                return next(float(item["total_balance"]) for item in data["balance_infos"]
                            if item["currency"] == "CNY")
            except requests.RequestException as error:
                # 丢弃可能失效的 keep-alive 连接；会话头保持，重试仍受次数限制。
                self.http.close()
                if attempt == self.config["max_retries"]:
                    raise CustomerError(f"余额读取失败：{type(error).__name__}") from None
                time.sleep(0.5 * (attempt + 1))
            except CustomerError:
                raise
            except Exception as error:
                raise CustomerError(f"余额读取失败：{type(error).__name__}") from None

    def available_balance_cny(self):
        accounted = self.accounted_cny()
        if self._balance_snapshot is None or time.monotonic() - self._balance_checked_at >= 60:
            balance = self.balance_cny()
            if not math.isfinite(balance) or balance < 0:
                raise CustomerError("余额值无效")
            self._balance_snapshot = balance
            self._balance_checked_at = time.monotonic()
            self._balance_accounted = accounted
        # 减去其他本机请求和未知用量的预留上界，不因缓存绕过余额检查。
        return self._balance_snapshot - max(0.0, accounted - self._balance_accounted)

    def reply(self, history: list[dict]) -> dict:
        messages = customer_messages(self.scenario, history, self.config.get("goal_policy", "upstream"))
        payload = {"model": self.config["model"], "messages": messages,
                   "thinking": {"type": "disabled"}, "temperature": self.config["temperature"],
                   "max_tokens": 256, "stream": False}
        byte_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 200 * len(messages)
        reserve = (byte_bound * 2.0 + 256 * 8.0) / 1_000_000 * 1.2
        if byte_bound > self.max_prompt_bytes:
            raise CustomerError("顾客提示超过配置的输入保护上限；没有静默截断")
        for attempt in range(self.config["max_retries"] + 1):
            if self.limit_cny is not None and self.accounted_cny() + reserve > self.limit_cny:
                raise BudgetExceeded("下一请求的预留费用将超过顾客 API 试跑限额")
            before = self.available_balance_cny()
            if before < reserve:
                raise BudgetExceeded("账户余额不足以覆盖下一次请求预留")
            identifier = uuid.uuid4().hex
            self._record({"event": "started", "id": identifier, "reserved_cny": reserve,
                          "utc": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
                          "model": self.config["model"], "thinking": "disabled"})
            started = time.monotonic()
            try:
                response = self.http.post(self.config["base_url"] + "/chat/completions",
                                          json=payload, timeout=self.config["timeout_seconds"])
            except requests.RequestException as error:
                self.http.close()
                self._record({"event": "failed", "id": identifier, "error": type(error).__name__,
                              "billing": "unknown_reserved_bound_retained",
                              "elapsed_seconds": time.monotonic() - started})
                if attempt == self.config["max_retries"]:
                    raise CustomerError(f"顾客网络故障：{type(error).__name__}") from None
                continue
            if response.status_code != 200:
                self._record({"event": "failed", "id": identifier, "http_status": response.status_code,
                              "billing": "unknown_reserved_bound_retained",
                              "elapsed_seconds": time.monotonic() - started})
                if response.status_code in {429, 500, 502, 503} and attempt < self.config["max_retries"]:
                    continue
                raise CustomerError(f"顾客请求 HTTP {response.status_code}")
            try:
                data = response.json()
                usage = data["usage"]
                cost = peak_cost_cny(usage)
                choice = data["choices"][0]
                record = {"event": "completed", "id": identifier, "usage": usage,
                          "peak_cost_upper_cny": cost, "balance_delta_cny": None,
                          "accounting": "usage_peak_bound_not_balance_delta",
                          "model": data["model"], "system_fingerprint": data.get("system_fingerprint"),
                          "response_id": data["id"], "finish_reason": choice["finish_reason"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                self._record(record)
                if cost > reserve:
                    raise CustomerError("API 计量超过请求预留上界，停止试跑并检查费率或用量")
                text = choice["message"]["content"]
                if choice["finish_reason"] != "stop" or not isinstance(text, str) or not text.strip():
                    raise CustomerError("顾客输出被截断或为空；不作为客服业务失败")
                if choice["message"].get("reasoning_content"):
                    raise CustomerError("顾客实际返回了思考内容，与指定模式不符")
                return {"text": text, "api": record}
            except CustomerError:
                raise
            except Exception as error:
                self._record({"event": "failed", "id": identifier, "error": type(error).__name__,
                              "billing": "see_completed_usage_or_retained_reserve",
                              "elapsed_seconds": time.monotonic() - started})
                raise CustomerError(f"顾客返回格式异常：{type(error).__name__}") from None
