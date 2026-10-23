"""固定版本 Retail 环境；业务逻辑来自上游，每条轨迹独立持有数据库。"""

import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from retail_agent.data import make_system_prompt, parse_task

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/tau2-bench"
if not (UPSTREAM / "src/tau2").is_dir():
    raise FileNotFoundError("缺少固定版本 tau2 源码，请先运行 scripts/download_data.py")
sys.path.insert(0, str(UPSTREAM / "src"))

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.environment.environment import Environment


@lru_cache(maxsize=4)
def _snapshot(path: Path) -> RetailDB:
    # 缓存只作为初始模板；绝不将这个对象交给工具执行。
    return RetailDB.model_validate_json(path.read_text(encoding="utf-8"))


class RetailEnvironment(Environment):
    """补充调用边界验证；不改写订单、退款和换货逻辑。"""

    def __init__(self, path: Path):
        super().__init__(domain_name="retail",
                         policy=(UPSTREAM / "data/tau2/domains/retail/policy.md").read_text(),
                         tools=RetailTools(_snapshot(path).model_copy(deep=True)))
        self.tool_defs = self.tools.get_tools()
        self.tool_errors = []

    def make_tool_call(self, tool_name: str, requestor="assistant", **kwargs):
        try:
            if requestor != "assistant":
                raise ValueError("Retail 顾客不能调用客服工具")
            if tool_name not in self.tool_defs:
                raise ValueError(f"Unknown tool: {tool_name}")
            params = self.tool_defs[tool_name].params
            if set(kwargs) - set(params.model_fields):
                raise ValueError("工具参数包含未定义字段")
            params.model_validate(kwargs, strict=True)
            return super().make_tool_call(tool_name, requestor=requestor, **kwargs)
        except Exception as error:
            # 上游金标准评估会捕获异常；保留诊断，评分端据此拒绝坏参考答案。
            self.tool_errors.append({"tool": tool_name, "error": str(error),
                                     "type": type(error).__name__})
            raise

    def get_response(self, message: ToolCall) -> ToolMessage:
        error = False
        try:
            response = self.make_tool_call(message.name, message.requestor, **message.arguments)
            self.sync_tools()
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as failure:
            response, error = f"Error: {failure}", True
        # 内部程序错误继续抛出，不能冒充模型的业务失败。
        return ToolMessage(id=message.id, role="tool", requestor=message.requestor,
                           content=self.to_json_str(response), error=error)


def create_environment(row: dict, solo_mode=False) -> RetailEnvironment:
    if solo_mode:
        raise ValueError("第一版只支持与顾客对话的 Retail 模式")
    task = parse_task(row)
    return RetailEnvironment(ROOT / "data/databases" / Path(task["db_path"]).name)


class RetailSession:
    """一次交互的状态与消息；agent_messages 仅暴露公开规则和实际对话。"""

    def __init__(self, row: dict):
        self._row = deepcopy(parse_task(row))
        self._task = Task.model_validate(self._row)
        self.reset()

    def reset(self):
        self._environment = create_environment(self._row)
        initial = self._task.initial_state
        self._history = deepcopy(initial.message_history or []) if initial else []
        self._environment.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=self._history,
        )

    @property
    def trajectory(self):
        return deepcopy(self._history)

    @property
    def tool_schemas(self):
        return [tool.openai_schema for tool in self._environment.tool_defs.values()]

    def db_hash(self):
        return self._environment.get_db_hash()

    def add_user_message(self, text: str):
        self._history.append(UserMessage(role="user", content=text))

    def add_assistant_message(self, text: str):
        self._history.append(AssistantMessage(role="assistant", content=text))

    def call_tool(self, name: str, arguments: dict) -> ToolMessage:
        call = ToolCall(id=f"call_{len(self._history)}", name=name,
                        arguments=deepcopy(arguments))
        response = self._environment.get_response(call)
        self._history.extend([AssistantMessage(role="assistant", tool_calls=[call]), response])
        return response.model_copy(deep=True)

    def agent_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": make_system_prompt(self._environment.get_policy())}]
        calls = {}
        for message in self._history:
            item = {"role": message.role, "content": message.content or ""}
            if isinstance(message, AssistantMessage) and message.tool_calls:
                item["tool_calls"] = []
                for call in message.tool_calls:
                    calls[call.id] = call.name
                    item["tool_calls"].append({"id": call.id, "type": "function",
                        "function": {"name": call.name, "arguments": deepcopy(call.arguments)}})
            elif isinstance(message, ToolMessage):
                item.update(name=calls[message.id], tool_call_id=message.id)
            messages.append(item)
        return messages
