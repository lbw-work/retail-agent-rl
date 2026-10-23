"""现成数据解析、非思考消息转换与可复现分组。"""

import json
import re
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: JSON 解析失败") from error


def parse_task(row: dict) -> dict:
    task = dict(row)
    criteria = task["evaluation_criteria"]
    if isinstance(criteria, str):
        criteria = json.loads(criteria)
    if not isinstance(criteria, dict) or not isinstance(criteria.get("actions"), list):
        raise ValueError("任务缺少 evaluation_criteria.actions")
    task["evaluation_criteria"] = criteria
    path = Path(task["db_path"])
    if path.parent.as_posix() != "tau2_rl_database" or not path.name.startswith("tau2_retail_"):
        raise ValueError("Retail 数据库路径不符合预期")
    return task


def extract_policy(system: str) -> str:
    # instructions 中也可能出现字面量 <policy>，读取最后一个真实政策块。
    if "<policy>" not in system or "</policy>" not in system:
        raise ValueError("缺少政策块")
    return system.rsplit("<policy>", 1)[1].split("</policy>", 1)[0].strip()


def make_system_prompt(policy: str) -> str:
    return (
        "You are a retail customer service agent. Follow the policy below. "
        "In each turn, either send a message to the customer or make exactly one tool call; "
        "do not do both. Use the provided tool-call format for tools, and ordinary text "
        "when speaking to the customer.\n<policy>\n" + policy.strip() + "\n</policy>"
    )


def normalize_message(message: dict, tools: dict) -> dict:
    role = message["role"]
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"未知角色 {role}")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串")
    if "<think>" in content or "</think>" in content:
        raise ValueError("可见内容混有思考标签，需要单独审查")
    result = {"role": role, "content": content}
    calls = message.get("tool_calls") or []
    if calls:
        if role != "assistant" or len(calls) != 1 or content.strip():
            raise ValueError("每回合只能发消息或调用一个工具")
        result["tool_calls"] = []
        for call in calls:
            function = call.get("function", call)
            name, arguments = function.get("name"), function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name not in tools or not isinstance(arguments, dict):
                raise ValueError("未知工具或非法参数")
            tool = tools[name]
            if set(arguments) - set(tool.params.model_fields):
                raise ValueError("工具参数包含未定义字段")
            tool.params.model_validate(arguments)
            result["tool_calls"].append({
                "type": "function", "function": {"name": name, "arguments": arguments},
            })
    if role == "tool":
        if message.get("name") not in tools:
            raise ValueError("工具回复缺少已知工具名")
        result["name"] = message["name"]
    return result


def entity_keys(value) -> set[str]:
    text = json.dumps(value, ensure_ascii=False)
    emails = re.findall(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    users = [x for x in re.findall(r"\b[a-z]+_[a-z]+_\d+\b", text)
             if not x.startswith(("credit_card_", "gift_card_", "retail_dialog_", "retail_train_"))]
    orders = re.findall(r"#W\d+", text)
    # 产品和规格可由不同顾客共享，不用它们连接分组。
    return {"email:" + x.lower() for x in emails} | {"user:" + x for x in users} | {
        "order:" + x for x in orders
    }


def normalize_history(messages: list[dict], tools: dict) -> tuple[list[dict], int]:
    """将历史中的并行只读查询按已有返回拆为顺序查询；不拆多写操作。"""
    result, batches, index = [], 0, 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if len(calls) > 1:
            functions = [call.get("function", call) for call in calls]
            if any(not f.get("name", "").startswith(("get_", "find_", "list_", "calculate")) for f in functions):
                raise ValueError("历史包含并行写操作，不能自动改写")
            responses = messages[index + 1:index + 1 + len(calls)]
            if len(responses) != len(calls):
                raise ValueError("并行查询缺少匹配返回")
            for function, response in zip(functions, responses, strict=True):
                if response["role"] != "tool" or response.get("name") != function["name"]:
                    raise ValueError("并行查询返回次序/工具名不匹配")
                result.append(normalize_message({"role": "assistant", "tool_calls": [function]}, tools))
                result.append(normalize_message(response, tools))
            # 原查询前的口头说明不生成新的顾客回合，只保留已有查询及结果。
            batches += 1
            index += len(calls) + 1
        else:
            result.append(normalize_message(message, tools))
            index += 1
    return result, batches


def normalized_reason(text: str) -> str:
    text = re.sub(r"#w\d+|\b\d+\b|[\w.+-]+@[\w.-]+", "<entity>", text.lower())
    return " ".join(re.findall(r"[a-z]+|<entity>", text))


class Groups:
    """共享身份、订单、场景或词面模板的样本进入同一连通分组。"""

    def __init__(self):
        self.parents = {}
        self.key_owner = {}

    def find(self, node):
        self.parents.setdefault(node, node)
        if self.parents[node] != node:
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            smaller, larger = sorted((first, second))
            self.parents[larger] = smaller

    def add(self, node, keys):
        self.find(node)
        for key in sorted(keys):
            if key in self.key_owner:
                self.union(node, self.key_owner[key])
            else:
                self.key_owner[key] = node
