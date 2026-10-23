"""第一版实验输入冻结检查；更换条件须另立协议，不能静默复用旧成绩。"""

import hashlib
import json

from retail_agent.environment import ROOT

PROTOCOL = ROOT / "configs/grpo_baseline_protocol.json"


def check_protocol():
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    if protocol["version"] != "grpo_baseline_v1":
        raise ValueError("未验收的实验协议版本")
    for relative, expected in protocol["frozen_sha256"].items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("实验协议冻结输入发生变化：" + relative)
    return {"protocol": protocol, "sha256": hashlib.sha256(raw).hexdigest()}
