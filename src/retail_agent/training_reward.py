"""05A 离线训练奖励候选；不替换固定评测器，不调用模型或 API。"""

import hashlib
import json
from functools import lru_cache
from itertools import combinations

from retail_agent.data import parse_task
from retail_agent.dataset import WRITES
from retail_agent.environment import ROOT, RetailSession
from retail_agent.reward import RewardEvaluationError, evaluate_task
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

LEGACY_VERSION = "business_subset_v1"
VERSION = "business_subset_v2_us_country"
COUNTRY_ALIASES = ("USA", "US", "United States")
PARTIAL_WEIGHT = 0.3
MANIFEST = ROOT / "data/versions/retail_full_v2/reviewed/phase05a_manifest.json"
IDENTITY_MANIFEST = ROOT / "data/versions/retail_full_v2/reviewed/phase05c_identity_manifest.json"


def row_fingerprint(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def goal_key(name: str, arguments: dict) -> tuple[str, str]:
    target = arguments["user_id"] if name == "modify_user_address" else arguments["order_id"]
    return name, target


def reference_subset(row: dict, goals: set[tuple[str, str]]) -> RetailSession:
    """保留参考只读操作，只执行指定原子业务；仅供隐藏评分与离线验收。"""
    session = RetailSession(row)
    for action in parse_task(row)["evaluation_criteria"]["actions"]:
        name, args = action["name"], action["arguments"]
        if name in WRITES and goal_key(name, args) not in goals:
            continue
        response = session.call_tool(name, args)
        if response.error:
            raise RewardEvaluationError(f"参考子集执行失败：{name}: {response.content}")
    return session


def training_db_hash(session: RetailSession, version=VERSION) -> str:
    if version == LEGACY_VERSION:
        return session.db_hash()
    if version != VERSION:
        raise RewardEvaluationError("未知训练奖励版本：" + str(version))
    # 只归一化独立序列化副本的地址国家；整个数据库的其他字段仍精确比较。
    db = session._environment.tools.db.model_dump()
    for collection in ["users", "orders"]:
        for entity in db[collection].values():
            address = entity["address"]
            if address["country"] in COUNTRY_ALIASES:
                address["country"] = "USA"
    return row_fingerprint(db)


@lru_cache(maxsize=64)
def _compile(row_json: str, version: str) -> dict:
    if version not in {LEGACY_VERSION, VERSION}:
        raise RewardEvaluationError("未知训练奖励版本：" + str(version))
    row = json.loads(row_json)
    task = parse_task(row)
    criteria = task["evaluation_criteria"]
    if task.get("initial_state") or set(criteria.get("reward_basis", ["DB", "COMMUNICATE"])) != {"DB", "COMMUNICATE"} or criteria.get("communicate_info"):
        raise RewardEvaluationError("首批子目标奖励只支持无初始历史的 DB / 空 COMMUNICATE 任务")
    writes = [a for a in criteria["actions"] if a["name"] in WRITES]
    goals = tuple(dict.fromkeys(goal_key(a["name"], a["arguments"]) for a in writes))
    if not 1 <= len(goals) <= 4 or len(goals) != len(writes):
        raise RewardEvaluationError("首批任务须有 1～4 个不重复的原子业务目标；过程型任务需单独复核")
    initial = RetailSession(row)
    db = initial._environment.tools.db
    owners = {a["arguments"]["user_id"] if a["name"] == "modify_user_address"
              else db.orders[a["arguments"]["order_id"]].user_id for a in writes}
    if len(owners) != 1:
        raise RewardEvaluationError("参考操作涉及多个用户")
    variants, rejected = [], []
    for size in range(len(goals) + 1):
        for subset in combinations(goals, size):
            try:
                session = reference_subset(row, set(subset))
            except RewardEvaluationError as error:
                rejected.append({"goals": subset, "reason": str(error)})
                continue
            variants.append({"goals": subset, "db_hash": training_db_hash(session, version)})
    full = next((v for v in variants if len(v["goals"]) == len(goals)), None)
    if full is None or evaluate_task(row, reference_subset(row, set(goals)).trajectory).reward != 1:
        raise RewardEvaluationError("完整参考业务不可达")
    if len({v["db_hash"] for v in variants}) != len(variants):
        raise RewardEvaluationError("子目标状态无区分度或存在还原初始状态的目标")
    return {"goals": goals, "owner": next(iter(owners)), "variants": variants,
            "rejected_subsets": rejected}


def compile_business_goals(row: dict, version=VERSION) -> dict:
    return _compile(json.dumps(row, sort_keys=True, ensure_ascii=False), version)


def audit_scenario_identity(row: dict) -> dict:
    """只读核对顾客实际持有的认证信息；参考动作正确不代表场景信息正确。"""
    session = RetailSession(row)
    db = session._environment.tools.db
    writes = [a for a in parse_task(row)["evaluation_criteria"]["actions"] if a["name"] in WRITES]
    owners = {a["arguments"]["user_id"] if a["name"] == "modify_user_address"
              else db.orders[a["arguments"]["order_id"]].user_id for a in writes}
    if len(owners) != 1:
        raise RewardEvaluationError("场景身份审计要求单一业务归属用户")
    user = db.users[next(iter(owners))]
    known = row["user_scenario"]["instructions"].get("known_info")
    known = json.loads(known) if isinstance(known, str) else known
    if not isinstance(known, dict):
        raise RewardEvaluationError("场景 known_info 不能直接核验，需人工复查")
    issues, routes = [], []
    if known.get("user_id") and known["user_id"] != user.user_id:
        issues.append("known_user_id_mismatch")
    if known.get("name") and known["name"] != user.name.model_dump():
        issues.append("known_name_mismatch")
    calls = []
    if known.get("email"):
        calls.append(("find_user_id_by_email", {"email": known["email"]}))
    if known.get("zip") and isinstance(known.get("name"), dict):
        calls.append(("find_user_id_by_name_zip", {**known["name"], "zip": known["zip"]}))
    for name, args in calls:
        response = session.call_tool(name, args)
        identity = _content(response)
        routes.append({"tool": name, "arguments": args, "returned_identity": identity, "error": response.error})
        if response.error or identity != user.user_id:
            issues.append("known_" + ("email" if name.endswith("email") else "name_zip") + "_does_not_authenticate_target")
    return {"id": row["id"], "owner": user.user_id, "database_email": user.email,
            "issues": issues, "routes": routes, "row_sha256": row_fingerprint(row)}


def ready_training_rows(path, review_manifest=None) -> list[dict]:
    if review_manifest is not None:
        from retail_agent.rl_dataset import reviewed_rl_rows
        rows = reviewed_rl_rows(path, 'train', review_manifest)
        for row in rows:
            if audit_scenario_identity(row)['issues']:
                raise RewardEvaluationError('训练任务身份矛盾：' + row['id'])
        return rows
    manifest = json.loads(IDENTITY_MANIFEST.read_text())
    phase05a = json.loads(MANIFEST.read_text())
    if (str(path.relative_to(ROOT)) != manifest["pool"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["pool_sha256"]
            or hashlib.sha256((ROOT / manifest["source"]).read_bytes()).hexdigest() != manifest["source_sha256"]
            or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != manifest["phase05a_manifest_sha256"]
            or manifest["immutable_sha256"] != phase05a["immutable_sha256"]
            or manifest["source_sha256"] != phase05a["pool_sha256"]):
        raise RewardEvaluationError("身份复核训练池或历史来源变化")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    approved = _approved_rows()
    accepted = {e["id"]: e["row_sha256"] for e in manifest["entries"] if not e["issues"]}
    if len(rows) != len(accepted) or len(rows) != len({r["id"] for r in rows}):
        raise RewardEvaluationError("身份复核训练池计数或任务 ID 不符")
    for row in rows:
        if (accepted.get(row["id"]) != row_fingerprint(row) or approved.get(row["id"]) != row_fingerprint(row)
                or audit_scenario_identity(row)["issues"]):
            raise RewardEvaluationError("训练任务身份矛盾或未经复核：" + row["id"])
    return rows


@lru_cache(maxsize=1)
def _approved_rows() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["immutable_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RewardEvaluationError(f"已复核的来源发生变化：{relative}")
    return {entry["id"]: entry["row_sha256"] for entry in manifest["entries"]}


def _content(message):
    try:
        return json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return message.content


def _replay_checks(row: dict, history: list, spec: dict) -> tuple[RetailSession, set, list]:
    """重放核验真实返回，同时检查身份、归属与成功的非目标写入。"""
    session = RetailSession(row)
    env = session._environment
    authenticated, pending, seen_ids = None, None, set()
    successful, violations = set(), []
    for message in history:
        if pending is not None:
            call, expected = pending
            if not isinstance(message, ToolMessage) or message.id != call.id:
                raise RewardEvaluationError("轨迹缺少匹配的工具返回")
            if (_content(message) != _content(expected) or message.error != expected.error
                    or message.requestor != expected.requestor):
                raise RewardEvaluationError("轨迹工具返回与真实重放不一致")
            if call.name.startswith("find_user_id_") and not expected.error:
                identity = _content(expected)
                if identity != spec["owner"]:
                    violations.append("identity_not_task_customer")
                if authenticated is not None and identity != authenticated:
                    violations.append("identity_switch")
                authenticated = identity
            pending = None
            continue
        if isinstance(message, ToolMessage):
            raise RewardEvaluationError("出现没有调用的工具返回")
        if isinstance(message, UserMessage):
            if message.tool_calls:
                raise RewardEvaluationError("Retail 顾客不能调用工具")
            continue
        if not isinstance(message, AssistantMessage):
            raise RewardEvaluationError("轨迹包含不支持的消息类型")
        if not message.tool_calls:
            continue
        if len(message.tool_calls) != 1:
            raise RewardEvaluationError("轨迹应使用单工具调用协议")
        call = message.tool_calls[0]
        if call.id in seen_ids or call.requestor != "assistant":
            raise RewardEvaluationError("工具调用 ID 重复或调用方非法")
        seen_ids.add(call.id)
        if (message.content or "").strip():
            violations.append("mixed_text_and_tool")
        name, args = call.name, call.arguments
        # 查询失败也不能绕过身份边界；公开商品与库存查询不需要账户归属。
        if name in WRITES or name in {"get_order_details", "get_user_details"}:
            if authenticated != spec["owner"]:
                violations.append("private_access_before_authentication")
            if isinstance(args.get("order_id"), str):
                order = env.tools.db.orders.get(args["order_id"])
                if order is not None and order.user_id != authenticated:
                    violations.append("order_not_owned_by_authenticated_user")
            if "user_id" in args and args["user_id"] != authenticated:
                violations.append("profile_not_owned_by_authenticated_user")
        before = session.db_hash() if name in WRITES else None
        expected = env.get_response(call)
        if name in WRITES and not expected.error:
            key = goal_key(name, args)
            successful.add(key)
            # 真正无状态变化的合法重复地址设置不算额外目标，也不增加进度。
            if key not in spec["goals"] and session.db_hash() != before:
                violations.append("successful_non_target_write")
        pending = call, expected
    if pending is not None:
        raise RewardEvaluationError("轨迹末尾缺少工具返回")
    return session, successful, sorted(set(violations))


def evaluate_training_task(row: dict, trajectory: list, termination="completed", *, version=VERSION,
                           review_manifest=None) -> dict:
    """固定业务指标与训练 A/B；当前清单或显式保留的历史 TRAIN 复核。"""
    try:
        if review_manifest is None:
            approved = _approved_rows()
        else:
            from retail_agent.rl_dataset import approved_training_rows
            approved = approved_training_rows(review_manifest)
        if approved.get(row["id"]) != row_fingerprint(row):
            raise RewardEvaluationError("任务未逐条复核，或已复核的原始任务被修改")
        spec = compile_business_goals(row, version)
        business = evaluate_task(row, trajectory, termination).reward
        session, successful, violations = _replay_checks(row, trajectory, spec)
    except RewardEvaluationError:
        raise
    except Exception as error:
        raise RewardEvaluationError(f"训练奖励配置或重放异常：{error}") from error
    completed = []
    if termination == "completed":
        final_hash = training_db_hash(session, version)
        match = next((v for v in spec["variants"] if v["db_hash"] == final_hash
                      and set(v["goals"]) <= successful), None)
        if match:
            completed = list(match["goals"])
    progress = len(completed) / len(spec["goals"])
    gate = float(not violations and termination == "completed")
    training_business = business if version == LEGACY_VERSION else float(progress == 1)
    result = {"version": version, "task_id": row["id"], "termination": termination,
            "business_reward": business, "gate": gate, "violations": violations,
            "goal_count": len(spec["goals"]), "completed_goals": completed,
            "progress": progress, "reward_a": gate * training_business,
            "reward_b": gate * (training_business + PARTIAL_WEIGHT * (1 - training_business) * progress),
            "partial_weight": PARTIAL_WEIGHT,
            "policy_scope": "identity_ownership_non_target_writes_protocol_only"}
    if version == VERSION:
        result["training_business_reward"] = training_business
    return result
