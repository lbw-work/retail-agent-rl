"""复用官方确定性评估器；隐藏标准只进入评分端，不进入客服消息。"""

from copy import deepcopy

from retail_agent.data import parse_task
from retail_agent.environment import create_environment
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


class RewardEvaluationError(RuntimeError):
    """环境、轨迹或评分配置异常；不能作为零奖励训练样本。"""


def evaluate_task(row: dict, trajectory: list, termination="completed") -> RewardInfo:
    task = Task.model_validate(parse_task(row))
    criteria = task.evaluation_criteria
    if criteria is None or not criteria.reward_basis:
        raise RewardEvaluationError("任务必须声明非空奖励标准")
    if RewardType.NL_ASSERTION in criteria.reward_basis:
        raise RewardEvaluationError("自然语言裁判未接入；禁止静默忽略或自动调用付费模型")
    if termination == "infrastructure_error":
        raise RewardEvaluationError("基础设施故障，不计算业务零奖励")
    if termination not in {"completed", "max_steps", "too_many_errors", "context_limit",
                           "generation_limit", "invalid_model_output"}:
        raise RewardEvaluationError(f"未知终止原因：{termination}")
    if termination != "completed":
        return RewardInfo(reward=0.0, reward_basis=criteria.reward_basis,
                          info={"termination": termination, "incomplete": True})

    environments = []

    def constructor(solo_mode=False):
        environment = create_environment(row, solo_mode=solo_mode)
        environments.append(environment)
        return environment

    history = deepcopy(trajectory)
    results = []
    try:
        if {RewardType.DB, RewardType.ENV_ASSERTION} & set(criteria.reward_basis):
            result = EnvironmentEvaluator.calculate_reward(constructor, task, history)
            # 预测环境和金标准环境由官方分别创建；不修改交互现场。
            if len(environments) == 2 and environments[1].tool_errors:
                raise RewardEvaluationError(f"参考操作执行失败：{environments[1].tool_errors}")
            results.append(result)
        if RewardType.COMMUNICATE in criteria.reward_basis:
            results.append(CommunicateEvaluator.calculate_reward(task, history))
        if RewardType.ACTION in criteria.reward_basis:
            results.append(ActionEvaluator.calculate_reward(task, history))
    except RewardEvaluationError:
        raise
    except Exception as failure:
        raise RewardEvaluationError(f"奖励评估异常：{type(failure).__name__}: {failure}") from failure

    breakdown = {key: value for result in results
                 for key, value in (result.reward_breakdown or {}).items()}
    if set(breakdown) != set(criteria.reward_basis):
        raise RewardEvaluationError("未完整计算任务声明的奖励组成")
    reward = 1.0
    for component in set(criteria.reward_basis):
        reward *= breakdown[component]
    details = {}
    for result in results:
        for field in ("db_check", "env_assertions", "action_checks", "communicate_checks"):
            if getattr(result, field) is not None:
                details[field] = getattr(result, field)
    return RewardInfo(reward=reward, reward_basis=criteria.reward_basis,
                      reward_breakdown=breakdown, info={"termination": termination}, **details)
