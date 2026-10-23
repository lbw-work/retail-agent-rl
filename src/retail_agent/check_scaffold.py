"""检查项目骨架与固定约束，不导入训练依赖，不进行 API 调用。"""

import tomllib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    configs = {}
    for name in ("project", "sft", "grpo", "eval"):
        with (root / "configs" / f"{name}.toml").open("rb") as file:
            configs[name] = tomllib.load(file)

    for name, config in configs.items():
        if config["model_id"] != "Qwen/Qwen3-4B":
            raise ValueError(f"{name}: 客服模型必须是 Qwen/Qwen3-4B")
        if config["enable_thinking"] is not False:
            raise ValueError(f"{name}: 第一版固定关闭思考模式")

    budget = configs["project"]["budget"]
    allocated = sum(budget[key] for key in ("gpu_limit", "api_limit", "reserve_limit"))
    if not 0 < allocated <= budget["total_limit"] <= 50:
        raise ValueError("费用分配必须为正数，且不超过 50 元")
    if not 0 < budget["stop_starting_new_work_at"] < budget["total_limit"]:
        raise ValueError("启动新工作的停止线必须低于总费用上限")
    if configs["grpo"]["num_generations"] < 2:
        raise ValueError("GRPO 分组至少需要两条轨迹")

    for name in (
        "data/raw", "data/databases", "data/provenance", "data/prepared", "models",
        configs["project"]["data"]["processed_dir"], configs["project"]["data"]["manifest_dir"],
        "outputs", "third_party", "tests", "docs",
    ):
        if not (root / name).is_dir():
            raise FileNotFoundError(f"缺少目录：{name}")

    print("骨架检查通过：4 份配置可读取；模型、模式、预算与目录约束一致。")
    print(f"当前状态：{configs['project']['status']}；骨架检查本身不验证环境奖励、模型推理或训练。")


if __name__ == "__main__":
    main()
