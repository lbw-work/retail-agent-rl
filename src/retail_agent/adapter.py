"""将本项目 PEFT LoRA 接到同一 MLX 量化底座，不合并或再次量化权重。"""

import json
from pathlib import Path


def load_peft_adapter(model, directory):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text())
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if (config["peft_type"] != "LORA" or set(config["target_modules"]) != targets
            or config["bias"] != "none" or config["r"] != 16 or config["lora_alpha"] != 32):
        raise ValueError("适配器与本项目固定 LoRA 配置不符")
    unsupported = ["use_dora", "use_rslora", "use_qalora", "fan_in_fan_out", "lora_bias",
                   "rank_pattern", "alpha_pattern", "modules_to_save", "layers_to_transform",
                   "layer_replication", "trainable_token_indices", "alora_invocation_tokens"]
    if any(config.get(key) for key in unsupported):
        raise ValueError("适配器包含尚未支持的 PEFT 变体")
    weights = mx.load(str(directory / "adapter_model.safetensors"))
    converted = {}
    for name, value in weights.items():
        if not name.startswith("base_model.model.model.layers."):
            raise ValueError(f"适配器权重名称不支持：{name}")
        name = name.removeprefix("base_model.model.")
        if name.endswith(".lora_A.weight"):
            name = name.removesuffix(".lora_A.weight") + ".lora_a"
        elif name.endswith(".lora_B.weight"):
            name = name.removesuffix(".lora_B.weight") + ".lora_b"
        else:
            raise ValueError("适配器包含非 LoRA 权重")
        if name in converted or value.ndim != 2 or not mx.all(mx.isfinite(value)).item():
            raise ValueError("适配器包含重复、错误形状或非有限权重")
        converted[name] = value.T
    if not any(mx.any(value != 0).item() for name, value in converted.items()
               if name.endswith(".lora_b")):
        raise ValueError("LoRA B 全为零，没有有效更新证据")
    linear_to_lora_layers(model, len(model.layers), {
        "rank": config["r"], "scale": config["lora_alpha"] / config["r"],
        "dropout": 0.0, "keys": ["self_attn." + key for key in sorted(targets)],
    })
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())
                if name.endswith((".lora_a", ".lora_b"))}
    if set(expected) != set(converted):
        raise ValueError("适配器权重缺失或多余，禁止部分加载")
    if any(converted[name].shape != shape for name, shape in expected.items()):
        raise ValueError("适配器维度与底座不符")
    model.load_weights(list(converted.items()), strict=False)
    model.eval()
    return {"tensor_count": len(converted), "scale": config["lora_alpha"] / config["r"],
            "conversion": "PEFT_A_B_transpose_to_MLX_no_merge", "finite": True,
            "nonzero_lora_B": True}
