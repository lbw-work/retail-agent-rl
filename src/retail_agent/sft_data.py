"""固定 Qwen 模板的目标回合编码；历史与 padding 不参与损失。"""

import hashlib


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_target(row, tokenizer, max_length):
    completion = row["completion"]
    if len(completion) != 1 or completion[0]["role"] != "assistant":
        raise ValueError("每条样本必须恰好包含一个目标客服回合")
    kwargs = dict(tools=row["tools"], enable_thinking=False, tokenize=True)
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(row["prompt"] + completion, **kwargs)
    if full[:len(prompt)] != prompt:
        raise ValueError("目标回合边界不是精确 token 前缀")
    if len(full) <= len(prompt):
        raise ValueError("没有可训练的目标 token")
    if len(full) > max_length:
        raise ValueError(f"样本 {row['id']} 长度 {len(full)} 超过 {max_length}，禁止截断")
    metadata = row.get("metadata", {})
    if metadata.get("total_tokens", len(full)) != len(full) or metadata.get(
        "target_tokens", len(full)-len(prompt)) != len(full)-len(prompt):
        raise ValueError("来源 token 计数与当前 tokenizer 不一致")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") not in full[len(prompt):]:
        raise ValueError("目标客服回合缺少结束标记")
    result = {"id": row["id"], "input_ids": full,
              "labels": [-100] * len(prompt) + full[len(prompt):],
              "prompt_tokens": len(prompt)}
    validate_encoded(result, max_length)
    return result


def validate_encoded(row, max_length):
    ids, labels, boundary = row["input_ids"], row["labels"], row["prompt_tokens"]
    if not 0 < boundary < len(ids) <= max_length or len(labels) != len(ids):
        raise ValueError("编码长度或目标边界无效")
    if labels[:boundary] != [-100]*boundary or labels[boundary:] != ids[boundary:]:
        raise ValueError("掩码必须只监督当前客服目标回合")


class TargetTurnCollator:
    """只补右侧 padding，原有 labels 保持不变。"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            length = len(row["input_ids"])
            if len(row["labels"]) != length or not any(x != -100 for x in row["labels"]):
                raise ValueError("样本 labels 长度错误或没有监督 token")
            ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[i, :length] = torch.tensor(row["labels"], dtype=torch.long)
            attention[i, :length] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attention}
