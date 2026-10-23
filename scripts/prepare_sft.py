"""只处理 v2 train/dev；离线编码、全量验收掩码，封存 test 不进入训练输入。"""

import json
import sys
import tempfile
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer
from retail_agent.data import read_jsonl
from retail_agent.sft_data import encode_target, file_sha256, validate_encoded


def main():
    config = tomllib.loads((ROOT / "configs/sft.toml").read_text())
    project = tomllib.loads((ROOT / "configs/project.toml").read_text())
    inputs = {"train": config["train_file"], "dev": config["validation_file"]}
    expected = {"train": "sft_train.jsonl", "dev": "sft_dev.jsonl"}
    base = ROOT / project["data"]["processed_dir"]
    for split, relative in inputs.items():
        if (ROOT / relative).resolve() != (base / expected[split]).resolve():
            raise ValueError("SFT 只允许当前版本 train/dev；禁止读取最终测试示范")
    tokenizer_dir = ROOT / "data/raw/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    out = ROOT / config["prepared_dir"]
    if out.exists():
        raise FileExistsError(f"准备数据已存在：{out}；检查并移走旧产物后才能重建")
    out.parent.mkdir(parents=True, exist_ok=True)
    hashes = {p: file_sha256(ROOT / p) for p in inputs.values()}
    for p in [ROOT / "configs/sft.toml", ROOT / "src/retail_agent/sft_data.py",
              ROOT / "scripts/prepare_sft.py", *sorted(tokenizer_dir.glob("*.json"))]:
        hashes[str(p.relative_to(ROOT))] = file_sha256(p)
    counters = {}
    features = Features({"id": Value("string"), "input_ids": Sequence(Value("int32")),
                         "labels": Sequence(Value("int32")), "prompt_tokens": Value("int32")})
    with tempfile.TemporaryDirectory(prefix="sft-build-", dir=out.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for split, relative in inputs.items():
            def generate(path):
                for number, row in read_jsonl(Path(path)):
                    yield encode_target(row, tokenizer, config["max_sequence_length"])
                    if number % 500 == 0:
                        print(f"{split}: {number} 条完整历史已编码", flush=True)
            dataset = Dataset.from_generator(generate, gen_kwargs={"path": str(ROOT / relative)},
                                             features=features, cache_dir=str(Path(temporary) / "cache"))
            dataset.save_to_disk(str(staging / split))
            stored = load_from_disk(str(staging / split))
            counts, ids = Counter(), set()
            for row in stored:
                validate_encoded(row, config["max_sequence_length"])
                if row["id"] in ids:
                    raise ValueError("重复样本 ID")
                ids.add(row["id"])
                counts["rows"] += 1
                counts["total_tokens"] += len(row["input_ids"])
                counts["supervised_tokens"] += len(row["input_ids"])-row["prompt_tokens"]
                counts["max_tokens"] = max(counts["max_tokens"], len(row["input_ids"]))
            counters[split] = dict(counts)
            audit = json.loads((ROOT / project["data"]["manifest_dir"]).parent.joinpath(
                "reports/data_audit.json").read_text())["splits"]["sft_"+split]
            if any(counts[k] != audit[k] for k in ("rows", "total_tokens", "supervised_tokens")):
                raise ValueError("准备数据计数与独立数据审计不一致")
        if any(file_sha256(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError("编码过程中输入改变")
        report = {"status": "passed", "scope": "sft_preparation_not_formal_training",
                  "model_id": project["model_id"], "model_revision": project["model_revision"],
                  "dataset_version": project["data"]["dataset_version"], "splits": counters,
                  "max_sequence_length": config["max_sequence_length"], "input_sha256": hashes,
                  "output_sha256": {str(p.relative_to(staging)): file_sha256(p)
                                    for p in sorted(staging.rglob("*")) if p.is_file()},
                  "dependencies": {n: version(n) for n in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")},
                  "test_used": False, "truncated_samples": 0, "packing": False,
                  "loss_scope": "current_target_assistant_turn_including_terminator",
                  "paid_cost_cny": 0}
        (staging / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        staging.rename(out)
    (ROOT / "outputs/reports/sft_preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
