"""打包 GRPO 必需代码与来源证据，不含密钥、环境、底座权重或旧训练输出。"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/transfer"
MANIFEST = DEST / "grpo_upload_manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("上传文件缺失或哈希不同：" + relative)
    print(f"上传包文件验收通过：{len(manifest['sha256'])} 个文件；底座另由 preflight 校验")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify()
        return
    sft = ROOT / "outputs/downloads/sft_first_20261011_200837/outputs/sft/sft_first_20261011_200837"
    paths = [ROOT / name for name in ["src", "scripts", "tests", "configs", "requirements", "docs",
        "pyproject.toml", "README.md", "AGENTS.md", "data/databases", "data/provenance",
        "data/versions/retail_full_v2", "data/versions/retail_rl_v3", "data/raw/tokenizer",
        "data/prepared/retail_full_v2_sft/manifest.json", "models/manifest.json",
        "outputs/evaluation/dev_original_sft_20261012/manifest.json",
        "third_party/tau2-bench/src", "third_party/tau2-bench/data/tau2/domains/retail",
        "third_party/tau2-bench/data/tau2/user_simulator", "third_party/tau2-bench/LICENSE"]]
    paths += [sft / "adapter", sft / "run.json"]
    paths += [ROOT / "outputs/review/rl_full_20261015" / name
              for name in ["facts.json", "semantic_checks.json", "manual_decisions.json"]]
    files = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for source in path.rglob("*") if path.is_dir() else [path]:
            if source.is_file() and not any(p in {"__pycache__", ".cache", ".git"} for p in source.parts):
                if source.is_symlink() or source.name == ".env" or source.suffix == ".pyc":
                    continue
                files.add(source)
    DEST.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"scope": "grpo_code_data_evidence_no_secrets_no_base_weights",
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}, ensure_ascii=False, indent=2) + "\n")
    archive = DEST / "grpo_upload.tar.gz"
    temporary = archive.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in sorted(files | {MANIFEST}):
            bundle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    temporary.replace(archive)
    print(f"上传包：{archive}；{archive.stat().st_size / 2**20:.1f} MiB；另传 models/qwen3-4b")


if __name__ == "__main__":
    main()
