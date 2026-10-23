"""下载固定 Qwen3-4B 原权重，逐文件验证后派生本机 MLX 4-bit 副本。"""

import hashlib
import json
import tomllib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = tomllib.loads((ROOT / "configs/project.toml").read_text())
    repo, revision = config["model_id"], config["model_revision"]
    original = ROOT / "models/qwen3-4b"
    converted = ROOT / "models/qwen3-4b-mlx4bit"
    manifest_path = ROOT / "models/manifest.json"
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    files = []
    for item in info.siblings:
        if item.rfilename == ".gitattributes":
            continue
        path = Path(hf_hub_download(repo, item.rfilename, revision=revision,
                                   local_dir=original, token=False))
        digest = sha256(path)
        if item.lfs:
            if digest != item.lfs.sha256:
                raise ValueError(f"模型 SHA256 不符：{item.rfilename}")
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != item.blob_id:
                raise ValueError(f"模型 git blob 不符：{item.rfilename}")
        files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": digest, "upstream_verified": True})
        print(f"已验证 {item.rfilename} ({path.stat().st_size:,} bytes)", flush=True)
    if converted.exists():
        if not manifest_path.exists():
            raise ValueError("已有量化目录但无完成清单，请检查中断产物；不会覆盖已有文件")
        previous = json.loads(manifest_path.read_text())
        for item in previous["converted_files"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("量化副本校验失败")
    else:
        from mlx_lm import convert
        converted.parent.mkdir(parents=True, exist_ok=True)
        convert(str(original), mlx_path=str(converted), quantize=True,
                q_group_size=64, q_bits=4, dtype="float16")
    converted_files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                        "sha256": sha256(p)} for p in sorted(converted.glob("*")) if p.is_file()]
    manifest = {"model_id": repo, "model_revision": revision, "backend": "mlx-lm",
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "purpose": "local_phase3_inference_not_formal_model_comparison",
                "original_files": files, "converted_files": converted_files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("固定原权重与本机量化副本准备完成。", flush=True)


if __name__ == "__main__":
    main()
