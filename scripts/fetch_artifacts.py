"""Restore pinned public artifacts; verify hashes and refuse conflicting local files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/downloads.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_destination(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p in {"..", "."} for p in path.parts):
        raise ValueError("Unsafe artifact path: " + relative)
    target = root / path
    if any(p.is_symlink() for p in [target, *target.parents] if p == root or root in p.parents):
        raise ValueError("Symlink in artifact destination: " + relative)
    return target


def restore_archive(archive_path, root, allowed_prefixes):
    """Validate the entire archive before creating any missing files."""
    with tarfile.open(archive_path, "r:gz") as archive:
        pending, names = [], set()
        for member in archive.getmembers():
            if not member.isfile() or member.name in names:
                raise ValueError("Unexpected archive member: " + member.name)
            names.add(member.name)
            if not any(member.name.startswith(prefix + "/") for prefix in allowed_prefixes):
                raise ValueError("Unexpected artifact destination: " + member.name)
            target = check_destination(root, member.name)
            if target.exists():
                if not target.is_file():
                    raise ValueError("Artifact path is not a file: " + member.name)
                with archive.extractfile(member) as source:
                    expected = hashlib.file_digest(source, "sha256").hexdigest()
                if digest(target) != expected:
                    raise ValueError("Existing file differs; no files restored: " + member.name)
            else:
                pending.append(member)
        archive.extractall(root, members=pending, filter="data")
    return len(pending)


def restore_bundle(item, root, download):
    with tempfile.TemporaryDirectory(prefix="retail-artifact-") as temporary:
        archive_path = Path(temporary) / item["filename"]
        with archive_path.open("wb") as output:
            for part in item["parts"]:
                cached = Path(download(item["repo_id"], part["filename"], repo_type="dataset",
                                       revision=item["revision"], token=False))
                if cached.stat().st_size != part["bytes"] or digest(cached) != part["sha256"]:
                    raise ValueError("Artifact part SHA256/size mismatch: " + part["filename"])
                with cached.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if archive_path.stat().st_size != item["bytes"] or digest(archive_path) != item["sha256"]:
            raise ValueError("Reassembled bundle SHA256/size mismatch: " + item["filename"])
        return restore_archive(archive_path, root, item["allowed_prefixes"])


def restore_community_model(item):
    """Reproduce the pinned conversion; never accept a different model silently."""
    import sys
    from huggingface_hub import snapshot_download

    destination = check_destination(ROOT, item["destination"])
    manifest_path = ROOT / item["conversion_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if destination.exists():
        for name, expected in {**manifest["files"], "CONVERSION_MANIFEST.json": digest(manifest_path)}.items():
            if not (destination / name).is_file() or digest(destination / name) != expected:
                raise ValueError("Existing model file differs or is missing: " + name)
        print("community-model: existing exact conversion verified")
        return
    if sys.platform != "darwin":
        raise ValueError("The MLX conversion requires Apple Silicon macOS; use the upstream model for other backends")
    from mlx_lm.convert import convert

    pending = destination.with_name(destination.name + ".partial")
    if pending.exists():
        raise ValueError("An earlier conversion is preserved; inspect it before retrying: " + str(pending))
    source = Path(snapshot_download(item["repo_id"], revision=item["revision"],
                                   allow_patterns=list(manifest["source_files_sha256"]), token=False))
    for name, expected in manifest["source_files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("Source model SHA256 mismatch: " + name)
    convert(hf_path=str(source), mlx_path=str(pending), quantize=True,
            q_group_size=64, q_bits=4, q_mode="affine")
    for name, expected in manifest["files"].items():
        if not (pending / name).is_file() or digest(pending / name) != expected:
            raise ValueError("Conversion differs from frozen evaluation; preserved in " + str(pending))
    shutil.copyfile(manifest_path, pending / "CONVERSION_MANIFEST.json")
    pending.rename(destination)
    print("community-model: pinned source converted and every frozen file hash verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=["data", "sft", "verl", "community-model"],
                        help="May be repeated; default: data and sft")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    manifest = json.loads(MANIFEST.read_text())
    for name in args.only or ["data", "sft"]:
        item = manifest["artifacts"][name]
        if name == "community-model":
            restore_community_model(item)
        else:
            count = restore_bundle(item, ROOT, hf_hub_download)
            print(f"{name}: {count} files restored, parts and full archive SHA256 verified")



if __name__ == "__main__":
    main()
