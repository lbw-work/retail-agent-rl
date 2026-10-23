"""下载固定 revision 的数据和 tokenizer 文件；不下载模型权重。"""

import hashlib
import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_REV = "86971dc03da6e7c1a7933295e05b84aab8215386"
MODEL_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
ENV_REV = "f8de30c298689cbe0117d76a378e7315a17e5bd8"
MANIFEST = ROOT / "data/provenance/raw_sources.json"
PREVIOUS = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def download(item):
    url, relative_path = item
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    previous = next((f for f in PREVIOUS.get('files', [])
                     if f['path'] == relative_path and f['url'] == url), None)
    if path.exists() and previous:
        with path.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        if path.stat().st_size == previous['bytes'] and digest.hexdigest() == previous['sha256']:
            print(f"复用已校验 {relative_path}", flush=True)
            return previous
        digest = hashlib.sha256()
    # 暂存文件避免中途退出留下被当作完整数据的 JSONL。
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    temporary.replace(path)
    print(f"完成 {relative_path} ({size:,} bytes)", flush=True)
    return {"path": relative_path, "url": url, "bytes": size, "sha256": digest.hexdigest()}


def main():
    dataset_url = f"https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data/resolve/{DATA_REV}"
    model_url = f"https://huggingface.co/Qwen/Qwen3-4B/resolve/{MODEL_REV}"
    items = [(f"{dataset_url}/{name}", f"data/raw/{name}") for name in (
        "README.md", "tau2_rl_train.jsonl", "tau2_sft_train.jsonl",
    )]
    items += [(f"{dataset_url}/tau2_rl_database/{name}", f"data/databases/{name}") for name in (
        "tau2_retail_new_db_1.json", "tau2_retail_new_db_2.json",
        "tau2_retail_new_db_3.json", "tau2_retail_new_db_4.json",
    )]
    items += [(f"{model_url}/{name}", f"data/raw/tokenizer/{name}") for name in (
        "tokenizer.json", "tokenizer_config.json", "config.json", "README.md", "LICENSE",
    )]
    archive_url = f"https://codeload.github.com/sierra-research/tau2-bench/tar.gz/{ENV_REV}"
    items.append((archive_url, "data/raw/tau2-v0.2.0.tar.gz"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, items))
    checkout = ROOT / 'third_party/tau2-bench'
    if not checkout.exists():
        with tarfile.open(ROOT / 'data/raw/tau2-v0.2.0.tar.gz') as archive:
            archive.extractall(ROOT / 'third_party', filter='data')
        (ROOT / 'third_party' / f'tau2-bench-{ENV_REV}').rename(checkout)
    manifest = {**PREVIOUS,
        "dataset_id": "inclusionAI/AReaL-tau2-data", "dataset_revision": DATA_REV,
        "dataset_license": "apache-2.0", "model_id": "Qwen/Qwen3-4B",
        "model_revision": MODEL_REV, "model_license": "apache-2.0",
        "files": files,
    }
    manifest.setdefault('environment', {
        'repository': 'https://github.com/sierra-research/tau2-bench', 'tag': 'v0.2.0',
        'commit': ENV_REV, 'license': 'MIT', 'archive_url': archive_url,
        'status': 'candidate_pending_audit',
    })
    source_paths = sorted((checkout / 'src/tau2').rglob('*.py'))
    source_paths.append(checkout / 'data/tau2/domains/retail/policy.md')
    source_files = [{'path': str(p.relative_to(ROOT)),
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]
    previous_sources = {f['path']: f['sha256'] for f in PREVIOUS.get('environment', {}).get('source_files', [])}
    if previous_sources and previous_sources != {f['path']: f['sha256'] for f in source_files}:
        raise ValueError('已固定的上游源码改变；不会静默刷新来源记录')
    manifest['environment'].update(source_files=source_files,
        archive_sha256=next(f['sha256'] for f in files if f['path']=='data/raw/tau2-v0.2.0.tar.gz'))
    dependencies = {}
    for line in (ROOT / 'requirements/environment.lock.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==')
        dependencies[name] = version(name)
        if dependencies[name] != expected:
            raise ValueError('请先安装固定审计环境：'+name)
    manifest.setdefault('environment_dependencies', dependencies)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
