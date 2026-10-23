"""独立 RL 审计版本：只投影原始划分，SFT 及历史评测继续使用 v2。"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from retail_agent.environment import ROOT

REVIEW_MANIFEST = ROOT / 'data/versions/retail_rl_v3/manifests/review.json'


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_review(path=REVIEW_MANIFEST):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.resolve() != REVIEW_MANIFEST.resolve():
        raise ValueError('只允许当前冻结的 RL 复核清单')
    manifest = json.loads(path.read_text())
    if manifest['version'] != 'retail_rl_v3' or manifest['source_version'] != 'retail_full_v2':
        raise ValueError('RL 复核版本不符')
    for relative, expected in manifest['immutable_sha256'].items():
        if file_digest(ROOT / relative) != expected:
            raise ValueError('RL 原始来源、SFT 或数据库发生变化：' + relative)
    for relative, expected in manifest['evidence_sha256'].items():
        if file_digest(ROOT / relative) != expected:
            raise ValueError('RL 复核证据发生变化：' + relative)
    entries = manifest['entries']
    if len(entries) != 514 or len({e['id'] for e in entries}) != 514:
        raise ValueError('RL 全量复核条目计数不符')
    for side in ['train', 'dev', 'test']:
        source = ROOT / manifest['source_files'][side]
        rows = [json.loads(l) for l in source.read_text().splitlines()]
        records = {e['id']: e for e in entries if e['split'] == side}
        if len(rows) != len(records) or {r['id'] for r in rows} != set(records):
            raise ValueError('RL 复核遗漏任务或改变原划分')
        for row in rows:
            fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            entry = records[row['id']]
            if entry['row_sha256'] != fingerprint or entry['status'] not in {'accepted', 'quarantined'}:
                raise ValueError('RL 来源行变化或复核状态非法')
            if not entry['decision_basis'] or not entry['checks_complete']:
                raise ValueError('RL 任务缺少完整复核记录')
            if entry['status'] == 'accepted' and entry['unresolved_issues']:
                raise ValueError('未解决疑点进入 RL 可用池')
    return manifest


def reviewed_rl_rows(path, split, manifest_path=REVIEW_MANIFEST):
    if split not in {'train', 'dev', 'test'}:
        raise ValueError('未知 RL 划分')
    manifest = load_review(manifest_path)
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    expected = manifest['files'][split]
    if path.resolve() != (ROOT / expected['path']).resolve() or file_digest(path) != expected['sha256']:
        raise ValueError('RL 可用池路径或哈希不符')
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    accepted = {e['id']: e['row_sha256'] for e in manifest['entries']
                if e['split'] == split and e['status'] == 'accepted'}
    if len(rows) != expected['count'] or len(rows) != len(accepted) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('RL 可用池计数或 ID 不符')
    for row in rows:
        fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if accepted.get(row['id']) != fingerprint:
            raise ValueError('RL 可用池任务被修改或跨划分')
    return rows


@lru_cache(maxsize=1)
def approved_training_rows(manifest_path):
    # 与历史奖励清单相同：启动时验证冻结来源，每条轨迹再核对原始行指纹。
    manifest = load_review(manifest_path)
    return {e['id']: e['row_sha256'] for e in manifest['entries']
            if e['split'] == 'train' and e['status'] == 'accepted'}
