"""05C 身份复核：保留历史 30 条，另导出逐字段不变的可用子集；无 API。"""
import json

from retail_agent.environment import ROOT
from retail_agent.evaluate import sha256
from retail_agent.training_reward import IDENTITY_MANIFEST, MANIFEST, audit_scenario_identity


def main():
    source = MANIFEST.parent / 'rl_train_verified.jsonl'
    phase05a = json.loads(MANIFEST.read_text())
    if sha256(source) != phase05a['pool_sha256']:
        raise ValueError('历史任务池已变化')
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    entries = [audit_scenario_identity(row) for row in rows]
    approved = {e['id']: e['row_sha256'] for e in phase05a['entries']}
    if any(approved.get(e['id']) != e['row_sha256'] for e in entries):
        raise ValueError('历史任务行未经复核')
    for path, expected in phase05a['immutable_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('原始数据或数据库变化：' + path)
    accepted = {e['id'] for e in entries if not e['issues']}
    pool = MANIFEST.parent / 'rl_train_ready.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows if row['id'] in accepted)
    if pool.exists() and pool.read_text() != content:
        raise ValueError('已有复核训练池不同，禁止覆盖')
    pool.write_text(content)
    report = {'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source),
              'phase05a_manifest_sha256': sha256(MANIFEST), 'pool': str(pool.relative_to(ROOT)),
              'pool_sha256': sha256(pool), 'approved_count': len(accepted),
              'excluded_task_ids': [e['id'] for e in entries if e['issues']], 'entries': entries,
              'immutable_sha256': phase05a['immutable_sha256'], 'paid_cost_cny': 0,
              'scope': 'Known customer identity routes checked against initial DB; not full policy or all prose semantics.'}
    if IDENTITY_MANIFEST.exists() and json.loads(IDENTITY_MANIFEST.read_text()) != report:
        raise ValueError('已有身份复核记录不同，禁止覆盖')
    IDENTITY_MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['approved_count', 'excluded_task_ids', 'pool_sha256', 'paid_cost_cny']}))


if __name__ == '__main__':
    main()
