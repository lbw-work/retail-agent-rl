"""发布全量 RL 复核投影；固定原划分，禁止覆盖已有版本，不触碰 SFT。"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from build_dataset import digest, save, save_rows
from retail_agent.training_reward import row_fingerprint

OUT=ROOT/'outputs/review/rl_full_20261015'
DEST=ROOT/'data/versions/retail_rl_v3'


def main():
    if DEST.exists():
        raise ValueError('RL 审计版本已存在；禁止覆盖冻结数据')
    facts=json.loads((OUT/'facts.json').read_text())
    semantic=json.loads((OUT/'semantic_checks.json').read_text())
    manual=json.loads((OUT/'manual_decisions.json').read_text())
    by_id={c['id']:c for c in facts['cases']}
    semantics={c['id']:c for c in semantic['cases']}
    assert len(by_id)==len(semantics)==514
    for relative,expected in facts['source_sha256'].items():
        assert digest(ROOT/relative)==expected, '原始文件改变：'+relative
    accepted,quarantined={s:[] for s in ['train','dev','test']},{s:[] for s in ['train','dev','test']}
    entries=[];source_files={}
    for side in accepted:
        source=ROOT/f'data/versions/retail_full_v2/processed/rl_{side}.jsonl'
        source_files[side]=str(source.relative_to(ROOT))
        for line in source.read_text().splitlines():
            row=json.loads(line);task_id=row['id'];case=by_id[task_id];sem=semantics[task_id]
            assert case['split']==sem['split']==side and case['row_sha256']==row_fingerprint(row)
            issues=sorted(set(case['issues']+sem['flags']))
            decision=manual.get(task_id)
            if decision:
                assert decision['decision']=='quarantine', '不允许无证据的人工放行'
                issues.append(decision['reason'])
            complete='audit_error' not in case and all(k in case for k in
                ['reference','identity','writes','owned_orders','reward_support'])
            assert complete, '离线复核不完整：'+task_id
            status='quarantined' if issues else 'accepted'
            (quarantined if issues else accepted)[side].append(row)
            entries.append({'id':task_id,'split':side,'group':row['group'],'business':row['business'],
                'businesses':row['businesses'],'difficulty':row['difficulty'],'row_sha256':row_fingerprint(row),
                'status':status,'checks_complete':complete,'unresolved_issues':issues,
                'decision_basis':decision['evidence'] if decision else
                    ('真实参考重放、身份/快照和原子奖励检查通过，固定语义约束未发现未解决矛盾。' if not issues else
                     '保守隔离；理由为检查线索或奖励支持边界，不把所有线索声称为已确认错误。'),
                'evidence_task_id':task_id})
    files={}
    for side in accepted:
        for key,rows,path in [(side,accepted[side],DEST/'processed'/f'rl_{side}.jsonl'),
                              ('quarantine_'+side,quarantined[side],DEST/'quarantine'/f'rl_{side}.jsonl')]:
            path.parent.mkdir(parents=True,exist_ok=True);save_rows(path,rows)
            files[key]={'path':str(path.relative_to(ROOT)),'sha256':digest(path),'count':len(rows)}
    evidence={str((OUT/n).relative_to(ROOT)):digest(OUT/n) for n in ['facts.json','semantic_checks.json','manual_decisions.json']}
    immutable=dict(facts['source_sha256'])
    for relative in ['data/versions/retail_full_v2/manifests/splits.json','data/provenance/legacy_holdout_nodes.json']:
        immutable[relative]=digest(ROOT/relative)
    manifest={'version':'retail_rl_v3','source_version':'retail_full_v2','scope':'RL_only_all_514_selected_v2_tasks',
        'row_version_note':'任务行保持 v2 原始字段与指纹；v3 是复核后的独立数据视图。',
        'paid_cost_cny':0,'model_called':False,'sft_changed':False,'gold_or_scenario_changed':False,
        'split_reassigned':False,'test_model_evaluation_used':False,
        'review_method':'all_task_execution_identity_snapshot_atomic_reward_checks_plus_fixed_semantic_contracts_and_targeted_source_review',
        'limitations':['固定语义规则和人工重点复核不构成全部自然语言语义或逐笔确认的形式证明。',
            '隔离包含待澄清线索，不等于所有隔离任务确定错误。',
            '测试内容为质量审查已被查看，仍留出且不用于训练；不再称为完全盲测。',
            'SFT 仍为 v2；既有 SFT 训练可能含与本轮疑点相似的示范，未开展 SFT 语义审查。'],
        'source_files':source_files,'immutable_sha256':immutable,'evidence_sha256':evidence,
        'files':files,'source_counts':dict(Counter(e['split'] for e in entries)),
        'accepted_counts':{s:len(rs) for s,rs in accepted.items()},
        'quarantined_counts':{s:len(rs) for s,rs in quarantined.items()},
        'entries':entries}
    save(DEST/'manifests/review.json',manifest)
    stats={'version':'retail_rl_v3','source_counts':manifest['source_counts'],
        'accepted_counts':manifest['accepted_counts'],'quarantined_counts':manifest['quarantined_counts'],
        'business_primary':{s:dict(Counter(r['business'] for r in rs)) for s,rs in accepted.items()},
        'business_coverage':{s:dict(Counter(b for r in rs for b in r['businesses'])) for s,rs in accepted.items()},
        'difficulty':{s:dict(Counter(r['difficulty'] for r in rs)) for s,rs in accepted.items()},
        'source_groups':{s:len({r['group'] for r in rs}) for s,rs in accepted.items()},
        'unresolved_issue_counts':dict(Counter(i for e in entries for i in e['unresolved_issues'])),
        'old_metrics_note':'v2 3/14 -> 8/14 为历史结果，新版需在相同新任务上重新对照。'}
    save(DEST/'reports/counts.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
