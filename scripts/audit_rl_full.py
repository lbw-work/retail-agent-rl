"""全量 RL 离线事实审计；保留 v2/SFT，不调用模型或付费服务。"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from build_dataset import digest, reference_probe, save
from finalize_rl_review import (MANUAL_QUARANTINE, address_without_source, mentioned_missing_products)
from review_rl_data import inspect_task
from retail_agent.data import parse_task
from retail_agent.dataset import WRITES
from retail_agent.environment import RetailSession
from retail_agent.training_reward import audit_scenario_identity, compile_business_goals, row_fingerprint

SOURCE = ROOT / 'data/versions/retail_full_v2'
OUT = ROOT / 'outputs/review/rl_full_20261015'


def inspect_full(item):
    split, row = item
    result = {'id': row['id'], 'split': split, 'row_sha256': row_fingerprint(row),
              'businesses': row['businesses'], 'group': row['group'], 'issues': [],
              'instructions': row['user_scenario']['instructions']}
    try:
        case = inspect_task(row)
        result['reference'] = case['reference_checks']
        result['changes'] = case['changes']
        result['writes'] = case['writes']
        result['issues'] += [f['type'] for f in case['flags']]
        if case['reference_checks']['issues']:
            result['issues'].append('reference_execution_or_reward_failure')
        result['issues'] += case['reference_checks']['policy_flags']
        if case['omit_all_profile_writes_reward'] == 1:
            result['issues'].append('profile_write_omission_still_full_reward')
        identity = audit_scenario_identity(row)
        result['identity'] = identity
        result['issues'] += identity['issues']
        session = RetailSession(row)
        db = session._environment.tools.db
        user = db.users[identity['owner']]
        orders = [db.orders[o] for o in user.orders]
        owned_names = {i.name for o in orders for i in o.items}
        missing = mentioned_missing_products(case['instructions'], owned_names, {p.name for p in db.products.values()})
        result['mentioned_missing_products'] = missing
        if missing:
            result['issues'].append('mentioned_product_not_owned_requires_review')
        addresses = [user.address.model_dump()] + [o.address.model_dump() for o in orders]
        result['ungrounded_addresses'] = address_without_source(case['instructions'], case['writes'], addresses)
        if result['ungrounded_addresses']:
            result['issues'].append('gold_address_not_fully_grounded_requires_review')
        if row['id'] in MANUAL_QUARANTINE:
            result['issues'].append(MANUAL_QUARANTINE[row['id']])
        if row['id'] == 'retail_500':
            result['issues'].append('gold_omits_explicit_phone_goals')
        result['owned_orders'] = [{'id': o.order_id, 'status': o.status, 'address': o.address.model_dump(),
            'items': [{'id': i.item_id, 'product': i.name, 'options': i.options, 'price': i.price} for i in o.items],
            'payment_history': [p.model_dump() for p in o.payment_history]} for o in orders]
        result['profile_address'] = user.address.model_dump()
        result['payment_methods'] = {k:v.model_dump() for k,v in user.payment_methods.items()}
        result['reference_authentication'] = [a for a in parse_task(row)['evaluation_criteria']['actions']
            if a['name'].startswith('find_user_id_')]
        # 每条任务都检查当前原子奖励的支持边界；不能把不可区分过程混入训练。
        try:
            spec = compile_business_goals(row)
            result['reward_support'] = {'supported': True, 'goals': spec['goals'],
                'reachable_subsets': len(spec['variants']), 'rejected_subsets': spec['rejected_subsets']}
        except Exception as error:
            result['reward_support'] = {'supported': False, 'error': str(error), 'type': type(error).__name__}
            result['issues'].append('atomic_reward_unsupported')
    except Exception as error:
        result['audit_error'] = {'type': type(error).__name__, 'message': str(error)}
        result['issues'].append('audit_incomplete')
    result['issues'] = sorted(set(result['issues']))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / 'processed' / f'{kind}_{side}.jsonl'
             for kind in ['rl', 'sft'] for side in ['train', 'dev', 'test']]
    paths += sorted((ROOT / 'data/databases').glob('*.json'))
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    items = [(s, json.loads(line)) for s in ['train', 'dev', 'test']
             for line in (SOURCE / 'processed' / f'rl_{s}.jsonl').read_text().splitlines()]
    assert len(items) == 514 and len({r['id'] for _,r in items}) == 514
    cases = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect_full, items, chunksize=1):
            cases.append(result)
            if len(cases) % 20 == 0 or len(cases) == len(items):
                save(OUT / 'facts.partial.json', {'source_sha256': before, 'cases': cases})
                print(f"离线事实复核 {len(cases)}/{len(items)}；线索任务 {sum(bool(c['issues']) for c in cases)}", flush=True)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in paths}, '原始数据/SFT/数据库发生变化'
    save(OUT / 'facts.json', {'scope': 'all_514_rl_train_dev_test_no_sft_changes',
        'source_sha256': before, 'script_sha256': digest(Path(__file__)),
        'paid_cost_cny': 0, 'model_called': False,
        'counts': dict(Counter(c['split'] for c in cases)),
        'issue_counts': dict(Counter(i for c in cases for i in c['issues'])), 'cases': cases})
    print(json.dumps({'count': len(cases), 'audit_incomplete': sum('audit_incomplete' in c['issues'] for c in cases),
                      'flagged': sum(bool(c['issues']) for c in cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
