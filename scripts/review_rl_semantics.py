"""全量任务语义线索：固定规则、不使用模型表现；歧义保守隔离。"""
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/review/rl_full_20261015'


def has_value(text, value):
    value = str(value).lower()
    aliases = {'xs': ['xs', 'extra small', 'extra-small'], 's': ['size s', 'small-size', 'small size'],
               'm': ['size m', 'medium-size', 'medium size'], 'l': ['size l', 'large size', 'large-size'],
               'xl': ['xl', 'extra large', 'extra-large'], 'xxl': ['xxl', 'extra extra large', 'extra-extra-large']}
    options = aliases.get(value, [value])
    return any(re.search(r'(?<![a-z0-9])'+re.escape(v)+r'(?![a-z0-9])', text) for v in options)


def semantic_flags(row, facts, database):
    instructions = row['user_scenario']['instructions']
    text = instructions['task_instructions'].lower()
    goal_text = re.split(r'\*\*details|\*\*constraints', text)[0]
    if '**user goals' not in text:
        goal_text = re.split(r'\n\s*\n(?:user behavior|authentication|behavior|key expectations|key flow|the agent)', goal_text)[0]
    positive_goals = re.sub(r"\b(?:do(?:es)? not|doesn't|don't|never)\s+(?:want to |request any |request |wish to )?(?:cancel|exchange|return|modify)\b[^.\n]*", '', goal_text)
    issues, evidence = [], []
    writes = facts.get('writes', [])
    owner = facts.get('identity', {}).get('owner')
    if not owner:
        return ['semantic_owner_unresolved'], []
    user = database['users'][owner]
    orders = [database['orders'][o] for o in user['orders']]
    pending = [o for o in orders if o['status'] == 'pending']
    for match in re.finditer(r'(?:name|full name)\s*(?:plus|\+|and|with)\s*(?:new\s+)?zip(?: code)?[^.\n]{0,40}(\d{5})', text):
        if match.group(1) != user['address']['zip']:
            issues.append('scenario_prose_authentication_zip_conflicts_with_snapshot')
            evidence.append({'claimed_authentication_zip': match.group(1), 'actual_zip': user['address']['zip']})
    if re.search(r'\bcomplain|investigate the courier|compensat', goal_text):
        issues.append('explicit_escalation_or_compensation_goal_not_scored_by_database')
    if re.search(r'after[^.\n]{0,80}(?:process|complete)[^.\n]{0,80}exchange[^.\n]{0,120}(?:realiz|mistake|instead)', text):
        issues.append('post_execution_exchange_target_change_requires_review')
    # 明确陈述初始订单数量与快照矛盾；不以事后模型行为判断任务质量。
    for number, count in [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('1', 1), ('2', 2), ('3', 3), ('4', 4)]:
        if re.search(r'(?:has|have)\s+'+number+r'\s+orders total', text) and len(orders) != count:
            issues.append('claimed_total_order_count_conflicts_with_snapshot')
            evidence.append({'claim': count, 'actual_total': len(orders)})
        for status in ['pending', 'delivered']:
            actual = sum(o['status'] == status for o in orders)
            if re.search(r'(?:has|have)\s+(?:all\s+)?'+number+r'\s+'+status+r' orders', text) and actual != count:
                issues.append('claimed_'+status+'_order_count_conflicts_with_snapshot')
                evidence.append({'claim': count, 'actual_'+status: actual})
    # 顾客要求一种可执行业务，但金标准完全缺少它，交由复核或隔离。
    patterns = {'return_delivered_order_items': r'\breturn\b',
                'exchange_delivered_order_items': r'\bexchange\b',
                'cancel_pending_order': r'\bcancel\b',
                'modify_pending_order_payment': r'\b(?:payment method|payment to|pay with)\b'}
    names = {w['tool'] for w in writes}
    for name, pattern in patterns.items():
        if re.search(pattern, positive_goals) and name not in names:
            issues.append('requested_business_without_reference_requires_review:'+name)
    item_request = re.search(r'(?:modify|change|swap)[^.\n]{0,160}(?:items?|sizes?|specs?|specifications|variant)', positive_goals)
    illegal_product_change = re.search(r'(?:different product type|completely different product)[^.\n]{0,160}(?:reject|not allow)|(?:reject|not allow)[^.\n]{0,160}(?:different product type|completely different product)', text)
    if item_request and not illegal_product_change and not {'modify_pending_order_items', 'exchange_delivered_order_items'} & names:
        issues.append('item_modification_goal_without_reference_requires_review')
    if re.search(r'(?:update|change|fix|modify)[^.\n]*(?:default|profile|user)[^.\n]*address', positive_goals) and 'modify_user_address' not in names:
        issues.append('profile_address_goal_without_reference_requires_review')
    address_request = re.search(r'orders?[^.\n]{0,60}address|address[^.\n]{0,60}(?:pending|delivered) order', positive_goals)
    if re.search(r'not asking to change any orders|only (?:her|his|their|the) profile address', positive_goals):
        address_request = None
    explicit_refusal = ('delivered' in positive_goals or 'processed' in positive_goals) and re.search(
        r'(?:delivered|processed|shipped)[^.\n]{0,180}(?:refus|reject|cannot|not allow)|(?:cannot|refus|reject)[^.\n]{0,120}(?:delivered|processed)', text)
    if address_request and not explicit_refusal and 'modify_pending_order_address' not in names:
        issues.append('order_address_goal_without_reference_requires_review')
    all_payment = re.search(r'payment[^.\n]{0,70}(?:all (?:their |the )?pending orders|both pending orders)|(?:all (?:their |the )?pending orders|both pending orders)[^.\n]{0,70}payment', positive_goals)
    if all_payment and 'modify_pending_order_payment' in names:
        targets = {w['arguments']['order_id'] for w in writes if w['tool']=='modify_pending_order_payment'}
        cancellations = {w['arguments']['order_id'] for w in writes if w['tool']=='cancel_pending_order'}
        required = {o['order_id'] for o in pending} - cancellations
        if not required <= targets or not targets <= {o['order_id'] for o in pending}:
            issues.append('all_pending_payment_goal_incomplete_requires_review')
            evidence.append({'payment_targets': sorted(targets), 'initial_pending': sorted(o['order_id'] for o in pending)})
    for change in facts.get('changes', []):
        order = database['orders'][change['order_id']]
        old_item = next(i for i in order['items'] if i['item_id'] == change['old_id'])
        variants = database['products'][old_item['product_id']]['variants']
        specified = {k:v for k,v in change['new'].items()
                     if v != change['old'].get(k) and has_value(text, v)}
        candidates = [k for k,v in variants.items() if v['available'] and k != change['old_id']
                      and all(v['options'].get(a) == b for a,b in specified.items())]
        missing = {k:v for k,v in change['new'].items()
                   if v != change['old'].get(k) and not has_value(text, v)}
        if missing and specified and candidates == [change['new_id']]:
            evidence.append({'order_id': change['order_id'], 'product': change['product'],
                             'unique_available_variant_for_explicit_options': specified})
            missing = {}
        if missing:
            issues.append('replacement_options_not_explicitly_grounded_requires_review')
            evidence.append({'order_id': change['order_id'], 'product': change['product'], 'ungrounded_options': missing})
        for key in ['size', 'style', 'material', 'color']:
            if re.search(r'\bsame\s+'+key+r'\b|keeping\s+(?:the\s+)?same\s+'+key, text) and change['old'].get(key) != change['new'].get(key):
                issues.append('explicit_same_'+key+'_violated')
        if re.search(r'\b(?:different color|different size|different variant|cheaper|bigger|larger|smaller|fewer pieces|easier|rgb or white)\b', text) and missing:
            issues.append('open_variant_choice_with_single_gold_requires_review')
        for key in ['color', 'size', 'backlight', 'difficulty level']:
            values = {str(v['options'][key]).lower() for v in variants.values() if key in v['options']}
            # 明确允许两个规格时，即使金标准规格在原文出现也不能自动视为唯一答案。
            for a in values:
                for b in values - {a}:
                    if len(a) > 1 and len(b) > 1 and re.search(r'\b'+re.escape(a)+r'\s+or\s+'+re.escape(b)+r'\b', text):
                        issues.append('explicit_alternative_variants_with_single_gold_requires_review')
    if re.search(r'\b(?:already|previously)\s+(?:been\s+)?modified|modification_history', text):
        # 官方快照没有相应历史时，模型无法从工具获知叙述中要求的先前修改。
        matching = [o for o in orders if o['status'].startswith('pending')]
        if matching and not any(o.get('modification_history') or o['status'] != 'pending' for o in matching):
            issues.append('claimed_prior_modification_not_in_snapshot')
    if re.search(r'change (?:it |them |the payment method )?back|switch back|revert.*(?:payment|credit card)', text) and any(w['tool']=='modify_pending_order_payment' for w in writes):
        if sum(w['tool']=='modify_pending_order_payment' for w in writes) == 1:
            issues.append('payment_revert_request_not_represented_requires_review')
    # 隐藏标准仅检查写结果，无法确认纯咨询或拒绝行为；记录边界，不伪造新奖励。
    return sorted(set(issues)), evidence


def main():
    facts = json.loads((OUT/'facts.json').read_text())
    by_id = {x['id']:x for x in facts['cases']}
    dbs = {p.name:json.loads(p.read_text()) for p in (ROOT/'data/databases').glob('*.json')}
    result = []
    for side in ['train','dev','test']:
        for line in (ROOT/f'data/versions/retail_full_v2/processed/rl_{side}.jsonl').read_text().splitlines():
            row=json.loads(line);case=by_id[row['id']]
            flags,evidence=semantic_flags(row,case,dbs[Path(row['db_path']).name])
            result.append({'id':row['id'],'split':side,'flags':flags,'evidence':evidence})
    (OUT/'semantic_checks.json').write_text(json.dumps({'method':'fixed_prose_and_snapshot_contracts_not_complete_natural_language_proof',
        'counts':dict(Counter(f for r in result for f in r['flags'])),'cases':result},ensure_ascii=False,indent=2)+'\n')
    print('semantic checks',len(result),'flagged',sum(bool(r['flags']) for r in result))


if __name__ == '__main__':main()
