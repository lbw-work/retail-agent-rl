"""独立验收 RL v3 的投影、划分与真实奖励反例；无模型/API/训练。"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from retail_agent.dataset import WRITES
from retail_agent.rl_dataset import REVIEW_MANIFEST, file_digest, load_review, reviewed_rl_rows
from retail_agent.training_reward import compile_business_goals, evaluate_training_task, reference_subset, row_fingerprint
from retail_agent.reward import evaluate_task
from tau2.data_model.message import AssistantMessage, ToolMessage


def reward_probe(item):
    side,row=item
    spec=compile_business_goals(row)
    full=reference_subset(row,set(spec['goals']))
    empty=reference_subset(row,set())
    assert evaluate_task(row,full.trajectory).reward==1
    assert evaluate_task(row,empty.trajectory).reward==0
    result={'id':row['id'],'split':side,'reference_reward':1,'query_only_reward':0,
            'reachable_subsets':len(spec['variants'])}
    if side=='train':
        score=evaluate_training_task(row,full.trajectory,review_manifest=str(REVIEW_MANIFEST))
        assert score['reward_a']==score['reward_b']==score['progress']==1, (row['id'],score)
        zero=evaluate_training_task(row,empty.trajectory,review_manifest=str(REVIEW_MANIFEST))
        assert zero['reward_a']==zero['reward_b']==zero['progress']==0
        history=[];omit_ids=set()
        for message in full.trajectory:
            if isinstance(message,AssistantMessage) and message.tool_calls and message.tool_calls[0].name.startswith('find_user_id_'):
                omit_ids.add(message.tool_calls[0].id);continue
            if isinstance(message,ToolMessage) and message.id in omit_ids:continue
            history.append(message)
        denied=evaluate_training_task(row,history,review_manifest=str(REVIEW_MANIFEST))
        assert denied['gate']==denied['reward_a']==denied['reward_b']==0
        result.update(full_training_reward=1,query_only_training_reward=0,unauthenticated_training_reward=0)
    return result


def main():
    manifest=load_review();accepted={};groups={s:set() for s in ['train','dev','test']}
    for side in groups:
        path=ROOT/manifest['files'][side]['path'];rows=reviewed_rl_rows(path,side)
        accepted[side]=rows;groups[side]={r['group'] for r in rows}
        rejected_file=manifest['files']['quarantine_'+side]
        rejected_path=ROOT/rejected_file['path']
        assert file_digest(rejected_path)==rejected_file['sha256']
        rejected=[json.loads(l) for l in rejected_path.read_text().splitlines()]
        source={r['id']:r for r in map(json.loads,(ROOT/manifest['source_files'][side]).read_text().splitlines())}
        assert len(rejected)==rejected_file['count']
        combined=rows+rejected
        assert len(combined)==len(source) and {r['id'] for r in combined}==set(source)
        assert all(r==source[r['id']] for r in combined)
        assert {r['id'] for r in rows}.isdisjoint(r['id'] for r in rejected)
        expected={e['id'] for e in manifest['entries'] if e['split']==side and e['status']=='quarantined'}
        assert {r['id'] for r in rejected}==expected
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not groups[a]&groups[b]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for item in pool.map(reward_probe,[(s,r) for s,rs in accepted.items() for r in rs],chunksize=1):
            results.append(item)
            if len(results)%10==0:print('新版本奖励验收',len(results),flush=True)
    # 来源/SFT/环境直到执行完毕仍须保持同一哈希。
    load_review()
    report={'status':'passed','version':manifest['version'],'review_manifest_sha256':file_digest(REVIEW_MANIFEST),
        'all_514_tasks_accounted':len(manifest['entries']), 'accepted_counts':manifest['accepted_counts'],
        'quarantined_counts':manifest['quarantined_counts'],'original_rows_and_split_preserved':True,
        'sft_and_v2_immutable_hashes_verified':True,'cross_side_source_groups':0,
        'reference_and_query_counterexamples':len(results)*2,
        'train_reward_counterexamples':len(accepted['train'])*3,
        'model_called':False,'customer_api_called':False,'optimizer_steps':0,'paid_cost_cny':0,
        'cases':results}
    path=REVIEW_MANIFEST.parent.parent/'reports/verification.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
