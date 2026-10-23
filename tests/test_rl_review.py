"""新版 RL 只接收原侧原始行；语义线索不能被执行成功掩盖。"""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from retail_agent.environment import ROOT
from retail_agent.rl_dataset import REVIEW_MANIFEST, load_review, reviewed_rl_rows
from retail_agent.reward import RewardEvaluationError
from retail_agent.training_reward import evaluate_training_task, ready_training_rows, reference_subset, compile_business_goals
sys.path.insert(0,str(ROOT/'scripts'))
from review_rl_semantics import semantic_flags


class RLReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=load_review()
        cls.rows=reviewed_rl_rows(ROOT/cls.manifest['files']['train']['path'],'train')
        cls.source={r['id']:r for side in ['train','dev','test'] for r in map(json.loads,
            (ROOT/f'data/versions/retail_full_v2/processed/rl_{side}.jsonl').read_text().splitlines())}
        cls.facts={c['id']:c for c in json.loads((ROOT/'outputs/review/rl_full_20261015/facts.json').read_text())['cases']}

    def test_all_tasks_have_a_decision_and_preserved_source(self):
        entries=self.manifest['entries']
        self.assertEqual(len(entries),514)
        self.assertEqual({e['id'] for e in entries},set(self.source))
        self.assertTrue(all(e['checks_complete'] and e['decision_basis'] for e in entries))
        for side in ['train','dev','test']:
            rows=reviewed_rl_rows(ROOT/self.manifest['files'][side]['path'],side)
            self.assertTrue(all(r==self.source[r['id']] for r in rows))
        self.assertFalse(self.manifest['sft_changed'])

    def test_dev_cannot_be_loaded_as_train(self):
        with self.assertRaisesRegex(ValueError,'路径'):
            reviewed_rl_rows(ROOT/self.manifest['files']['dev']['path'],'train')

    def test_modified_or_heldout_task_cannot_receive_training_reward(self):
        changed=deepcopy(self.rows[0]);changed['user_scenario']['instructions']['reason_for_call']='changed'
        with self.assertRaisesRegex(RewardEvaluationError,'未逐条复核'):
            evaluate_training_task(changed,[],review_manifest=str(REVIEW_MANIFEST))
        dev=reviewed_rl_rows(ROOT/self.manifest['files']['dev']['path'],'dev')[0]
        with self.assertRaisesRegex(RewardEvaluationError,'未逐条复核'):
            evaluate_training_task(dev,[],review_manifest=str(REVIEW_MANIFEST))

    def test_identity_conflict_is_excluded_from_current_reward_scope(self):
        self.assertNotIn('retail_494',{r['id'] for r in self.rows})
        with self.assertRaisesRegex(RewardEvaluationError,'未逐条复核'):
            evaluate_training_task(self.source['retail_494'],[],review_manifest=str(REVIEW_MANIFEST))

    def test_new_task_real_success_and_query_only_reward(self):
        old={r['id'] for r in ready_training_rows(ROOT/'data/versions/retail_full_v2/reviewed/rl_train_ready.jsonl')}
        row=next(r for r in self.rows if r['id'] not in old)
        goals=set(compile_business_goals(row)['goals'])
        full=evaluate_training_task(row,reference_subset(row,goals).trajectory,review_manifest=str(REVIEW_MANIFEST))
        empty=evaluate_training_task(row,reference_subset(row,set()).trajectory,review_manifest=str(REVIEW_MANIFEST))
        self.assertEqual(full['reward_a'],1)
        self.assertEqual(empty['reward_a'],0)

    def flags(self,identifier):
        row=self.source[identifier]
        db=json.loads((ROOT/'data/databases'/Path(row['db_path']).name).read_text())
        return semantic_flags(row,self.facts[identifier],db)[0]

    def test_explicit_missing_item_goal_is_flagged(self):
        self.assertIn('item_modification_goal_without_reference_requires_review',self.flags('retail_334'))

    def test_denial_and_multi_order_dependencies_are_not_missing_goals(self):
        self.assertNotIn('requested_business_without_reference_requires_review:cancel_pending_order',self.flags('retail_264'))
        self.assertNotIn('all_pending_payment_goal_incomplete_requires_review',self.flags('retail_196'))
        self.assertNotIn('order_address_goal_without_reference_requires_review',self.flags('retail_381'))

    def test_missing_payment_and_post_exchange_goal_are_flagged(self):
        self.assertIn('all_pending_payment_goal_incomplete_requires_review',self.flags('retail_181'))
        self.assertIn('post_execution_exchange_target_change_requires_review',self.flags('retail_479'))


if __name__=='__main__':unittest.main()
