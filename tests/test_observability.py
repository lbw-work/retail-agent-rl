"""计时对账、异常持久化与冻结协议；不进行模型/API 业务评测。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from retail_agent.observability import StageTimer
from retail_agent.experiment_protocol import check_protocol


class ObservationTests(unittest.TestCase):
    def test_nested_self_time_does_not_double_count(self):
        clock = [0.0]
        with patch('retail_agent.observability.time.perf_counter', side_effect=lambda: clock[0]):
            timer = StageTimer()
            with timer.measure('outer'):
                clock[0] = 2
                with timer.measure('inner'):
                    clock[0] = 5
                clock[0] = 10
            report = timer.summary()
        self.assertEqual(report['wall_seconds'], 10)
        self.assertEqual(report['stages']['outer']['inclusive_seconds'], 10)
        self.assertEqual(report['stages']['outer']['self_seconds'], 7)
        self.assertEqual(report['stages']['inner']['self_seconds'], 3)
        self.assertEqual(report['unaccounted_or_active_seconds'], 0)

    def test_failure_is_persisted_without_swallowing_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            timer = StageTimer(temporary)
            with self.assertRaisesRegex(RuntimeError, 'service fault'):
                with timer.measure('customer_reply'):
                    raise RuntimeError('service fault')
            report = json.loads((timer.directory / f'{timer.run_id}.json').read_text())
            events = [json.loads(l) for l in (timer.directory / f'{timer.run_id}.jsonl').read_text().splitlines()]
            self.assertEqual(report['active_stages'], [])
            self.assertEqual(report['stages']['customer_reply']['failed_calls'], 1)
            self.assertEqual(events[-1]['error_type'], 'RuntimeError')
            self.assertNotIn('service fault', json.dumps(events))

    def test_gpu_boundary_hook_only_for_gpu_stages(self):
        sync = Mock()
        timer = StageTimer(synchronize=sync)
        with timer.measure('gpu', gpu=True):
            pass
        with timer.measure('api'):
            pass
        self.assertEqual(sync.call_count, 2)

    def test_protocol_rejects_changed_frozen_input(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input.json';source.write_text('original')
            protocol = root / 'protocol.json'
            protocol.write_text(json.dumps({'version': 'grpo_baseline_v1',
                'frozen_sha256': {'input.json': hashlib.sha256(source.read_bytes()).hexdigest()}}))
            with patch('retail_agent.experiment_protocol.PROTOCOL', protocol), patch('retail_agent.experiment_protocol.ROOT', root):
                self.assertEqual(check_protocol()['protocol']['version'], 'grpo_baseline_v1')
                source.write_text('changed')
                with self.assertRaisesRegex(ValueError, '冻结输入'):
                    check_protocol()

    def test_current_protocol_preserves_reward_and_split_counts(self):
        protocol = check_protocol()['protocol']
        self.assertEqual(protocol['counts'], {'train': 94, 'dev': 16, 'test': 22})
        self.assertEqual(len(set(protocol['development_task_ids'])), 16)
        self.assertEqual(protocol['training_config']['reward_mode'], 'reward_a')
        self.assertEqual(protocol['training_config']['scale_rewards'], 'group')
        self.assertEqual(protocol['training_config']['beta'], 0)
        self.assertEqual(protocol['implemented_evaluation_arms'], ['original', 'sft'])


if __name__ == '__main__':
    unittest.main()
