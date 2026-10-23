"""MLX采样桥的EOS/概率检查，不调用真实模型或API。"""

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from retail_agent.run_mmopd_local import MLXSampler


class SamplerTests(unittest.TestCase):
    def test_real_final_eos_and_selected_probability_are_kept(self):
        probabilities = Mock()
        probabilities.__getitem__ = Mock(side_effect=lambda i: SimpleNamespace(item=lambda: -float(i)))
        responses = [SimpleNamespace(token=i, logprobs=probabilities) for i in [7, 151645]]
        mlx = SimpleNamespace(stream_generate=Mock(return_value=iter(responses)))
        sample = SimpleNamespace(make_sampler=Mock(return_value='sampler'))
        with patch.dict(sys.modules, {'mlx_lm': mlx, 'mlx_lm.sample_utils': sample}):
            result = MLXSampler('model', 'tokenizer', 1.).generate([2, 3], 64)
        self.assertEqual(result, {'token_ids': [7, 151645], 'logprobs': [-7., -151645.]})
        self.assertEqual(mlx.stream_generate.call_args.kwargs['max_tokens'], 64)

    def test_length_stop_does_not_fabricate_eos(self):
        probability = SimpleNamespace(item=lambda: -1.)
        logits = Mock()
        logits.__getitem__ = Mock(return_value=probability)
        mlx = SimpleNamespace(stream_generate=Mock(return_value=iter([
            SimpleNamespace(token=7, logprobs=logits)])))
        with patch.dict(sys.modules, {'mlx_lm': mlx,
                'mlx_lm.sample_utils': SimpleNamespace(make_sampler=Mock())}):
            self.assertEqual(MLXSampler(None, None, 1.).generate([1], 1)['token_ids'], [7])
