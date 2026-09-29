"""Prediction alignment, source shielding and causal history-reset invariants."""
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .native import full_states, reset_states
from .readout import distribution_scores, surprise_tail


class TokenEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, attention_dropout=0., _attn_implementation='sdpa')).eval()
        self.prompt = [1, 2, 3, 4, 5]
        self.answer = [6, 7, 8, 9, 10, 11]

    def test_masked_source_cannot_leak_through_prompt_cache(self):
        mask = [1, 0, 0, 1, 1]
        original, _ = full_states(self.model, self.prompt, self.answer, mask)
        changed, _ = full_states(self.model, [1, 30, 31, 4, 5], self.answer, mask)
        np.testing.assert_array_equal(original.numpy(), changed.numpy())

    def test_full_history_and_reset_identical_before_window_fills(self):
        mask = [1] * 5
        full, cache = full_states(self.model, self.prompt, self.answer, mask)
        for target in range(5):
            reset = reset_states(self.model, self.prompt, self.answer, mask, cache, [target], 4)
            np.testing.assert_allclose(reset.cpu().numpy(), full[target:target + 1].numpy(), atol=1e-6)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_batch_isolation_and_no_future_tokens(self):
        mask = [1, 0, 0, 1, 1]
        _, cache = full_states(self.model, self.prompt, self.answer, mask)
        batched = reset_states(self.model, self.prompt, self.answer, mask, cache, [3, 4, 5], 2)
        individual = torch.cat([reset_states(self.model, self.prompt, self.answer, mask, cache, [t], 2)
                                for t in [3, 4, 5]])
        np.testing.assert_allclose(batched.cpu().numpy(), individual.cpu().numpy(), atol=1e-6)
        altered = self.answer[:4] + [40, 41]
        original = reset_states(self.model, self.prompt, self.answer, mask, cache, [4], 2)
        changed = reset_states(self.model, self.prompt, altered, mask, cache, [4], 2)
        np.testing.assert_array_equal(original.cpu().numpy(), changed.cpu().numpy())

    def test_tail_exact_mass_and_rare_token_numerics(self):
        logits = torch.tensor([[.5, .25, .25]], dtype=torch.float64).log()
        expected = -np.log([.75, .25, .25])
        actual = [float(surprise_tail(logits, torch.tensor([target]))) for target in range(3)]
        np.testing.assert_allclose(actual, expected)
        rare = surprise_tail(torch.tensor([[0., -1000.]]), torch.tensor([1]))
        self.assertTrue(torch.isfinite(rare).all())
        self.assertGreater(float(rare), 1000)

    def test_no_source_change_gives_constant_ratio_tail(self):
        logits = torch.randn(4, 64)
        result, _ = distribution_scores(logits, logits, torch.tensor([1, 2, 3, 4]))
        np.testing.assert_allclose(result['source_tail'].numpy(), np.log(2), atol=1e-6)
        np.testing.assert_allclose(result['cad_tail'].numpy(), result['confidence_tail'].numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
