"""Causal contracts: caching, feedback timing, all-layer KV retention."""
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .native import full_margin, stream


class SequentialResponseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, attention_dropout=0., _attn_implementation='sdpa')).eval()
        self.prefix = [1, 2, 3, 4]
        self.continuation = [5, 6, 7, 8, 9]
        self.probe = dict(layer=0, head=1, receiver=3)

    def run_stream(self, dose, **kwargs):
        return stream(self.model, self.prefix, self.continuation, [5, 6],
                      self.probe, [0, 1], dose, **kwargs)

    def test_cache_matches_full_recomputation_at_each_dose(self):
        for dose in [0., -.5, .5]:
            sequential = self.run_stream(dose)
            full = full_margin(self.model, self.prefix + self.continuation[:-1],
                               3, [5, 6], self.probe, [0, 1], dose)
            np.testing.assert_allclose(sequential['margin'], full, atol=2e-7)
            self.assertEqual(sequential['kv'].shape, (5, 3, 2, 2, 8))

    def test_feedback_cannot_change_its_own_preceding_prediction(self):
        original = self.run_stream(0.)
        changed = self.run_stream(0., first_token=10)
        self.assertEqual(original['margin'][0], changed['margin'][0])
        np.testing.assert_array_equal(original['kv'][0], changed['kv'][0])
        self.assertGreater(np.linalg.norm(original['kv'][1:] - changed['kv'][1:]), .01)

    def test_zero_dose_matches_native_and_restores_model(self):
        tokens = torch.tensor([self.prefix + self.continuation[:-1]])
        with torch.no_grad():
            logits = self.model(tokens).logits[0, 3:]
        expected = (logits[:, 5] - logits[:, 6]).numpy()
        np.testing.assert_allclose(self.run_stream(0.)['margin'], expected, atol=2e-7)
        self.run_stream(.5)
        with torch.no_grad():
            restored = self.model(tokens).logits
        np.testing.assert_array_equal(restored[0, 3:].numpy(), logits.numpy())


if __name__ == '__main__':
    unittest.main()
