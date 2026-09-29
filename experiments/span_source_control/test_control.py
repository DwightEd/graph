"""Mathematical and native causal checks needed for interpreting the experiment."""
import unittest
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .measure import centered_effect
from .native import intervene, observe
from .memory import update_memory


class SourceControlTest(unittest.TestCase):
    def test_softmax_gradient(self):
        scores = torch.tensor([.3, -.7, .9], dtype=torch.float64, requires_grad=True)
        values = torch.tensor([2., -3., 1.], dtype=torch.float64)
        attention = scores.softmax(-1)
        output = (attention * values).sum()
        gradient = torch.autograd.grad(output, scores)[0]
        result = centered_effect(attention.detach().numpy(), (attention * values).detach().numpy())
        np.testing.assert_allclose(result, gradient.numpy(), atol=1e-12)
        self.assertAlmostEqual(result.sum(), 0., places=12)

    def test_memory_keeps_signs_and_source_identity(self):
        previous = np.array([[[1., 0.], [0., 0.]]])
        flipped = np.array([[[0., 0.], [1., 0.]]])
        memory, observed = update_memory(previous, flipped, np.array([1.]), np.array([2.]))
        np.testing.assert_allclose(observed, [[1., 0., 1.]])
        np.testing.assert_allclose(memory, [[[.5, 0.], [.5, 0.]]])
        shifted = np.array([[[0., 1.], [0., 0.]]])
        _, observed = update_memory(previous, shifted, np.array([1.]), np.array([2.]))
        np.testing.assert_allclose(observed, [[0., 0., 0.]])

    def test_full_prefix_intervention(self):
        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0., _attn_implementation='sdpa')
        model = LlamaForCausalLM(config).eval()
        prompt, answer = [1, 2, 3], [4, 5, 6, 7, 8]
        tokens = prompt + answer[:-1]
        probe = dict(layer=0, head=1, receiver=4)
        treatment = dict(keys=[0, 1])
        base = observe(model, tokens, len(prompt), answer)
        with intervene(model, probe, treatment, 0.):
            zero = observe(model, tokens, len(prompt), answer, base['alternatives'])
        np.testing.assert_allclose(zero['margin'], base['margin'], atol=1e-6)
        with intervene(model, probe, treatment, .5) as patch:
            changed = observe(model, tokens, len(prompt), answer, base['alternatives'])
            self.assertTrue(0 < patch.mass < 1)
        np.testing.assert_allclose(changed['hidden'][:2], zero['hidden'][:2], atol=0.)
        self.assertGreater(np.linalg.norm(changed['hidden'][3:] - zero['hidden'][3:]), 1e-7)
        restored = observe(model, tokens, len(prompt), answer)
        np.testing.assert_array_equal(restored['margin'], base['margin'])


if __name__ == '__main__':
    unittest.main()
