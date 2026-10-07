"""Toy causal-attention checks; these are not an 8B or hallucination experiment."""
import unittest

import torch

from .response_decomposition import decompose_response, matrix_readout


def attention(states, query_weight, key_weight, value_weight):
    query = states @ query_weight
    key = states @ key_weight
    value = states @ value_weight
    scores = query @ key.T / states.shape[-1] ** .5
    causal = torch.ones_like(scores, dtype=torch.bool).tril()
    weights = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    return weights @ value, weights.diagonal()


def toy_forward(states, weights, perturbation, sender, cut, observation='residual'):
    """Two native attention layers; all later Q/K/V recomputed after the cut."""
    diagonals = []
    for layer in range(2):
        message, diagonal = attention(states, *weights[layer])
        if layer == cut:
            injected = torch.zeros_like(message)
            injected[sender] = perturbation
            message = message + injected
        states = states + message
        diagonals.append(diagonal)
    if observation == 'self_attention':
        return torch.stack(diagonals, dim=-1)
    return states


class NativeResponseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.states = torch.randn(5, 4, dtype=torch.float64)
        self.weights = torch.randn(2, 3, 4, 4, dtype=torch.float64) * .2
        self.direction = torch.randn(4, dtype=torch.float64)
        self.zero = torch.zeros(4, dtype=torch.float64)
        self.readout = torch.randn(4, 4, dtype=torch.float64)

    def test_full_score_decomposes_without_freezing_intermediate_nodes(self):
        forward = lambda perturbation: toy_forward(
            self.states, self.weights, perturbation, sender=1, cut=0)
        _, response = torch.func.jvp(forward, (self.zero,), (self.direction,))
        parts = decompose_response(response, self.readout, sender=1, receiver=4)
        score = lambda perturbation: matrix_readout(forward(perturbation), self.readout, receiver=4)
        _, score_response = torch.func.jvp(score, (self.zero,), (self.direction,))
        torch.testing.assert_close(parts['total'], score_response)
        torch.testing.assert_close(parts['total'], parts['direct'] + parts['transport'])
        self.assertGreater(float(parts['transport'].abs()), 1e-8)
        self.assertGreater(float((parts['transport'] - parts['endpoint']).abs()), 1e-8)
        torch.testing.assert_close(response[:1], torch.zeros_like(response[:1]))

    def test_last_attention_output_has_direct_shortcut_but_no_transport(self):
        forward = lambda perturbation: toy_forward(
            self.states, self.weights, perturbation, sender=1, cut=1)
        _, response = torch.func.jvp(forward, (self.zero,), (self.direction,))
        parts = decompose_response(response, self.readout, sender=1, receiver=4)
        self.assertGreater(float(parts['direct'].abs()), 1e-8)
        torch.testing.assert_close(parts['transport'], torch.zeros_like(parts['transport']))

    def test_self_attention_observation_precedes_its_output_cut(self):
        forward = lambda perturbation: toy_forward(self.states, self.weights, perturbation,
            sender=1, cut=1, observation='self_attention')
        _, response = torch.func.jvp(forward, (self.zero,), (self.direction,))
        torch.testing.assert_close(response, torch.zeros_like(response))

    def test_finite_difference_and_jvp_vjp_duality(self):
        forward = lambda perturbation: toy_forward(
            self.states, self.weights, perturbation, sender=1, cut=0)
        _, response = torch.func.jvp(forward, (self.zero,), (self.direction,))
        epsilon = 1e-5
        finite = (forward(epsilon * self.direction) - forward(-epsilon * self.direction)) / (2 * epsilon)
        torch.testing.assert_close(response, finite, rtol=1e-7, atol=1e-9)
        _, pullback = torch.func.vjp(forward, self.zero)
        target_direction = torch.randn_like(response)
        reverse = pullback(target_direction)[0]
        torch.testing.assert_close((target_direction * response).sum(),
                                   (reverse * self.direction).sum())


if __name__ == '__main__':
    unittest.main()
