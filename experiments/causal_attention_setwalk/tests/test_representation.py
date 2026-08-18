import unittest

import numpy as np
import torch

from experiments.causal_attention_setwalk.representation import (
    SetWalkConfig,
    _dct_basis,
    _head_pool,
    _propagate_view,
)


class SetWalkRepresentationTests(unittest.TestCase):
    def test_head_pool_is_permutation_invariant(self):
        values = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.5, 0.5]],
            ]
        )
        weights = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.25, 0.25]])
        first = _head_pool(values, weights, 1e-8)
        permutation = torch.tensor([2, 0, 1])
        second = _head_pool(values[permutation], weights[permutation], 1e-8)
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left, right)

    def test_two_hops_reach_a_nonadjacent_response_source(self):
        # Three layers, one head, three response tokens, one scalar set feature.
        base = torch.tensor(
            [
                [[[2.0], [4.0], [8.0]]],
                [[[1.0], [1.0], [1.0]]],
                [[[1.0], [1.0], [1.0]]],
            ]
        )
        mass = torch.ones((3, 1, 3))
        empty = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0)
        edges = [
            (empty, empty, empty, empty_weight),
            (
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([1.0]),
            ),
            (
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([1]),
                torch.tensor([1.0]),
            ),
        ]
        _, state = _propagate_view(
            base,
            mass,
            edges,
            np.arange(3),
            SetWalkConfig(fourier_features=1, dct_components=1),
        )
        self.assertAlmostEqual(float(state["hop1_survival"][2, 2]), 1.0)
        self.assertAlmostEqual(float(state["hop2_survival"][2, 2]), 1.0)

    def test_layer_permutation_changes_ordered_walk(self):
        base = torch.arange(1, 7, dtype=torch.float32).reshape(3, 1, 2, 1)
        mass = torch.ones((3, 1, 2))
        empty = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0)
        edges = [(empty, empty, empty, empty_weight)] * 3
        config = SetWalkConfig(fourier_features=1, dct_components=2)
        ordered, _ = _propagate_view(base, mass, edges, np.arange(3), config)
        shuffled, _ = _propagate_view(
            base, mass, edges, np.asarray([2, 0, 1]), config
        )
        self.assertFalse(torch.allclose(ordered, shuffled))

    def test_dct_basis_is_orthonormal(self):
        basis = _dct_basis(3, 3, device="cpu", dtype=torch.float32)
        torch.testing.assert_close(basis @ basis.T, torch.eye(3), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

