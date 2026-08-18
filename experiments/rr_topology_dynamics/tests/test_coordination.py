import unittest

import torch

from experiments.rr_topology_dynamics.features import (
    _layer_source_profiles,
    _relation_coordination_profiles,
)


class CoordinationProfilesTest(unittest.TestCase):
    def test_exact_source_concentration_coalesces_head_repetitions(self):
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[2, 2, 2], [0, 0, 1]]),
            torch.tensor([0.2, 0.3, 0.5]),
            size=(3, 3),
        ).coalesce()
        profiles = _layer_source_profiles(
            sparse.indices()[0],
            sparse.indices()[1],
            sparse.values(),
            response_count=3,
            layers=1,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(float(profiles["top1_share"][2, 0]), 0.5)
        self.assertAlmostEqual(float(profiles["effective_number"][2, 0]), 2.0)
        self.assertEqual(float(profiles["mass"][0, 0]), 0.0)

    def test_relation_set_signature_is_head_permutation_invariant(self):
        prompt = torch.tensor(
            [[[0.8, 0.2], [0.7, 0.1], [0.1, 0.1]]], dtype=torch.float32
        )
        rr = torch.tensor(
            [[[0.2, 0.8], [0.3, 0.9], [0.9, 0.9]]], dtype=torch.float32
        )
        recent = torch.tensor(
            [[[0.1, 0.4], [0.2, 0.6], [0.9, 0.9]]], dtype=torch.float32
        )
        effective = torch.tensor([[2.0, 2.0, 1.0]])
        top1 = torch.tensor([[0.5, 0.5, 1.0]])
        original = _relation_coordination_profiles(
            prompt, rr, recent, effective, top1, epsilon=1e-8
        )

        permutation = torch.tensor([1, 0])
        permuted = _relation_coordination_profiles(
            prompt[:, :, permutation],
            rr[:, :, permutation],
            recent[:, :, permutation],
            effective,
            top1,
            epsilon=1e-8,
        )
        for name in original:
            torch.testing.assert_close(original[name], permuted[name])

    def test_local_collapse_retains_layer_of_onset(self):
        prompt = torch.tensor([[[0.0], [1.0], [1.0]]])
        rr = torch.tensor([[[1.0], [0.0], [0.0]]])
        recent = rr.clone()
        effective = torch.tensor([[1.0, 0.0, 0.0]])
        top1 = torch.tensor([[1.0, 0.0, 0.0]])
        profiles = _relation_coordination_profiles(
            prompt, rr, recent, effective, top1, epsilon=1e-8
        )
        self.assertAlmostEqual(float(profiles["early_local_rr_collapse"][0]), 1.0)
        self.assertAlmostEqual(float(profiles["late_local_rr_collapse"][0]), 0.0)
        self.assertAlmostEqual(float(profiles["local_rr_collapse_depth"][0]), 0.0)
        torch.testing.assert_close(
            profiles["layer_local_rr_collapse"], torch.tensor([[1.0, 0.0, 0.0]])
        )

    def test_true_layer_adjacency_is_compared_with_all_layer_pairs(self):
        prompt = torch.tensor([[[1.0], [0.5], [0.0]]])
        rr = torch.tensor([[[0.0], [0.5], [1.0]]])
        recent = rr.clone()
        effective = torch.tensor([[0.0, 1.0, 1.0]])
        top1 = torch.tensor([[0.0, 1.0, 1.0]])
        profiles = _relation_coordination_profiles(
            prompt, rr, recent, effective, top1, epsilon=1e-8
        )
        self.assertLess(
            float(profiles["cross_layer_adjacency_gap_vs_all_pairs"][0]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
