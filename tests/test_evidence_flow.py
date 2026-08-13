import unittest

import numpy as np
import torch

from attention_graph.evidence_flow import (
    anomaly_components,
    evidence_flow_fields,
    fixed_head_projection,
)


class EvidenceFlowTests(unittest.TestCase):
    def test_fixed_head_projection_is_reproducible_and_orthonormal(self):
        first = fixed_head_projection(8, 4, 19)
        second = fixed_head_projection(8, 4, 19)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.T @ first, torch.eye(4), atol=1e-6, rtol=1e-6)

    def test_weak_path_cannot_transmit_a_full_conditional_residual(self):
        lookback = torch.tensor([[[0.0]], [[1.0]]])
        projection = torch.ones((1, 1))

        def field(weight):
            route = {
                "layer": torch.tensor([0]),
                "source": torch.tensor([1]),
                "target": torch.tensor([2]),
                "weight": torch.tensor([weight]),
            }
            return evidence_flow_fields(
                lookback, route, prompt_count=1, prompt_bins=1,
                head_projection=projection, sample_id="weak", seed=1,
            )[1]

        weak = field(1e-8)
        strong = field(1.0)
        self.assertAlmostEqual(float(weak[1, 0]), 1e-8, places=12)
        self.assertAlmostEqual(float(strong[1, 0]), 1.0, places=6)

    def test_two_hop_filter_retains_path_products_and_signed_innovations(self):
        lookback = torch.tensor([[[.2]], [[.6]], [[1.0]]])
        # One prompt edge per response plus the chain R0 -> R1 -> R2.
        route = {
            "layer": torch.zeros(5, dtype=torch.long),
            "source": torch.tensor([0, 0, 0, 1, 2]),
            "target": torch.tensor([1, 2, 3, 2, 3]),
            "weight": torch.tensor([.5, .2, .1, .8, .5]),
        }
        projection = torch.ones((1, 1))
        direct, propagation, diagnostics = evidence_flow_fields(
            lookback, route, prompt_count=1, prompt_bins=1,
            head_projection=projection, sample_id="chain", seed=7,
        )
        torch.testing.assert_close(
            direct,
            torch.tensor([
                [.5, 0.0], [.2, float(np.log1p(.8))],
                [.1, float(np.log1p(.5))],
            ]),
            atol=1e-6, rtol=1e-6,
        )
        # Schema: node local, node scale, prompt local, prompt scale, log mass2.
        torch.testing.assert_close(
            propagation[2],
            torch.tensor([.2, .16, -.05, -.12, float(np.log1p(.4))]),
            atol=1e-6, rtol=1e-6,
        )
        self.assertAlmostEqual(float(diagnostics["rr_mass_hop2"][2, 0]), .4, places=6)
        self.assertFalse(bool(diagnostics["reachable_hop1"][0, 0]))

    def test_randomized_topology_is_deterministic_and_preserves_direct_field(self):
        lookback = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1) / 4
        route = {
            "layer": torch.zeros(8, dtype=torch.long),
            "source": torch.tensor([0, 0, 0, 0, 1, 2, 2, 4]),
            "target": torch.tensor([1, 2, 4, 5, 2, 4, 5, 5]),
            "weight": torch.tensor([.4, .3, .2, .1, .8, .6, .5, .7]),
        }
        projection = fixed_head_projection(1, 1, 11)
        direct, true_field, true_diagnostics = evidence_flow_fields(
            lookback, route, prompt_count=1, prompt_bins=2,
            head_projection=projection, sample_id="random-null", seed=11,
        )
        randomized_direct, first, randomized_diagnostics = evidence_flow_fields(
            lookback, route, prompt_count=1, prompt_bins=2,
            head_projection=projection, sample_id="random-null", seed=11,
            randomize_rr=True,
        )
        _, second, _ = evidence_flow_fields(
            lookback, route, prompt_count=1, prompt_bins=2,
            head_projection=projection, sample_id="random-null", seed=11,
            randomize_rr=True,
        )
        torch.testing.assert_close(direct, randomized_direct)
        torch.testing.assert_close(
            true_diagnostics["rr_mass_hop1"],
            randomized_diagnostics["rr_mass_hop1"],
        )
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(true_field, first))

    def test_anomaly_components_use_rr_connectivity_not_token_adjacency(self):
        route = {
            "source": np.asarray([2, 4, 3]),
            "target": np.asarray([4, 5, 5]),
        }
        active, component = anomaly_components(
            route, prompt_count=2,
            scores=np.asarray([.99, .98, .10, .97]), threshold=.95,
        )
        np.testing.assert_array_equal(active, [True, True, False, True])
        # R0 -> R2 connects token 0 only to token 2 (inactive); R1 -> R3 connects 1 and 3.
        self.assertNotEqual(int(component[0]), int(component[1]))
        self.assertEqual(int(component[1]), int(component[3]))
        self.assertEqual(int(component[2]), -1)


if __name__ == "__main__":
    unittest.main()
