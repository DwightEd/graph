import unittest

import torch

from experiments.rr_topology_dynamics.attractor import AttractorFeatureExtractor
from experiments.rr_topology_dynamics.routing_state import RoutingState


def _column(features, name):
    return features.values[:, features.names.index(name)]


class AttractorFeatureExtractorTest(unittest.TestCase):
    def test_separates_prompt_concentration_stability_and_grounded_relay(self):
        state = RoutingState(
            prompt_source_mass=torch.tensor(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.0, 0.0],
                ]
            ),
            response_source_mass=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            retained_edge_count=torch.tensor([2.0, 2.0, 1.0]),
        )

        features = AttractorFeatureExtractor(recent_lag_max=1).extract(state)

        self.assertEqual(
            features.names,
            (
                "prompt_attention_share",
                "prompt_source_effective_fraction",
                "prompt_source_top1_share",
                "response_source_effective_fraction",
                "response_source_top1_share",
                "recent_response_share",
                "source_stability",
                "prompt_groundedness",
            ),
        )
        torch.testing.assert_close(
            _column(features, "prompt_attention_share"),
            torch.tensor([1.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(
            _column(features, "prompt_source_effective_fraction"),
            torch.tensor([1.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(
            _column(features, "prompt_source_top1_share"),
            torch.tensor([0.5, 0.5, 0.0]),
        )
        torch.testing.assert_close(
            _column(features, "response_source_effective_fraction"),
            torch.tensor([0.0, 0.0, 0.5]),
        )
        torch.testing.assert_close(
            _column(features, "response_source_top1_share"),
            torch.tensor([0.0, 0.0, 1.0]),
        )
        torch.testing.assert_close(
            _column(features, "recent_response_share"),
            torch.tensor([0.0, 0.0, 1.0]),
        )
        torch.testing.assert_close(
            _column(features, "source_stability"),
            torch.tensor([0.0, 1.0, 0.0]),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            _column(features, "prompt_groundedness"),
            torch.tensor([1.0, 1.0, 1.0]),
        )

        self.assertEqual(
            features.control_names,
            ("retained_attention_mass", "retained_edge_count"),
        )
        torch.testing.assert_close(
            features.controls,
            torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 2.0],
                    [1.0, 1.0],
                ]
            ),
        )

    def test_response_only_chain_remains_ungrounded(self):
        state = RoutingState(
            prompt_source_mass=torch.zeros((3, 1)),
            response_source_mass=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            retained_edge_count=torch.tensor([0.0, 1.0, 1.0]),
        )

        features = AttractorFeatureExtractor().extract(state)

        torch.testing.assert_close(
            _column(features, "prompt_groundedness"), torch.zeros(3)
        )


if __name__ == "__main__":
    unittest.main()
