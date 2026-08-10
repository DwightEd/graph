import math
import unittest

import torch

from descriptors import temporal_summary, token_routing_features


class DescriptorTests(unittest.TestCase):
    def test_routing_features_match_hand_calculated_relation_weights(self):
        graph = {
            "num_nodes": 5,
            "response_idx": 2,
            "edge_index": torch.tensor(
                [[0, 1, 0, 2, 3, 2], [2, 2, 3, 3, 4, 4]]
            ),
            "edge_type": torch.tensor([0, 0, 0, 1, 1, 1]),
            "edge_weight": torch.tensor([2.0, 2.0, 1.0, 3.0, 2.0, 2.0]),
        }

        features = token_routing_features(graph, num_channels=2)
        entropy = -0.25 * math.log(0.25) / math.log(2) - 0.75 * math.log(0.75) / math.log(2)
        expected = torch.tensor(
            [[4.0, 1.0, 1.0, 0.0], [4.0, 0.25, entropy, 0.5], [4.0, 0.0, 1.0, 0.75]]
        )
        torch.testing.assert_close(features, expected)

    def test_sparse_channel_values_are_averaged_over_all_channels(self):
        graph = {
            "num_nodes": 5,
            "response_idx": 2,
            "edge_index": torch.tensor(
                [[0, 1, 0, 2, 3, 2], [2, 2, 3, 3, 4, 4]]
            ),
            "edge_type": torch.tensor([0, 0, 0, 1, 1, 1]),
            "edge_ptr": torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            "edge_channel": torch.zeros(6, dtype=torch.int32),
            "edge_value": torch.tensor([4.0, 4.0, 2.0, 6.0, 4.0, 4.0]),
        }

        sparse_features = token_routing_features(graph, num_channels=2)
        weighted_graph = {**graph, "edge_weight": torch.tensor([2.0, 2.0, 1.0, 3.0, 2.0, 2.0])}
        torch.testing.assert_close(sparse_features, token_routing_features(weighted_graph, num_channels=2))

    def test_empty_incoming_rows_are_finite_and_zero(self):
        graph = {
            "num_nodes": 5,
            "response_idx": 2,
            "edge_index": torch.tensor([[0], [2]]),
            "edge_type": torch.tensor([0]),
            "edge_weight": torch.tensor([1.0]),
        }

        features = token_routing_features(graph, num_channels=1)
        self.assertEqual(features.shape, (3, 4))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[1:], torch.zeros((2, 4)))

    def test_routing_features_reject_invalid_graph_edges(self):
        graph = {
            "num_nodes": 4,
            "response_idx": 2,
            "edge_index": torch.tensor([[0], [2]]),
            "edge_weight": torch.tensor([1.0]),
        }
        invalid_edges = (
            ("endpoint below zero", torch.tensor([[-1], [2]])),
            ("endpoint beyond node count", torch.tensor([[0], [4]])),
            ("target in prompt", torch.tensor([[0], [1]])),
            ("source is not earlier than target", torch.tensor([[2], [2]])),
        )

        for description, edge_index in invalid_edges:
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    token_routing_features({**graph, "edge_index": edge_index}, num_channels=1)

    def test_routing_features_reject_invalid_edge_weights(self):
        graph = {
            "num_nodes": 3,
            "response_idx": 1,
            "edge_index": torch.tensor([[0], [1]]),
        }
        invalid_weights = (float("nan"), float("inf"), -1.0)

        for value in invalid_weights:
            with self.subTest(weight=value):
                with self.assertRaises(ValueError):
                    token_routing_features({**graph, "edge_weight": torch.tensor([value])}, num_channels=1)
                with self.assertRaises(ValueError):
                    token_routing_features(
                        {**graph, "edge_ptr": torch.tensor([0, 1]), "edge_value": torch.tensor([value])},
                        num_channels=1,
                    )

    def test_temporal_summary_matches_population_statistics_and_ols_slope(self):
        entropy = -0.25 * math.log(0.25) / math.log(2) - 0.75 * math.log(0.75) / math.log(2)
        features = torch.tensor([[4.0, 1.0, 1.0, 0.0], [4.0, 0.25, entropy, 0.5], [4.0, 0.0, 1.0, 0.75]])

        summary = temporal_summary(features)
        expected_mean = torch.tensor([4.0, 5.0 / 12.0, (2.0 + entropy) / 3.0, 5.0 / 12.0])
        expected_std = features.std(dim=0, correction=0)
        expected_slope = torch.tensor([0.0, -1.0, 0.0, 0.75])
        torch.testing.assert_close(summary, torch.cat((expected_mean, expected_std, expected_slope)))

    def test_temporal_summary_is_finite_for_one_response_token(self):
        features = torch.tensor([[2.0, 1.0, 0.0, 0.0]])

        summary = temporal_summary(features)
        self.assertEqual(summary.shape, (12,))
        self.assertTrue(torch.isfinite(summary).all())
        torch.testing.assert_close(summary, torch.tensor([2.0, 1.0, 0.0, 0.0] + [0.0] * 8))

    def test_temporal_summary_rejects_nonfinite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    temporal_summary(torch.tensor([[value, 0.0]]))


if __name__ == "__main__":
    unittest.main()
