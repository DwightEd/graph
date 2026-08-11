import unittest

import torch

from behavior import (
    BEHAVIOR_FEATURE_NAMES,
    align_error_onsets,
    centered_window,
    positive_mask,
    summarize_run_windows,
    token_behavior_features,
    validate_positive_runs,
)


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "num_nodes": 5,
            "response_idx": 2,
            "edge_index": torch.tensor([[0, 1, 0, 2, 3, 2], [2, 2, 3, 3, 4, 4]]),
            "edge_weight": torch.tensor([2.0, 2.0, 1.0, 3.0, 2.0, 2.0]),
        }

    def test_behavior_features_extend_routing_with_topology(self):
        features = token_behavior_features(self.graph, num_channels=2)
        self.assertEqual(features.shape, (3, len(BEHAVIOR_FEATURE_NAMES)))
        expected_topology = torch.tensor([
            [2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 1.0, 2.0 / 3.0, 0.5, 1.0, 0.5],
            [2.0, 0.0, 2.0, 0.5, 0.0, 1.0, 1.0],
        ])
        torch.testing.assert_close(features[:, 4:], expected_topology)

    def test_empty_graph_behavior_is_finite_zero(self):
        graph = {
            "num_nodes": 4,
            "response_idx": 2,
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_weight": torch.empty(0),
        }
        features = token_behavior_features(graph, num_channels=1)
        self.assertEqual(features.shape, (2, len(BEHAVIOR_FEATURE_NAMES)))
        torch.testing.assert_close(features, torch.zeros_like(features))

    def test_positive_runs_and_mask(self):
        self.assertEqual(validate_positive_runs(7, [[1, 3], [5, 7]]), ((1, 3), (5, 7)))
        mask = positive_mask(7, [[1, 3], [5, 7]])
        torch.testing.assert_close(mask, torch.tensor([False, True, True, False, False, True, True]))
        with self.assertRaises(ValueError):
            validate_positive_runs(7, [[2, 4], [3, 5]])

    def test_centered_window_nan_pads_boundaries(self):
        features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        window, valid = centered_window(features, center=0, radius=2)
        self.assertEqual(window.shape, (5, 2))
        torch.testing.assert_close(valid, torch.tensor([False, False, True, True, True]))
        self.assertTrue(torch.isnan(window[:2]).all())
        torch.testing.assert_close(window[2:], features[:3])

    def test_align_error_onsets_first_and_all(self):
        features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        first, first_valid = align_error_onsets(features, [[1, 2], [4, 5]], radius=1, policy="first")
        all_windows, all_valid = align_error_onsets(features, [[1, 2], [4, 5]], radius=1, policy="all")
        self.assertEqual(first.shape, (1, 3, 4))
        self.assertEqual(first_valid.shape, (1, 3))
        self.assertEqual(all_windows.shape, (2, 3, 4))
        self.assertEqual(all_valid.shape, (2, 3))

    def test_summarize_pre_error_post(self):
        features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        summary = summarize_run_windows(features, [[2, 4]], pre_window=2, post_window=2)
        expected = torch.stack((features[:2].mean(0), features[2:4].mean(0), features[4:].mean(0)))
        torch.testing.assert_close(summary[0], expected)


if __name__ == "__main__":
    unittest.main()
