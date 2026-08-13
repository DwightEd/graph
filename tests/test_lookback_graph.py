import unittest

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.patterns import (
    PatternDiscoveryConfig,
    _apply_position_calibration,
    _fit_patterns,
    _fit_position_calibration,
    _landmark_tsne,
    graph_lookback_trajectories,
    lookback_trajectories,
)
from cache import AttentionSample
from main import parse_args


def _attention_sample():
    # In each layer:
    # R0 <- P0(.2), P1(.2), diagonal(.1)
    # R1 <- P0(.2), R0(.6), diagonal(.2)
    diagonal = torch.zeros((2, 1, 4), dtype=torch.float16)
    diagonal[:, :, 2] = .1
    diagonal[:, :, 3] = .2
    return AttentionSample(
        sample_id="sample",
        source_id="source",
        response_idx=2,
        token_ids=torch.arange(4, dtype=torch.int32),
        attention_diagonal=diagonal,
        response_row_ptr=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        response_column_indices=torch.tensor(
            [0, 1, 0, 2, 0, 1, 0, 2], dtype=torch.int32
        ),
        response_values=torch.tensor(
            [.2, .2, .2, .6, .2, .2, .2, .6], dtype=torch.float16
        ),
        attention_floor=.01,
    )


def _two_layer_two_head_sample():
    # CSR row order is (layer, head, response token). Distinct per-head masses
    # catch accidental reshapes that interchange layers, heads, and tokens.
    masses = [.1, .1, .2, .2, .3, .3, .4, .4]
    diagonal = torch.zeros((2, 2, 4), dtype=torch.float16)
    diagonal[:, :, 2:] = .1
    return AttentionSample(
        sample_id="multi-head",
        source_id="source-multi-head",
        response_idx=2,
        token_ids=torch.arange(4, dtype=torch.int32),
        attention_diagonal=diagonal,
        response_row_ptr=torch.arange(9, dtype=torch.int32),
        response_column_indices=torch.zeros(8, dtype=torch.int32),
        response_values=torch.tensor(masses, dtype=torch.float16),
        attention_floor=.01,
    )


class LookbackTrajectoryTests(unittest.TestCase):
    def test_exact_length_normalized_formula_includes_diagonal(self):
        values, unresolved = lookback_trajectories(
            _attention_sample(), layer_bins=2, csr_row_block=1
        )
        self.assertEqual(tuple(values.shape), (2, 2))
        expected = torch.tensor([[2.0 / 3.0, 2.0 / 3.0], [.2, .2]])
        torch.testing.assert_close(values, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(
            unresolved, torch.tensor([[.5, .5], [0., 0.]]),
            atol=2e-3, rtol=2e-3,
        )

    def test_diagonal_only_control_has_zero_lookback(self):
        sample = _attention_sample()
        sample.response_row_ptr = torch.zeros(5, dtype=torch.int32)
        sample.response_column_indices = torch.empty(0, dtype=torch.int32)
        sample.response_values = torch.empty(0, dtype=torch.float16)
        values, unresolved = lookback_trajectories(sample, layer_bins=2)
        torch.testing.assert_close(values, torch.zeros_like(values))
        expected = torch.tensor([[.9, .9], [.8, .8]])
        torch.testing.assert_close(unresolved, expected, atol=2e-3, rtol=2e-3)

    def test_full_graph_and_direct_csr_paths_are_identical(self):
        sample = _attention_sample()
        direct, direct_control = lookback_trajectories(sample, layer_bins=2)
        graph = build_attention_graph(sample, GraphBuildConfig())
        transformed, transformed_control = graph_lookback_trajectories(
            graph, layer_bins=2
        )
        torch.testing.assert_close(direct, transformed)
        torch.testing.assert_close(direct_control, transformed_control)

    def test_layer_head_response_order_is_preserved(self):
        sample = _two_layer_two_head_sample()
        values, _ = lookback_trajectories(sample, layer_bins=2)
        # R0 denominator uses one generated-side token (the diagonal).
        # R1 denominator uses two generated-side tokens.
        expected = torch.tensor([
            [(1 / 3 + 1 / 2) / 2, (3 / 5 + 2 / 3) / 2],
            [(1 / 2 + 2 / 3) / 2, (3 / 4 + 4 / 5) / 2],
        ])
        torch.testing.assert_close(values, expected, atol=2e-3, rtol=2e-3)
        graph = build_attention_graph(sample, GraphBuildConfig())
        graph_values, _ = graph_lookback_trajectories(graph, layer_bins=2)
        torch.testing.assert_close(values, graph_values)


class LookbackProjectionTests(unittest.TestCase):
    def test_repeated_trajectories_do_not_request_too_many_patterns(self):
        values = np.repeat(
            np.asarray([[0., 0.], [1., 1.], [3., -1.]], dtype=np.float64),
            repeats=[40, 30, 30], axis=0,
        )
        model, scores = _fit_patterns(
            values,
            PatternDiscoveryConfig(fit_reference_size=100, tsne_landmarks=20),
        )
        self.assertGreaterEqual(model.n_clusters, 2)
        self.assertLessEqual(model.n_clusters, 3)
        self.assertTrue(all(np.isfinite(list(scores.values()))))

    def test_position_calibration_is_train_only_and_keeps_low_direction(self):
        train = np.ones((40, 2), dtype=np.float64)
        train_position = np.linspace(0, 1, len(train))
        center, scale, _ = _fit_position_calibration(
            train, train_position, bins=2
        )
        test = np.asarray([[0., 0.], [1., 1.]])
        calibrated = _apply_position_calibration(
            test, np.asarray([0., 1.]), center, scale
        )
        score = -calibrated.mean(axis=1)
        self.assertGreater(score[0], score[1])

    def test_landmark_projection_returns_every_node(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(60, 6)).astype(np.float64)
        config = PatternDiscoveryConfig(
            fit_reference_size=20, tsne_landmarks=20, perplexity=5
        )
        coordinates, diagnostics = _landmark_tsne(values, config)
        self.assertEqual(coordinates.shape, (60, 2))
        self.assertTrue(np.isfinite(coordinates).all())
        self.assertEqual(diagnostics["all_nodes"], 60)
        self.assertEqual(diagnostics["landmarks"], 20)

    def test_cli_exposes_lookback_and_specific_samples(self):
        args = parse_args([
            "discover-patterns", "--train-split", "train",
            "--test-split", "test", "--output-dir", "output",
            "--sample-id", "42",
        ])
        self.assertEqual(args.command, "discover-patterns")
        self.assertEqual(args.layer_bins, 8)
        self.assertEqual(args.csr_row_block, 4096)
        self.assertEqual(args.sample_id, ["42"])
        self.assertFalse(hasattr(args, "components_per_mechanism"))


if __name__ == "__main__":
    unittest.main()
