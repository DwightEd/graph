import unittest

import numpy as np
import torch

from attention_graph.graph import AttentionGraph, GraphBuildConfig, RP, RR
from attention_graph.patterns import (
    PatternDiscoveryConfig,
    _fit_patterns,
    _landmark_tsne,
    _response_graph_report,
    provenance_curves,
)
from main import parse_args


def _chain_graph():
    # P0 -> R0 -> R1 -> R2 in every layer.  Tracing R2 backwards therefore
    # reaches prompt after exactly three layer steps, while R0 is direct.
    layers, heads = 3, 1
    edge_index = torch.tensor([[0, 2, 3], [2, 3, 4]], dtype=torch.long)
    edge_type = torch.tensor([RP, RR, RR], dtype=torch.long)
    trace_edge_id = torch.arange(3).repeat_interleave(layers)
    trace_channel = torch.arange(layers).repeat(3)
    return AttentionGraph(
        sample_id="chain",
        source_id="source",
        response_idx=2,
        num_layers=layers,
        num_heads=heads,
        attention_floor=0.01,
        token_ids=torch.arange(5),
        node_attr=torch.zeros((5, layers)),
        node_context=torch.zeros((5, 4)),
        response_mask=torch.tensor([False, False, True, True, True]),
        edge_index=edge_index,
        edge_type=edge_type,
        edge_score=torch.ones(3),
        trace_edge_id=trace_edge_id,
        trace_channel=trace_channel,
        trace_value=torch.ones(9),
        build_config=GraphBuildConfig(),
    )


class ProvenanceCurveTests(unittest.TestCase):
    def test_curves_capture_direct_and_delayed_prompt_grounding(self):
        signature, unresolved = provenance_curves(_chain_graph(), checkpoints=3)
        prompt = signature[:, :3]
        torch.testing.assert_close(prompt[0], torch.ones(3))
        torch.testing.assert_close(prompt[2], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(unresolved, torch.zeros_like(unresolved))
        concentration, _ = provenance_curves(
            _chain_graph(),
            checkpoints=3,
            signature_view="response_concentration",
        )
        self.assertEqual(float(concentration[2, 0]), 1.0)
        self.assertEqual(float(concentration[2, -1]), 0.0)

    def test_missing_attention_is_control_not_primary_grounding(self):
        graph = _chain_graph()
        graph = AttentionGraph(**{
            **graph.__dict__,
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_type": torch.empty(0, dtype=torch.long),
            "edge_score": torch.empty(0),
            "trace_edge_id": torch.empty(0, dtype=torch.long),
            "trace_channel": torch.empty(0, dtype=torch.long),
            "trace_value": torch.empty(0),
        })
        signature, unresolved = provenance_curves(graph, checkpoints=3)
        torch.testing.assert_close(signature, torch.zeros_like(signature))
        torch.testing.assert_close(unresolved, torch.ones_like(unresolved))


class PatternProjectionTests(unittest.TestCase):
    def test_repeated_curves_do_not_collapse_pattern_fitting(self):
        values = np.repeat(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            repeats=[40, 30, 30],
            axis=0,
        )
        model, scores = _fit_patterns(
            values,
            PatternDiscoveryConfig(
                checkpoints=3,
                min_patterns=2,
                max_patterns=6,
                fit_reference_size=100,
                tsne_landmarks=20,
            ),
        )
        self.assertGreaterEqual(model.n_clusters, 2)
        self.assertLessEqual(model.n_clusters, 3)
        self.assertTrue(all(np.isfinite(list(scores.values()))))

    def test_cli_exposes_training_free_pattern_discovery(self):
        args = parse_args([
            "discover-patterns",
            "--train-split", "train",
            "--test-split", "test",
            "--output-dir", "output",
        ])
        self.assertEqual(args.command, "discover-patterns")
        self.assertEqual(args.checkpoints, 8)
        self.assertEqual(args.signature_view, "prompt_absorption")

    def test_response_report_compares_graph_modes_after_node_clustering(self):
        metadata = {
            "sample_id": np.asarray(["a", "a", "b", "b", "b"]),
            "source_id": np.asarray(["sa", "sa", "sb", "sb", "sb"]),
        }
        summaries, report = _response_graph_report(
            metadata,
            np.asarray([0, 0, 1, 0, 1]),
            np.asarray([0, 0, 0, 1, 0]),
            2,
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(report["response_graphs"], 2)
        self.assertEqual(report["hallucination_graphs"], 1)
        self.assertEqual(summaries[1]["pattern_sequence"], [1, 0, 1])

    def test_landmark_projection_returns_every_node(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(60, 6)).astype(np.float32)
        config = PatternDiscoveryConfig(
            fit_reference_size=20,
            tsne_landmarks=20,
            perplexity=5,
        )
        coordinates, diagnostics = _landmark_tsne(values, config)
        self.assertEqual(coordinates.shape, (60, 2))
        self.assertTrue(np.isfinite(coordinates).all())
        self.assertEqual(diagnostics["all_nodes"], 60)
        self.assertEqual(diagnostics["landmarks"], 20)


if __name__ == "__main__":
    unittest.main()
