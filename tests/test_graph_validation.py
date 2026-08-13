import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.graph_validation import (
    GraphValidationConfig,
    GraphValidator,
    _spans,
    evaluate_graph_artifacts,
)
from attention_graph.graph_variants import transform_graph
from attention_graph.patterns import graph_lookback_trajectories
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from main import parse_args
from research_dataset import ResearchDataset


def _sample(sample_id="sample", source_id="source"):
    return AttentionSample(
        sample_id, source_id, 2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[0.1, 0.1, 0.2, 0.3]], [[0.1, 0.1, 0.1, 0.2]]], dtype=torch.float16),
        torch.tensor([0, 2, 4, 6, 8]),
        torch.tensor([0, 1, 0, 2, 0, 1, 1, 2]),
        torch.tensor([0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.2, 0.3], dtype=torch.float16),
        0.01,
    )


def _split(root, split, samples):
    (root / "attention").mkdir(parents=True)
    rows = []
    labels = []
    for sample in samples:
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata={
            "split": split, "task_type": "QA", "data_source": "MARCO",
            "generator_model": "generator",
        }))
        labels.append({"sample_id": sample.sample_id, "positive_runs": [[1, 2]]})
    label_path = root / "labels.jsonl"
    label_path.write_text("\n".join(__import__("json").dumps(row) for row in labels) + "\n")
    write_split_index(
        root, rows, attention_floor=0.01, num_layers=2, num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={"schema": "ragtruth-attention-split-v1", "split": split,
               "observer_model": "observer", "generator_model": "generator",
               "labels_sha256": sha256(label_path)},
    )


class GraphTransformTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_attention_graph(_sample(), GraphBuildConfig())

    def test_variants_are_causal_and_deterministic(self):
        for variant in GraphValidationConfig().variants:
            first = transform_graph(self.graph, variant, seed=9)
            second = transform_graph(self.graph, variant, seed=9)
            self.assertTrue(torch.equal(first.edge_index, second.edge_index))
            self.assertTrue(torch.equal(first.trace_value, second.trace_value))
            if first.num_edges:
                self.assertTrue(bool((first.edge_index[0] < first.edge_index[1]).all()))

    def test_rewire_preserves_trace_mass_by_target_relation_and_channel(self):
        graph = type(self.graph)(**{
            **self.graph.__dict__,
            "response_idx": 4,
            "token_ids": torch.arange(6),
            "node_attr": torch.zeros((6, 2)),
            "node_context": torch.zeros((6, 4)),
            "response_mask": torch.tensor([False, False, False, False, True, True]),
            "edge_index": torch.tensor([[0, 1], [4, 5]]),
            "edge_type": torch.tensor([0, 0]),
            "edge_score": torch.tensor([0.5, 0.5]),
            "trace_edge_id": torch.tensor([0, 1]),
            "trace_channel": torch.tensor([0, 0]),
            "trace_value": torch.tensor([0.5, 0.5]),
        })
        rewired = transform_graph(graph, "source_rewire", seed=2)

        def grouped(graph):
            edge = graph.trace_edge_id
            target = graph.edge_index[1, edge]
            relation = graph.edge_type[edge]
            key = torch.stack((target, relation, graph.trace_channel), dim=1)
            rows = {}
            for key_row, value in zip(key.tolist(), graph.trace_value.tolist()):
                rows.setdefault(tuple(key_row), []).append(value)
            return {key: sorted(values) for key, values in rows.items()}

        self.assertEqual(grouped(graph), grouped(rewired))
        self.assertEqual(
            sorted(graph.edge_index[0].tolist()),
            sorted(rewired.edge_index[0].tolist()),
        )
        self.assertFalse(torch.equal(graph.edge_index, rewired.edge_index))
        pair = rewired.edge_index[1] * rewired.num_nodes + rewired.edge_index[0]
        self.assertEqual(len(pair), len(torch.unique(pair)))

    def test_marginals_remove_source_endpoint_information(self):
        changed = self.graph
        edge_index = changed.edge_index.clone()
        edge_index[0, 0] = 1
        altered = type(changed)(**{**changed.__dict__, "edge_index": edge_index})
        left = transform_graph(changed, "marginals", seed=0)
        right = transform_graph(altered, "marginals", seed=0)
        self.assertTrue(torch.equal(left.edge_index, right.edge_index))
        self.assertTrue(torch.equal(left.trace_value, right.trace_value))

    def test_relation_collapse_is_an_explicit_noop_for_lookback_endpoint_boundary(self):
        collapsed = transform_graph(self.graph, "collapse_relations", seed=0)
        self.assertTrue(torch.equal(collapsed.edge_type, torch.zeros_like(self.graph.edge_type)))
        original, _ = graph_lookback_trajectories(self.graph, layer_bins=2)
        transformed, _ = graph_lookback_trajectories(collapsed, layer_bins=2)
        torch.testing.assert_close(original, transformed)

    def test_layer_shuffle_moves_node_attributes_with_trace_layers(self):
        sample = _sample()
        sample.attention_diagonal[1] += 0.2
        graph = build_attention_graph(sample, GraphBuildConfig())
        shuffled = transform_graph(graph, "shuffle_layers", seed=1)
        self.assertFalse(torch.equal(shuffled.node_attr, graph.node_attr))
        self.assertEqual(sorted(shuffled.trace_value.tolist()), sorted(graph.trace_value.tolist()))

    def test_binary_removes_attention_magnitudes_but_keeps_support(self):
        binary = transform_graph(self.graph, "binary", seed=0)
        self.assertTrue(torch.equal(binary.edge_index, self.graph.edge_index))
        self.assertTrue(torch.equal(binary.edge_type, self.graph.edge_type))
        self.assertTrue(torch.equal(binary.trace_value, torch.ones_like(self.graph.trace_value)))
        self.assertTrue(torch.equal(binary.node_attr, (self.graph.node_attr != 0).float()))

    def test_config_requires_full_unique_variants_and_two_layer_bins(self):
        with self.assertRaises(ValueError):
            GraphValidationConfig(variants=("no_edges",)).validate()
        with self.assertRaises(ValueError):
            GraphValidationConfig(variants=("full", "full")).validate()
        with self.assertRaises(ValueError):
            GraphValidationConfig(layer_bins=1).validate()

    def test_short_responses_are_excluded_from_fixed_width_spans(self):
        metadata = {
            "sample_id": np.asarray(["a", "a"]), "source_id": np.asarray(["s", "s"]),
            "token_index": np.asarray([0, 1]), "task_type": np.asarray(["q", "q"]),
            "data_source": np.asarray(["d", "d"]), "generator_model": np.asarray(["m", "m"]),
        }
        spans, rows, skipped = _spans(np.ones((2, 3), dtype=np.float32), metadata, width=3)
        self.assertEqual(spans.shape, (0, 6))
        self.assertEqual(len(rows["span_start"]), 0)
        self.assertEqual(skipped, 1)

    def test_fixed_span_window_counts_cover_width_one_equal_and_too_large(self):
        metadata = {"sample_id": np.asarray(["a"] * 3), "source_id": np.asarray(["s"] * 3),
                    "token_index": np.arange(3), "task_type": np.asarray(["q"] * 3),
                    "data_source": np.asarray(["d"] * 3), "generator_model": np.asarray(["m"] * 3)}
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        self.assertEqual(len(_spans(values, metadata, 1)[0]), 3)
        self.assertEqual(len(_spans(values, metadata, 3)[0]), 1)
        self.assertEqual(len(_spans(values, metadata, 4)[0]), 0)


class GraphValidationPipelineTests(unittest.TestCase):
    def test_cli_and_label_free_fit_then_evaluation(self):
        args = parse_args([
            "validate-graphs", "--train-split", "train", "--test-split", "test",
            "--output-dir", "output", "--variants", "full", "no_edges",
        ])
        self.assertEqual(args.command, "validate-graphs")
        self.assertNotIn("mean_heads", parse_args(["validate-graphs", "--train-split", "train", "--test-split", "test", "--output-dir", "output"]).variants)
        self.assertNotIn("collapse_relations", parse_args(["validate-graphs", "--train-split", "train", "--test-split", "test", "--output-dir", "output"]).variants)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _split(train_root, "train", [_sample("a", "sa"), _sample("b", "sb")])
            _split(test_root, "test", [_sample("c", "sc"), _sample("d", "sd")])
            output = root / "artifacts"
            validator = GraphValidator(GraphValidationConfig(
                variants=("full", "no_edges"), neighbors=1, span_width=2,
            ))
            with mock.patch.object(ResearchDataset, "labels", side_effect=AssertionError("labels leaked")):
                result = validator.run(ResearchDataset(train_root), ResearchDataset(test_root), output)
            self.assertFalse(result["labels_consumed"])
            self.assertTrue((output / "full.npz").exists())
            with np.load(output / "full.npz", allow_pickle=False) as artifact:
                self.assertEqual(artifact["token_score"].shape[0], 4)
                self.assertEqual(artifact["span_start"].tolist(), [0, 0])
                self.assertEqual(artifact["span_end"].tolist(), [2, 2])
            report = evaluate_graph_artifacts(ResearchDataset(test_root), output, root / "evaluation.json")
            self.assertEqual(report["labels_consumed_during"], "evaluation_only")
            self.assertIn("full", report["variants"])
            self.assertIn("data_source", report["variants"]["full"]["token"])
            self.assertIn("full_minus_variant", report["variants"]["no_edges"])

    def test_evaluation_rejects_missing_fixed_width_span_after_hash_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _split(train_root, "train", [_sample("a", "sa"), _sample("b", "sb")])
            _split(test_root, "test", [_sample("c", "sc"), _sample("d", "sd")])
            output = root / "artifacts"
            GraphValidator(GraphValidationConfig(variants=("full",), neighbors=1, span_width=1)).run(
                ResearchDataset(train_root), ResearchDataset(test_root), output
            )
            path = output / "full.npz"
            with np.load(path, allow_pickle=False) as values:
                arrays = {name: values[name] for name in values.files}
            for name, value in arrays.items():
                if name.startswith("span_"):
                    arrays[name] = value[:-1]
            np.savez_compressed(path, **arrays)
            manifest_path = output / "label_free_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["variants"][0]["sha256"] = sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fixed-width span"):
                evaluate_graph_artifacts(ResearchDataset(test_root), output, root / "evaluation.json")

    def test_validation_without_rewire_marks_rewire_non_estimable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _split(train_root, "train", [_sample("a", "sa"), _sample("b", "sb")])
            _split(test_root, "test", [_sample("c", "sc"), _sample("d", "sd")])
            output = root / "artifacts"
            GraphValidator(GraphValidationConfig(variants=("full",), neighbors=1, span_width=1)).run(
                ResearchDataset(train_root), ResearchDataset(test_root), output
            )
            manifest = json.loads((output / "label_free_manifest.json").read_text())
            self.assertEqual(manifest["source_rewire"]["status"], "non_estimable")

    def test_nonestimable_rewire_has_no_inference_but_keeps_descriptive_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _split(train_root, "train", [_sample("a", "sa"), _sample("b", "sb")])
            _split(test_root, "test", [_sample("c", "sc"), _sample("d", "sd")])
            output = root / "artifacts"
            GraphValidator(GraphValidationConfig(variants=("full", "source_rewire"), neighbors=1, span_width=1)).run(
                ResearchDataset(train_root), ResearchDataset(test_root), output
            )
            report = evaluate_graph_artifacts(ResearchDataset(test_root), output, root / "evaluation.json")
            result = report["variants"]["source_rewire"]
            self.assertIn("overall", result["token"])
            self.assertEqual(result["full_minus_variant"]["token"]["auroc"]["status"], "non_estimable")
            self.assertIsNone(result["full_minus_variant"]["token"]["auroc"]["point"])

    def test_evaluation_rejects_tampered_token_metadata_after_hash_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _split(train_root, "train", [_sample("a", "sa"), _sample("b", "sb")])
            _split(test_root, "test", [_sample("c", "sc"), _sample("d", "sd")])
            output = root / "artifacts"
            GraphValidator(GraphValidationConfig(variants=("full",), neighbors=1, span_width=1)).run(
                ResearchDataset(train_root), ResearchDataset(test_root), output
            )
            path = output / "full.npz"
            with np.load(path, allow_pickle=False) as values:
                arrays = {name: values[name] for name in values.files}
            arrays["source_id"] = np.asarray(["tampered"] * len(arrays["source_id"]))
            np.savez_compressed(path, **arrays)
            manifest_path = output / "label_free_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["variants"][0]["sha256"] = sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source_id"):
                evaluate_graph_artifacts(ResearchDataset(test_root), output, root / "evaluation.json")


if __name__ == "__main__":
    unittest.main()
