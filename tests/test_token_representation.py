import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from attention_graph.token_representation import (
    TokenRepresentationConfig,
    build_node_representation,
    compact_layer_structure,
    direct_lookback_channels,
    discover_token_representations,
    exact_channel_route,
    render_saved_sample,
    representation_feature_names,
    structure_names,
    _read_dataset_labels,
    _cluster_bootstrap_difference,
    _PositionReservoir,
    _route_edges_by_relation,
)
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from main import _require_llama31_geometry, parse_args
from research_dataset import ResearchDataset


def _sample(sample_id="sample", source_id="source"):
    # R0: P0 .2 + P1 .2 + self .1
    # R1: P0 .2 + R0 .6 + self .2
    diagonal = torch.zeros((1, 1, 4), dtype=torch.float16)
    diagonal[:, :, 2] = .1
    diagonal[:, :, 3] = .2
    return AttentionSample(
        sample_id, source_id, 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([.2, .2, .2, .6], dtype=torch.float16), .01,
    )


def _multi_channel_sample():
    # Row order is channel-major: C0R0,C0R1,C1R0,C1R1,...
    masses = torch.tensor(
        [.1, .2, .3, .4, .5, .6, .7, .8], dtype=torch.float16
    )
    diagonal = torch.zeros((2, 2, 4), dtype=torch.float16)
    diagonal[:, :, 2:] = .1
    return AttentionSample(
        "multi", "multi-source", 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.arange(9, dtype=torch.int32), torch.zeros(8, dtype=torch.int32),
        masses, .01,
    )


def _varied_sample(sample_id, source_id, variant):
    prompt_source = variant % 3
    prompt = .14 + .015 * variant
    history = .22 + .012 * variant
    diagonal = torch.zeros((1, 1, 6), dtype=torch.float16)
    diagonal[:, :, 3:] = torch.tensor(
        [.08 + .005 * variant, .10 + .004 * variant, .12 + .003 * variant],
        dtype=torch.float16,
    )
    return AttentionSample(
        sample_id, source_id, 3, torch.arange(6, dtype=torch.int32), diagonal,
        torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        torch.tensor([prompt_source, prompt_source, 3, prompt_source, 3, 4], dtype=torch.int32),
        torch.tensor([prompt, prompt + .03, history, prompt + .05,
                      history - .04, history + .02], dtype=torch.float16), .01,
    )


def _write_split(root, split, count, source_prefix):
    (root / "attention").mkdir(parents=True)
    rows, labels = [], []
    for index in range(count):
        sample = _varied_sample(
            f"{split}-{index}", f"{source_prefix}-{index}",
            index + (0 if split == "train" else 7),
        )
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata={
            "split": split, "task_type": "QA" if index % 2 else "Summary",
            "data_source": "MARCO" if index % 2 else "CNN/DM",
            "generator_model": "generator",
        }))
        labels.append({
            "sample_id": sample.sample_id,
            "positive_runs": [[1, 2]] if index % 2 else [],
        })
    label_path = root / "labels.jsonl"
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    write_split_index(
        root, rows, attention_floor=.01, num_layers=1, num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={
            "schema": "ragtruth-attention-split-v1", "split": split,
            "observer_model": "observer", "generator_model": "generator",
            "labels_sha256": sha256(label_path),
        },
    )


class TokenGraphRepresentationTests(unittest.TestCase):
    def test_undefined_lookback_channel_uses_attention_floor(self):
        attention = AttentionSample(
            "undefined-lookback", "source", 1,
            torch.arange(3, dtype=torch.int32),
            torch.zeros((2, 2, 3), dtype=torch.float16),
            torch.zeros(9, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float16), .01,
        )

        lookback = direct_lookback_channels(attention)

        self.assertEqual(tuple(lookback.shape), (2, 2, 2))
        torch.testing.assert_close(
            lookback, torch.full((2, 2, 2), attention.attention_floor)
        )

    def test_lookback_is_preserved_before_any_channel_average(self):
        values = direct_lookback_channels(_sample())
        self.assertEqual(tuple(values.shape), (2, 1, 1))
        torch.testing.assert_close(
            values[:, 0, 0], torch.tensor([2 / 3, .2]), atol=2e-3, rtol=2e-3
        )

    def test_multi_channel_order_matches_layer_head_geometry(self):
        sample = _multi_channel_sample()
        values = direct_lookback_channels(sample)
        masses = torch.tensor([
            [[.1, .3], [.5, .7]],
            [[.2, .4], [.6, .8]],
        ])
        expected = (masses / 2) / (masses / 2 + torch.tensor([.1, .05])[:, None, None])
        self.assertEqual(tuple(values.shape), (2, 2, 2))
        torch.testing.assert_close(values, expected, atol=2e-3, rtol=2e-3)

    def test_prompt_range_history_change_and_provenance_are_explicit(self):
        sample = _sample()
        names = structure_names(2)
        matrix = compact_layer_structure(sample, provenance_hops=2)
        self.assertEqual(tuple(matrix.shape), (2, len(names), 1))
        feature = {name: matrix[:, index, 0] for index, name in enumerate(names)}
        torch.testing.assert_close(
            feature["retained_prompt_coverage"], torch.tensor([1., .5]),
            atol=2e-3, rtol=2e-3,
        )
        torch.testing.assert_close(
            feature["retained_prompt_span"], torch.tensor([1., .5]),
            atol=2e-3, rtol=2e-3,
        )
        self.assertAlmostEqual(float(feature["history_lag"][1]), 1.0, places=3)
        self.assertAlmostEqual(
            float(feature["prompt_provenance_log_mass_hop1"][1]),
            float(np.log10(.6 * .4)), places=3,
        )
        self.assertEqual(float(feature["prompt_provenance_log_mass_hop2"][1]), -12.0)

    def test_top_head_route_is_compact_per_layer_not_all_head_average(self):
        names = structure_names(1)
        matrix = compact_layer_structure(
            _multi_channel_sample(), provenance_hops=1
        )
        prompt_mass = matrix[:, names.index("retained_prompt_mass")]
        torch.testing.assert_close(
            prompt_mass,
            torch.tensor([[.3, .7], [.4, .8]]),
            atol=2e-3, rtol=2e-3,
        )

    def test_strong_minority_head_is_not_diluted(self):
        diagonal = torch.zeros((1, 4, 2), dtype=torch.float16)
        sample = AttentionSample(
            "sparse-head", "source", 1, torch.arange(2, dtype=torch.int32),
            diagonal, torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([.8], dtype=torch.float16), .01,
        )
        names = structure_names(1)
        matrix, route = compact_layer_structure(
            sample, provenance_hops=1, return_route=True
        )
        self.assertAlmostEqual(
            float(matrix[0, names.index("retained_prompt_mass"), 0]), .8, places=3
        )
        self.assertAlmostEqual(float(route["weight"][0]), .8, places=3)

    def test_node_vector_is_exact_flattened_lookback(self):
        sample = _multi_channel_sample()
        lookback = direct_lookback_channels(sample)
        node = build_node_representation(
            lookback, num_layers=2, num_heads=2
        )
        expected_width = 4
        self.assertEqual(tuple(node.shape), (2, expected_width))
        schema = representation_feature_names(2, 2)
        self.assertEqual(len(schema), expected_width)
        torch.testing.assert_close(
            node, lookback.reshape(2, 4)
        )

    def test_llama31_node_vector_keeps_all_1024_layer_head_coordinates(self):
        lookback = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 32, 32)

        node = build_node_representation(
            lookback, num_layers=32, num_heads=32
        )

        self.assertEqual(tuple(node.shape), (2, 1024))
        torch.testing.assert_close(node, lookback.reshape(2, 1024))

    def test_exact_channel_route_keeps_each_csr_edge_in_its_layer_head_channel(self):
        route = exact_channel_route(_multi_channel_sample())
        torch.testing.assert_close(
            route["channel"], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        )
        torch.testing.assert_close(
            route["layer"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        )
        torch.testing.assert_close(
            route["head"], torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        )
        torch.testing.assert_close(
            route["source"], torch.zeros(8, dtype=torch.long)
        )
        torch.testing.assert_close(
            route["target"], torch.tensor([2, 3, 2, 3, 2, 3, 2, 3])
        )
        torch.testing.assert_close(
            route["weight"], torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8]),
            atol=2e-3, rtol=2e-3,
        )

    def test_sparse_plot_route_preserves_source_target_distance(self):
        route = {
            "layer": np.asarray([0, 1, 0, 0]),
            "source": np.asarray([0, 0, 2, 3]),
            "target": np.asarray([3, 3, 4, 4]),
            "weight": np.asarray([.2, .7, .5, .4], dtype=np.float32),
        }
        edges = _route_edges_by_relation(route, response_idx=2, response_count=3)
        self.assertEqual(len(edges["selected"]), 3)
        np.testing.assert_array_equal(edges["rp_source"], [0])
        np.testing.assert_array_equal(edges["rp_target"], [1])
        np.testing.assert_allclose(edges["rp_weight"], [.7])
        np.testing.assert_array_equal(edges["rr_source"], [0, 1])
        np.testing.assert_array_equal(edges["rr_target"], [2, 2])
        np.testing.assert_allclose(edges["rr_weight"], [.5, .4])


class PipelineContractTests(unittest.TestCase):
    def test_frozen_cli_requires_llama31_8b_geometry_and_point01_floor(self):
        valid = type("Dataset", (), {
            "manifest": {"num_layers": 32, "num_heads": 32, "attention_floor": .01}
        })()
        invalid = type("Dataset", (), {
            "manifest": {"num_layers": 32, "num_heads": 32, "attention_floor": 0.0}
        })()

        _require_llama31_geometry(valid)
        with self.assertRaisesRegex(ValueError, "Llama-3.1-8B geometry"):
            _require_llama31_geometry(invalid)

    def test_reference_size_is_a_total_position_reservoir_budget(self):
        reservoir = _PositionReservoir(bins=3, size=10, seed=0)
        values = np.arange(36, dtype=np.float32).reshape(12, 3)
        reservoir.add(values, np.linspace(0.0, .99, 12))
        retained, _ = reservoir.matrix()
        self.assertLessEqual(len(retained), 12)
        self.assertEqual(reservoir.maximum_rows, 12)

    def test_paired_bootstrap_keeps_response_clusters_and_detects_gain(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        sample_ids = np.asarray(["a", "a", "b", "b", "c", "c"])
        better = np.asarray([.1, .9, .2, .8, .3, .7])
        worse = np.asarray([.9, .1, .8, .2, .7, .3])
        result = _cluster_bootstrap_difference(
            labels, better, worse, sample_ids, seed=4, replicates=20,
        )
        self.assertGreater(result["auroc_difference"]["ci95"][0], 0)
        self.assertGreater(result["auprc_difference"]["ci95"][0], 0)

    def test_saved_render_unlocks_a_formal_style_label_seal(self):
        class Sample:
            def __init__(self, dataset, sample_id):
                self.dataset, self.sample_id = dataset, sample_id

            def attention(self):
                self.dataset.processed.add(self.sample_id)

            def release_attention(self):
                pass

        class Store:
            def response_labels(self, sample):
                return torch.tensor([int(sample.sample_id == "b")])

        class Dataset:
            sample_ids = ["a", "b"]

            def __init__(self):
                self.processed = set()

            def __getitem__(self, sample_id):
                return Sample(self, sample_id)

            def labels(self):
                if len(self.processed) != len(self.sample_ids):
                    raise RuntimeError(
                        "formal labels become available only after every attention "
                        "sample has been processed"
                    )
                return Store()

        dataset = Dataset()
        labels = _read_dataset_labels(dataset, "test label seal")
        np.testing.assert_array_equal(labels, np.asarray([0, 1], dtype=np.int8))
        self.assertEqual(dataset.processed, {"a", "b"})

    def test_cli_exposes_only_current_compact_graph_controls(self):
        args = parse_args([
            "represent-tokens", "--train-split", "train", "--test-split", "test",
            "--output-dir", "output", "--sample-id", "42",
        ])
        self.assertFalse(hasattr(args, "lookback_window"))
        self.assertEqual(args.provenance_hops, 2)
        self.assertFalse(hasattr(args, "route_top_heads"))
        self.assertFalse(hasattr(args, "prompt_bins"))
        self.assertFalse(hasattr(args, "graph_head_components"))
        self.assertEqual(args.bootstrap_replicates, 200)
        self.assertEqual(args.csr_row_block, 65536)
        self.assertEqual(args.anomaly_quantile, .95)
        self.assertFalse(hasattr(args, "layer_bins"))
        self.assertFalse(hasattr(args, "diffusion_hops"))
        self.assertFalse(hasattr(args, "visual_reference_size"))

    def test_end_to_end_freezes_graph_state_before_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _write_split(train_root, "train", 5, "train-source")
            _write_split(test_root, "test", 4, "test-source")
            output = root / "output"
            with patch(
                "attention_graph.token_representation.exact_channel_route",
                side_effect=AssertionError("production pipeline must stream CSR"),
            ), patch(
                "attention_graph.token_representation.compact_layer_structure",
                side_effect=AssertionError(
                    "population pipeline must not materialize visualization routes"
                ),
            ):
                result = discover_token_representations(
                    ResearchDataset(train_root), ResearchDataset(test_root),
                    ResearchDataset(test_root), output_dir=output,
                    config=TokenRepresentationConfig(
                        position_bins=2, provenance_hops=2,
                        reference_size=8, subspace_components=2,
                        bootstrap_replicates=10,
                        sample_ids=("test-1",), seed=7,
                    ),
                )
            self.assertEqual(result["test_nodes"], 12)
            self.assertEqual(result["primary_score"], "true_graph")
            self.assertEqual(
                set(result["score_evaluation"]),
                {
                    "scalar_only", "token_only", "prompt_graph",
                    "response_graph", "true_graph", "rewired_graph",
                    "direct_marginals",
                },
            )
            self.assertIn(
                "true_graph_vs_rewired_graph", result["structural_comparisons"]
            )
            nodes = np.load(output / "token_node_representations.float16.npy", mmap_mode="r")
            self.assertFalse((output / "compact_layer_structure.float16.npy").exists())
            self.assertEqual(nodes.shape[0], 12)
            true_graph = np.load(
                output / "true_graph_node_representations.float16.npy", mmap_mode="r"
            )
            self.assertEqual(nodes.shape, (12, 1))
            self.assertEqual(true_graph.shape, (12, 3))
            with np.load(output / "token_representations_label_free.npz", allow_pickle=False) as artifact:
                self.assertEqual(str(artifact["schema"]), "token-graph-representation-v2")
                self.assertFalse(bool(artifact["labels_included"]))
                self.assertNotIn("label", artifact.files)
                self.assertNotIn("exact_token_features", artifact.files)
                self.assertIn("scalar_only_score", artifact.files)
                self.assertIn("true_graph_score", artifact.files)
                self.assertIn("rewired_graph_score", artifact.files)
                self.assertIn("response_graph_score", artifact.files)
                self.assertIn("prompt_graph_score", artifact.files)
                self.assertIn("exact_route_canonical_split", artifact.files)
                self.assertNotIn("evidence_flow_score", artifact.files)
                self.assertEqual(str(artifact["true_graph_representation_file"]), "true_graph_node_representations.float16.npy")
            with np.load(output / "train_reference_model.npz", allow_pickle=False) as model:
                self.assertFalse(bool(model["labels_included"]))
                self.assertIn("token_only_pca_components", model.files)
                self.assertIn("true_graph_pca_components", model.files)
            with np.load(output / "sample_graphs" / "sample_test-1.npz", allow_pickle=False) as graph:
                self.assertEqual(str(graph["schema"]), "token-graph-representation-v2")
                self.assertFalse(bool(graph["labels_included"]))
                self.assertIn("global_row_start", graph.files)
                self.assertNotIn("compact_route_layer", graph.files)
                self.assertIn("exact_route_sha256", graph.files)
                self.assertIn("true_graph_score", graph.files)
                self.assertIn("anomaly_component", graph.files)
                self.assertIn("rewire_rr_edges", graph.files)
                self.assertIn("rewire_changed_fraction", graph.files)
                self.assertEqual(str(graph["exact_route_sample_id"]), "test-1")
                self.assertEqual(int(graph["rewire_seed"]), 7)
            report = json.loads((output / "token_representation_report.json").read_text())
            self.assertFalse(report["primary_node_state"]["layer_head_averaged"])
            self.assertFalse(report["evidence_flow_graph"]["all_head_mean_used"])
            self.assertFalse(report["evidence_flow_graph"]["trainable"])
            self.assertIn("rewire_audit", report["evidence_flow_graph"])
            self.assertEqual(
                report["compact_layer_structure"]["role"],
                "computed_on_demand_for_explicit_sample_visualization_only",
            )
            self.assertFalse(report["sample_visualization"]["automatic"])
            self.assertEqual(report["exact_route_audit"]["canonical_test_split"], str(test_root))
            self.assertEqual(report["unsupervised_scores"]["primary_score"], "true_graph")
            self.assertEqual(report["labels_used"], {"train": False, "test": "evaluation_only"})
            self.assertIn("structural_validation", report)
            self.assertIn("graph_pattern_score_evaluation", report)
            self.assertIn(
                "anomaly_component_localization", report["structural_validation"]
            )
            self.assertEqual(report["labels_read_during"], "evaluation_and_plot_coloring_only")
            self.assertTrue((output / "evaluation_token_labels.npy").exists())
            del nodes, true_graph
            rendered = render_saved_sample(
                ResearchDataset(test_root), output_dir=output,
                sample_id="test-1", layer=0,
            )
            self.assertFalse(rendered["features_recomputed"])
            self.assertTrue(rendered["visualization_structure_recomputed"])
            self.assertEqual(rendered["label_source"], "saved_evaluation_cache")
            self.assertTrue(Path(rendered["attention_structure_figure"]).exists())
            self.assertIn("layer_0", rendered["attention_structure_figure"])
            graph_path = output / "sample_graphs" / "sample_test-1.npz"
            with np.load(graph_path, allow_pickle=False) as graph:
                graph_payload = {name: np.array(graph[name]) for name in graph.files}
            graph_payload["schema"] = np.asarray("token-graph-representation-v1")
            np.savez_compressed(graph_path, **graph_payload)
            with self.assertRaisesRegex(ValueError, "unsupported sample graph schema"):
                render_saved_sample(
                    ResearchDataset(test_root), output_dir=output,
                    sample_id="test-1", layer=0,
                )


if __name__ == "__main__":
    unittest.main()
