import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from graph_tsne import GraphTSNEAnalysis
from research_dataset import ResearchDataset, ResearchSample


def write_canonical_split(root: Path) -> Path:
    rows = []
    (root / "attention").mkdir(parents=True)
    for index in range(6):
        sample = AttentionSample(
            f"r{index}",
            f"s{index // 2}",
            2,
            torch.tensor([10, 11, 12, 13, 14]),
            torch.tensor([[[1.0, 0.5, 0.2 + index * 0.01, 0.1, 0.1]]], dtype=torch.float16),
            torch.tensor([0, 2, 5, 9]),
            torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=torch.int32),
            torch.tensor([
                0.02 + index * 0.001, 0.08 + index * 0.002, 0.03 + index * 0.002,
                0.12 + index * 0.003, 0.04 + index * 0.001, 0.03 + index * 0.001,
                0.05 + index * 0.002, 0.09 + index * 0.003, 0.11 + index * 0.004,
            ], dtype=torch.float16),
            0.01,
        )
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path))
    write_split_index(
        root, rows, attention_floor=0.01, num_layers=1, num_heads=1,
        alignment="post_token_query_at_same_position",
    )
    write_labels(root, [[], [[0, 1]], [], [[1, 2]], [], [[0, 3]]])
    return root


def write_labels(root: Path, runs: list[list[list[int]]]) -> None:
    labels = root / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps({"sample_id": f"r{index}", "positive_runs": value}) + "\n" for index, value in enumerate(runs)),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256(labels)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GraphTSNETests(unittest.TestCase):
    def test_live_original_graph_writes_all_outputs_and_ignores_labels_during_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            first = GraphTSNEAnalysis(root, Path(directory) / "first", tau=0.01, seed=7).run()
            with np.load(first["coordinates"]) as arrays:
                first_coordinates = {name: arrays[name].copy() for name in arrays.files}

            self.assertEqual(first["samples"], 6)
            self.assertEqual(set(first), {"samples", "figure", "length_figure", "coordinates"})
            self.assertTrue(all(Path(first[name]).is_file() for name in ("figure", "length_figure", "coordinates")))
            self.assertEqual(set(first_coordinates), {"sample_id", "response_tokens", "topology", "node", "combined"})
            self.assertEqual(first_coordinates["topology"].shape, (6, 2))
            self.assertEqual(first_coordinates["node"].shape, (6, 2))
            self.assertEqual(first_coordinates["combined"].shape, (6, 2))

            write_labels(root, [[[0, 3]], [], [[0, 1]], [], [[1, 2]], []])
            second = GraphTSNEAnalysis(root, Path(directory) / "second", tau=0.01, seed=7).run()
            with np.load(second["coordinates"]) as arrays:
                second_coordinates = {name: arrays[name].copy() for name in arrays.files}
            for name in ("topology", "node", "combined"):
                np.testing.assert_allclose(first_coordinates[name], second_coordinates[name])

    def test_topology_descriptor_has_36_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            dimensions = []

            def embed(matrix, pre_scaled=False):
                dimensions.append(matrix.shape[1])
                return np.zeros((len(matrix), 2))

            with patch.object(GraphTSNEAnalysis, "_embed", side_effect=embed):
                GraphTSNEAnalysis(root, Path(directory) / "output").run()

            self.assertEqual(dimensions[0], 36)

    def test_none_node_features_are_rejected_at_the_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            with self.assertRaisesRegex(ValueError, "node_feature_mode"):
                GraphTSNEAnalysis(root, Path(directory) / "output", node_feature_mode="none").run()

    def test_research_sample_memoizes_attention_for_related_accessors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            sample = ResearchDataset(root)["r0"]
            original_loader = __import__("research_dataset").load_attention_sample

            with patch("research_dataset.load_attention_sample", wraps=original_loader) as loader:
                attention = sample.attention()
                self.assertIs(sample.attention(), attention)
                sample.original_graph(0.01)
                sample.node_features("attention")
                _ = sample.response_slice

            self.assertEqual(loader.call_count, 1)

    def test_cached_original_graph_is_loaded_through_research_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            graph_root = Path(directory) / "graphs"
            GraphDatasetBuilder(BuildConfig(root, graph_root, kind="original", tau=0.01, device="cpu")).run()
            original_graph = ResearchSample.graph
            calls = []

            def traced_graph(sample, name):
                calls.append((sample.sample_id, name))
                return original_graph(sample, name)

            with patch.object(ResearchSample, "graph", new=traced_graph):
                GraphTSNEAnalysis(root, Path(directory) / "output", graph_root=graph_root, tau=0.01).run()

            self.assertEqual(calls, [(f"r{index}", "graph") for index in range(6)])

    def test_cached_original_graph_requires_requested_tau_before_graph_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            graph_root = Path(directory) / "graphs"
            GraphDatasetBuilder(BuildConfig(root, graph_root, kind="original", tau=0.05, device="cpu")).run()

            with patch.object(ResearchSample, "graph", side_effect=AssertionError("graph should not load")):
                with self.assertRaisesRegex(ValueError, "tau"):
                    GraphTSNEAnalysis(root, Path(directory) / "output", graph_root=graph_root, tau=0.01).run()

    def test_non_original_graph_cache_is_rejected_before_graph_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = write_canonical_split(Path(directory) / "split")
            graph_root = Path(directory) / "topk"
            GraphDatasetBuilder(BuildConfig(root, graph_root, kind="relation_topk", tau=0.01, device="cpu")).run()

            with patch.object(ResearchSample, "graph", side_effect=AssertionError("graph should not load")):
                with self.assertRaisesRegex(ValueError, "original graph cache"):
                    GraphTSNEAnalysis(root, Path(directory) / "output", graph_root=graph_root, tau=0.01).run()


if __name__ == "__main__":
    unittest.main()
