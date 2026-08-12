import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig
from attention_graph.graph_view import _edge_display_strength, _sample_scores, visualize_graph
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from research_dataset import ResearchDataset


class GraphViewTests(unittest.TestCase):
    def test_edge_display_strength_uses_schema_fixed_log_scale(self):
        strength, baseline = _edge_display_strength(
            np.asarray([0.01 / 1024, 0.5], dtype=np.float32),
            attention_floor=0.01, num_channels=1024,
        )
        self.assertGreater(baseline, 0.0)
        self.assertGreater(strength[0], 0.0)
        self.assertGreater(strength[1], strength[0])
        self.assertLessEqual(strength[1], 1.0)

    def test_score_rows_must_exactly_match_selected_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            cases = {
                "duplicate": ([0, 0, 2], ["source", "source", "source"]),
                "negative": ([-1, 1, 2], ["source", "source", "source"]),
                "wrong-source": ([0, 1, 2], ["source", "other", "source"]),
            }
            for name, (token_index, source_id) in cases.items():
                with self.subTest(name=name):
                    np.savez_compressed(
                        path, sample_id=np.asarray(["case"] * 3),
                        source_id=np.asarray(source_id),
                        token_index=np.asarray(token_index, dtype=np.int32),
                        score=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                    )
                    with self.assertRaises(ValueError):
                        _sample_scores(path, "case", "source", 3)

    def test_graph_view_aligns_scores_and_exports_visible_typed_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            attention_dir = root / "attention"
            attention_dir.mkdir(parents=True)
            sample = AttentionSample(
                "case-1", "source-1", 2, torch.tensor([10, 11, 12, 13, 14]),
                torch.ones((1, 1, 5), dtype=torch.float16) * 0.1,
                torch.tensor([0, 2, 4, 7]), torch.tensor([0, 1, 0, 1, 0, 2, 3]),
                torch.tensor([0.4, 0.2, 0.3, 0.2, 0.2, 0.5, 0.3], dtype=torch.float16), 0.01,
            )
            attention_path = attention_dir / "case.npz"
            save_attention_sample(sample, attention_path)
            labels = root / "labels.jsonl"
            labels.write_text('{"sample_id":"case-1","positive_runs":[[1,2]]}\n', encoding="utf-8")
            write_split_index(
                root, [index_row(root, sample, attention_path, metadata={"split": "test"})],
                attention_floor=0.01, num_layers=1, num_heads=1,
                alignment="post_token_query_at_same_position", extra={"labels_sha256": sha256(labels)},
            )
            scores = root / "scores.npz"
            np.savez_compressed(
                scores, representation=np.asarray("mart_mechanism_pca_embedding"),
                embedding=np.zeros((3, 2), dtype=np.float32), score=np.asarray([0.2, 0.9, 0.4], dtype=np.float32),
                sample_id=np.asarray(["case-1", "case-1", "case-1"]),
                source_id=np.asarray(["source-1", "source-1", "source-1"]),
                token_index=np.asarray([0, 1, 2], dtype=np.int32),
            )
            result = visualize_graph(
                ResearchDataset(root), score_path=scores, sample_id="case-1", output_dir=root / "view",
                graph_config=GraphBuildConfig(selection="typed_mass_cover", mass_cover=0.8),
                window=2, display_top_k=1,
            )
            self.assertEqual(result["center_token"], 1)
            self.assertEqual(result["display_top_k"], 1)
            self.assertTrue(Path(result["figure"]).is_file())
            with np.load(result["data"], allow_pickle=False) as data:
                self.assertEqual(data["sample_id"].item(), "case-1")
                self.assertEqual(data["source_id"].item(), "source-1")
                self.assertEqual(data["schema"].item(), "attention-graph-view-v1")
                self.assertEqual(data["selection"].item(), "typed_mass_cover")
                self.assertEqual(data["display_top_k"].item(), 1)
                self.assertEqual(data["graph_top_k"].item(), 8)
                self.assertTrue(np.isnan(data["graph_threshold"]).item())
                self.assertEqual(data["labels_read_during"].item(), "rendering_only")
                self.assertEqual(data["window"].tolist(), [0, 3])
                self.assertEqual(data["node_role"].tolist(), ["prompt", "response", "response", "response"])
                np.testing.assert_allclose(data["node_score"], [np.nan, 0.2, 0.9, 0.4], equal_nan=True)
                self.assertEqual(data["node_label"].tolist(), [-1, 0, 1, 0])
                self.assertIn("RP", data["edge_type"].tolist())
                self.assertIn("RR", data["edge_type"].tolist())
                self.assertEqual(data["trace_edge_id"].min(), 0)
                self.assertLess(data["trace_edge_id"].max(), len(data["edge_type"]))
                self.assertEqual(data["trace_layer"].tolist(), [0] * len(data["trace_value"]))
                self.assertEqual(data["trace_head"].tolist(), [0] * len(data["trace_value"]))
                self.assertTrue(np.isfinite(data["score_vmin"]).item())

    def test_graph_view_renders_empty_edge_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            attention_dir = root / "attention"
            attention_dir.mkdir(parents=True)
            sample = AttentionSample(
                "empty", "source", 2, torch.tensor([10, 11, 12, 13]),
                torch.ones((1, 1, 4), dtype=torch.float16) * 0.1,
                torch.tensor([0, 0, 0]), torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.float16), 0.01,
            )
            attention_path = attention_dir / "empty.npz"
            save_attention_sample(sample, attention_path)
            labels = root / "labels.jsonl"
            labels.write_text('{"sample_id":"empty","positive_runs":[]}\n', encoding="utf-8")
            write_split_index(
                root, [index_row(root, sample, attention_path)], attention_floor=0.01,
                num_layers=1, num_heads=1, alignment="post_token_query_at_same_position",
                extra={"labels_sha256": sha256(labels)},
            )
            scores = root / "scores.npz"
            np.savez_compressed(
                scores, score=np.asarray([0.1, 0.2], dtype=np.float32),
                sample_id=np.asarray(["empty", "empty"]), source_id=np.asarray(["source", "source"]),
                token_index=np.asarray([0, 1], dtype=np.int32),
            )
            result = visualize_graph(
                ResearchDataset(root), score_path=scores, sample_id="empty", output_dir=root / "view",
                graph_config=GraphBuildConfig(selection="typed_mass_cover"), window=1,
            )
            self.assertEqual(result["visible_edges"], 0)
            self.assertTrue(Path(result["figure"]).is_file())


if __name__ == "__main__":
    unittest.main()
