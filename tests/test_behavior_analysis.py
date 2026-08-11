import inspect
import json
import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from behavior_analysis import BehaviorAnalysis, token_tsne_coordinates, token_tsne_perplexity
from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from research_dataset import ResearchSample


def make_split(root: Path, *, with_control: bool = False) -> Path:
    split = root / "canonical" / "test"
    sample = AttentionSample(
        "r1", "s1", 2,
        torch.tensor([10, 11, 12, 13, 14, 15]),
        torch.ones((1, 1, 6), dtype=torch.float16),
        torch.tensor([0, 2, 5, 9, 14]),
        torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=torch.int32),
        torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float16),
        0.01,
    )
    path = split / "attention" / "r1.npz"
    save_attention_sample(sample, path)
    rows = [index_row(split, sample, path)]
    labels_rows = [{"sample_id": "r1", "positive_runs": [[1, 3]]}]
    if with_control:
        control = AttentionSample(
            "r2", "s2", sample.response_idx, sample.token_ids,
            sample.attention_diagonal, sample.response_row_ptr,
            sample.response_column_indices, sample.response_values, sample.attention_floor,
        )
        control_path = split / "attention" / "r2.npz"
        save_attention_sample(control, control_path)
        rows.append(index_row(split, control, control_path))
        labels_rows.append({"sample_id": "r2", "positive_runs": []})
    write_split_index(
        split, rows, attention_floor=0.01,
        num_layers=1, num_heads=1, alignment="post_token_query_at_same_position",
    )
    labels = split / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in labels_rows), encoding="utf-8")
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256(labels)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return split


class TokenTsneTests(unittest.TestCase):
    def test_requires_four_response_tokens(self):
        with self.assertRaisesRegex(ValueError, "at least four"):
            token_tsne_coordinates(np.ones((3, 11), dtype=np.float32))

    def test_uses_bounded_perplexity_and_returns_finite_coordinates(self):
        self.assertEqual(token_tsne_perplexity(4), 2)
        self.assertEqual(token_tsne_perplexity(100), 30)
        coordinates = token_tsne_coordinates(np.arange(44, dtype=np.float32).reshape(4, 11))
        self.assertEqual(coordinates.shape, (4, 2))
        self.assertTrue(np.isfinite(coordinates).all())

    def test_embedding_api_does_not_accept_labels(self):
        self.assertNotIn("labels", inspect.signature(token_tsne_coordinates).parameters)


class BehaviorAnalysisTests(unittest.TestCase):
    def test_single_builds_original_graph_from_canonical_split_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = make_split(root)
            output = root / "output"

            result = BehaviorAnalysis(split, output).single("r1")

            self.assertEqual(result["response_tokens"], 4)
            for name in ("behavior.csv", "run_summary.csv", "behavior.png", "token_tsne.png", "token_tsne.npz", "metadata.json"):
                self.assertTrue((output / name).is_file(), name)

    def test_single_loads_labels_only_after_label_free_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = BehaviorAnalysis(make_split(root), root / "output")
            real_labels = analysis.dataset.labels
            embedded = False

            def coordinates(features):
                nonlocal embedded
                embedded = True
                return np.zeros((len(features), 2), dtype=np.float32)

            def labels():
                self.assertTrue(embedded)
                return real_labels()

            with patch("behavior_analysis.token_tsne_coordinates", side_effect=coordinates), patch.object(analysis.dataset, "labels", side_effect=labels):
                analysis.single("r1")

    def test_rejects_non_original_graph_manifest_for_topology_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = make_split(root)
            graph_root = root / "relation_topk"
            GraphDatasetBuilder(BuildConfig(split, graph_root, kind="relation_topk", k_prompt=1, k_history=1, device="cpu")).run()

            with self.assertRaisesRegex(ValueError, "kind == original"):
                BehaviorAnalysis(split, root / "output", graph_root=graph_root)

    def test_cached_original_graph_requires_explicit_tau_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = make_split(root)
            graph_root = root / "original"
            GraphDatasetBuilder(BuildConfig(split, graph_root, kind="original", tau=0.05, device="cpu")).run()

            with patch.object(ResearchSample, "graph", side_effect=AssertionError("graph should not load")):
                with self.assertRaisesRegex(ValueError, "tau"):
                    BehaviorAnalysis(split, root / "output", graph_root=graph_root, tau=0.01)

            analysis = BehaviorAnalysis(split, root / "output", graph_root=graph_root)
            self.assertEqual(analysis.tau, 0.05)

    def test_alignment_loads_each_selected_control_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = make_split(root, with_control=True)
            labels = split / "labels.jsonl"
            labels.write_text(
                json.dumps({"sample_id": "r1", "positive_runs": [[0, 1], [2, 3]]}) + "\n"
                + json.dumps({"sample_id": "r2", "positive_runs": []}) + "\n",
                encoding="utf-8",
            )
            manifest_path = split / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["labels_sha256"] = sha256(labels)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            analysis = BehaviorAnalysis(split, root / "output")
            original_features = analysis._features
            calls = []

            def features(sample):
                calls.append(sample.sample_id)
                return original_features(sample)

            with patch.object(analysis, "_features", side_effect=features):
                analysis.align(run_policy="all")

            self.assertEqual(calls.count("r2"), 1)

    def test_alignment_writes_control_columns_only_when_controls_are_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = make_split(root, with_control=True)
            output = root / "with-controls"
            result = BehaviorAnalysis(split, output).align(radius=1, controls=True)
            self.assertEqual(result["events"], 1)
            with (output / "matched_events.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows[0]), 9)
            self.assertEqual(len(rows[1]), 9)

            without_controls = root / "without-controls"
            BehaviorAnalysis(split, without_controls).align(radius=1, controls=False)
            with (without_controls / "matched_events.csv").open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(len(header), 5)


if __name__ == "__main__":
    unittest.main()
