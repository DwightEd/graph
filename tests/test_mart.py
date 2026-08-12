import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from cache import (
    AttentionSample, index_row, load_attention_sample, save_attention_sample, sha256,
    write_split_index,
)
from attention_graph.evaluate import evaluate_scores
from attention_graph.mart import (
    MART_FEATURES, MartDetector, fit_mart, mart_features, load_mart, score_mart,
    save_mart,
)
from attention_graph.score import load_score_records
from main import _graph_config, parse_args
from research_dataset import ResearchDataset


def _sample():
    # Two channels, prompt tokens 0/1 and response tokens 2/3.  CSR rows are
    # channel * response_token + response_token.
    return AttentionSample(
        "sample", "source", 2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[0.10, 0.10, 0.20, 0.30]], [[0.10, 0.10, 0.10, 0.20]]], dtype=torch.float16),
        torch.tensor([0, 2, 4, 6, 8]),
        torch.tensor([0, 1, 0, 2, 0, 1, 1, 2]),
        torch.tensor([0.20, 0.30, 0.40, 0.10, 0.10, 0.10, 0.20, 0.30], dtype=torch.float16),
        0.01,
    )


def _layer_sample():
    return AttentionSample(
        "layers", "source", 1,
        torch.tensor([10, 11]),
        torch.tensor([[[0.20, 0.20]], [[0.20, 0.20]]], dtype=torch.float16),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.tensor([0.20, 0.60], dtype=torch.float16),
        0.01,
    )


class MartFeatureTests(unittest.TestCase):
    def test_features_preserve_prompt_mass_other_and_entropy(self):
        matrix = mart_features(_sample()).numpy()
        names = {name: index for index, name in enumerate(MART_FEATURES)}
        # At response token 1: channel prompt fractions are .8 and .4.
        self.assertAlmostEqual(
            matrix[1, names["retained_prompt_fraction_mean"]], 0.60, places=3
        )
        self.assertAlmostEqual(matrix[1, names["retained_mass_mean"]], 0.50, places=3)
        # OTHER is 1 - diagonal - retained: .2 and .3.
        self.assertAlmostEqual(
            matrix[1, names["censored_other_mass_mean"]], 0.25, places=3
        )
        channel_0 = -sum(value * np.log(value) for value in (0.4, 0.1, 0.3, 0.2)) / np.log(4)
        channel_1 = -sum(value * np.log(value) for value in (0.2, 0.3, 0.2, 0.3)) / np.log(4)
        self.assertAlmostEqual(
            matrix[1, names["censored_row_entropy_mean"]],
            (channel_0 + channel_1) / 2,
            places=3,
        )

    def test_innovation_uses_only_previous_response_tokens(self):
        base = mart_features(_sample()).numpy()
        changed = _sample()
        changed.response_values[-2:] = torch.tensor([0.90, 0.01], dtype=torch.float16)
        changed_matrix = mart_features(changed).numpy()
        names = {name: index for index, name in enumerate(MART_FEATURES)}
        self.assertEqual(base[0, names["innovation_norm"]], 0.0)
        self.assertEqual(changed_matrix[0, names["innovation_norm"]], 0.0)
        self.assertNotEqual(base[1, names["innovation_norm"]], changed_matrix[1, names["innovation_norm"]])

    def test_layer_drift_preserves_signed_mechanism_changes(self):
        matrix = mart_features(_layer_sample()).numpy()
        names = {name: index for index, name in enumerate(MART_FEATURES)}
        self.assertAlmostEqual(
            matrix[0, names["layer_drift_retained_mass_mean"]], 0.40, places=3
        )
        self.assertAlmostEqual(
            matrix[0, names["layer_drift_retained_prompt_fraction_mean"]], 0.0,
            places=6,
        )

    def test_zero_diagonal_is_not_counted_as_entropy_support(self):
        sample = _layer_sample()
        sample.attention_diagonal[:, :, 1] = 0.0
        matrix = mart_features(sample).numpy()
        names = {name: index for index, name in enumerate(MART_FEATURES)}
        expected_0 = -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)) / np.log(2)
        expected_1 = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4)) / np.log(2)
        self.assertAlmostEqual(
            matrix[0, names["censored_row_entropy_mean"]],
            (expected_0 + expected_1) / 2,
            places=3,
        )


class MartDetectorTests(unittest.TestCase):
    def test_fit_score_is_finite_and_score_does_not_refit(self):
        train = [np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)]
        detector = MartDetector(neighbors=1, position_bins=2).fit(train)
        center = detector.center.copy()
        reference = detector.reference.copy()
        embedding, score = detector.score([np.array([[9.0, 9.0]], dtype=np.float32)])
        self.assertTrue(np.isfinite(embedding).all())
        self.assertTrue(np.isfinite(score).all())
        np.testing.assert_array_equal(center, detector.center)
        np.testing.assert_array_equal(reference, detector.reference)

    def test_position_selects_calibration_bin_but_is_not_scored(self):
        train = [np.array([
            [0.0, -1.0], [0.1, 1.0], [0.9, -1.0], [1.0, 1.0],
        ], dtype=np.float32)]
        detector = MartDetector(neighbors=1, position_bins=2).fit(train)
        self.assertEqual(detector.feature_dim, 1)
        _embedding, score = detector.score([np.array([
            [0.0, -1.0], [1.0, 1.0],
        ], dtype=np.float32)])
        np.testing.assert_allclose(score, 0.0, atol=1e-7)

    def test_rank_deficient_training_features_stay_finite(self):
        positions = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        train = np.stack((positions, positions, np.ones_like(positions)), axis=1)
        detector = MartDetector(neighbors=2, position_bins=2).fit([train])
        embedding, score = detector.score([train[:3]])
        self.assertTrue(np.isfinite(embedding).all())
        self.assertTrue(np.isfinite(score).all())

    def test_reference_size_bounds_knn_memory_deterministically(self):
        train = np.stack((
            np.linspace(0, 1, 20), np.linspace(-1, 1, 20), np.arange(20)
        ), axis=1).astype(np.float32)
        detector = MartDetector(
            neighbors=2, position_bins=2, reference_size=5
        ).fit([train])
        self.assertEqual(detector.reference.shape[0], 5)

    def test_checkpoint_roundtrip_preserves_scores(self):
        detector = MartDetector(neighbors=1, position_bins=2).fit([
            np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
        ])
        expected_embedding, expected_score = detector.score([
            np.array([[0.15, 0.15]], dtype=np.float32)
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mart.npz"
            save_mart(detector, path)
            restored = load_mart(path)
            embedding, score = restored.score([
                np.array([[0.15, 0.15]], dtype=np.float32)
            ])
        np.testing.assert_allclose(embedding, expected_embedding, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(score, expected_score, rtol=1e-6, atol=1e-6)


class MartCliTests(unittest.TestCase):
    def test_cli_parses_fit_and_score_mart(self):
        fit = parse_args(["fit-mart", "--train-split", "train", "--output", "model.json"])
        score = parse_args(["score-mart", "--canonical-split", "test", "--checkpoint", "model.json", "--output", "score.npz"])
        self.assertEqual(fit.command, "fit-mart")
        self.assertEqual(score.command, "score-mart")

    def test_cli_graph_view_defaults_to_typed_mass_cover(self):
        args = parse_args([
            "visualize-graph", "--canonical-split", "test", "--scores", "score.npz",
            "--sample-id", "case", "--output-dir", "view",
        ])
        self.assertEqual(args.selection, "typed_mass_cover")
        self.assertEqual(args.mass_cover, 0.8)
        self.assertEqual(args.display_top_k, 4)
        self.assertIsNone(_graph_config(args).max_edges_per_target)


class MartPipelineTests(unittest.TestCase):
    @staticmethod
    def _split(root, split, sample):
        (root / "attention").mkdir(parents=True)
        path = root / "attention/sample.npz"
        save_attention_sample(sample, path)
        labels = root / "labels.jsonl"
        labels.write_text(
            '{"sample_id":"sample","positive_runs":[[1,2]]}\n', encoding="utf-8"
        )
        write_split_index(
            root,
            [index_row(root, sample, path, metadata={"split": split})],
            attention_floor=sample.attention_floor,
            num_layers=sample.num_layers,
            num_heads=sample.num_heads,
            alignment="post_token_query_at_same_position",
            extra={
                "schema": "ragtruth-attention-split-v1",
                "split": split,
                "observer_model": "observer",
                "generator_model": "generator",
                "labels_sha256": sha256(labels),
            },
        )

    def test_fit_score_and_evaluate_artifacts_are_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            self._split(train_root, "train", _sample())
            self._split(test_root, "test", _sample())
            checkpoint, scores, report = root / "mart.npz", root / "scores.npz", root / "report.json"
            fit_mart(ResearchDataset(train_root), output_path=checkpoint, neighbors=1, position_bins=1)
            score_mart(ResearchDataset(test_root), checkpoint=checkpoint, output_path=scores)
            with mock.patch(
                "research_dataset.load_attention_sample", wraps=load_attention_sample
            ) as loader:
                result = evaluate_scores(
                    ResearchDataset(test_root), score_path=scores, output_path=report
                )
            self.assertEqual(result["token"]["overall"]["n"], 2)
            self.assertEqual(loader.call_count, 1)
            with np.load(scores, allow_pickle=False) as artifact:
                self.assertEqual(artifact["representation"].item(), "mart_mechanism_pca_embedding")

    def test_score_loader_decompresses_each_npz_array_once(self):
        class CountingArchive:
            def __init__(self):
                self.values = {
                    "representation": np.asarray("mart_mechanism_pca_embedding"),
                    "embedding": np.zeros((2, 3), dtype=np.float32),
                    "score": np.zeros(2, dtype=np.float32),
                    "sample_id": np.asarray(["a", "a"]),
                    "source_id": np.asarray(["s", "s"]),
                    "token_index": np.asarray([0, 1], dtype=np.int32),
                }
                self.files = list(self.values)
                self.reads = {name: 0 for name in self.files}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __getitem__(self, name):
                self.reads[name] += 1
                if self.reads[name] > 1:
                    raise AssertionError(f"{name} was decompressed more than once")
                return self.values[name]

        archive = CountingArchive()
        with mock.patch("attention_graph.score.np.load", return_value=archive):
            records = load_score_records("unused.npz")
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
