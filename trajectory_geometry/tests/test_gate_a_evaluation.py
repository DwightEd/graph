import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from trajectory_geometry.evaluation import (
    GateAConfig,
    GateAEvaluator,
    _records,
    bootstrap_delta,
    continuation_selection_mask,
    nearest_position_concordance,
    onset_selection_mask,
    token_metrics,
)
from trajectory_geometry.cli import _parser


class GateAEvaluationTests(unittest.TestCase):
    def _write_split(self, root: Path, name: str, source_id: str, sample_id: str, labels: list[int]) -> Path:
        import torch

        split = root / name
        split.mkdir()
        source = split / f"attention_{sample_id}.pt"
        torch.save({"response_id": sample_id, "source_id": source_id, "task_type": "QA", "data_source": "MARCO", "response_idx": 2, "y_token": torch.tensor([0, 0, *labels])}, source)
        feature = split / f"not_the_sample_id_{sample_id}.npz"
        length = len(labels)
        np.savez_compressed(
            feature,
            sample_id=np.asarray(sample_id),
            response_idx=np.asarray(2),
            token_count=np.asarray(length + 2),
            route_embedding=np.arange(length * 4, dtype=np.float32).reshape(length, 4),
            prompt_mass=np.linspace(0.1, 0.9, length),
            history_mass=np.full(length, 0.2),
            self_mass=np.full(length, 0.1),
            unresolved_mass=np.full(length, 0.1),
            temporal_js=np.linspace(0.1, 0.4, length),
            depth_js=np.linspace(0.2, 0.5, length),
            head_js=np.linspace(0.3, 0.6, length),
            route_acceleration=np.linspace(0.4, 0.7, length),
        )
        (split / "manifest.json").write_text(json.dumps({"schema": "trajectory-geometry-route-dynamics-v1", "state": "complete", "embedding_dim": 4, "projection_seed": 7, "prompt_bins": 8, "records": [{"sample_id": sample_id, "source": str(source), "output": str(feature)}]}))
        return split

    def test_known_token_and_onset_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        metrics = token_metrics(labels, scores)
        self.assertEqual(metrics["n"], 6)
        self.assertEqual(metrics["positives"], 3)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)
        np.testing.assert_array_equal(
            onset_selection_mask(labels, np.asarray([0, 4])),
            [True, True, True, False, True, True],
        )
        np.testing.assert_array_equal(
            continuation_selection_mask(labels, np.asarray([0, 4])),
            [True, True, False, True, True, False],
        )

    def test_empirical_low_prompt_tail_scores_low_mass_as_more_anomalous(self) -> None:
        evaluator = GateAEvaluator(GateAConfig(pca_components=2, neighbors=1))
        reference = np.asarray([[0.1], [0.2], [0.7], [0.9]], dtype=np.float32)
        score = evaluator.low_prompt_score(reference, np.asarray([[0.1], [0.9]], dtype=np.float32))
        self.assertGreater(score[0], score[1])

    def test_nearest_position_concordance_skips_all_positive_response(self) -> None:
        labels = np.asarray([1, 1, 0, 1], dtype=np.int8)
        scores = np.asarray([0.8, 0.9, 0.1, 0.7])
        self.assertTrue(np.isfinite(nearest_position_concordance(labels, scores, np.asarray([0, 2]))))
        self.assertTrue(np.isnan(nearest_position_concordance(np.asarray([1, 1]), np.asarray([0.2, 0.3]), np.asarray([0]))))

    def test_cluster_bootstrap_matches_bruteforce_with_ties(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        full = np.asarray([0.2, 0.8, 0.5, 0.5, 0.2, 0.9])
        baseline = np.asarray([0.1, 0.7, 0.4, 0.6, 0.3, 0.7])
        sources = np.asarray(["a", "a", "b", "b", "c", "c"])
        draws = np.random.default_rng(4).integers(0, 3, size=(20, 3))
        result = bootstrap_delta(labels, full, baseline, sources, draws=draws)
        unique = np.unique(sources)
        aurocs, auprcs = [], []
        for draw in draws:
            rows = np.concatenate([np.flatnonzero(sources == unique[index]) for index in draw])
            aurocs.append(roc_auc_score(labels[rows], full[rows]) - roc_auc_score(labels[rows], baseline[rows]))
            auprcs.append(average_precision_score(labels[rows], full[rows]) - average_precision_score(labels[rows], baseline[rows]))
        for name, expected in (("auroc", aurocs), ("auprc", auprcs)):
            self.assertAlmostEqual(result[name]["point"], (token_metrics(labels, full)[name] - token_metrics(labels, baseline)[name]))
            self.assertAlmostEqual(result[name]["ci_low"], float(np.quantile(expected, 0.025)))
            self.assertAlmostEqual(result[name]["ci_high"], float(np.quantile(expected, 0.975)))
            self.assertEqual(result[name]["valid_replicates"], len(expected))

    def test_manifest_uses_npz_sample_id_and_rejects_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._write_split(Path(directory), "train", "source-a", "response-a", [0, 1])
            self.assertEqual(_records(split)[0].sample_id, "response-a")
            source = split / "attention_response-a.pt"
            import torch
            payload = torch.load(source, weights_only=True)
            payload["response_id"] = "different"
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, "identifiers"):
                _records(split)

    def test_labels_are_opened_only_after_label_free_scores_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [self._write_split(root, f"train-{index}", f"source-{index}", f"train-{index}", [0, 1]) for index in range(3)]
            train_dir = root / "train"
            train_dir.mkdir()
            records = []
            for split in train:
                row = json.loads((split / "manifest.json").read_text())["records"][0]
                records.append(row)
                for path in list(split.glob("*.npz")) + list(split.glob("*.pt")):
                    path.replace(train_dir / path.name)
                row["source"] = str(train_dir / Path(row["source"]).name)
                row["output"] = str(train_dir / Path(row["output"]).name)
            (train_dir / "manifest.json").write_text(json.dumps({"schema": "trajectory-geometry-route-dynamics-v1", "state": "complete", "embedding_dim": 4, "projection_seed": 7, "prompt_bins": 8, "records": records}))
            test = self._write_split(root, "test", "test-source", "test-response", [0, 1])
            output = root / "output"
            evaluator = GateAEvaluator(GateAConfig(position_bins=1, length_bins=1, positions_per_sample=2, references_per_group=16, pca_components=2, neighbors=1, bootstrap=4), device="cpu")
            original = evaluator._labels
            def checked(records, score_path):
                self.assertTrue(score_path.is_file())
                return original(records, score_path)
            evaluator._labels = checked  # type: ignore[method-assign]
            report = evaluator.evaluate(train_dir, test, output)
            self.assertTrue((output / "scores_label_free.npz").is_file())
            self.assertIn("overall", report["views"]["full"]["by_task"]["QA"])
            self.assertIn("onset", report["views"]["full"]["by_task"]["QA"])

    def test_label_boundary_rejects_nonbinary_and_prompt_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._write_split(Path(directory), "test", "source", "response", [0, 1])
            record = _records(split)
            score_path = split / "scores_label_free.npz"
            np.savez_compressed(score_path, sample_id=np.asarray(["response", "response"]))
            evaluator = GateAEvaluator(device="cpu")
            import torch
            source = split / "attention_response.pt"
            payload = torch.load(source, weights_only=True)
            payload["y_token"] = torch.tensor([1, 0, 0, 1])
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, "normal prompt"):
                evaluator._labels(record, score_path)
            payload["y_token"] = torch.tensor([0, 0, 0, 2])
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, "binary"):
                evaluator._labels(record, score_path)

    def test_evaluate_cli_and_shell_expose_only_paths_and_device(self) -> None:
        arguments = _parser().parse_args(["evaluate", "--train-features", "train", "--test-features", "test", "--output-dir", "out"])
        self.assertEqual(arguments.device, "cuda")
        script = (Path(__file__).parents[1] / "run_feature_effects.sh").read_text(encoding="utf-8")
        self.assertIn("python -m trajectory_geometry.cli evaluate", script)
        self.assertIn("TRAIN_FEATURE_DIR TEST_FEATURE_DIR [OUTPUT_DIR]", script)


if __name__ == "__main__":
    unittest.main()
