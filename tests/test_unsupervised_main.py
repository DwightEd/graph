import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import unsupervised_main


class _Sample:
    def __init__(self, sample_id, source_id):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.data_source = "RAGTruth"
        self.generator_model = "llama31-8b"


class _Dataset:
    instances = []

    def __init__(self, canonical_split, *, device):
        self.canonical_split = canonical_split
        self.device = device
        self.manifest = {"num_layers": 2, "num_heads": 3}
        self.sample_ids = ["train", "heldout", "extra"]
        self.samples = {
            "train": _Sample("train", "source-a"),
            "heldout": _Sample("heldout", "source-b"),
            "extra": _Sample("extra", "source-c"),
        }
        type(self).instances.append(self)

    def __getitem__(self, sample_id):
        return self.samples[sample_id]


class _Evaluator:
    instances = []

    def __init__(self, dataset, *, folds, seed):
        self.dataset = dataset
        self.folds = folds
        self.seed = seed
        self.records_were_label_blind = False
        type(self).instances.append(self)

    def run(self, fit_fold):
        outputs = fit_fold(
            [self.dataset["train"]], [self.dataset["heldout"]], fold=1
        )
        output = outputs["heldout"]
        return [
            {
                "sample_id": "heldout",
                "source_id": "source-b",
                "fold": 1,
                "token_index": index,
                "embedding": embedding,
                "score": score,
                "nll": float(score) + 1.0,
                "task_type": "QA",
                "data_source": "RAGTruth",
                "generator_model": "llama31-8b",
            }
            for index, (embedding, score) in enumerate(
                zip(output["embedding"], output["score"], strict=True)
            )
        ]

    def evaluate(self, records):
        self.records_were_label_blind = all("label" not in row for row in records)
        return [{**row, "label": index % 2} for index, row in enumerate(records)]


class _Method:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_samples = None
        type(self).instances.append(self)

    def fit(self, samples, *, progress=False):
        self.fit_samples = [sample.sample_id for sample in samples]
        self.progress = progress
        return self

    def score(self, samples):
        return {
            sample.sample_id: {
                "embedding": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "score": np.asarray([0.25, 0.75], dtype=np.float32),
                "nll": np.asarray([1.25, 1.75], dtype=np.float32),
            }
            for sample in samples
        }

    def embed(self, samples):
        return {
            sample.sample_id: np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            for sample in samples
        }


class UnsupervisedMainTests(unittest.TestCase):
    def test_main_flows_arguments_through_oof_evaluation_and_writes_artifacts(self):
        _Dataset.instances.clear()
        _Evaluator.instances.clear()
        _Method.instances.clear()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            unsupervised_main, "ResearchDataset", _Dataset
        ), patch.object(unsupervised_main, "AllDataEvaluator", _Evaluator), patch.object(
            unsupervised_main, "UnsupervisedGraphMethod", _Method
        ):
            output_dir = Path(directory) / "outputs"
            summary = unsupervised_main.main([
                "--canonical-split", "canonical/test",
                "--output-dir", str(output_dir),
                "--device", "cpu",
                "--folds", "2",
                "--embedding-dim", "2",
                "--message-steps", "4",
                "--epochs", "7",
                "--density-steps", "9",
                "--seed", "11",
                "--limit", "3",
            ])

            dataset = _Dataset.instances[0]
            evaluator = _Evaluator.instances[0]
            methods = _Method.instances
            self.assertEqual((dataset.canonical_split, dataset.device), ("canonical/test", "cpu"))
            self.assertEqual((evaluator.folds, evaluator.seed), (2, 11))
            self.assertEqual(len(methods), 4)
            self.assertTrue(all(method.fit_samples == ["train"] for method in methods))
            self.assertTrue(all(method.progress for method in methods))
            self.assertEqual(methods[0].kwargs, {
                "num_channels": 6,
                "embedding_dim": 2,
                "message_passing_steps": 4,
                "graph_variant": "full",
                "epochs": 7,
                "fit_steps": 9,
                "seed": 12,
            })
            self.assertEqual(
                [(method.kwargs["graph_variant"], method.kwargs["message_passing_steps"])
                 for method in methods],
                [("full", 4), ("full", 0), ("rewired", 4), ("channel_mean", 4)],
            )
            self.assertTrue(evaluator.records_were_label_blind)

            with np.load(output_dir / "full" / "results.npz") as results:
                self.assertEqual(
                    set(results.files),
                    {
                        "embedding", "score", "nll", "label", "fold", "sample_id",
                        "source_id", "token_index", "task_type", "data_source",
                        "generator_model",
                    },
                )
                np.testing.assert_allclose(results["embedding"], [[1.0, 2.0], [3.0, 4.0]])
                np.testing.assert_allclose(results["score"], [0.25, 0.75])
                np.testing.assert_allclose(results["nll"], [1.25, 1.75])
                np.testing.assert_array_equal(results["label"], [0, 1])
                self.assertEqual(results["sample_id"].tolist(), ["heldout", "heldout"])
                self.assertEqual(results["source_id"].tolist(), ["source-b", "source-b"])
                np.testing.assert_array_equal(results["fold"], [1, 1])
                np.testing.assert_array_equal(results["token_index"], [0, 1])

            persisted = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, summary)
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["positive_labels"], 1)
            self.assertEqual(summary["token_metrics"]["n"], 2)
            for variant in ("full", "no_message", "rewired", "channel_mean"):
                variant_dir = output_dir / variant
                self.assertTrue((variant_dir / "metrics.json").is_file())
                self.assertTrue((variant_dir / "token_scores.csv").is_file())
                self.assertTrue((variant_dir / "embedding_fold_1.png").is_file())
                self.assertTrue((variant_dir / "embedding_fold_1.npz").is_file())

    def test_main_rejects_a_limit_with_fewer_source_groups_than_folds(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            unsupervised_main, "ResearchDataset", _Dataset
        ):
            with self.assertRaisesRegex(ValueError, "at least 3 source groups"):
                unsupervised_main.main([
                    "--canonical-split", "canonical/test",
                    "--output-dir", str(Path(directory) / "outputs"),
                    "--folds", "2",
                    "--limit", "2",
                ])


if __name__ == "__main__":
    unittest.main()
