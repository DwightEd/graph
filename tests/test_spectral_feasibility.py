import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from experiments.spectral_feasibility.experiment import (
    collect_representations,
    evaluate_score_artifact,
    save_representation_artifact,
    score_representation_artifacts,
)
from research_dataset import ResearchDataset


def _write_dataset(root: Path) -> ResearchDataset:
    (root / "attention").mkdir()
    sample = AttentionSample(
        "r1",
        "s1",
        2,
        torch.tensor([1, 2, 3, 4, 5, 6]),
        torch.tensor(
            [[[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]],
            dtype=torch.float16,
        ),
        torch.tensor([0, 2, 4, 7, 10]),
        torch.tensor([0, 1, 0, 2, 1, 2, 3, 0, 3, 4], dtype=torch.int32),
        torch.tensor(
            [0.2, 0.1, 0.1, 0.3, 0.2, 0.15, 0.25, 0.1, 0.2, 0.3],
            dtype=torch.float16,
        ),
        0.01,
    )
    path = root / "attention" / "r1.npz"
    save_attention_sample(sample, path)
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"sample_id": "r1", "positive_runs": [[2, 3]]}) + "\n",
        encoding="utf-8",
    )
    write_split_index(
        root,
        [
            index_row(
                root,
                sample,
                path,
                metadata={
                    "split": "test",
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                    "quality": "good",
                },
            )
        ],
        attention_floor=0.01,
        num_layers=1,
        num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={"split": "test", "labels_sha256": sha256(labels)},
    )
    return ResearchDataset(root)


class SpectralFeasibilityTests(unittest.TestCase):
    def test_representation_and_unlabeled_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            dataset = _write_dataset(root)
            artifact = collect_representations(dataset)
            self.assertEqual(artifact["features"].shape[0], 4)
            self.assertEqual(artifact["features"].shape[1], 22)
            self.assertTrue(np.isfinite(artifact["features"]).all())
            self.assertNotIn("label", artifact)

            train_path = Path(directory) / "train.npz"
            test_path = Path(directory) / "test.npz"
            score_path = Path(directory) / "scores.npz"
            report_path = Path(directory) / "report.json"
            save_representation_artifact(artifact, train_path)
            save_representation_artifact(artifact, test_path)
            result = score_representation_artifacts(
                train_path,
                test_path,
                score_path,
                trim_fraction=0.75,
            )
            self.assertEqual(result["test_tokens"], 4)
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertEqual(arrays["score"].shape, (4,))

            report = evaluate_score_artifact(dataset, score_path, report_path)
            self.assertEqual(report["metrics"]["tokens"], 4)
            self.assertEqual(report["metrics"]["positive_tokens"], 1)
            self.assertTrue(0.0 <= report["metrics"]["auroc"] <= 1.0)
            self.assertTrue(0.0 <= report["metrics"]["auprc"] <= 1.0)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
