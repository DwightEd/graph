import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from experiments.rr_signal_audit.artifacts import (
    load_reference,
    load_score_artifact,
)
from experiments.rr_signal_audit.components import (
    RRSignalConfig,
    extract_rr_signal_features,
)
from experiments.rr_signal_audit.experiment import (
    evaluate_rr_signal_audit,
    fit_rr_signal_audit,
    score_rr_signal_audit,
)
from experiments.rr_signal_audit.geometry import (
    RRGeometryConfig,
    shuffle_channel_blocks,
)
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_causal_attention_spectrum,
)
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    diagonal = torch.tensor(
        [[[0.8, 0.7, 0.4, 0.3, 0.2], [0.6, 0.5, 0.2, 0.4, 0.1]]],
        dtype=torch.float16,
    )
    columns = torch.tensor(
        [0, 2, 1, 2, 3, 1, 2, 0, 2, 3],
        dtype=torch.int32,
    )
    values = (
        torch.tensor(
            [0.2, 0.2, 0.15, 0.1, 0.3, 0.25, 0.05, 0.05, 0.4, 0.1],
            dtype=torch.float32,
        )
        * float(multiplier)
    ).to(torch.float16)
    return AttentionSample(
        sample_id,
        source_id,
        2,
        torch.tensor([10, 11, 12, 13, 14]),
        diagonal,
        torch.tensor([0, 1, 2, 5, 6, 7, 10]),
        columns,
        values,
        0.01,
    )


def _write_dataset(
    root: Path,
    multipliers,
    *,
    positive_sample: int | None = None,
    source_prefix: str,
):
    (root / "attention").mkdir(parents=True)
    rows = []
    labels = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(
            f"r{index}",
            f"{source_prefix}{index}",
            float(multiplier),
        )
        path = root / "attention" / f"r{index}.npz"
        save_attention_sample(sample, path)
        rows.append(
            index_row(
                root,
                sample,
                path,
                metadata={
                    "split": "test" if positive_sample is not None else "train",
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                },
            )
        )
        labels.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": [[1, 2]] if positive_sample == index else [],
            }
        )
    label_path = root / "labels.jsonl"
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for row in labels),
        encoding="utf-8",
    )
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=1,
        num_heads=2,
        alignment="post_token_query_at_same_position",
        extra={
            "split": "test" if positive_sample is not None else "train",
            "labels_sha256": sha256(label_path),
        },
    )
    return ResearchDataset(root)


class RRSignalAuditTests(unittest.TestCase):
    def test_mixed_block_reproduces_historical_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(
                Path(directory),
                [1.0],
                source_prefix="one-",
            )
            sample = dataset["r0"]
            config = RRSignalConfig(top_k=2, causal_position_bins=3)
            features = extract_rr_signal_features(sample, config=config)
            expected = prefix_causal_attention_spectrum(
                sample,
                positions=range(3),
                config=SpectralConfig(top_k=2),
            )
            np.testing.assert_allclose(
                features.blocks["mixed_topk"],
                expected,
                atol=3e-4,
            )
            self.assertEqual(features.blocks["received_topk"].shape, (3, 4))
            self.assertEqual(features.blocks["diagonal_topk"].shape, (3, 4))
            self.assertEqual(features.blocks["ratio_topk"].shape, (3, 4))
            self.assertEqual(features.blocks["collapse_channel"].shape, (3, 12))
            self.assertEqual(features.collapse_global.shape, (3, 12))
            sample.release_attention()

    def test_channel_shuffle_preserves_every_conditional_channel_marginal(self):
        values = np.arange(6 * 3 * 2, dtype=np.float32).reshape(6, 6)
        conditions = np.asarray(["a", "a", "a", "b", "b", "b"])
        shuffled = shuffle_channel_blocks(
            values,
            num_channels=3,
            features_per_channel=2,
            conditions=conditions,
            seed=7,
        )
        original = values.reshape(6, 3, 2)
        changed = shuffled.reshape(6, 3, 2)
        for condition in np.unique(conditions):
            rows = np.flatnonzero(conditions == condition)
            for channel in range(3):
                left = sorted(map(tuple, original[rows, channel].tolist()))
                right = sorted(map(tuple, changed[rows, channel].tolist()))
                self.assertEqual(left, right)

    def test_label_free_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 0.95, 1.05],
                source_prefix="train-",
            )
            test = _write_dataset(
                root / "test",
                [1.0, 1.1],
                positive_sample=1,
                source_prefix="test-",
            )
            reference_path = root / "reference.npz"
            score_path = root / "test_scores.npz"
            evaluation_dir = root / "evaluation"

            fitted = fit_rr_signal_audit(
                train,
                reference_path,
                signal_config=RRSignalConfig(
                    top_k=2,
                    lag_bins=3,
                    local_lag_max=1,
                    anchor_count=2,
                    causal_position_bins=3,
                ),
                geometry_config=RRGeometryConfig(
                    relative_position_bins=2,
                    reservoir_rows=32,
                    pca_dim=2,
                    min_condition_rows=2,
                    trim_fraction=1.0,
                    calibration_fraction=0.25,
                    bootstrap_replicates=10,
                    seed=7,
                ),
            )
            self.assertFalse(fitted["labels_read"])
            reference = load_reference(reference_path)
            self.assertNotIn("label", reference)
            self.assertNotIn("y_token", reference)

            scored = score_rr_signal_audit(
                test,
                reference_path,
                score_path,
            )
            self.assertFalse(scored["labels_read"])
            artifact = load_score_artifact(score_path)
            self.assertEqual(len(artifact["sample_id"]), 6)
            self.assertNotIn("label", artifact)
            self.assertNotIn("y_token", artifact)

            report = evaluate_rr_signal_audit(
                test,
                score_path,
                evaluation_dir,
                onset_window=1,
                bootstrap_replicates=10,
                seed=7,
            )
            self.assertTrue(report["labels_read"])
            self.assertEqual(report["tokens"], 6)
            self.assertEqual(report["positive_tokens"], 1)
            self.assertTrue((evaluation_dir / "score_metrics.csv").is_file())
            self.assertTrue((evaluation_dir / "onset_effects.csv").is_file())


if __name__ == "__main__":
    unittest.main()
