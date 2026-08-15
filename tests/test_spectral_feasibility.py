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
from experiments.spectral_feasibility.experiment import (
    _empirical_upper_tail,
    _localized_channel_anomaly,
    evaluate_score_artifact,
    fit_spectral_reference,
    score_spectral_dataset,
)
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_laplacian_spectrum,
    prompt_transport_profile,
    rr_spectral_dimension,
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


def _write_dataset(root: Path, multipliers, *, positive_sample: int | None = None):
    (root / "attention").mkdir(parents=True)
    rows = []
    label_rows = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(f"r{index}", f"s{index}", float(multiplier))
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
                    "quality": "good",
                },
            )
        )
        runs = [[1, 2]] if positive_sample == index else []
        label_rows.append({"sample_id": sample.sample_id, "positive_runs": runs})

    labels = root / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows),
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
            "labels_sha256": sha256(labels),
        },
    )
    return ResearchDataset(root)


class SpectralFeasibilityTests(unittest.TestCase):
    def test_prefix_lapeigvals_keep_signed_strongest_magnitude(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            config = SpectralConfig(top_k=2)
            spectrum = prefix_laplacian_spectrum(
                sample, positions=[1, 2], config=config
            )
            np.testing.assert_allclose(
                spectrum[0], [-0.1, 0.0, -0.075, 0.0], atol=2e-4
            )
            np.testing.assert_allclose(
                spectrum[1], [-1.0 / 6.0, 0.0, -0.15, 1.0 / 60.0], atol=3e-4
            )
            self.assertEqual(rr_spectral_dimension(1, 2, 2), 4)
            sample.release_attention()

    def test_prompt_profile_remains_separate_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            profile = prompt_transport_profile(
                sample,
                positions=[1, 2],
                prompt_bins=2,
            ).reshape(2, 2, 2)
            np.testing.assert_allclose(profile[0], 0.0, atol=1e-6)
            np.testing.assert_allclose(profile[1, 0], [0.0, 0.15], atol=2e-4)
            np.testing.assert_allclose(profile[1, 1], [0.05, 0.0], atol=2e-4)
            np.testing.assert_allclose(
                profile[1].sum(axis=1), [0.15, 0.05], atol=2e-4
            )
            sample.release_attention()

    def test_empirical_tail_is_monotone_with_anomaly_magnitude(self):
        reference = np.asarray([1.0, 2.0, 3.0, 4.0])
        bins = np.zeros(4, dtype=np.int16)
        score = _empirical_upper_tail(
            reference,
            bins,
            np.asarray([1.5, 5.0]),
            np.zeros(2, dtype=np.int16),
        )
        self.assertGreater(score[1], score[0])

    def test_localized_channel_anomaly_uses_fixed_upper_tail(self):
        energy = np.asarray(
            [[0.0, 1.0, 2.0, 9.0], [0.0, 0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        bins = np.zeros(2, dtype=np.int16)
        center = np.zeros((1, 4), dtype=np.float32)
        scale = np.ones((1, 4), dtype=np.float32)
        aggregate, normalized, count = _localized_channel_anomaly(
            energy,
            bins,
            center,
            scale,
            tail_fraction=0.25,
        )
        self.assertEqual(count, 1)
        np.testing.assert_allclose(aggregate, [9.0, 0.5])
        np.testing.assert_allclose(normalized, energy)

    def test_label_free_rr_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(root / "train", [0.8, 1.0, 1.2, 1.1])
            test = _write_dataset(
                root / "test", [1.0, 1.1], positive_sample=1
            )
            reference_path = root / "reference.npz"
            score_path = root / "scores.npz"
            report_path = root / "report.json"
            config = SpectralConfig(
                top_k=2,
                position_bins=2,
                pca_dim=2,
                reference_per_sample=3,
                trim_fraction=0.9,
                channel_tail_fraction=0.5,
                attribution_topk=2,
            )

            fit = fit_spectral_reference(train, reference_path, config=config)
            self.assertFalse(fit["labels_read"])
            self.assertEqual(fit["rr_spectral_dim"], 4)
            self.assertLessEqual(
                fit["trimmed_reference_tokens"], fit["reference_tokens"]
            )
            with np.load(reference_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("rr_pca_components", arrays.files)
                self.assertIn("rr_untrimmed_pca_components", arrays.files)
                self.assertIn("channel_center", arrays.files)
                self.assertIn("calibration_rr_global", arrays.files)
                self.assertNotIn("dynamic_coef", arrays.files)
                self.assertNotIn("calibration_prompt", arrays.files)

            scored = score_spectral_dataset(test, reference_path, score_path)
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            self.assertEqual(
                scored["primary_detector"], "rr_trimmed_subspace_tail"
            )
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("score_rr_global", arrays.files)
                self.assertIn("score_rr_localized", arrays.files)
                self.assertIn("score_rr_untrimmed_ablation", arrays.files)
                self.assertIn("top_channel_index", arrays.files)
                self.assertEqual(arrays["top_channel_index"].shape, (6, 2))
                np.testing.assert_allclose(
                    arrays["score"], arrays["score_rr_global"]
                )
                self.assertNotIn("score_dynamic", arrays.files)
                self.assertNotIn("prompt_channel_volume", arrays.files)

            report = evaluate_score_artifact(test, score_path, report_path)
            self.assertEqual(report["metrics"]["tokens"], 6)
            self.assertEqual(report["metrics"]["positive_tokens"], 1)
            self.assertEqual(
                report["primary_detector"], "rr_trimmed_subspace_tail"
            )
            self.assertIn("rr_trimmed_subspace", report["components"])
            self.assertIn("rr_untrimmed_pca_ablation", report["components"])
            self.assertIn("rr_localized_channel_tail", report["components"])
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
