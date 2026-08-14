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
    evaluate_score_artifact,
    fit_spectral_reference,
    score_spectral_dataset,
)
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    causal_spectral_state,
    prefix_laplacian_spectrum,
    spectral_volume,
)
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    diagonal = torch.tensor(
        [[[0.8, 0.7, 0.4, 0.3, 0.2], [0.6, 0.5, 0.2, 0.4, 0.1]]],
        dtype=torch.float16,
    )
    # Canonical CSR requires source columns to be strictly increasing inside
    # every row.  The six rows below are:
    #   [0], [2], [1, 2, 3], [1], [2], [0, 2, 3].
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
):
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
    def test_prefix_lapeigvals_match_direct_causal_degree(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            config = SpectralConfig(top_k=2, prompt_sketch_dim=3)
            spectrum = prefix_laplacian_spectrum(
                sample, positions=[1, 2], config=config
            )
            np.testing.assert_allclose(
                spectrum[0],
                [0.0, -0.1, 0.0, -0.075],
                atol=2e-4,
            )
            np.testing.assert_allclose(
                spectrum[1],
                [0.0, 0.0, 1.0 / 60.0, 0.0],
                atol=3e-4,
            )
            sample.release_attention()

    def test_dual_state_preserves_channel_prompt_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            config = SpectralConfig(top_k=2, prompt_sketch_dim=3)
            state, prompt_volume = causal_spectral_state(
                sample, positions=[1, 2], config=config
            )
            self.assertEqual(state.shape, (2, 10))
            self.assertEqual(prompt_volume.shape, (2,))
            self.assertTrue(np.isfinite(prompt_volume).all())
            # First four dimensions are RR LapEigvals. Prompt sketches then
            # flatten as [head0:3, head1:3], with coordinate 0 exact mass.
            self.assertAlmostEqual(float(state[0, 4]), 0.0, places=6)
            self.assertAlmostEqual(float(state[0, 7]), 0.0, places=6)
            self.assertAlmostEqual(float(state[1, 4]), 0.15, places=3)
            self.assertAlmostEqual(float(state[1, 7]), 0.05, places=3)
            sample.release_attention()

    def test_spectral_logdet_volume_increases_for_nonconstant_trajectory(self):
        constant = np.zeros((4, 3), dtype=np.float32)
        varied = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
            dtype=np.float32,
        )
        constant_volume = spectral_volume(
            constant, window=4, alpha=1e-3
        )[-1]
        varied_volume = spectral_volume(
            varied, window=4, alpha=1e-3
        )[-1]
        self.assertGreater(float(varied_volume), float(constant_volume))

    def test_label_free_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.8, 1.0, 1.2],
            )
            test = _write_dataset(
                root / "test",
                [1.0, 1.1],
                positive_sample=1,
            )
            reference_path = root / "reference.npz"
            score_path = root / "scores.npz"
            report_path = root / "report.json"
            config = SpectralConfig(
                top_k=2,
                prompt_sketch_dim=3,
                position_bins=2,
                pca_dim=2,
                reference_per_sample=2,
                neighbors=1,
                spectral_window=2,
            )

            fit = fit_spectral_reference(
                train, reference_path, config=config
            )
            self.assertFalse(fit["labels_read"])
            self.assertEqual(fit["raw_spectral_dim"], 10)
            with np.load(reference_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("reference_manifold", arrays.files)

            scored = score_spectral_dataset(
                test, reference_path, score_path
            )
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertEqual(arrays["embedding"].shape, (6, 2))
                self.assertEqual(arrays["innovation"].shape, (6, 2))
                self.assertIn("prompt_channel_volume", arrays.files)

            report = evaluate_score_artifact(
                test, score_path, report_path
            )
            self.assertEqual(report["metrics"]["tokens"], 6)
            self.assertEqual(report["metrics"]["positive_tokens"], 1)
            self.assertIn("manifold_knn", report["components"])
            self.assertIn("prompt_channel_volume", report["components"])
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
