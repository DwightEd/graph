import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from experiment_protocol import dataset_manifest_sha256
from experiments.spectral_feasibility import experiment as spectral_experiment
from experiments.spectral_feasibility.artifacts import (
    load_score_artifact,
    load_spectral_reference,
)
from experiments.spectral_feasibility.experiment import (
    _localized_channel_anomaly,
    evaluate_score_artifact,
    fit_spectral_reference,
    score_spectral_dataset,
)
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_causal_attention_spectrum,
    rr_spectral_dimension,
)
from experiments.spectral_feasibility.subspace import (
    empirical_upper_tail,
    project_subspace,
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
    source_prefix: str = "s",
):
    (root / "attention").mkdir(parents=True)
    rows = []
    label_rows = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(
            f"r{index}", f"{source_prefix}{index}", float(multiplier)
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


def _minimal_score_artifact():
    residual_score = np.asarray([0.1, 0.2], dtype=np.float32)
    return {
        "schema": np.asarray("rr-spectral-score-v2"),
        "reference_path": np.asarray("reference.npz"),
        "reference_sha256": np.asarray("a" * 64),
        "dataset_manifest_sha256": np.asarray("b" * 64),
        "fit_group_id": np.asarray(["fit-source"]),
        "calibration_group_id": np.asarray(["calibration-source"]),
        "test_group_id": np.asarray(["test-source"]),
        "test_sample_id": np.asarray(["sample"]),
        "audit_scope": np.asarray("complete_split"),
        "sample_id": np.asarray(["sample", "sample"]),
        "source_id": np.asarray(["test-source", "test-source"]),
        "token_index": np.asarray([0, 1], dtype=np.int32),
        "response_length": np.asarray([2, 2], dtype=np.int32),
        "task_type": np.asarray(["QA", "QA"]),
        "data_source": np.asarray(["synthetic", "synthetic"]),
        "generator_model": np.asarray(["generator", "generator"]),
        "rr_embedding": np.zeros((2, 2), dtype=np.float32),
        "rr_residual_energy": np.asarray([1.0, 2.0], dtype=np.float32),
        "rr_latent_energy": np.asarray([1.0, 2.0], dtype=np.float32),
        "rr_ppca_energy": np.asarray([1.0, 2.0], dtype=np.float32),
        "rr_localized_residual": np.asarray([1.0, 2.0], dtype=np.float32),
        "top_channel_index": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        "top_channel_score": np.asarray(
            [[1.0, 0.5], [2.0, 0.25]], dtype=np.float32
        ),
        "score_rr_residual": residual_score,
        "score_rr_latent": np.asarray([0.2, 0.3], dtype=np.float32),
        "score_rr_ppca": np.asarray([0.3, 0.4], dtype=np.float32),
        "score_rr_localized": np.asarray([0.4, 0.5], dtype=np.float32),
        "score": residual_score.copy(),
    }


class SpectralFeasibilityTests(unittest.TestCase):
    def test_v2_score_loader_rejects_malformed_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.npz"
            np.savez_compressed(valid_path, **_minimal_score_artifact())
            loaded = load_score_artifact(valid_path)
            self.assertEqual(str(loaded["schema"].item()), "rr-spectral-score-v2")

            missing = _minimal_score_artifact()
            missing.pop("task_type")
            short_row = _minimal_score_artifact()
            short_row["data_source"] = np.asarray(["synthetic"])
            bad_embedding_shape = _minimal_score_artifact()
            bad_embedding_shape["rr_embedding"] = np.zeros(2, dtype=np.float32)
            mismatched_attribution = _minimal_score_artifact()
            mismatched_attribution["top_channel_score"] = np.ones(
                (2, 1), dtype=np.float32
            )
            fractional_token = _minimal_score_artifact()
            fractional_token["token_index"] = np.asarray([0.0, 1.0])
            integer_embedding = _minimal_score_artifact()
            integer_embedding["rr_embedding"] = np.zeros((2, 2), dtype=np.int32)
            non_finite = _minimal_score_artifact()
            non_finite["score_rr_latent"] = np.asarray([np.nan, 0.2])
            different_primary = _minimal_score_artifact()
            different_primary["score"] = np.asarray([0.1, 0.200001])
            overlapping_audit = _minimal_score_artifact()
            overlapping_audit["test_group_id"] = np.asarray(["fit-source"])
            inconsistent_sample_group = _minimal_score_artifact()
            inconsistent_sample_group["source_id"] = np.asarray(
                ["test-source", "other-test-source"]
            )
            inconsistent_sample_group["test_group_id"] = np.asarray(
                ["test-source", "other-test-source"]
            )
            incomplete_response = _minimal_score_artifact()
            incomplete_response["token_index"] = np.asarray([0, 0], dtype=np.int32)

            cases = (
                ("missing", missing, "misses fields"),
                ("row", short_row, "row columns"),
                ("embedding", bad_embedding_shape, "matrix"),
                ("attribution", mismatched_attribution, "geometry"),
                ("token-dtype", fractional_token, "integer"),
                ("embedding-dtype", integer_embedding, "floating"),
                ("finite", non_finite, "non-finite"),
                ("primary", different_primary, "primary score"),
                ("audit", overlapping_audit, "source-group audit"),
                (
                    "sample-group",
                    inconsistent_sample_group,
                    "source-group audit",
                ),
                ("coverage", incomplete_response, "complete token rows"),
            )
            for name, artifact, message in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.npz"
                    np.savez_compressed(path, **artifact)
                    with self.assertRaisesRegex(ValueError, message):
                        load_score_artifact(path)

    def test_prefix_causal_attention_keeps_signed_strongest_magnitude(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            config = SpectralConfig(top_k=2)
            spectrum = prefix_causal_attention_spectrum(
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

    def test_empirical_tail_is_monotone_with_anomaly_magnitude(self):
        reference = np.asarray([1.0, 2.0, 3.0, 4.0])
        score = empirical_upper_tail(
            reference,
            np.asarray([1.5, 5.0]),
        )
        self.assertGreater(score[1], score[0])

    def test_localized_channel_anomaly_uses_fixed_upper_tail(self):
        energy = np.asarray(
            [[0.0, 1.0, 2.0, 9.0], [0.0, 0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        center = np.zeros(4, dtype=np.float32)
        scale = np.ones(4, dtype=np.float32)
        aggregate, normalized, count = _localized_channel_anomaly(
            energy,
            center,
            scale,
            tail_fraction=0.25,
        )
        self.assertEqual(count, 1)
        np.testing.assert_allclose(aggregate, [9.0, 0.5])
        np.testing.assert_allclose(normalized, energy)

    def test_ppca_retains_in_subspace_distance(self):
        reference = {
            "rr_pca_mean": np.zeros(2, dtype=np.float32),
            "rr_pca_components": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "rr_pca_explained_variance": np.ones(1, dtype=np.float32),
            "rr_pca_noise_variance": np.asarray(1.0, dtype=np.float32),
        }
        projected = project_subspace(
            np.asarray([[0.0, 1.0], [10.0, 0.0]], dtype=np.float32),
            reference,
        )
        self.assertGreater(
            float(projected.residual_energy[0]),
            float(projected.residual_energy[1]),
        )
        self.assertGreater(
            float(projected.latent_energy[1]),
            float(projected.latent_energy[0]),
        )
        self.assertGreater(
            float(projected.ppca_energy[1]),
            float(projected.ppca_energy[0]),
        )

    def test_score_and_evaluation_reject_a_reference_changed_after_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train", [0.8, 1.0, 1.2, 1.1], source_prefix="train"
            )
            test = _write_dataset(
                root / "test", [1.0], positive_sample=0, source_prefix="test"
            )
            reference_path = root / "reference.npz"
            score_path = root / "scores.npz"
            config = SpectralConfig(
                top_k=2,
                position_bins=2,
                pca_dim=2,
                reference_per_sample=3,
                trim_fraction=0.9,
                channel_tail_fraction=0.5,
                attribution_topk=2,
            )
            fit_spectral_reference(train, reference_path, config=config)
            original_reference = reference_path.read_bytes()
            original_spectrum = spectral_experiment.prefix_causal_attention_spectrum

            def score_then_mutate(*args, **kwargs):
                result = original_spectrum(*args, **kwargs)
                reference_path.write_bytes(reference_path.read_bytes() + b"changed")
                return result

            with patch.object(
                spectral_experiment,
                "prefix_causal_attention_spectrum",
                side_effect=score_then_mutate,
            ), self.assertRaisesRegex(ValueError, "frozen file digest"):
                score_spectral_dataset(test, reference_path, score_path)

            reference_path.write_bytes(original_reference)
            score_spectral_dataset(test, reference_path, score_path)
            reference_path.write_bytes(original_reference + b"changed")
            with self.assertRaisesRegex(ValueError, "reference digest"):
                evaluate_score_artifact(test, score_path, root / "report.json")

    def test_label_free_rr_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.8, 1.0, 1.2, 1.1],
                source_prefix="train-s",
            )
            test = _write_dataset(
                root / "test",
                [1.0, 1.1],
                positive_sample=1,
                source_prefix="test-s",
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
            self.assertEqual(fit["fit_groups"], 3)
            self.assertEqual(fit["calibration_groups"], 1)
            self.assertLessEqual(
                fit["retained_fit_tokens"], fit["fit_reference_tokens"]
            )
            with np.load(reference_path, allow_pickle=False) as arrays:
                self.assertEqual(
                    str(arrays["schema"].item()), "rr-spectral-reference-v2"
                )
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("rr_pca_components", arrays.files)
                self.assertIn("rr_pca_explained_variance", arrays.files)
                self.assertIn("rr_pca_noise_variance", arrays.files)
                self.assertIn("channel_center", arrays.files)
                self.assertIn("calibration_rr_residual", arrays.files)
                self.assertIn("calibration_rr_ppca", arrays.files)
                self.assertEqual(
                    str(arrays["train_dataset_manifest_sha256"].item()),
                    dataset_manifest_sha256(train),
                )
                self.assertTrue(
                    set(arrays["fit_group_id"].tolist()).isdisjoint(
                        arrays["calibration_group_id"].tolist()
                    )
                )
                self.assertNotIn("dynamic_coef", arrays.files)
                self.assertNotIn("calibration_prompt", arrays.files)
                self.assertNotIn("rr_untrimmed_pca_components", arrays.files)

            scored = score_spectral_dataset(test, reference_path, score_path)
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            self.assertEqual(
                scored["primary_detector"], "rr_subspace_residual_tail"
            )
            with np.load(reference_path, allow_pickle=False) as reference_arrays:
                expected_fit_groups = set(reference_arrays["fit_group_id"].tolist())
                expected_calibration_groups = set(
                    reference_arrays["calibration_group_id"].tolist()
                )
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertEqual(str(arrays["schema"].item()), "rr-spectral-score-v2")
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("score_rr_residual", arrays.files)
                self.assertIn("score_rr_latent", arrays.files)
                self.assertIn("score_rr_ppca", arrays.files)
                self.assertIn("score_rr_localized", arrays.files)
                self.assertIn("top_channel_index", arrays.files)
                self.assertIn("reference_sha256", arrays.files)
                self.assertEqual(
                    set(arrays["fit_group_id"].tolist()),
                    expected_fit_groups,
                )
                self.assertEqual(
                    set(arrays["calibration_group_id"].tolist()),
                    expected_calibration_groups,
                )
                self.assertEqual(
                    set(arrays["test_group_id"].tolist()),
                    {"test-s0", "test-s1"},
                )
                self.assertEqual(arrays["test_sample_id"].tolist(), ["r0", "r1"])
                self.assertEqual(str(arrays["audit_scope"].item()), "complete_split")
                self.assertEqual(arrays["top_channel_index"].shape, (6, 2))
                np.testing.assert_allclose(
                    arrays["score"], arrays["score_rr_residual"]
                )
                self.assertNotIn("score_dynamic", arrays.files)
                self.assertNotIn("prompt_channel_volume", arrays.files)

            partial_path = root / "partial_scores.npz"
            score_spectral_dataset(test, reference_path, partial_path, limit=1)
            with np.load(partial_path, allow_pickle=False) as arrays:
                self.assertEqual(str(arrays["audit_scope"].item()), "selected_samples")
                self.assertEqual(arrays["test_sample_id"].tolist(), ["r0"])

            with np.load(reference_path, allow_pickle=False) as arrays:
                reference_artifact = {name: arrays[name].copy() for name in arrays.files}
            missing_manifest = dict(reference_artifact)
            missing_manifest.pop("train_dataset_manifest_sha256")
            missing_manifest_path = root / "missing_reference.npz"
            np.savez_compressed(missing_manifest_path, **missing_manifest)
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_spectral_reference(missing_manifest_path)
            malformed_manifest = dict(reference_artifact)
            malformed_manifest["train_dataset_manifest_sha256"] = np.asarray(
                "not-a-digest"
            )
            malformed_manifest_path = root / "malformed_reference.npz"
            np.savez_compressed(malformed_manifest_path, **malformed_manifest)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_spectral_reference(malformed_manifest_path)

            test.manifest["split"] = "train"
            with self.assertRaisesRegex(ValueError, "split"):
                evaluate_score_artifact(test, score_path, report_path)
            test.manifest["split"] = "test"

            report = evaluate_score_artifact(test, score_path, report_path)
            self.assertEqual(report["schema"], "rr-spectral-evaluation-v2")
            self.assertEqual(report["metrics"]["tokens"], 6)
            self.assertEqual(report["metrics"]["positive_tokens"], 1)
            self.assertEqual(
                report["primary_detector"], "rr_subspace_residual_tail"
            )
            self.assertFalse(report["online_causal_score"])
            self.assertEqual(
                report["future_length_conditioned_fields"],
                ["relative_position", "position_bin"],
            )
            self.assertIn("rr_subspace_residual_tail", report["components"])
            self.assertIn("rr_in_subspace_tail", report["components"])
            self.assertIn("rr_ppca_tail", report["components"])
            self.assertIn("rr_localized_channel_tail", report["components"])
            self.assertEqual(len(report["reference_sha256"]), 64)
            self.assertTrue(report_path.is_file())

    def test_score_rejects_a_test_source_reserved_by_the_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.8, 1.0, 1.2, 1.1],
                source_prefix="shared-s",
            )
            test = _write_dataset(
                root / "test",
                [1.0],
                positive_sample=0,
                source_prefix="shared-s",
            )
            reference_path = root / "reference.npz"
            fit_spectral_reference(
                train,
                reference_path,
                config=SpectralConfig(
                    top_k=2,
                    position_bins=2,
                    pca_dim=2,
                    reference_per_sample=3,
                ),
            )

            with self.assertRaisesRegex(ValueError, "disjoint"):
                score_spectral_dataset(
                    test,
                    reference_path,
                    root / "scores.npz",
                )

    def test_tiny_train_smoke_is_rejected_before_pca_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            train = _write_dataset(Path(directory), [0.8, 1.0, 1.2, 1.1])
            config = SpectralConfig(
                top_k=2,
                pca_dim=4,
                reference_per_sample=3,
            )
            with self.assertRaisesRegex(ValueError, "underdetermined"):
                fit_spectral_reference(
                    train,
                    Path(directory) / "reference.npz",
                    config=config,
                )


if __name__ == "__main__":
    unittest.main()
