from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.conditioned_benchmark.artifacts import ArtifactSpec, load_score_artifact
from experiments.conditioned_benchmark.dataset import align_artifacts


def _spectral_v2_artifact():
    residual = np.asarray([0.1, 0.2], dtype=np.float32)
    return {
        "schema": np.asarray("rr-spectral-score-v2"),
        "reference_path": np.asarray("reference.npz"),
        "reference_sha256": np.asarray("a" * 64),
        "dataset_manifest_sha256": np.asarray("b" * 64),
        "fit_group_id": np.asarray(["fit-source"]),
        "calibration_group_id": np.asarray(["cal-source"]),
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
        "score_rr_residual": residual,
        "score_rr_latent": np.asarray([0.2, 0.3], dtype=np.float32),
        "score_rr_ppca": np.asarray([0.3, 0.4], dtype=np.float32),
        "score_rr_localized": np.asarray([0.4, 0.5], dtype=np.float32),
        "score": residual.copy(),
    }


def _cmrp_v2_artifact():
    rows = 2
    return {
        "schema": np.asarray("cmrp-score-v2"),
        "reference_sha256": np.asarray("a" * 64),
        "model_sha256": np.asarray("b" * 64),
        "dataset_manifest_sha256": np.asarray("c" * 64),
        "fit_group_id": np.asarray(["fit-source"]),
        "calibration_group_id": np.asarray(["cal-source"]),
        "test_group_id": np.asarray(["test-source"]),
        "test_sample_id": np.asarray(["sample"]),
        "audit_scope": np.asarray("complete_split"),
        "sample_id": np.asarray(["sample"] * rows),
        "source_id": np.asarray(["test-source"] * rows),
        "token_index": np.asarray([0, 1], dtype=np.int32),
        "response_length": np.asarray([2, 2], dtype=np.int32),
        "task_type": np.asarray(["QA"] * rows),
        "data_source": np.asarray(["synthetic"] * rows),
        "generator_model": np.asarray(["generator"] * rows),
        "score": np.asarray([0.1, 0.2], dtype=np.float32),
        "raw_route_surprise": np.asarray([1.0, 2.0], dtype=np.float32),
        "presence_nll": np.asarray([1.0, 2.0], dtype=np.float32),
        "source_nll": np.asarray([1.0, 2.0], dtype=np.float32),
        "weight_error": np.asarray([1.0, 2.0], dtype=np.float32),
        "rewired_source_nll": np.asarray([1.1, 2.1], dtype=np.float32),
        "rewire_gap": np.asarray([0.1, 0.1], dtype=np.float32),
        "selected_rr_edges": np.asarray([1, 1], dtype=np.int32),
    }


class ArtifactAdapterTests(unittest.TestCase):
    def test_cmrp_v2_requires_owner_validation_then_registers_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmrp-scores.npz"
            np.savez_compressed(
                path,
                schema=np.asarray("cmrp-score-v2"),
                sample_id=np.asarray(["a", "a", "b"]),
                token_index=np.asarray([0, 1, 0]),
                score=np.asarray([0.1, 0.2, 0.3]),
                raw_route_surprise=np.asarray([1.0, 2.0, 3.0]),
            )
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_score_artifact(ArtifactSpec("cmrp", str(path)))

            np.savez_compressed(path, **_cmrp_v2_artifact())
            artifact = load_score_artifact(ArtifactSpec("cmrp", str(path)))

            self.assertEqual(set(artifact.methods), {"cmrp.primary"})
            method = artifact.methods["cmrp.primary"]
            self.assertEqual(method.protocol, "label_free_frozen_score")
            self.assertEqual(method.source_field, "score")
            np.testing.assert_array_equal(
                method.values, np.asarray([0.1, 0.2], dtype=np.float32)
            )

    def test_legacy_spectral_schema_registers_canonical_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-spectral.npz"
            rows = 3
            np.savez_compressed(
                path,
                schema=np.asarray("rr-spectral-score"),
                sample_id=np.asarray(["a", "a", "b"]),
                token_index=np.asarray([0, 1, 0]),
                score=np.asarray([9.0, 9.0, 9.0]),
                score_rr_residual=np.asarray([0.1, 0.2, 0.3]),
                rr_residual_energy=np.asarray([1.0, 2.0, 3.0]),
                score_rr_latent=np.asarray([0.3, 0.2, 0.1]),
                score_rr_ppca=np.asarray([0.2, 0.3, 0.4]),
                score_rr_localized=np.asarray([0.4, 0.3, 0.2]),
                top_channel_score=np.arange(rows * 2).reshape(rows, 2),
            )
            artifact = load_score_artifact(ArtifactSpec("rr", str(path)))
            self.assertIn("rr.primary", artifact.methods)
            self.assertEqual(
                artifact.methods["rr.primary"].protocol,
                "label_free_frozen_score",
            )
            self.assertEqual(
                artifact.methods["rr.primary"].source_field,
                "score",
            )
            np.testing.assert_array_equal(
                artifact.methods["rr.primary"].values,
                np.asarray([9.0, 9.0, 9.0]),
            )

    def test_spectral_v2_requires_owner_validation_then_registers_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectral-v2.npz"
            fake = _spectral_v2_artifact()
            fake.pop("audit_scope")
            fake["score"] = np.asarray([9.0, 9.0], dtype=np.float32)
            np.savez_compressed(path, **fake)
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_score_artifact(ArtifactSpec("rr", str(path)))

            np.savez_compressed(path, **_spectral_v2_artifact())
            artifact = load_score_artifact(ArtifactSpec("rr", str(path)))
            self.assertIn("rr.primary", artifact.methods)
            self.assertIn("rr.peak_channel", artifact.methods)
            primary = artifact.methods["rr.primary"]
            self.assertEqual(primary.protocol, "label_free_frozen_score")
            self.assertEqual(primary.source_field, "score_rr_residual")
            np.testing.assert_array_equal(
                primary.values, np.asarray([0.1, 0.2], dtype=np.float32)
            )

    def test_named_feature_column_and_lower_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez_compressed(
                path,
                sample_id=np.asarray(["a", "a"]),
                token_index=np.asarray([0, 1]),
                feature_names=np.asarray(["grounding", "rank"]),
                features_z=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            )
            spec = ArtifactSpec.from_mapping(
                {
                    "name": "topology",
                    "path": str(path),
                    "protocol": "fixed_feature",
                    "methods": [
                        {
                            "name": "low_grounding",
                            "field": "features_z",
                            "column": "grounding",
                            "direction": "lower",
                        }
                    ],
                }
            )
            artifact = load_score_artifact(spec)
            np.testing.assert_array_equal(
                artifact.methods["topology.low_grounding"].values,
                np.asarray([-1.0, -3.0]),
            )

    def test_alignment_uses_same_rows_and_reorders(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            np.savez_compressed(
                first,
                sample_id=np.asarray(["a", "a", "b"]),
                token_index=np.asarray([0, 1, 0]),
                score=np.asarray([1.0, 2.0, 3.0]),
            )
            np.savez_compressed(
                second,
                sample_id=np.asarray(["b", "a"]),
                token_index=np.asarray([0, 1]),
                score=np.asarray([30.0, 20.0]),
            )
            artifacts = [
                load_score_artifact(ArtifactSpec("left", str(first))),
                load_score_artifact(ArtifactSpec("right", str(second))),
            ]
            sample, token, methods, _ = align_artifacts(artifacts)
            np.testing.assert_array_equal(sample, np.asarray(["a", "b"]))
            np.testing.assert_array_equal(token, np.asarray([1, 0]))
            np.testing.assert_array_equal(
                methods["right.score"].values, np.asarray([20.0, 30.0])
            )


if __name__ == "__main__":
    unittest.main()
