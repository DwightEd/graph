from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.conditioned_benchmark.artifacts import ArtifactSpec, load_score_artifact
from experiments.conditioned_benchmark.dataset import align_artifacts


class ArtifactAdapterTests(unittest.TestCase):
    def test_spectral_schema_registers_canonical_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            rows = 3
            np.savez_compressed(
                path,
                schema=np.asarray("rr-spectral-score"),
                sample_id=np.asarray(["a", "a", "b"]),
                token_index=np.asarray([0, 1, 0]),
                score=np.asarray([0.1, 0.2, 0.3]),
                score_rr_residual=np.asarray([0.1, 0.2, 0.3]),
                rr_residual_energy=np.asarray([1.0, 2.0, 3.0]),
                score_rr_latent=np.asarray([0.3, 0.2, 0.1]),
                score_rr_ppca=np.asarray([0.2, 0.3, 0.4]),
                score_rr_localized=np.asarray([0.4, 0.3, 0.2]),
                top_channel_score=np.arange(rows * 2).reshape(rows, 2),
            )
            artifact = load_score_artifact(ArtifactSpec("rr", str(path)))
            self.assertIn("rr.primary", artifact.methods)
            self.assertIn("rr.peak_channel", artifact.methods)
            self.assertEqual(
                artifact.methods["rr.primary"].protocol,
                "label_free_frozen_score",
            )
            np.testing.assert_array_equal(
                artifact.methods["rr.peak_channel"].values,
                np.asarray([0.0, 2.0, 4.0]),
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
