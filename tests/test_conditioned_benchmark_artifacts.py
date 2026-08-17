from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment_protocol import FrozenFile
from experiments.conditioned_benchmark.artifacts import (
    ArtifactSpec,
    load_score_artifact,
)


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
        "top_channel_score": np.asarray([[1.0, 0.5], [2.0, 0.25]], dtype=np.float32),
        "score_rr_residual": residual,
        "score_rr_latent": np.asarray([0.2, 0.3], dtype=np.float32),
        "score_rr_ppca": np.asarray([0.3, 0.4], dtype=np.float32),
        "score_rr_localized": np.asarray([0.4, 0.5], dtype=np.float32),
        "score": residual.copy(),
    }


def _topology_v2_artifact(root: Path):
    return {
        "schema": np.asarray("rr-topology-dynamics-features-v2"),
        "spectral_reference_path": np.asarray(str((root / "spectral.npz").resolve())),
        "spectral_reference_sha256": np.asarray("a" * 64),
        "topology_reference_path": np.asarray(str((root / "topology.npz").resolve())),
        "topology_reference_sha256": np.asarray("b" * 64),
        "dataset_manifest_sha256": np.asarray("c" * 64),
        "reference_source_id": np.asarray(["fit-source", "cal-source"]),
        "test_group_id": np.asarray(["test-source"]),
        "test_sample_id": np.asarray(["sample"]),
        "audit_scope": np.asarray("complete_split"),
        "feature_names": np.asarray(["grounding", "rank"]),
        "sample_id": np.asarray(["sample", "sample"]),
        "source_id": np.asarray(["test-source", "test-source"]),
        "task_type": np.asarray(["QA", "QA"]),
        "data_source": np.asarray(["synthetic", "synthetic"]),
        "generator_model": np.asarray(["generator", "generator"]),
        "token_index": np.asarray([0, 1], dtype=np.int32),
        "response_length": np.asarray([2, 2], dtype=np.int32),
        "position_bin": np.asarray([0, 1], dtype=np.int32),
        "relative_position": np.asarray([0.0, 1.0], dtype=np.float32),
        "features_raw": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "features_z": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "layer_route_effective_rank": np.ones((2, 1), dtype=np.float32),
        "layer_route_consensus": np.ones((2, 1), dtype=np.float32),
        "layer_residual_energy": np.ones((2, 1), dtype=np.float32),
        "spectral_rank_residual_energy": np.ones((2, 1), dtype=np.float32),
        "rr_embedding": np.ones((2, 1), dtype=np.float32),
    }


class StrictArtifactTests(unittest.TestCase):
    def test_example_configuration_uses_the_live_artifact_spec_contract(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "conditioned_benchmark"
            / "config.example.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        specs = [ArtifactSpec.from_mapping(value) for value in payload["artifacts"]]

        self.assertEqual(specs[2].column, "prompt_groundedness")
        self.assertEqual(specs[2].direction, "lower")
        for obsolete in ("adapter", "protocol", "methods", "features_z"):
            with (
                self.subTest(obsolete=obsolete),
                self.assertRaisesRegex(ValueError, "unsupported artifact settings"),
            ):
                ArtifactSpec.from_mapping(
                    {"name": "obsolete", "path": "scores.npz", obsolete: "x"}
                )

    def test_spectral_v2_uses_owner_validation_and_exposes_only_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectral.npz"
            malformed = _spectral_v2_artifact()
            malformed.pop("audit_scope")
            np.savez_compressed(path, **malformed)
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_score_artifact(
                    ArtifactSpec("spectral", str(path)), FrozenFile.capture(path)
                )

            incomplete = _spectral_v2_artifact()
            incomplete["token_index"] = np.asarray([0, 0], dtype=np.int32)
            np.savez_compressed(path, **incomplete)
            with self.assertRaisesRegex(ValueError, "complete token rows"):
                load_score_artifact(
                    ArtifactSpec("spectral", str(path)), FrozenFile.capture(path)
                )

            np.savez_compressed(path, **_spectral_v2_artifact())
            artifact = load_score_artifact(
                ArtifactSpec("spectral", str(path)), FrozenFile.capture(path)
            )

        self.assertEqual(set(artifact.methods), {"spectral.primary"})
        self.assertEqual(
            artifact.methods["spectral.primary"].source_field,
            "score_rr_residual",
        )
        self.assertEqual(artifact.dataset_manifest_sha256, "b" * 64)
        self.assertEqual(
            set(artifact.evaluation_rows()),
            {
                "dataset_manifest_sha256",
                "sample_id",
                "source_id",
                "token_index",
                "response_length",
            },
        )

    def test_rr_requires_owner_validity_and_an_explicit_features_z_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "topology-features.npz"
            spec = ArtifactSpec(
                "topology",
                str(path),
                column="grounding",
                direction="lower",
            )
            for missing in ("audit_scope", "topology_reference_sha256"):
                malformed = _topology_v2_artifact(root)
                malformed.pop(missing)
                np.savez_compressed(path, **malformed)
                with (
                    self.subTest(missing=missing),
                    self.assertRaisesRegex(ValueError, "misses fields"),
                ):
                    load_score_artifact(spec, FrozenFile.capture(path))

            np.savez_compressed(path, **_topology_v2_artifact(root))
            artifact = load_score_artifact(spec, FrozenFile.capture(path))

        method = artifact.methods["topology.grounding"]
        self.assertEqual(method.protocol, "label_free_feature_fixed_direction")
        self.assertEqual(method.source_direction, "lower")
        np.testing.assert_array_equal(method.values, [-1.0, -3.0])

    def test_legacy_and_unversioned_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for schema in ("rr-spectral-score", "cmrp-score-v1", "unversioned"):
                path = root / f"{schema}.npz"
                np.savez_compressed(
                    path,
                    schema=np.asarray(schema),
                    sample_id=np.asarray(["sample"]),
                    token_index=np.asarray([0]),
                    score=np.asarray([0.1]),
                )
                with (
                    self.subTest(schema=schema),
                    self.assertRaisesRegex(ValueError, "unsupported.*schema"),
                ):
                    load_score_artifact(
                        ArtifactSpec("legacy", str(path)),
                        FrozenFile.capture(path),
                    )


if __name__ == "__main__":
    unittest.main()
