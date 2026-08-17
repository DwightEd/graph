import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiment_protocol import dataset_manifest_sha256, file_sha256
from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from experiments.rr_topology_dynamics.artifacts import (
    EVALUATION_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_topology_artifact,
    load_topology_reference,
)
from experiments.rr_topology_dynamics.experiment import (
    TopologyAuditConfig,
    evaluate_topology_artifact,
    first_onset_effects,
    fit_topology_reference,
    score_topology_dataset,
)
from experiments.rr_topology_dynamics.features import (
    TopologyDynamicsConfig,
    _batched_route_spectrum,
    _mean_pairwise_cosine,
    _prompt_groundedness,
    extract_sample_topology_dynamics,
    load_rr_reference,
)
from experiments.spectral_feasibility.experiment import fit_spectral_reference
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_causal_attention_modes,
)
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    diagonal = torch.tensor(
        [[[0.8, 0.7, 0.4, 0.3, 0.2], [0.6, 0.5, 0.2, 0.4, 0.1]]],
        dtype=torch.float16,
    )
    columns = torch.tensor(
        [0, 2, 1, 2, 3, 1, 2, 0, 2, 3], dtype=torch.int32
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
    source_ids=None,
):
    (root / "attention").mkdir(parents=True)
    rows = []
    labels = []
    for index, multiplier in enumerate(multipliers):
        source_id = (
            f"{source_prefix}{index}"
            if source_ids is None
            else str(source_ids[index])
        )
        sample = _sample(f"r{index}", source_id, multiplier)
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
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
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


class RRTopologyDynamicsTests(unittest.TestCase):
    def test_first_onset_uses_only_the_first_zero_to_one_transition(self):
        effects = first_onset_effects(
            np.asarray([0.0, 2.0, 0.0, 10.0, 0.0]),
            np.asarray([0, 1, 0, 1, 0]),
            np.asarray(["response"] * 5),
            np.arange(5),
            window=1,
        )

        np.testing.assert_array_equal(effects, [2.0])

    def test_prefix_modes_preserve_source_identity_and_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            modes = prefix_causal_attention_modes(
                sample,
                positions=[1, 2],
                config=SpectralConfig(top_k=2),
            )
            self.assertEqual(modes.values.shape, (2, 2, 2))
            self.assertEqual(int(modes.source_index[0, 0, 0]), 0)
            self.assertEqual(int(modes.lag[0, 0, 0]), 1)
            self.assertEqual(int(modes.source_index[1, 0, 0]), 0)
            self.assertEqual(int(modes.lag[1, 0, 0]), 2)
            self.assertEqual(int(modes.source_index[1, 1, 0]), 1)
            self.assertEqual(int(modes.lag[1, 1, 0]), 1)
            self.assertEqual(int(modes.source_index[1, 1, 1]), 0)
            self.assertEqual(int(modes.lag[1, 1, 1]), 2)
            sample.release_attention()

    def test_route_rank_and_consensus_distinguish_collapsed_routes(self):
        collapsed = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
        )
        diverse = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
        )
        collapsed_rank = _batched_route_spectrum(collapsed, 1e-8)[0]
        diverse_rank = _batched_route_spectrum(diverse, 1e-8)[0]
        self.assertGreater(float(diverse_rank[0]), float(collapsed_rank[0]))
        active = torch.ones((1, 2), dtype=torch.bool)
        collapsed_consensus = _mean_pairwise_cosine(collapsed, active, 1e-8)
        diverse_consensus = _mean_pairwise_cosine(diverse, active, 1e-8)
        self.assertGreater(
            float(collapsed_consensus[0]), float(diverse_consensus[0])
        )

    def test_prompt_grounding_separates_relay_from_feedback(self):
        prompt = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        grounded_chain = np.zeros((3, 3), dtype=np.float32)
        grounded_chain[1, 0] = 1.0
        grounded_chain[2, 1] = 1.0
        _, grounded, relay, feedback = _prompt_groundedness(
            prompt, grounded_chain, 1e-8
        )
        self.assertGreater(float(grounded[2]), 0.99)
        self.assertGreater(float(relay[2]), 0.99)
        self.assertLess(float(feedback[2]), 0.01)

        no_prompt = np.zeros(3, dtype=np.float32)
        _, ungrounded, _, ungrounded_feedback = _prompt_groundedness(
            no_prompt, grounded_chain, 1e-8
        )
        self.assertLess(float(ungrounded[2]), 0.01)
        self.assertGreater(float(ungrounded_feedback[2]), 0.99)

    def test_label_free_fit_score_then_posthoc_topology_audit(self):
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
            spectral_path = root / "spectral_reference.npz"
            topology_path = root / "topology_reference.npz"
            feature_path = root / "test_features.npz"
            evaluation_dir = root / "evaluation"

            spectral_config = SpectralConfig(
                top_k=2,
                position_bins=2,
                pca_dim=2,
                reference_per_sample=3,
                trim_fraction=0.9,
                channel_tail_fraction=0.5,
                attribution_topk=2,
            )
            spectral_fit = fit_spectral_reference(
                train, spectral_path, config=spectral_config
            )
            self.assertFalse(spectral_fit["labels_read"])

            topology_config = TopologyDynamicsConfig(
                lag_bins=3,
                spectral_top_k=2,
                position_bins=2,
                top_source_count=2,
                recent_lag_max=1,
                mid_lag_max=2,
            )
            audit_config = TopologyAuditConfig(
                reference_per_sample=3,
                min_task_bin_rows=2,
                phase_bins=2,
                onset_window=1,
                bootstrap_replicates=10,
                seed=7,
            )
            fitted = fit_topology_reference(
                train,
                spectral_path,
                topology_path,
                topology_config=topology_config,
                audit_config=audit_config,
            )
            self.assertFalse(fitted["labels_read"])
            self.assertEqual(fitted["feature_dim"], 37)
            with np.load(topology_path, allow_pickle=False) as arrays:
                self.assertEqual(str(arrays["schema"].item()), REFERENCE_SCHEMA)
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("task_center", arrays.files)
                self.assertIn("feature_names", arrays.files)
                self.assertEqual(
                    str(arrays["spectral_reference_path"].item()),
                    str(spectral_path.resolve()),
                )
                self.assertEqual(len(str(arrays["spectral_reference_sha256"].item())), 64)
                self.assertEqual(
                    set(arrays["reference_source_id"].tolist()),
                    {"train-s0", "train-s1", "train-s2", "train-s3"},
                )

            topology_reference = load_topology_reference(topology_path)
            broken_reference = dict(topology_reference)
            broken_reference.pop("feature_names")
            broken_reference_path = root / "broken_topology_reference.npz"
            np.savez_compressed(broken_reference_path, **broken_reference)
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_topology_reference(broken_reference_path)

            other_spectral_path = root / "other_spectral_reference.npz"
            other_spectral_path.write_bytes(spectral_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "identity"):
                score_topology_dataset(
                    test,
                    other_spectral_path,
                    topology_path,
                    root / "wrong_identity_features.npz",
                )

            scored = score_topology_dataset(
                test,
                spectral_path,
                topology_path,
                feature_path,
            )
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            with np.load(feature_path, allow_pickle=False) as arrays:
                self.assertEqual(str(arrays["schema"].item()), SCORE_SCHEMA)
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertEqual(
                    str(arrays["spectral_reference_path"].item()),
                    str(spectral_path.resolve()),
                )
                self.assertEqual(
                    str(arrays["topology_reference_path"].item()),
                    str(topology_path.resolve()),
                )
                self.assertEqual(
                    str(arrays["spectral_reference_sha256"].item()),
                    file_sha256(spectral_path),
                )
                self.assertEqual(
                    str(arrays["topology_reference_sha256"].item()),
                    file_sha256(topology_path),
                )
                self.assertEqual(
                    set(arrays["reference_source_id"].tolist()),
                    {"train-s0", "train-s1", "train-s2", "train-s3"},
                )
                self.assertEqual(
                    set(arrays["test_group_id"].tolist()),
                    {"test-s0", "test-s1"},
                )
                self.assertEqual(arrays["test_sample_id"].tolist(), ["r0", "r1"])
                self.assertEqual(str(arrays["audit_scope"].item()), "complete_split")
                self.assertEqual(
                    str(arrays["dataset_manifest_sha256"].item()),
                    dataset_manifest_sha256(test),
                )
                np.testing.assert_array_equal(arrays["response_length"], [3] * 6)
                self.assertEqual(arrays["features_raw"].shape, (6, 37))
                self.assertEqual(arrays["layer_residual_energy"].shape, (6, 1))
                self.assertEqual(
                    arrays["spectral_rank_residual_energy"].shape, (6, 2)
                )

            artifact = load_topology_artifact(feature_path)
            incomplete_artifact = dict(artifact)
            incomplete_artifact["token_index"] = artifact["token_index"].copy()
            incomplete_artifact["token_index"][1] = 0
            incomplete_artifact_path = root / "incomplete_features.npz"
            np.savez_compressed(incomplete_artifact_path, **incomplete_artifact)
            with self.assertRaisesRegex(ValueError, "complete token rows"):
                load_topology_artifact(incomplete_artifact_path)

            different_dataset = _write_dataset(
                root / "different-test",
                [1.0, 1.1],
                positive_sample=0,
                source_prefix="other-s",
            )
            with self.assertRaisesRegex(ValueError, "dataset manifest"):
                evaluate_topology_artifact(
                    different_dataset, feature_path, evaluation_dir
                )

            same_rows_other_manifest = _write_dataset(
                root / "same-rows-other-manifest",
                [1.0, 1.1],
                positive_sample=1,
                source_prefix="test-s",
            )
            manifest_path = same_rows_other_manifest.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_binding_marker"] = "different"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            same_rows_other_manifest.manifest["dataset_binding_marker"] = "different"
            with self.assertRaisesRegex(ValueError, "dataset manifest"):
                evaluate_topology_artifact(
                    same_rows_other_manifest, feature_path, evaluation_dir
                )

            broken_artifact = dict(artifact)
            broken_artifact.pop("features_z")
            broken_artifact_path = root / "broken_features.npz"
            np.savez_compressed(broken_artifact_path, **broken_artifact)
            with self.assertRaisesRegex(ValueError, "misses fields"):
                load_topology_artifact(broken_artifact_path)

            copied_spectral_path = root / "copied_spectral_reference.npz"
            copied_spectral_path.write_bytes(spectral_path.read_bytes())
            identity_mismatch = dict(artifact)
            identity_mismatch["spectral_reference_path"] = np.asarray(
                str(copied_spectral_path.resolve())
            )
            identity_mismatch_path = root / "identity_mismatch_features.npz"
            np.savez_compressed(identity_mismatch_path, **identity_mismatch)
            with self.assertRaisesRegex(ValueError, "spectral reference identity"):
                evaluate_topology_artifact(
                    test, identity_mismatch_path, evaluation_dir
                )

            digest_mismatch = dict(artifact)
            digest_mismatch["spectral_reference_sha256"] = np.asarray("b" * 64)
            digest_mismatch_path = root / "digest_mismatch_features.npz"
            np.savez_compressed(digest_mismatch_path, **digest_mismatch)
            with self.assertRaisesRegex(ValueError, "spectral reference digest"):
                evaluate_topology_artifact(
                    test, digest_mismatch_path, evaluation_dir
                )

            partial_path = root / "partial_features.npz"
            score_topology_dataset(
                test,
                spectral_path,
                topology_path,
                partial_path,
                limit=1,
            )
            with np.load(partial_path, allow_pickle=False) as arrays:
                self.assertEqual(str(arrays["audit_scope"].item()), "selected_samples")
                self.assertEqual(arrays["test_sample_id"].tolist(), ["r0"])

            original_spectral = spectral_path.read_bytes()
            with np.load(spectral_path, allow_pickle=False) as arrays:
                changed_spectral = {
                    name: arrays[name].copy() for name in arrays.files
                }
            changed_spectral["split_seed"] = np.asarray(999, dtype=np.int64)
            np.savez_compressed(spectral_path, **changed_spectral)
            with self.assertRaisesRegex(ValueError, "digest"):
                score_topology_dataset(
                    test,
                    spectral_path,
                    topology_path,
                    root / "wrong_digest_features.npz",
                )
            spectral_path.write_bytes(original_spectral)

            spectral_reference = load_rr_reference(spectral_path)
            sample = test["r0"]
            extracted = extract_sample_topology_dynamics(
                sample, spectral_reference, config=topology_config
            )
            self.assertEqual(extracted["features"].shape, (3, 37))
            sample.release_attention()

            original_topology = topology_path.read_bytes()
            with np.load(topology_path, allow_pickle=False) as arrays:
                changed_topology = {
                    name: arrays[name].copy() for name in arrays.files
                }
            changed_topology["seed"] = np.asarray(999, dtype=np.int64)
            np.savez_compressed(topology_path, **changed_topology)
            with self.assertRaisesRegex(ValueError, "digest"):
                evaluate_topology_artifact(test, feature_path, evaluation_dir)
            topology_path.write_bytes(original_topology)

            test.manifest["split"] = "train"
            with self.assertRaisesRegex(ValueError, "split"):
                evaluate_topology_artifact(test, feature_path, evaluation_dir)
            test.manifest["split"] = "test"

            report = evaluate_topology_artifact(
                test,
                feature_path,
                evaluation_dir,
                bootstrap_replicates=10,
                onset_window=1,
                phase_bins=2,
                seed=7,
            )
            self.assertEqual(report["schema"], EVALUATION_SCHEMA)
            self.assertEqual(report["overall"]["tokens"], 6)
            self.assertEqual(report["overall"]["positive_tokens"], 1)
            self.assertFalse(
                report["claim_boundaries"]["confidence_available"]
            )
            self.assertIn(
                "route_effective_rank", report["feature_metrics_raw"]
            )
            self.assertIn(
                "route_effective_rank",
                report["within_sample_effects_train_standardized"],
            )
            self.assertIn(
                "route_effective_rank",
                report["first_hallucination_onset_effects_train_standardized"],
            )
            self.assertNotIn("within_sample_effects", report)
            self.assertNotIn("hallucination_onset_effects", report)
            self.assertEqual(
                report["claim_boundaries"]["effect_representation"],
                "train_standardized_features_z",
            )
            self.assertEqual(
                report["claim_boundaries"]["onset_definition"],
                "first_0_to_1_transition_per_response",
            )
            self.assertTrue((evaluation_dir / "report.json").is_file())
            self.assertTrue((evaluation_dir / "onset_effects.csv").is_file())
            self.assertIn(
                "first_0_to_1_transition_per_response",
                (evaluation_dir / "onset_effects.csv").read_text(encoding="utf-8"),
            )

    def test_score_rejects_a_source_used_by_the_topology_reference(self):
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
            spectral_path = root / "spectral_reference.npz"
            topology_path = root / "topology_reference.npz"
            fit_spectral_reference(
                train,
                spectral_path,
                config=SpectralConfig(
                    top_k=2,
                    position_bins=2,
                    pca_dim=2,
                    reference_per_sample=3,
                ),
            )
            fit_topology_reference(
                train,
                spectral_path,
                topology_path,
                topology_config=TopologyDynamicsConfig(
                    lag_bins=3,
                    spectral_top_k=2,
                    position_bins=2,
                    top_source_count=2,
                ),
                audit_config=TopologyAuditConfig(
                    reference_per_sample=3,
                    min_task_bin_rows=2,
                ),
            )

            with self.assertRaisesRegex(ValueError, "disjoint"):
                score_topology_dataset(
                    test,
                    spectral_path,
                    topology_path,
                    root / "features.npz",
                )

    def test_limited_topology_fit_reserves_all_spectral_reference_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.8, 1.0, 1.2, 1.1],
                source_prefix="train-s",
            )
            spectral_path = root / "spectral_reference.npz"
            topology_path = root / "topology_reference.npz"
            fit_spectral_reference(
                train,
                spectral_path,
                config=SpectralConfig(
                    top_k=2,
                    position_bins=2,
                    pca_dim=2,
                    reference_per_sample=3,
                ),
            )
            fit_topology_reference(
                train,
                spectral_path,
                topology_path,
                topology_config=TopologyDynamicsConfig(
                    lag_bins=3,
                    spectral_top_k=2,
                    position_bins=2,
                    top_source_count=2,
                ),
                audit_config=TopologyAuditConfig(
                    reference_per_sample=3,
                    min_task_bin_rows=2,
                ),
                limit=1,
            )

            reference = load_topology_reference(topology_path)
            self.assertEqual(
                set(reference["reference_source_id"].tolist()),
                {"train-s0", "train-s1", "train-s2", "train-s3"},
            )
            test = _write_dataset(
                root / "test",
                [1.0],
                positive_sample=0,
                source_ids=["train-s3"],
            )
            with self.assertRaisesRegex(ValueError, "disjoint"):
                score_topology_dataset(
                    test,
                    spectral_path,
                    topology_path,
                    root / "features.npz",
                )


if __name__ == "__main__":
    unittest.main()
