from __future__ import annotations

import csv
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
from experiment_protocol import (
    EvaluationLabels,
    FrozenFile,
    TemporalScope,
    dataset_manifest_sha256,
    file_sha256,
)
from experiments.causal_isomorphism_trajectory.artifacts import (
    score_temporal_scope as cmrp_scope,
)
from experiments.conditioned_benchmark.artifacts import ArtifactSpec
from experiments.conditioned_benchmark.conditions import (
    aggregate_responses,
    prevalence_weights,
)
from experiments.conditioned_benchmark.dataset import build_benchmark_frame
from experiments.conditioned_benchmark.metrics import METRICS, evaluate_metrics
from experiments.conditioned_benchmark.runner import (
    REPORT_SCHEMA,
    BenchmarkConfig,
    ConditionedBenchmark,
)
from experiments.conditioned_benchmark.types import (
    EvaluatedArtifact,
    MethodScore,
    ScoreArtifact,
)
from experiments.rr_topology_dynamics.artifacts import (
    score_temporal_scope as topology_scope,
)
from experiments.spectral_feasibility.artifacts import (
    score_temporal_scope as spectral_scope,
)
from research_dataset import ResearchDataset, open_research_dataset


class _Sample:
    source_id = "test-source"
    task_type = "QA"
    data_source = "synthetic"
    generator_model = "generator"


class _Dataset:
    def __getitem__(self, sample_id):
        if str(sample_id) != "response":
            raise KeyError(sample_id)
        return _Sample()


class _TrackingResearchDataset(ResearchDataset):
    def __init__(self, root):
        super().__init__(root)
        self.labels_called = False

    def prepare_evaluation_labels(self):
        self.labels_called = True
        return super().prepare_evaluation_labels()


def _attention_sample(sample_id: str, source_id: str):
    return AttentionSample(
        sample_id,
        source_id,
        2,
        torch.tensor([10, 11, 12, 13, 14]),
        torch.ones((1, 1, 5), dtype=torch.float16),
        torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.float16),
        0.01,
    )


def _write_dataset(root: Path):
    (root / "attention").mkdir(parents=True)
    rows = []
    label_rows = []
    for index in range(4):
        sample = _attention_sample(f"r{index}", f"test-source-{index}")
        path = root / "attention" / f"r{index}.npz"
        save_attention_sample(sample, path)
        rows.append(
            index_row(
                root,
                sample,
                path,
                metadata={
                    "split": "test",
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                },
            )
        )
        label_rows.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": [[2, 3]] if index in {0, 2} else [],
            }
        )
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
        num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={"split": "test", "labels_sha256": sha256(labels)},
    )
    return ResearchDataset(root)


def _write_cmrp_scores(path: Path, dataset: ResearchDataset):
    """Write a strict CITG artifact while preserving the legacy helper name."""
    reference_path = path.parent / "reference.npz"
    reference_fields = {
        "schema": np.asarray("citg-reference-v1"),
        "train_dataset_manifest_sha256": np.asarray("a" * 64),
        "event_config_json": np.asarray("{}"),
        "signature_config_json": np.asarray("{}"),
        "geometry_config_json": np.asarray("{}"),
        "fit_group_id": np.asarray(["fit-source"]),
        "calibration_group_id": np.asarray(["calibration-source"]),
        "topology_gate_token_count": np.asarray(4, dtype=np.int32),
        "topology_gate_evaluated_tokens": np.asarray(4, dtype=np.int32),
        "topology_gate_coverage": np.asarray(1.0, dtype=np.float32),
        "topology_gate_source_groups": np.asarray(2, dtype=np.int32),
        "topology_gate_mean_gap": np.asarray(0.1, dtype=np.float32),
        "topology_gate_median_gap": np.asarray(0.1, dtype=np.float32),
        "topology_gate_positive_group_fraction": np.asarray(1.0, dtype=np.float32),
        "topology_gate_ci_low": np.asarray(0.05, dtype=np.float32),
        "topology_gate_ci_high": np.asarray(0.15, dtype=np.float32),
        "topology_gate_pass": np.asarray(True),
    }
    for variant in ("full", "static", "topology", "mass"):
        prefix = f"{variant}_"
        reference_fields.update(
            {
                f"calibration_energy_{variant}": np.asarray(
                    [0.1, 0.2], dtype=np.float32
                ),
                prefix + "condition_names": np.asarray(["QA\x1f0"]),
                prefix + "condition_center": np.zeros((1, 2), dtype=np.float32),
                prefix + "condition_scale": np.ones((1, 2), dtype=np.float32),
                prefix + "condition_count": np.asarray([4], dtype=np.int32),
                prefix + "global_center": np.zeros(2, dtype=np.float32),
                prefix + "global_scale": np.ones(2, dtype=np.float32),
                prefix + "rr_pca_mean": np.zeros(2, dtype=np.float32),
                prefix + "rr_pca_components": np.asarray(
                    [[1.0, 0.0]], dtype=np.float32
                ),
                prefix + "rr_pca_explained_variance": np.ones(
                    1, dtype=np.float32
                ),
                prefix + "rr_pca_noise_variance": np.asarray(
                    1.0, dtype=np.float32
                ),
                prefix + "feature_names": np.asarray(["f0", "f1"]),
                prefix + "fit_rows": np.asarray(4, dtype=np.int32),
                prefix + "retained_fit_rows": np.asarray(4, dtype=np.int32),
                prefix + "provisional_residual_median": np.asarray(
                    0.0, dtype=np.float32
                ),
            }
        )
    np.savez_compressed(reference_path, **reference_fields)

    sample_id = np.repeat(np.asarray(dataset.sample_ids, dtype=str), 3)
    token_index = np.tile(np.arange(3, dtype=np.int32), len(dataset.sample_ids))
    source_by_sample = {
        sample: dataset[sample].source_id for sample in dataset.sample_ids
    }
    source_id = np.asarray([source_by_sample[value] for value in sample_id])
    positive_sample = np.isin(sample_id, ["r0", "r2"])
    rows = len(sample_id)
    score = np.where(positive_sample, 0.9, 0.1).astype(np.float32)
    np.savez_compressed(
        path,
        schema=np.asarray("citg-score-v1"),
        reference_path=np.asarray(str(reference_path.resolve())),
        reference_sha256=np.asarray(file_sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(dataset_manifest_sha256(dataset)),
        fit_group_id=np.asarray(["fit-source"]),
        calibration_group_id=np.asarray(["calibration-source"]),
        test_group_id=np.asarray(sorted(set(source_id.tolist()))),
        test_sample_id=np.asarray(dataset.sample_ids),
        audit_scope=np.asarray("complete_split"),
        sample_id=sample_id,
        source_id=source_id,
        token_index=token_index,
        response_length=np.full(rows, 3, dtype=np.int32),
        task_type=np.asarray(["QA"] * rows),
        data_source=np.asarray(["synthetic"] * rows),
        generator_model=np.asarray(["generator"] * rows),
        score=score,
        score_static=score.copy(),
        score_topology=score.copy(),
        score_mass=score.copy(),
        energy_full=score.copy(),
        energy_static=score.copy(),
        energy_topology=score.copy(),
        energy_mass=score.copy(),
        rewired_energy_full=score.copy(),
        rewire_energy_gap=np.zeros(rows, dtype=np.float32),
        rewire_valid=np.ones(rows, dtype=bool),
    )


def _evaluated(name, token_index, token_label, response_positive):
    token_index = np.asarray(token_index, dtype=np.int32)
    rows = len(token_index)
    score = ScoreArtifact(
        name=name,
        path=f"{name}.npz",
        schema="test-v2",
        sample_id=np.asarray(["response"] * rows),
        source_id=np.asarray(["test-source"] * rows),
        token_index=token_index,
        response_length=np.asarray([3] * rows, dtype=np.int32),
        audit_scope="selected_samples",
        dataset_manifest_sha256="a" * 64,
        methods={
            f"{name}.primary": MethodScore(
                f"{name}.primary",
                np.arange(rows, dtype=np.float32),
                protocol="label_free_frozen_score",
            )
        },
    ).validate()
    labels = EvaluationLabels(
        token_label=np.asarray(token_label, dtype=np.int8),
        response_positive=np.asarray(response_positive, dtype=np.int8),
        source_id=np.asarray(["test-source"] * rows),
        response_length=np.asarray([3] * rows, dtype=np.int32),
    )
    return EvaluatedArtifact(
        score=score,
        labels=labels,
    )


class CanonicalBenchmarkFrameTests(unittest.TestCase):
    def test_owner_scopes_survive_intersection_subset_and_response_aggregation(self):
        scope = TemporalScope(
            online_causal_score=False,
            future_length_conditioned_fields=("relative_position", "position_bin"),
        )
        complete = _evaluated("complete", [0, 1, 2], [0, 0, 1], [1, 1, 1])
        method = next(iter(complete.score.methods.values()))
        complete.score.methods = {
            method.name: MethodScore(
                method.name, method.values, temporal_scope=scope
            )
        }

        frame = build_benchmark_frame([complete], _Dataset())
        aggregated = aggregate_responses(frame.subset(frame.token_index < 2))

        self.assertEqual(next(iter(aggregated.methods.values())).temporal_scope, scope)

    def test_owner_temporal_scopes_distinguish_causal_and_offline_scores(self):
        self.assertTrue(cmrp_scope().online_causal_score)
        self.assertFalse(spectral_scope().online_causal_score)
        self.assertEqual(
            spectral_scope().future_length_conditioned_fields,
            ("relative_position", "position_bin"),
        )
        self.assertEqual(
            topology_scope("offline_source_distance_to_final").offline_future_features,
            ("offline_source_distance_to_final",),
        )
    def test_intersection_cannot_erase_a_canonical_positive_response_label(self):
        complete = _evaluated(
            "complete",
            [0, 1, 2],
            [0, 0, 1],
            [1, 1, 1],
        )
        missing_hallucinated_token = _evaluated(
            "partial",
            [0, 1],
            [0, 0],
            [1, 1],
        )

        frame = build_benchmark_frame(
            [complete, missing_hallucinated_token], _Dataset()
        )
        aggregated = aggregate_responses(frame, aggregation="max")

        np.testing.assert_array_equal(frame.labels, [0, 0])
        np.testing.assert_array_equal(frame.response_positive, [1, 1])
        np.testing.assert_array_equal(aggregated.labels, [1])


class ConditionedMetricTests(unittest.TestCase):
    def test_prevalence_reweighting_hits_target(self):
        labels = np.asarray([0, 0, 0, 1])
        weights = prevalence_weights(labels, 0.5)
        self.assertAlmostEqual(float(np.average(labels, weights=weights)), 0.5)

    def test_operational_alert_metrics_remain_available(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.0, 0.1, 0.8, 0.9])
        values = evaluate_metrics(
            labels,
            scores,
            np.ones(4),
            ("precision_at_alert_10", "recall_at_alert_10", "f1_at_alert_10"),
        )
        self.assertTrue(all(value > 0.0 for value in values.values()))

    def test_auprc_lift_is_registered_as_prevalence_sensitive(self):
        labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
        native = float(labels.mean())

        def weights(target):
            result = np.ones(len(labels), dtype=np.float64)
            result[labels == 1] = target / native
            result[labels == 0] = (1.0 - target) / (1.0 - native)
            return result

        low = evaluate_metrics(labels, scores, weights(0.25), ("auprc_lift",))
        high = evaluate_metrics(labels, scores, weights(0.50), ("auprc_lift",))

        self.assertTrue(METRICS["auprc_lift"].prevalence_sensitive)
        self.assertNotAlmostEqual(low["auprc_lift"], high["auprc_lift"])

    def test_position_filter_cannot_erase_a_canonical_positive_response_label(self):
        complete = _evaluated(
            "complete",
            [0, 1, 2],
            [0, 0, 1],
            [1, 1, 1],
        )
        frame = build_benchmark_frame([complete], _Dataset())

        positioned = frame.subset(frame.relative_position <= 0.5)
        aggregated = aggregate_responses(positioned, aggregation="mean")

        np.testing.assert_array_equal(positioned.labels, [0, 0])
        np.testing.assert_array_equal(aggregated.labels, [1])


class ConditionedBenchmarkTests(unittest.TestCase):
    def test_benchmark_configuration_rejects_unknown_settings(self):
        with self.assertRaisesRegex(ValueError, "unsupported benchmark settings"):
            BenchmarkConfig.from_mapping({"ratio_modes": "subsample"})

    def test_main_rejects_unknown_top_level_configuration_settings(self):
        from experiments.conditioned_benchmark.main import main

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "split_root": "test",
                        "output_dir": "report",
                        "artifacts": [{"name": "cmrp", "path": "score.npz"}],
                        "benchmark": {},
                        "devcie": "cpu",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "unsupported top-level configuration settings"
            ):
                main(["--config", str(config_path)])

    def test_reweight_without_resamples_does_not_claim_a_confidence_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            output_dir = root / "report"
            _write_cmrp_scores(score_path, dataset)
            report = ConditionedBenchmark(
                BenchmarkConfig(
                    task_types=("all",),
                    positive_rates=("native",),
                    metrics=("auroc",),
                    bootstrap_replicates=0,
                )
            ).run(
                dataset.root,
                output_dir,
                [ArtifactSpec("cmrp", str(score_path))],
            )
            with (output_dir / "metrics_long.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_row = next(csv.DictReader(handle))

        complete = next(
            condition
            for condition in report["conditions"]
            if condition["state"] == "complete"
        )
        metric = complete["methods"]["cmrp.primary"]["metrics"]["auroc"]
        self.assertEqual(
            metric,
            {
                "value": metric["value"],
                "uncertainty_scope": "not_estimated_insufficient_resamples",
            },
        )
        self.assertEqual(csv_row["ci_low"], "")
        self.assertEqual(csv_row["ci_high"], "")

    def test_native_subsample_is_a_single_point_without_repeat_quantiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            output_dir = root / "report"
            _write_cmrp_scores(score_path, dataset)
            report = ConditionedBenchmark(
                BenchmarkConfig(
                    task_types=("all",),
                    positive_rates=("native",),
                    metrics=("auroc",),
                    ratio_mode="subsample",
                    ratio_repeats=20,
                    bootstrap_replicates=0,
                )
            ).run(
                dataset.root,
                output_dir,
                [ArtifactSpec("cmrp", str(score_path))],
            )
            with (output_dir / "metrics_long.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_row = next(csv.DictReader(handle))

        complete = next(
            condition
            for condition in report["conditions"]
            if condition["state"] == "complete"
        )
        metric = complete["methods"]["cmrp.primary"]["metrics"]["auroc"]
        self.assertEqual(
            metric,
            {
                "value": metric["value"],
                "uncertainty_scope": "not_estimated_native_prevalence",
            },
        )
        self.assertEqual(csv_row["repeat_q025"], "")
        self.assertEqual(csv_row["repeat_q975"], "")

    def test_all_artifact_bindings_are_validated_before_any_labels_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            valid_path = root / "valid.npz"
            invalid_path = root / "invalid.npz"
            _write_cmrp_scores(valid_path, dataset)
            _write_cmrp_scores(invalid_path, dataset)
            with np.load(invalid_path, allow_pickle=False) as arrays:
                invalid = {name: arrays[name].copy() for name in arrays.files}
            invalid["dataset_manifest_sha256"] = np.asarray("f" * 64)
            np.savez_compressed(invalid_path, **invalid)
            tracked = _TrackingResearchDataset(dataset.root)

            with (
                patch(
                    "experiments.conditioned_benchmark.runner.open_research_dataset",
                    return_value=tracked,
                ),
                self.assertRaisesRegex(ValueError, "dataset manifest"),
            ):
                ConditionedBenchmark(
                    BenchmarkConfig(
                        task_types=("all",),
                        positive_rates=("native",),
                        metrics=("auroc",),
                    )
                ).run(
                    dataset.root,
                    root / "report",
                    [
                        ArtifactSpec("valid", str(valid_path)),
                        ArtifactSpec("invalid", str(invalid_path)),
                    ],
                )

        self.assertFalse(tracked.labels_called)

    def test_forged_owner_groups_are_rejected_before_conditioned_labels_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            _write_cmrp_scores(score_path, dataset)
            with np.load(score_path, allow_pickle=False) as arrays:
                forged = {name: arrays[name].copy() for name in arrays.files}
            forged["calibration_group_id"] = np.asarray(["forged-calibration"])
            np.savez_compressed(score_path, **forged)
            tracked = _TrackingResearchDataset(dataset.root)

            with (
                patch(
                    "experiments.conditioned_benchmark.runner.open_research_dataset",
                    return_value=tracked,
                ),
                self.assertRaisesRegex(ValueError, "source groups differ"),
            ):
                ConditionedBenchmark(
                    BenchmarkConfig(
                        task_types=("all",),
                        positive_rates=("native",),
                        metrics=("auroc",),
                    )
                ).run(
                    dataset.root,
                    root / "report",
                    [ArtifactSpec("cmrp", str(score_path))],
                )

        self.assertFalse(tracked.labels_called)

    def test_benchmark_opens_a_hash_verified_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            _write_cmrp_scores(score_path, dataset)
            with patch(
                "experiments.conditioned_benchmark.runner.open_research_dataset",
                wraps=open_research_dataset,
            ) as open_dataset:
                ConditionedBenchmark(
                    BenchmarkConfig(
                        task_types=("all",),
                        positive_rates=("native",),
                        metrics=("auroc",),
                        bootstrap_replicates=0,
                    )
                ).run(
                    dataset.root,
                    root / "report",
                    [ArtifactSpec("cmrp", str(score_path))],
                )

        self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])

    def test_class_runs_the_sealed_workflow_with_full_response_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            output_dir = root / "report"
            _write_cmrp_scores(score_path, dataset)
            config = BenchmarkConfig(
                task_types=("all",),
                positive_rates=("native",),
                metrics=("auroc",),
                evaluation_unit="response",
                relative_position_max=0.5,
                bootstrap_replicates=4,
            )

            report = ConditionedBenchmark(config).run(
                dataset.root,
                output_dir,
                [ArtifactSpec("cmrp", str(score_path))],
            )
            expected_sha256 = FrozenFile.capture(score_path).sha256
            expected_manifest = dataset_manifest_sha256(dataset)

        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(REPORT_SCHEMA, "conditioned-detector-benchmark-v2")
        self.assertNotIn("aligned_rows", report)
        self.assertEqual(report["aligned_token_rows"], 12)
        self.assertEqual(report["aligned_samples"], 4)
        self.assertEqual(report["evaluated_rows"], 4)
        self.assertEqual(report["evaluated_samples"], 4)
        self.assertEqual(report["evaluation_unit"], "response")
        complete = next(
            condition
            for condition in report["conditions"]
            if condition["state"] == "complete"
        )
        self.assertEqual(complete["native_positives"], 2)
        metric = complete["methods"]["cmrp.primary"]["metrics"]["auroc"]
        self.assertEqual(
            set(metric),
            {"value", "uncertainty_scope", "ci_low", "ci_high"},
        )
        self.assertEqual(
            metric["uncertainty_scope"],
            "source_cluster_bootstrap_percentile_95",
        )
        self.assertEqual(report["artifacts"][0]["sha256"], expected_sha256)
        self.assertEqual(
            report["artifacts"][0]["dataset_manifest_sha256"],
            expected_manifest,
        )
        self.assertEqual(report["artifacts"][0]["evaluation_rows"], 12)
        self.assertEqual(
            report["methods"]["cmrp.primary"]["temporal_scope"],
            TemporalScope(online_causal_score=True).as_dict(),
        )
        self.assertTrue(
            report["evaluation_transforms"]["relative_position_filter"]
            ["uses_final_response_length"]
        )
        self.assertTrue(
            report["evaluation_transforms"]["response_aggregation"]
            ["uses_full_response"]
        )

    def test_subsample_reports_repeat_variability_without_ci_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            output_dir = root / "report"
            _write_cmrp_scores(score_path, dataset)
            report = ConditionedBenchmark(
                BenchmarkConfig(
                    task_types=("all",),
                    positive_rates=(0.5,),
                    metrics=("auprc",),
                    ratio_mode="subsample",
                    ratio_repeats=5,
                    bootstrap_replicates=0,
                )
            ).run(
                dataset.root,
                output_dir,
                [ArtifactSpec("cmrp", str(score_path))],
            )
            long_header = (
                (output_dir / "metrics_long.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

        complete = next(
            condition
            for condition in report["conditions"]
            if condition["state"] == "complete"
        )
        metric = complete["methods"]["cmrp.primary"]["metrics"]["auprc"]
        self.assertEqual(
            set(metric),
            {"value", "uncertainty_scope", "repeat_q025", "repeat_q975"},
        )
        self.assertEqual(
            metric["uncertainty_scope"],
            "repeated_row_subsample_variability",
        )
        self.assertNotIn("ci_low", long_header)
        self.assertIn("repeat_q025", long_header)

    def test_thin_main_runs_the_benchmark_class(self):
        from experiments.conditioned_benchmark.main import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root / "test")
            score_path = root / "cmrp.npz"
            output_dir = root / "report"
            _write_cmrp_scores(score_path, dataset)

            report = main(
                [
                    "--split-root",
                    str(dataset.root),
                    "--output-dir",
                    str(output_dir),
                    "--artifact",
                    f"cmrp={score_path}",
                    "--task-type",
                    "all",
                    "--positive-rate",
                    "native",
                    "--metric",
                    "auroc",
                    "--bootstrap",
                    "0",
                ]
            )

        self.assertEqual(report["schema"], REPORT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
