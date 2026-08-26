"""End-to-end audit of frozen GroundedRoute node representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from experiment_protocol import scalar_text, sha256_text, validate_source_audit

from ..artifacts import load_checkpoint, load_scores, save_npz, sha256
from ..detection import PCAKNNConfig, PCAWhitenedKNN
from ..pipeline import validate_calibration_provenance
from .data import GraphBundle, load_aligned_artifact_labels
from .control_audit import audit_controls
from .detectors import DETECTOR_NAMES, DetectorConfig, score_detectors
from .label_free import label_free_audit
from .metrics import binary_metrics, paired_source_delta, source_cluster_bootstrap
from .upper_bound import ProbeConfig, fit_readability_probes
from .views import AlignedEmbeddingViews, VIEW_PROTOCOL, load_embedding_views


SCORE_SCHEMA = "grounded-route-node-only-detector-benchmark"
OOF_SCHEMA = "grounded-route-node-only-readability-oof"
REPORT_SCHEMA = "grounded-route-saved-embedding-effectiveness-report"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class AuditConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    bootstrap_replicates: int = 2_000
    bootstrap_seed: int = 20260825
    minimum_auprc_gain: float = 0.005
    minimum_changed_fraction: float = 0.10
    minimum_sample_changed_fraction: float = 0.05
    minimum_effective_samples: float = 0.80
    minimum_bootstrap_valid_fraction: float = 0.90


def audit(
    calibration_indices: Mapping[str, str | Path],
    test_indices: Mapping[str, str | Path],
    test_root: str | Path,
    output_dir: str | Path,
    *,
    published_score_path: str | Path | None = None,
    config: AuditConfig | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Benchmark node-only readers, then open labels for evaluation/probes."""

    config = AuditConfig() if config is None else config
    calibration = load_embedding_views(calibration_indices)
    test = load_embedding_views(test_indices)
    _validate_variant_pairs(calibration, test)
    training_match = _validate_training_match(calibration)
    control_evidence = _control_evidence(
        calibration,
        test,
        training_match,
        config,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_arrays: dict[str, np.ndarray] = {}
    for variant in test.variants:
        variant_scores = score_detectors(
            calibration.embedding(variant),
            test.embedding(variant),
            config=config.detector,
            device=device,
        )
        for detector, score in variant_scores.items():
            score_arrays[_score_name(variant, detector)] = score
    score_arrays["position_pca_knn"] = _position_anomaly_score(
        calibration.reference.bundle.index,
        test.reference.bundle.index,
        config.detector,
    )

    published = None
    if published_score_path is not None:
        published = _load_published_score(
            published_score_path,
            test.reference.bundle,
        )
        score_arrays[_score_name("real", "published_pca_knn")] = published

    score_path = output_dir / "unsupervised_scores.npz"
    _save_score_artifact(
        score_path,
        calibration,
        test,
        score_arrays,
        config,
    )
    integrity_path = output_dir / "integrity.json"
    integrity = {
        "schema": "grounded-route-effectiveness-bundle-integrity",
        "version": ARTIFACT_VERSION,
        "labels_read": False,
        "calibration": {
            variant: view.bundle.integrity
            for variant, view in calibration.views.items()
        },
        "test": {
            variant: view.bundle.integrity
            for variant, view in test.views.items()
        },
    }
    integrity_path.write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sensitivity = label_free_audit(
        test,
        output_dir / "label_free_topology.npz",
        seed=config.bootstrap_seed,
    )

    for view in calibration.views.values():
        view.bundle.reverify()
    for view in test.views.values():
        view.bundle.reverify()
    labels, frozen_scores, frozen_score_rows = load_aligned_artifact_labels(
        test.reference.bundle,
        test_root,
        score_path,
    )
    label = labels.token_label.astype(np.int8)
    source_id = labels.source_id.astype(str)
    score_arrays = {
        name: np.asarray(frozen_score_rows[name], dtype=np.float32)
        for name in score_arrays
    }

    probes = fit_readability_probes(
        {variant: test.embedding(variant) for variant in test.variants},
        label,
        source_id,
        test.reference.bundle.index.token_index,
        test.reference.bundle.index.response_length,
        config=config.probe,
        device=device,
    )
    oof_path = output_dir / "oof_predictions.npz"
    _save_oof_artifact(oof_path, test, probes, config)

    unsupervised_metrics = _metric_table(
        label,
        source_id,
        score_arrays,
        config,
    )
    probe_metrics = _metric_table(label, source_id, probes.score, config)
    probe_seed_metrics = {
        name: binary_metrics(label, score)
        for name, score in probes.seed_score.items()
    }
    comparisons = _variant_comparisons(
        label,
        source_id,
        score_arrays,
        probes.score,
        probes.seed_score,
        test,
        control_evidence,
        config,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "experiment_type": "saved_node_embedding_effectiveness_audit",
        "primary_downstream_feature": "node_embedding_only",
        "edges_used_by_downstream_models": False,
        "position_control_fields": ["token_index", "response_length"],
        "position_control_online_causal": False,
        "attention_opened_for_features": False,
        "canonical_cache_opened_for_embedded_labels": True,
        "labels_read": True,
        "labels_used_during_unsupervised_fit": False,
        "labels_used_during": (
            "posthoc_unsupervised_evaluation_and_source_grouped_probe_only"
        ),
        "supervised_result_scope": "diagnostic_readability_ceiling",
        "construction_claim_scope": (
            "row_local_no_neighbour_embedding_ablation_and_exact_endpoint_or_weight_effects"
        ),
        "row_local_control_scope": (
            "node_embedding_path_only; endpoints_remain_training_targets_and_saved_lineage"
        ),
        "confirmatory_unsupervised_detector": "pca_knn",
        "confirmatory_position_baseline": "position_pca_knn_offline",
        "other_unsupervised_detectors": "exploratory_no_best_of_test_selection",
        "interval_scope": "conditional_on_frozen_scores_or_oof_predictions",
        "variants": list(test.variants),
        "variant_protocol": {
            name: {
                "graph_variant": test.views[name].graph_variant,
                "message_mode": test.views[name].message_mode,
            }
            for name in test.variants
        },
        "samples": len(test.reference.bundle.records),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "config": asdict(config),
        "unsupervised": unsupervised_metrics,
        "readability_probes": probe_metrics,
        "readability_probe_seeds": probe_seed_metrics,
        "comparisons": comparisons,
        "variant_training_match": training_match,
        "matched_control_evidence": control_evidence,
        "label_free_representation_sensitivity": sensitivity,
        "artifacts": {
            "bundle_integrity": str(integrity_path.resolve()),
            "bundle_integrity_sha256": sha256(integrity_path),
            "unsupervised_scores": str(score_path.resolve()),
            "unsupervised_scores_sha256": frozen_scores.sha256,
            "oof_predictions": str(oof_path.resolve()),
            "oof_predictions_sha256": sha256(oof_path),
        },
    }
    report_path = output_dir / "report.json"
    frozen_scores.verify(score_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "report": str(report_path.resolve())}


def _validate_variant_pairs(
    calibration: AlignedEmbeddingViews,
    test: AlignedEmbeddingViews,
) -> None:
    if set(calibration.variants) != set(test.variants):
        raise ValueError("calibration and test variant sets differ")
    for variant in test.variants:
        calibration_bundle = calibration.views[variant].bundle
        test_bundle = test.views[variant].bundle
        if scalar_text(calibration_bundle.metadata, "scope") != "calibration":
            raise ValueError("detector reference must use calibration embeddings")
        if scalar_text(calibration_bundle.metadata, "split") != "train":
            raise ValueError("calibration embeddings must come from the train split")
        if scalar_text(test_bundle.metadata, "split") != "test":
            raise ValueError("scored embeddings must come from the test split")
        if sha256_text(
            calibration_bundle.metadata,
            "checkpoint_sha256",
        ) != sha256_text(test_bundle.metadata, "checkpoint_sha256"):
            raise ValueError(f"{variant} calibration/test encoders differ")
        validate_calibration_provenance(
            calibration_bundle.index,
            calibration_bundle.metadata,
        )
        validate_source_audit(
            reserved_source_ids=test_bundle.metadata["reserved_source_ids"],
            test_source_ids=test_bundle.metadata["test_source_ids"],
            test_sample_ids=test_bundle.metadata["test_sample_ids"],
            row_sample_ids=test_bundle.index.sample_id,
            row_source_ids=test_bundle.index.source_id,
            audit_scope=scalar_text(test_bundle.metadata, "audit_scope"),
        )


def _validate_training_match(views: AlignedEmbeddingViews) -> dict[str, object]:
    """Require controlled runs to differ only in their graph intervention."""

    checkpoints = {}
    for variant, view in views.views.items():
        path = view.bundle.index_path.parent.parent / "model.pt"
        if not path.is_file():
            if len(views.variants) == 1:
                return {
                    "required": False,
                    "verified": False,
                    "reason": "single real bundle; no construction gate requested",
                }
            raise ValueError(f"control audit requires checkpoint beside {variant} bundle")
        expected = sha256_text(view.bundle.metadata, "checkpoint_sha256")
        if sha256(path) != expected:
            raise ValueError(f"{variant} checkpoint differs from index metadata")
        checkpoint = load_checkpoint(path, map_location="cpu")
        checkpoint_implementation = checkpoint.get("implementation_sha256")
        if checkpoint_implementation is not None:
            artifact_implementation = sha256_text(
                view.bundle.metadata,
                "implementation_sha256",
            )
            if artifact_implementation != checkpoint_implementation:
                raise ValueError(
                    f"{variant} embedding implementation differs from checkpoint"
                )
        model_config = checkpoint["config"]["model"]
        observed = (
            checkpoint["config"]["intervention"]["variant"],
            model_config.get("message_mode", "neighbor"),
        )
        if checkpoint["variant"] != observed[0]:
            raise ValueError(f"{variant} checkpoint graph-variant fields disagree")
        if checkpoint.get("message_mode", "neighbor") != observed[1]:
            raise ValueError(f"{variant} checkpoint message-mode fields disagree")
        if observed != VIEW_PROTOCOL[variant]:
            raise ValueError(f"{variant} checkpoint has the wrong construction mode")
        checkpoints[variant] = (path, checkpoint)

    reference = _checkpoint_signature(checkpoints["real"][1])
    for variant, (_, checkpoint) in checkpoints.items():
        if _checkpoint_signature(checkpoint) != reference:
            raise ValueError(
                f"{variant} training differs from real beyond graph intervention"
            )
    if len(views.variants) > 1 and any(
        "implementation_sha256" not in checkpoint
        for _, checkpoint in checkpoints.values()
    ):
        raise ValueError("control checkpoints must record their implementation SHA-256")
    return {
        "required": len(views.variants) > 1,
        "verified": True,
        "implementation_identity_verified": all(
            "implementation_sha256" in checkpoint
            for _, checkpoint in checkpoints.values()
        ),
        "checkpoint": {
            variant: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "training_changed_fraction": float(
                    checkpoint["changed_fraction"]
                ),
                "graph_variant": checkpoint["config"]["intervention"]["variant"],
                "message_mode": checkpoint["config"]["model"].get(
                    "message_mode",
                    "neighbor",
                ),
                "implementation_sha256": checkpoint.get(
                    "implementation_sha256",
                    "unrecorded",
                ),
                "training_health": _training_health(
                    checkpoint,
                    views.embedding(variant),
                ),
            }
            for variant, (path, checkpoint) in checkpoints.items()
        },
    }


def _checkpoint_signature(checkpoint) -> str:
    config = checkpoint["config"]
    model = {
        name: value
        for name, value in config["model"].items()
        if name != "message_mode"
    }
    intervention = {
        name: value
        for name, value in config["intervention"].items()
        if name != "variant"
    }
    signature = {
        "graph": config["graph"],
        "model": model,
        "learning": config["learning"],
        "train": config["train"],
        "intervention": intervention,
        "layer_count": int(checkpoint["layer_count"]),
        "head_count": int(checkpoint["head_count"]),
        "parameter_count": int(checkpoint["parameter_count"]),
        "implementation_sha256": checkpoint.get(
            "implementation_sha256",
            "unrecorded",
        ),
        "state_dict_schema": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in checkpoint["state_dict"].items()
        },
        "graph_spec_sha256": str(checkpoint["graph_spec_sha256"]),
        "fit_sample_ids": list(map(str, checkpoint["fit_sample_ids"])),
        "validation_sample_ids": list(map(str, checkpoint["validation_sample_ids"])),
        "calibration_sample_ids": list(map(str, checkpoint["calibration_sample_ids"])),
        "fit_source_ids": list(map(str, checkpoint["fit_source_ids"])),
        "validation_source_ids": list(map(str, checkpoint["validation_source_ids"])),
        "calibration_source_ids": list(map(str, checkpoint["calibration_source_ids"])),
        "train_source_ids": list(map(str, checkpoint["train_source_ids"])),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _training_health(checkpoint, calibration_embedding) -> dict[str, object]:
    history = checkpoint["history"]
    loss = np.asarray(
        [
            (float(row["train_loss"]), float(row["validation_loss"]))
            for row in history
        ],
        dtype=np.float64,
    )
    centered = np.asarray(calibration_embedding, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    spectrum = np.linalg.eigvalsh(covariance).clip(min=0.0)
    total = float(spectrum.sum())
    probability = spectrum[spectrum > 0.0] / max(total, 1e-12)
    effective_rank = float(np.exp(-(probability * np.log(probability)).sum()))
    return {
        "epochs": len(history),
        "loss_finite": bool(loss.size and np.isfinite(loss).all()),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "calibration_total_variance": total,
        "calibration_effective_rank": effective_rank,
        "embedding_noncollapsed": bool(total > 1e-8 and effective_rank > 1.0),
    }


def _control_evidence(calibration, test, training_match, config):
    calibration_report = audit_controls(
        calibration,
        minimum_changed_fraction=config.minimum_changed_fraction,
        minimum_sample_changed_fraction=config.minimum_sample_changed_fraction,
        minimum_effective_samples=config.minimum_effective_samples,
    )
    test_report = audit_controls(
        test,
        minimum_changed_fraction=config.minimum_changed_fraction,
        minimum_sample_changed_fraction=config.minimum_sample_changed_fraction,
        minimum_effective_samples=config.minimum_effective_samples,
    )
    result = {}
    for variant in test.variants:
        if variant == "real":
            continue
        training_fraction = training_match["checkpoint"][variant][
            "training_changed_fraction"
        ]
        training_health = training_match["checkpoint"][variant]["training_health"]
        message_ablation = variant == "no_message"
        result[variant] = {
            "control_type": (
                "no_neighbor_message" if message_ablation else "graph_intervention"
            ),
            "training_changed_fraction": training_fraction,
            "training_health": training_health,
            "calibration": calibration_report[variant],
            "test": test_report[variant],
            "intervention_sufficient": bool(
                (
                    message_ablation
                    or training_fraction >= config.minimum_changed_fraction
                )
                and training_health["loss_finite"]
                and training_health["embedding_noncollapsed"]
                and calibration_report[variant]["intervention_sufficient"]
                and test_report[variant]["intervention_sufficient"]
            ),
        }
    return result


def _load_published_score(path, reference: GraphBundle) -> np.ndarray:
    scores = load_scores(path)
    if sha256_text(scores, "test_embedding_sha256") != reference.index_sha256:
        raise ValueError("published score uses a different test embedding index")
    for name in (
        "sample_id",
        "source_id",
        "token_index",
        "response_length",
        "response_token_id",
    ):
        if not np.array_equal(scores[name], getattr(reference.index, name)):
            raise ValueError(f"published score rows disagree on {name}")
    return np.asarray(scores["score"], dtype=np.float32)


def _save_score_artifact(
    path,
    calibration,
    test,
    scores,
    config: AuditConfig,
) -> None:
    reference = test.reference.bundle
    index = reference.index
    save_npz(
        path,
        schema=np.asarray(SCORE_SCHEMA),
        version=np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
        labels_read=np.asarray(False),
        labels_used_during=np.asarray("none"),
        primary_downstream_feature=np.asarray("node_embedding_only"),
        real_index_sha256=np.asarray(reference.index_sha256),
        variants=np.asarray(test.variants),
        graph_variant=np.asarray(
            [test.views[name].graph_variant for name in test.variants]
        ),
        message_mode=np.asarray(
            [test.views[name].message_mode for name in test.variants]
        ),
        calibration_index_sha256=np.asarray(
            [calibration.views[name].bundle.index_sha256 for name in test.variants]
        ),
        test_index_sha256=np.asarray(
            [test.views[name].bundle.index_sha256 for name in test.variants]
        ),
        checkpoint_sha256=np.asarray(
            [
                sha256_text(test.views[name].bundle.metadata, "checkpoint_sha256")
                for name in test.variants
            ]
        ),
        dataset_manifest_sha256=np.asarray(
            sha256_text(reference.metadata, "dataset_manifest_sha256")
        ),
        audit_scope=np.asarray(scalar_text(reference.metadata, "audit_scope")),
        reserved_source_ids=reference.metadata["reserved_source_ids"],
        test_source_ids=reference.metadata["test_source_ids"],
        test_sample_ids=reference.metadata["test_sample_ids"],
        detector_names=np.asarray(DETECTOR_NAMES),
        nuisance_score_names=np.asarray(("position_pca_knn",)),
        detector_config=np.asarray(json.dumps(asdict(config.detector), sort_keys=True)),
        **_identity_arrays(index),
        **scores,
    )


def _save_oof_artifact(path, views, probes, config: AuditConfig) -> None:
    index = views.reference.bundle.index
    save_npz(
        path,
        schema=np.asarray(OOF_SCHEMA),
        version=np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
        labels_used_during=np.asarray("source_grouped_probe_training_only"),
        primary_downstream_feature=np.asarray("node_embedding_only"),
        position_control_fields=np.asarray(("token_index", "response_length")),
        position_control_online_causal=np.asarray(False),
        real_index_sha256=np.asarray(views.reference.bundle.index_sha256),
        variants=np.asarray(views.variants),
        graph_variant=np.asarray(
            [views.views[name].graph_variant for name in views.variants]
        ),
        message_mode=np.asarray(
            [views.views[name].message_mode for name in views.variants]
        ),
        fold_id=np.asarray(probes.fold_id, dtype=np.int16),
        probe_config=np.asarray(json.dumps(asdict(config.probe), sort_keys=True)),
        **_identity_arrays(index),
        **{f"score__{name}": value for name, value in probes.score.items()},
        **{f"score__{name}": value for name, value in probes.seed_score.items()},
    )


def _identity_arrays(index) -> dict[str, np.ndarray]:
    arrays = index.arrays()
    arrays.pop("embedding")
    return arrays


def _metric_table(label, source_id, scores, config: AuditConfig):
    result = {}
    for name, score in scores.items():
        result[name] = {
            **binary_metrics(label, score),
            "source_bootstrap": source_cluster_bootstrap(
                label,
                score,
                source_id,
                replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed,
            ),
        }
    return result


def _variant_comparisons(
    label,
    source_id,
    unsupervised,
    probe,
    seed_probe,
    views,
    control_evidence,
    config: AuditConfig,
):
    comparisons: dict[str, object] = {}
    absolute_cache = {}

    def absolute(name, score):
        if name not in absolute_cache:
            absolute_cache[name] = _absolute_evidence(
                label,
                score,
                source_id,
                config,
            )
        return absolute_cache[name]

    position_comparison = {
        "pca_knn": paired_source_delta(
            label,
            unsupervised["pca_knn__real"],
            unsupervised["position_pca_knn"],
            source_id,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        ),
        "linear_node": paired_source_delta(
            label,
            probe["linear_node__real"],
            probe["linear_position"],
            source_id,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        ),
        "node_mlp": paired_source_delta(
            label,
            probe["node_mlp__real"],
            probe["position_mlp"],
            source_id,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        ),
    }
    for control in views.variants:
        if control == "real":
            continue
        detector_comparison = {}
        for detector in DETECTOR_NAMES:
            left = unsupervised[_score_name("real", detector)]
            right = unsupervised[_score_name(control, detector)]
            delta = paired_source_delta(
                label,
                left,
                right,
                source_id,
                replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed,
            )
            detector_comparison[detector] = {
                **delta,
                "gate": _gate(
                    delta,
                    control_evidence[control],
                    config,
                    absolute(f"unsupervised:{detector}", left),
                    confirmatory=detector == "pca_knn",
                    position_delta=(
                        position_comparison["pca_knn"]
                        if detector == "pca_knn"
                        else None
                    ),
                ),
            }

        probe_comparison = {}
        for probe_name in ("linear_node", "node_mlp"):
            left_name = f"{probe_name}__real"
            right_name = f"{probe_name}__{control}"
            delta = paired_source_delta(
                label,
                probe[left_name],
                probe[right_name],
                source_id,
                replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed,
            )
            probe_seed_consistency = None
            if probe_name == "node_mlp":
                directions = []
                for seed in config.probe.seeds:
                    left = seed_probe[f"node_mlp__real__seed_{seed}"]
                    right = seed_probe[f"node_mlp__{control}__seed_{seed}"]
                    left_ap = binary_metrics(label, left)["auprc"]
                    right_ap = binary_metrics(label, right)["auprc"]
                    directions.append(bool(left_ap > right_ap))
                probe_seed_consistency = {
                    "positive_seeds": int(sum(directions)),
                    "total_seeds": len(directions),
                    "required_positive_seeds": math_ceil_four_fifths(len(directions)),
                }
            probe_comparison[probe_name] = {
                **delta,
                "probe_seed_consistency": probe_seed_consistency,
                "gate": _gate(
                    delta,
                    control_evidence[control],
                    config,
                    absolute(f"probe:{probe_name}", probe[left_name]),
                    confirmatory=probe_name == "node_mlp",
                    probe_seed_consistency=probe_seed_consistency,
                    position_delta=position_comparison[probe_name],
                ),
            }
        comparisons[f"real_minus_{control}"] = {
            "control_type": control_evidence[control]["control_type"],
            "control_changed_fraction": views.views[control].changed_fraction,
            "unsupervised": detector_comparison,
            "readability_ceiling": probe_comparison,
        }

    comparisons["real_minus_position_controls"] = position_comparison
    required = ("no_message", "endpoint_rewire", "weight_shuffle")
    complete = all(control in views.variants for control in required)
    comparisons["joint_construction_evidence"] = {
        "required_controls": list(required),
        "all_required_controls_present": complete,
        "confirmatory_pca_knn_all_controls_passed": bool(
            complete
            and all(
                comparisons[f"real_minus_{control}"]["unsupervised"]["pca_knn"][
                    "gate"
                ]["paired_run_gate_passed"]
                for control in required
            )
        ),
        "diagnostic_node_mlp_all_controls_passed": bool(
            complete
            and all(
                comparisons[f"real_minus_{control}"]["readability_ceiling"][
                    "node_mlp"
                ]["gate"]["paired_run_gate_passed"]
                for control in required
            )
        ),
        "multi_encoder_seed_confirmation": False,
        "paper_claim_ready": False,
    }
    return comparisons


def _absolute_evidence(label, score, source_id, config):
    point = binary_metrics(label, score)
    interval = source_cluster_bootstrap(
        label,
        score,
        source_id,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    return {
        "auroc": point["auroc"],
        "auprc_lift": point["auprc_lift"],
        "auroc_ci_low": interval["auroc_ci_low"],
        "auprc_lift_ci_low": interval["auprc_lift_ci_low"],
        "replicates_requested": interval["replicates_requested"],
        "replicates_valid": interval["replicates_valid"],
    }


def _gate(
    delta,
    control_evidence,
    config,
    absolute,
    confirmatory: bool,
    probe_seed_consistency=None,
    position_delta=None,
):
    lower = delta["auprc_delta_ci_low"]
    gain = delta["auprc_delta"]
    auroc = delta["auroc_delta"]
    seed_pass = True
    if probe_seed_consistency is not None:
        seed_pass = (
            probe_seed_consistency["total_seeds"] >= 3
            and
            probe_seed_consistency["positive_seeds"]
            >= probe_seed_consistency["required_positive_seeds"]
        )
    position_pass = bool(
        position_delta is None
        or (
            position_delta["auprc_delta_ci_low"] is not None
            and position_delta["auprc_delta_ci_low"] > 0.0
        )
    )
    absolute_pass = bool(
        absolute["auroc_ci_low"] is not None
        and absolute["auroc_ci_low"] > 0.5
        and absolute["auprc_lift_ci_low"] is not None
        and absolute["auprc_lift_ci_low"] > 1.0
    )
    bootstrap_pass = bool(
        _bootstrap_valid(delta, config)
        and _bootstrap_valid(absolute, config)
        and (
            position_delta is None
            or _bootstrap_valid(position_delta, config)
        )
    )
    evidence_passed = bool(
        lower is not None
        and lower > 0.0
        and gain is not None
        and gain >= config.minimum_auprc_gain
        and auroc is not None
        and auroc >= 0.0
        and control_evidence["intervention_sufficient"]
        and seed_pass
        and position_pass
        and absolute_pass
        and bootstrap_pass
    )
    return {
        "evidence_passed": evidence_passed,
        "confirmatory_comparison": confirmatory,
        "paired_run_gate_passed": bool(evidence_passed and confirmatory),
        "multi_encoder_seed_confirmation": False,
        "paper_claim_ready": False,
        "requires_auprc_ci_low_above_zero": True,
        "requires_real_auroc_ci_low_above_0_5": True,
        "requires_real_auprc_lift_ci_low_above_1": True,
        "absolute_real_evidence": absolute,
        "minimum_auprc_gain": config.minimum_auprc_gain,
        "minimum_changed_fraction": config.minimum_changed_fraction,
        "control_type": control_evidence["control_type"],
        "control_intervention_sufficient": control_evidence[
            "intervention_sufficient"
        ],
        "probe_seed_consistency_passed": seed_pass,
        "position_control_passed": position_pass,
        "bootstrap_validity_passed": bootstrap_pass,
        "interval_scope": "conditional_on_frozen_predictions",
    }


def _bootstrap_valid(report, config: AuditConfig) -> bool:
    required = int(
        np.ceil(
            config.minimum_bootstrap_valid_fraction
            * int(report["replicates_requested"])
        )
    )
    return int(report["replicates_valid"]) >= required


def _position_anomaly_score(calibration_index, test_index, config) -> np.ndarray:
    reference = PCAWhitenedKNN.fit(
        _position_features(calibration_index),
        PCAKNNConfig(
            components=2,
            neighbors=config.neighbors,
            max_reference=config.max_reference,
            seed=config.seed,
        ),
    )
    return reference.score(_position_features(test_index))


def _position_features(index) -> np.ndarray:
    length = np.asarray(index.response_length, dtype=np.float32)
    return np.column_stack(
        (
            np.asarray(index.token_index, dtype=np.float32)
            / np.maximum(length - 1.0, 1.0),
            np.log1p(length),
        )
    ).astype(np.float32)


def _score_name(variant: str, detector: str) -> str:
    return f"{detector}__{variant}"


def math_ceil_four_fifths(value: int) -> int:
    return (4 * value + 4) // 5
