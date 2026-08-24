"""Post-hoc evaluation of frozen causal typed-path De Bruijn scores.

This module never imports an attention cache and exposes no fit or scoring
entrypoint.  It first freezes and validates a complete score artifact, verifies
the exact reference and dataset manifest, and only then asks the canonical
research dataset to unlock formal labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import FrozenEvaluation, scalar_text

from .artifacts import (
    EVALUATION_SCHEMA,
    SCORE_SCHEMA,
    atomic_write_json,
    load_score_artifact,
    score_temporal_scope,
    verify_score_provenance,
)


HYBRID_EVALUATION_SCHEMA = "causal-typed-path-rr-hybrid-evaluation-v1"


def _frozen_score_schema(path) -> str:
    """Read only the pickle-free schema discriminator before choosing a loader."""

    with np.load(path, allow_pickle=False) as arrays:
        return scalar_text(arrays, "schema")


def _binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, object]:
    """Return token-level AUROC/AUPRC, or explicit nulls for one class."""

    labels = np.asarray(labels, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if labels.ndim != 1 or score.ndim != 1 or len(labels) != len(score):
        raise ValueError("labels and score must be aligned vectors")
    if len(labels) < 1 or not bool(np.isfinite(score).all()):
        raise ValueError("evaluation vectors must be non-empty and finite")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("formal token labels must be binary")

    result: dict[str, object] = {
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
    }
    if np.unique(labels).size < 2:
        result.update(
            auroc=None,
            auprc=None,
            undefined_reason="formal labels contain only one class",
        )
        return result
    result.update(
        auroc=float(roc_auc_score(labels, score)),
        auprc=float(average_precision_score(labels, score)),
        undefined_reason=None,
    )
    return result


def _sample_clusters(sample_id: np.ndarray) -> list[np.ndarray]:
    """Return row indices grouped by response, preserving first-seen order."""

    values = np.asarray(sample_id)
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"} or not len(values):
        raise ValueError("sample_id must be a non-empty text vector")
    text = values.astype(str, copy=False)
    return [
        np.flatnonzero(text == sample)
        for sample in dict.fromkeys(text.tolist())
    ]


def _sample_cluster_bootstrap(
    labels: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Percentile confidence intervals from whole-response resampling.

    A sampled response contributes all of its tokens and may occur repeatedly.
    Replicates containing only one token-label class are undefined and skipped,
    with the skipped count retained in the report.
    """

    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    clusters = _sample_clusters(sample_id)
    rng = np.random.default_rng(int(seed))
    estimates: list[tuple[float, float]] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        row_index = np.concatenate([clusters[index] for index in selected])
        current_labels = labels[row_index]
        if np.unique(current_labels).size < 2:
            continue
        estimates.append(
            (
                float(roc_auc_score(current_labels, score[row_index])),
                float(average_precision_score(current_labels, score[row_index])),
            )
        )

    valid = len(estimates)
    result: dict[str, object] = {
        "unit": "sample_response",
        "confidence_level": 0.95,
        "replicates_requested": replicates,
        "replicates_valid": valid,
        "replicates_skipped_single_class": replicates - valid,
    }
    if not estimates:
        result.update(
            auroc_ci_low=None,
            auroc_ci_high=None,
            auprc_ci_low=None,
            auprc_ci_high=None,
        )
        return result

    values = np.asarray(estimates, dtype=np.float64)
    result.update(
        auroc_ci_low=float(np.quantile(values[:, 0], 0.025)),
        auroc_ci_high=float(np.quantile(values[:, 0], 0.975)),
        auprc_ci_low=float(np.quantile(values[:, 1], 0.025)),
        auprc_ci_high=float(np.quantile(values[:, 1], 0.975)),
    )
    return result


def _task_conditioned_reports(
    labels: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
    task_type: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Report every observed task without using task identity during scoring."""

    tasks = np.asarray(task_type)
    if tasks.ndim != 1 or len(tasks) != len(labels) or tasks.dtype.kind not in {
        "U",
        "S",
    }:
        raise ValueError("task_type must be an aligned text vector")
    normalized = tasks.astype(str, copy=False)
    result: dict[str, object] = {}
    for index, raw_name in enumerate(sorted(set(normalized.tolist()))):
        output_name = raw_name if raw_name else "__missing__"
        selected = normalized == raw_name
        result[output_name] = {
            "token_metrics": _binary_metrics(labels[selected], score[selected]),
            "sample_cluster_bootstrap": _sample_cluster_bootstrap(
                labels[selected],
                score[selected],
                np.asarray(sample_id)[selected],
                replicates=replicates,
                seed=seed + 1009 * (index + 1),
            ),
        }
    return result


def evaluate_scores(
    dataset,
    score_path,
    output_path,
    bootstrap_replicates: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate one frozen held-out score artifact against formal token labels.

    The authorization boundary is intentionally visible in the call order:

    1. capture the score file identity;
    2. strict-load rows and verify their reference provenance;
    3. verify the held-out dataset manifest and complete token geometry;
    4. only then open and align canonical/formal labels.

    Parameters
    ----------
    dataset:
        An already-open canonical research dataset for the ``test`` split.
    score_path:
        Path to either a path-only score NPZ or the strictly aligned causal-RR
        hybrid produced by :mod:`spectral_bridge`.
    output_path:
        JSON destination.  It is replaced atomically after strict serialization.
    bootstrap_replicates:
        Number of response-cluster bootstrap draws.
    seed:
        NumPy random seed used only for post-hoc confidence intervals.
    """

    bootstrap_replicates = int(bootstrap_replicates)
    seed = int(seed)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")

    evaluation = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact_schema = _frozen_score_schema(evaluation.artifact.path)
    if artifact_schema == SCORE_SCHEMA:
        artifact = load_score_artifact(evaluation.artifact.path)
        reference = verify_score_provenance(artifact)
        report_schema = EVALUATION_SCHEMA
        provenance = {
            "reference_path": str(
                Path(str(np.asarray(artifact["reference_path"]).item())).resolve()
            ),
            "reference_sha256": str(
                np.asarray(artifact["reference_sha256"]).item()
            ),
            "train_dataset_manifest_sha256": str(
                np.asarray(reference["train_dataset_manifest_sha256"]).item()
            ),
        }
    else:
        # Imported lazily so the core path-only evaluator has no dependency on
        # the optional RR audit until a hybrid schema is actually requested.
        from .spectral_bridge import (
            HYBRID_SCHEMA,
            load_rr_hybrid,
            verify_rr_hybrid_provenance,
        )

        if artifact_schema != HYBRID_SCHEMA:
            raise ValueError("unsupported typed-path evaluation score schema")
        artifact = load_rr_hybrid(evaluation.artifact.path)
        reference, _rr_reference = verify_rr_hybrid_provenance(artifact)
        report_schema = HYBRID_EVALUATION_SCHEMA
        provenance = {
            "reference_path": str(
                Path(
                    str(np.asarray(artifact["path_reference_path"]).item())
                ).resolve()
            ),
            "reference_sha256": str(
                np.asarray(artifact["path_reference_sha256"]).item()
            ),
            "train_dataset_manifest_sha256": str(
                np.asarray(
                    artifact["path_train_dataset_manifest_sha256"]
                ).item()
            ),
            "rr_reference_path": str(
                Path(
                    str(np.asarray(artifact["rr_reference_path"]).item())
                ).resolve()
            ),
            "rr_reference_sha256": str(
                np.asarray(artifact["rr_reference_sha256"]).item()
            ),
            "rr_train_dataset_manifest_sha256": str(
                np.asarray(
                    artifact["rr_train_dataset_manifest_sha256"]
                ).item()
            ),
            "rr_score_name": str(np.asarray(artifact["rr_score_name"]).item()),
        }

    # This call validates the frozen score bytes, split, manifest digest,
    # canonical source groups, response lengths, and complete row coverage.  It
    # performs no label access.  ``align_loaded`` repeats those checks before it
    # invokes ``prepare_evaluation_labels`` inside FrozenEvaluation.
    evaluation.validate_loaded(dataset, artifact)
    aligned = evaluation.align_loaded(dataset, artifact)

    labels = np.asarray(aligned.token_label, dtype=np.int8)
    score = np.asarray(artifact["score"], dtype=np.float64)
    metrics = _binary_metrics(labels, score)
    bootstrap = _sample_cluster_bootstrap(
        labels,
        score,
        artifact["sample_id"],
        replicates=bootstrap_replicates,
        seed=seed,
    )
    task_reports = _task_conditioned_reports(
        labels,
        score,
        artifact["sample_id"],
        artifact["task_type"],
        replicates=bootstrap_replicates,
        seed=seed,
    )

    report: dict[str, object] = {
        "schema": report_schema,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_detector": "score",
        "token_metrics": metrics,
        "sample_cluster_bootstrap": bootstrap,
        "task_reports": task_reports,
        "samples": int(len(np.unique(np.asarray(artifact["sample_id"], dtype=str)))),
        "score_artifact_path": str(evaluation.artifact.path),
        "score_artifact_sha256": evaluation.artifact.sha256,
        "dataset_manifest_sha256": str(
            np.asarray(artifact["dataset_manifest_sha256"]).item()
        ),
        "topology_gate_mean_gap": float(reference["topology_gate_mean_gap"]),
        "topology_gate_pass": bool(reference["topology_gate_pass"]),
        "bootstrap_seed": seed,
        **provenance,
        **score_temporal_scope().as_dict(),
    }
    atomic_write_json(output_path, report)
    return report


__all__ = ["evaluate_scores"]
