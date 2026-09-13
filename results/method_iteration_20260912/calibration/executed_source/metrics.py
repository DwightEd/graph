"""Binary detection metrics with source-cluster uncertainty."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_detection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    source_ids: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
    source_balanced: bool = False,
) -> dict:
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or source_ids.shape != labels.shape
    ):
        raise ValueError("labels, scores, and source IDs must be aligned vectors")
    if len(np.unique(labels)) != 2:
        raise ValueError("evaluation requires both normal and hallucination labels")
    intervals, valid = _cluster_bootstrap(
        labels,
        scores,
        source_ids,
        replicates=bootstrap,
        seed=seed,
        source_balanced=source_balanced,
    )
    weights = _source_weights(source_ids) if source_balanced else None
    return {
        "auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "auprc": float(average_precision_score(labels, scores, sample_weight=weights)),
        "valid_bootstrap_replicates": valid,
        "confidence_intervals": intervals,
    }


def _cluster_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    source_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    source_balanced: bool,
) -> tuple[dict[str, list[float] | None], int]:
    if replicates < 0:
        raise ValueError("bootstrap must be non-negative")
    groups = np.unique(source_ids)
    by_group = {group: np.flatnonzero(source_ids == group) for group in groups}
    source_weights = _source_weights(source_ids) if source_balanced else None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if len(np.unique(labels[indices])) != 2:
            continue
        weights = source_weights[indices] if source_weights is not None else None
        estimates.append(
            (
                roc_auc_score(labels[indices], scores[indices], sample_weight=weights),
                average_precision_score(
                    labels[indices], scores[indices], sample_weight=weights
                ),
            )
        )
    if not estimates:
        return {"auroc": None, "auprc": None}, 0
    values = np.asarray(estimates)
    return {
        "auroc": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "auprc": np.quantile(values[:, 1], [0.025, 0.975]).tolist(),
    }, len(estimates)


def _source_weights(source_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(source_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]
