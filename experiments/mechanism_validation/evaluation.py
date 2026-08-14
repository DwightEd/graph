"""Evaluation helpers.  Labels enter only through the public evaluation calls."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None, "prevalence": prevalence, "lift": None}
    auprc = float(average_precision_score(labels, scores))
    auroc = float(roc_auc_score(labels, scores))
    return {"auroc": auroc, "auprc": auprc,
            "prevalence": prevalence, "lift": auprc / prevalence if prevalence else None}


def cluster_bootstrap(labels, scores, source_ids, *, n_resamples=1000, seed=0):
    """Bootstrap source clusters, retaining every token of each drawn source."""
    labels, scores, source_ids = map(np.asarray, (labels, scores, source_ids))
    _, inverse = np.unique(source_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    clusters = [order[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    generator = np.random.default_rng(seed)
    drawn = generator.integers(len(clusters), size=(n_resamples, len(clusters)))
    auroc = np.full(n_resamples, np.nan)
    auprc = np.full(n_resamples, np.nan)
    for index, selected_clusters in enumerate(drawn):
        selected = np.concatenate([clusters[cluster] for cluster in selected_clusters])
        metric = ranking_metrics(labels[selected], scores[selected])
        auroc[index] = np.nan if metric["auroc"] is None else metric["auroc"]
        auprc[index] = np.nan if metric["auprc"] is None else metric["auprc"]
    return {"auroc": auroc, "auprc": auprc, "valid_replicates": int(np.isfinite(auroc).sum())}


def paired_cluster_delta(labels, left_scores, right_scores, source_ids, *, n_resamples=200, seed=0):
    labels, left_scores, right_scores, source_ids = map(np.asarray, (labels, left_scores, right_scores, source_ids))
    _, inverse = np.unique(source_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    clusters = [order[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    draws = np.random.default_rng(seed).integers(len(clusters), size=(n_resamples, len(clusters)))
    point_left, point_right = ranking_metrics(labels, left_scores), ranking_metrics(labels, right_scores)
    deltas = {"auroc": [], "auprc": []}
    for draw in draws:
        selected = np.concatenate([clusters[index] for index in draw])
        left, right = ranking_metrics(labels[selected], left_scores[selected]), ranking_metrics(labels[selected], right_scores[selected])
        for metric, values in deltas.items():
            if left[metric] is not None and right[metric] is not None:
                values.append(left[metric] - right[metric])
    return {metric: {"point": point_left[metric] - point_right[metric] if point_left[metric] is not None and point_right[metric] is not None else None,
                     "ci_low": float(np.quantile(values, .025)) if values else None,
                     "ci_high": float(np.quantile(values, .975)) if values else None,
                     "valid_replicates": len(values)} for metric, values in deltas.items()}


def same_response_effect(labels, scores, response_ids, positions):
    labels, scores, response_ids, positions = map(np.asarray, (labels, scores, response_ids, positions))
    differences, gaps, positives = [], [], 0
    for response_id in np.unique(response_ids):
        rows = np.flatnonzero(response_ids == response_id)
        negative = rows[labels[rows] == 0].tolist()
        caliper = max(8, .05 * (positions[rows].max() + 1))
        for positive in rows[labels[rows] == 1]:
            positives += 1
            if negative:
                close = min(negative, key=lambda row: abs(positions[row] - positions[positive]))
                gap = abs(positions[positive] - positions[close])
                if gap <= caliper:
                    differences.append(scores[positive] - scores[close])
                    gaps.append(gap)
                    negative.remove(close)
    values = np.asarray(differences)
    return {"concordance": float(((values > 0) + .5 * (values == 0)).mean()) if len(values) else None,
            "mean_effect": float(values.mean()) if len(values) else None,
            "pair_coverage": len(values) / positives if positives else None,
            "mean_position_gap": float(np.mean(gaps)) if gaps else None}
