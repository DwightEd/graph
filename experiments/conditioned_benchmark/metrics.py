"""Extensible metric registry with prevalence-aware evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Metric:
    name: str
    function: MetricFunction
    prevalence_sensitive: bool
    description: str


def _auroc(labels, scores, weights):
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def _auprc(labels, scores, weights):
    return float(average_precision_score(labels, scores, sample_weight=weights))


def _weighted_prevalence(labels, weights):
    return float(np.average(labels, weights=weights))


def _lift(labels, scores, weights):
    prevalence = _weighted_prevalence(labels, weights)
    return _auprc(labels, scores, weights) / prevalence


def _tpr_at(max_fpr):
    def metric(labels, scores, weights):
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        selected = tpr[fpr <= max_fpr + 1e-12]
        return float(selected.max()) if len(selected) else 0.0

    return metric


def _partial_auroc(max_fpr):
    def metric(labels, scores, weights):
        return float(
            roc_auc_score(
                labels,
                scores,
                sample_weight=weights,
                max_fpr=max_fpr,
            )
        )

    return metric


def _alert_counts(labels, scores, weights, alert_rate):
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    count = max(1, int(np.searchsorted(cumulative, alert_rate * weights.sum()) + 1))
    selected = order[:count]
    true_positive = float(np.sum(weights[selected] * labels[selected]))
    predicted_positive = float(np.sum(weights[selected]))
    actual_positive = float(np.sum(weights * labels))
    return true_positive, predicted_positive, actual_positive


def _precision_at_alert(alert_rate):
    def metric(labels, scores, weights):
        true_positive, predicted_positive, _ = _alert_counts(
            labels, scores, weights, alert_rate
        )
        return true_positive / predicted_positive

    return metric


def _recall_at_alert(alert_rate):
    def metric(labels, scores, weights):
        true_positive, _, actual_positive = _alert_counts(
            labels, scores, weights, alert_rate
        )
        return true_positive / actual_positive

    return metric


def _f1_at_alert(alert_rate):
    def metric(labels, scores, weights):
        true_positive, predicted_positive, actual_positive = _alert_counts(
            labels, scores, weights, alert_rate
        )
        precision = true_positive / predicted_positive
        recall = true_positive / actual_positive
        return (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

    return metric


METRICS = {
    metric.name: metric
    for metric in (
        Metric("auroc", _auroc, False, "area under the ROC curve"),
        Metric("auprc", _auprc, True, "average precision / PR AUC"),
        Metric(
            "auprc_lift",
            _lift,
            True,
            "AUPRC divided by the evaluated positive prevalence",
        ),
        Metric(
            "tpr_at_fpr_05",
            _tpr_at(0.05),
            False,
            "maximum TPR at FPR <= 0.05",
        ),
        Metric(
            "tpr_at_fpr_10",
            _tpr_at(0.10),
            False,
            "maximum TPR at FPR <= 0.10",
        ),
        Metric(
            "partial_auroc_fpr_10",
            _partial_auroc(0.10),
            False,
            "standardized partial AUROC up to FPR 0.10",
        ),
        Metric(
            "precision_at_alert_05",
            _precision_at_alert(0.05),
            True,
            "precision when the top 5% weighted score mass is alerted",
        ),
        Metric(
            "recall_at_alert_05",
            _recall_at_alert(0.05),
            True,
            "recall when the top 5% weighted score mass is alerted",
        ),
        Metric(
            "f1_at_alert_05",
            _f1_at_alert(0.05),
            True,
            "F1 when the top 5% weighted score mass is alerted",
        ),
        Metric(
            "precision_at_alert_10",
            _precision_at_alert(0.10),
            True,
            "precision when the top 10% weighted score mass is alerted",
        ),
        Metric(
            "recall_at_alert_10",
            _recall_at_alert(0.10),
            True,
            "recall when the top 10% weighted score mass is alerted",
        ),
        Metric(
            "f1_at_alert_10",
            _f1_at_alert(0.10),
            True,
            "F1 when the top 10% weighted score mass is alerted",
        ),
    )
}

DEFAULT_METRICS = (
    "auroc",
    "auprc",
    "auprc_lift",
    "tpr_at_fpr_05",
    "tpr_at_fpr_10",
    "partial_auroc_fpr_10",
)


def evaluate_metrics(labels, scores, weights, names):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) != len(weights):
        raise ValueError("metric inputs have inconsistent row counts")
    if len(np.unique(labels)) != 2:
        raise ValueError("ranking metrics require both label classes")
    unknown = set(names).difference(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    return {name: METRICS[name].function(labels, scores, weights) for name in names}


def cluster_bootstrap_indices(cluster_ids, *, replicates, seed):
    cluster_ids = np.asarray(cluster_ids).astype(str)
    names, inverse = np.unique(cluster_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(names))]
    if not groups or int(replicates) <= 0:
        return []
    draws = np.random.default_rng(seed).integers(
        0, len(groups), size=(int(replicates), len(groups))
    )
    return [np.concatenate([groups[index] for index in draw]) for draw in draws]
