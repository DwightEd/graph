"""Metrics for the isolated saved-graph effectiveness diagnostic.

Confidence intervals resample complete ``source_id`` clusters.  The paired
comparison uses the same bootstrap draw for both views, which is essential
when deciding whether a graph intervention changes performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METRIC_NAMES = ("auroc", "auprc", "auprc_lift")
BOOTSTRAP_METRIC_NAMES = ("auroc", "auprc", "prevalence", "auprc_lift")


@dataclass(frozen=True)
class _MetricPlan:
    """A fixed score ordering reused across source-bootstrap draws."""

    label: np.ndarray
    order: np.ndarray
    starts: np.ndarray

    @classmethod
    def build(cls, label, score) -> "_MetricPlan":
        label, score = _binary_rows(label, score)
        order = np.argsort(score, kind="stable")
        ordered_score = score[order]
        starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
        return cls(label=label, order=order, starts=starts)

    def evaluate(self, row_weight=None) -> dict[str, float] | None:
        if row_weight is None:
            row_weight = np.ones(len(self.label), dtype=np.float64)
        else:
            row_weight = np.asarray(row_weight, dtype=np.float64)
            if row_weight.shape != self.label.shape:
                raise ValueError("row weights must align with labels")

        ordered_weight = row_weight[self.order]
        ordered_label = self.label[self.order]
        positive = np.add.reduceat(ordered_weight * ordered_label, self.starts)
        negative = np.add.reduceat(ordered_weight * (1 - ordered_label), self.starts)
        positive_total = float(positive.sum())
        negative_total = float(negative.sum())
        if positive_total <= 0.0 or negative_total <= 0.0:
            return None

        negative_before = np.cumsum(negative) - negative
        auroc = float(
            np.sum(positive * (negative_before + 0.5 * negative))
            / (positive_total * negative_total)
        )

        descending_positive = positive[::-1]
        descending_total = (positive + negative)[::-1]
        cumulative_positive = np.cumsum(descending_positive)
        cumulative_total = np.cumsum(descending_total)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0.0,
        )
        auprc = float(
            np.sum(precision * descending_positive) / positive_total
        )
        prevalence = positive_total / (positive_total + negative_total)
        return {
            "auroc": auroc,
            "auprc": auprc,
            "prevalence": float(prevalence),
            "auprc_lift": float(auprc / prevalence),
        }


def binary_metrics(label, score) -> dict[str, float | int | None]:
    """Return token-level discrimination metrics and the random AUPRC level."""

    label, score = _binary_rows(label, score)
    result = _MetricPlan.build(label, score).evaluate()
    base: dict[str, float | int | None] = {
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "auroc": None,
        "auprc": None,
        "auprc_lift": None,
    }
    if result is not None:
        base.update(result)
    return base


def source_cluster_bootstrap(
    label,
    score,
    source_id,
    *,
    replicates: int = 2_000,
    seed: int = 20260825,
) -> dict[str, float | int | None]:
    """Bootstrap complete sources and return percentile intervals."""

    label, score = _binary_rows(label, score)
    _, group_index = _source_rows(source_id, len(label))
    plan = _MetricPlan.build(label, score)
    draws = _bootstrap_group_counts(group_index, replicates, seed)
    values = {name: [] for name in BOOTSTRAP_METRIC_NAMES}
    for counts in draws:
        result = plan.evaluate(counts[group_index])
        if result is None:
            continue
        for name in BOOTSTRAP_METRIC_NAMES:
            values[name].append(result[name])

    valid = len(values["auroc"])
    report: dict[str, float | int | None] = {
        "replicates_requested": int(replicates),
        "replicates_valid": valid,
    }
    for name in BOOTSTRAP_METRIC_NAMES:
        low, high = _interval(values[name])
        report[f"{name}_ci_low"] = low
        report[f"{name}_ci_high"] = high
    return report


def paired_source_delta(
    label,
    left_score,
    right_score,
    source_id,
    *,
    replicates: int = 2_000,
    seed: int = 20260825,
) -> dict[str, float | int | None]:
    """Return ``left - right`` metrics under a paired source bootstrap."""

    label, left_score = _binary_rows(label, left_score)
    right_label, right_score = _binary_rows(label, right_score)
    if not np.array_equal(label, right_label):
        raise ValueError("paired predictions must use the same labels")
    _, group_index = _source_rows(source_id, len(label))
    left = _MetricPlan.build(label, left_score)
    right = _MetricPlan.build(label, right_score)

    left_point = left.evaluate()
    right_point = right.evaluate()
    report: dict[str, float | int | None] = {
        "replicates_requested": int(replicates),
        "replicates_valid": 0,
    }
    for name in METRIC_NAMES:
        report[f"{name}_delta"] = (
            None
            if left_point is None or right_point is None
            else float(left_point[name] - right_point[name])
        )

    values = {name: [] for name in METRIC_NAMES}
    for counts in _bootstrap_group_counts(group_index, replicates, seed):
        row_weight = counts[group_index]
        left_result = left.evaluate(row_weight)
        right_result = right.evaluate(row_weight)
        if left_result is None or right_result is None:
            continue
        for name in METRIC_NAMES:
            values[name].append(left_result[name] - right_result[name])

    report["replicates_valid"] = len(values["auroc"])
    for name in METRIC_NAMES:
        low, high = _interval(values[name])
        report[f"{name}_delta_ci_low"] = low
        report[f"{name}_delta_ci_high"] = high
    return report


def _binary_rows(label, score) -> tuple[np.ndarray, np.ndarray]:
    label = np.asarray(label)
    score = np.asarray(score, dtype=np.float64)
    if label.ndim != 1 or score.ndim != 1 or len(label) != len(score) or not len(label):
        raise ValueError("labels and scores must be aligned non-empty vectors")
    if not np.isin(label, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError("binary metrics require 0/1 labels and finite scores")
    return label.astype(np.int8, copy=False), score


def _source_rows(source_id, row_count: int) -> tuple[np.ndarray, np.ndarray]:
    source_id = np.asarray(source_id).astype(str)
    if source_id.ndim != 1 or len(source_id) != row_count:
        raise ValueError("source IDs must align with prediction rows")
    groups, inverse = np.unique(source_id, return_inverse=True)
    if not len(groups):
        raise ValueError("source bootstrap requires at least one source")
    return groups, inverse


def _bootstrap_group_counts(group_index, replicates: int, seed: int):
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    group_count = int(np.max(group_index)) + 1
    random = np.random.default_rng(seed)
    for _ in range(int(replicates)):
        yield np.bincount(
            random.integers(group_count, size=group_count),
            minlength=group_count,
        ).astype(np.float64)


def _interval(values) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))
