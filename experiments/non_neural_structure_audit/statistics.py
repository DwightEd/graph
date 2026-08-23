"""Small grouped statistical tests used by several structure gates."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if np.unique(labels).size < 2:
        return {"auprc": float("nan"), "auroc": float("nan")}
    return {
        "auprc": float(average_precision_score(labels, score)),
        "auroc": float(roc_auc_score(labels, score)),
    }


def grouped_bootstrap_delta(
    labels_by_group: list[np.ndarray],
    real_by_group: list[np.ndarray],
    baseline_by_group: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a metric difference by resampling complete groups."""

    labels = np.concatenate(labels_by_group)
    real = np.concatenate(real_by_group)
    baseline = np.concatenate(baseline_by_group)
    real_metrics = binary_metrics(labels, real)
    baseline_metrics = binary_metrics(labels, baseline)
    point = {
        "auprc_delta": real_metrics["auprc"] - baseline_metrics["auprc"],
        "auroc_delta": real_metrics["auroc"] - baseline_metrics["auroc"],
    }

    rng = np.random.default_rng(seed)
    draws = []
    groups = len(labels_by_group)
    for _ in range(replicates):
        selected = rng.integers(groups, size=groups)
        current_labels = np.concatenate([labels_by_group[index] for index in selected])
        if np.unique(current_labels).size < 2:
            continue
        current_real = np.concatenate([real_by_group[index] for index in selected])
        current_baseline = np.concatenate(
            [baseline_by_group[index] for index in selected]
        )
        real_metric = binary_metrics(current_labels, current_real)
        baseline_metric = binary_metrics(current_labels, current_baseline)
        draws.append(
            (
                real_metric["auprc"] - baseline_metric["auprc"],
                real_metric["auroc"] - baseline_metric["auroc"],
            )
        )
    draws = np.asarray(draws, dtype=np.float64)
    if not len(draws):
        return {
            **point,
            "auprc_delta_ci_low": float("nan"),
            "auprc_delta_ci_high": float("nan"),
            "auroc_delta_ci_low": float("nan"),
            "auroc_delta_ci_high": float("nan"),
        }
    return {
        **point,
        "auprc_delta_ci_low": float(np.quantile(draws[:, 0], 0.025)),
        "auprc_delta_ci_high": float(np.quantile(draws[:, 0], 0.975)),
        "auroc_delta_ci_low": float(np.quantile(draws[:, 1], 0.025)),
        "auroc_delta_ci_high": float(np.quantile(draws[:, 1], 0.975)),
    }


def grouped_metric_interval(
    labels_by_group: list[np.ndarray],
    scores_by_group: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    metrics = binary_metrics(
        np.concatenate(labels_by_group), np.concatenate(scores_by_group)
    )
    rng = np.random.default_rng(seed)
    draws = []
    groups = len(labels_by_group)
    for _ in range(replicates):
        selected = rng.integers(groups, size=groups)
        labels = np.concatenate([labels_by_group[index] for index in selected])
        if np.unique(labels).size < 2:
            continue
        scores = np.concatenate([scores_by_group[index] for index in selected])
        current = binary_metrics(labels, scores)
        draws.append((current["auprc"], current["auroc"]))
    draws = np.asarray(draws, dtype=np.float64)
    if not len(draws):
        return {
            **metrics,
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
        }
    return {
        **metrics,
        "auprc_ci_low": float(np.quantile(draws[:, 0], 0.025)),
        "auprc_ci_high": float(np.quantile(draws[:, 0], 0.975)),
        "auroc_ci_low": float(np.quantile(draws[:, 1], 0.025)),
        "auroc_ci_high": float(np.quantile(draws[:, 1], 0.975)),
    }


def circular_shift_p_value(
    labels_by_group: list[np.ndarray],
    scores_by_group: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> float:
    """Preserve each response label sequence while breaking token alignment."""

    observed = binary_metrics(
        np.concatenate(labels_by_group), np.concatenate(scores_by_group)
    )["auprc"]
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        shifted = []
        for labels in labels_by_group:
            if len(labels) < 2:
                shifted.append(labels)
                continue
            offset = int(rng.integers(1, len(labels)))
            shifted.append(np.roll(labels, offset))
        null.append(
            binary_metrics(np.concatenate(shifted), np.concatenate(scores_by_group))[
                "auprc"
            ]
        )
    return float((1 + np.sum(np.asarray(null) >= observed)) / (replicates + 1))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return FDR-adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adjusted = np.empty_like(ranked)
    adjusted[order] = ranked
    return adjusted
