"""Source-cluster statistics for event-level mechanism tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metric(label, score) -> dict[str, float | int | None]:
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    label, score = label[valid], score[valid]
    positives = int(label.sum())
    prevalence = float(label.mean()) if len(label) else None
    if not len(score) or np.unique(label).size < 2:
        return {
            "claims": int(len(score)),
            "positives": positives,
            "prevalence": prevalence,
            "auroc": None,
            "average_precision": None,
            "ap_lift": None,
        }
    average_precision = float(average_precision_score(label, score))
    return {
        "claims": int(len(score)),
        "positives": positives,
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": average_precision,
        "ap_lift": average_precision / prevalence if prevalence else None,
    }


def metric_with_cluster_ci(
    label,
    score,
    source,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Metric plus a whole-source bootstrap interval."""

    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    source = np.asarray(source).astype(str)
    valid = np.isfinite(score)
    label, score, source = label[valid], score[valid], source[valid]
    result: dict[str, object] = metric(label, score)
    result.update(
        auroc_ci95=[None, None],
        average_precision_ci95=[None, None],
        bootstrap_replicates=0,
    )
    groups = np.unique(source)
    if (
        len(groups) < 2
        or np.unique(label).size < 2
        or repeats <= 0
    ):
        return result

    rows = {group: np.flatnonzero(source == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(label[index], score[index]),
                average_precision_score(label[index], score[index]),
            )
        )
    if len(estimates) < np.ceil(0.9 * repeats):
        return result
    interval = np.quantile(np.asarray(estimates), (0.025, 0.975), axis=0)
    result.update(
        auroc_ci95=interval[:, 0].tolist(),
        average_precision_ci95=interval[:, 1].tolist(),
        bootstrap_replicates=len(estimates),
    )
    return result


def cluster_summary(
    value,
    source,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Equal-source mean, bootstrap CI, and source-level sign-flip p-value."""

    value = np.asarray(value, dtype=np.float64)
    source = np.asarray(source).astype(str)
    valid = np.isfinite(value)
    value, source = value[valid], source[valid]
    groups = np.unique(source)
    source_mean = np.asarray(
        [value[source == group].mean() for group in groups], dtype=np.float64
    )
    result: dict[str, object] = {
        "events": int(len(value)),
        "sources": int(len(groups)),
        "event_mean": float(value.mean()) if len(value) else None,
        "source_mean": float(source_mean.mean()) if len(source_mean) else None,
        "source_median": float(np.median(source_mean)) if len(source_mean) else None,
        "ci95": [None, None],
        "sign_flip_p": None,
        "bootstrap_replicates": 0,
    }
    if len(groups) < 2 or repeats <= 0:
        return result

    random = np.random.default_rng(seed)
    index = random.integers(0, len(groups), size=(repeats, len(groups)))
    bootstrap = source_mean[index].mean(axis=1)
    result["ci95"] = np.quantile(bootstrap, (0.025, 0.975)).tolist()
    result["bootstrap_replicates"] = repeats

    signs = random.choice((-1.0, 1.0), size=(repeats, len(groups)))
    null = (signs * source_mean).mean(axis=1)
    observed = abs(float(source_mean.mean()))
    result["sign_flip_p"] = float(
        (1 + np.count_nonzero(np.abs(null) >= observed)) / (repeats + 1)
    )
    return result


def cluster_group_contrast(
    value,
    positive,
    source,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Equal-source mean in the positive group minus the negative group.

    Only sources containing both groups enter the estimate; their within-source
    differences are resampled. Repeated claims from a document therefore do not
    masquerade as independent evidence or create a cross-source group confound.
    """

    value = np.asarray(value, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    source = np.asarray(source).astype(str)
    if not (len(value) == len(positive) == len(source)):
        raise ValueError("contrast values, labels, and sources must align")
    valid = np.isfinite(value)
    value, positive, source = value[valid], positive[valid], source[valid]
    groups = np.unique(source)

    positive_groups = [group for group in groups if ((source == group) & positive).any()]
    negative_groups = [group for group in groups if ((source == group) & ~positive).any()]
    paired = np.asarray(
        sorted(set(positive_groups) & set(negative_groups)), dtype=str
    )
    source_effect = np.asarray(
        [
            value[(source == group) & positive].mean()
            - value[(source == group) & ~positive].mean()
            for group in paired
        ],
        dtype=np.float64,
    )
    estimate = float(source_effect.mean()) if len(source_effect) else None
    result: dict[str, object] = {
        "definition": "mean within-source (positive-group minus negative-group)",
        "events": int(len(value)),
        "positive_events": int(positive.sum()),
        "negative_events": int((~positive).sum()),
        "sources": int(len(paired)),
        "all_sources": int(len(groups)),
        "positive_sources": int(len(positive_groups)),
        "negative_sources": int(len(negative_groups)),
        "paired_sources": int(len(paired)),
        "difference": estimate,
        "ci95": [None, None],
        "bootstrap_replicates": 0,
    }
    if estimate is None or len(paired) < 2 or repeats <= 0:
        return result

    random = np.random.default_rng(seed)
    index = random.integers(0, len(paired), size=(repeats, len(paired)))
    estimates = source_effect[index].mean(axis=1)
    result["ci95"] = np.quantile(estimates, (0.025, 0.975)).tolist()
    result["bootstrap_replicates"] = repeats
    return result


def cluster_curve(
    values,
    source,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Mean event-time curve with whole-source bootstrap bands."""

    values = np.asarray(values, dtype=np.float64)
    source = np.asarray(source).astype(str)
    if values.ndim != 2 or values.shape[0] != len(source):
        raise ValueError("curve rows and source IDs must align")
    groups = np.unique(source)
    if not len(groups):
        width = values.shape[1]
        return {
            "mean": np.full(width, np.nan).tolist(),
            "ci95_low": np.full(width, np.nan).tolist(),
            "ci95_high": np.full(width, np.nan).tolist(),
            "events": 0,
            "sources": 0,
        }
    source_curve = np.stack(
        [np.nanmean(values[source == group], axis=0) for group in groups]
    )
    mean = np.nanmean(source_curve, axis=0)
    low = high = np.full(values.shape[1], np.nan)
    if len(groups) >= 2 and repeats > 0:
        random = np.random.default_rng(seed)
        index = random.integers(0, len(groups), size=(repeats, len(groups)))
        draws = np.nanmean(source_curve[index], axis=1)
        low, high = np.nanquantile(draws, (0.025, 0.975), axis=0)
    return {
        "mean": mean.tolist(),
        "ci95_low": low.tolist(),
        "ci95_high": high.tolist(),
        "events": int(len(values)),
        "sources": int(len(groups)),
    }
