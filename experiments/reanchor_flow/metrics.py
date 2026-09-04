"""Claim-level metrics and source-cluster paired bootstrap."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metric(label, score) -> dict[str, float | int | None]:
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    label, score = label[valid], score[valid]
    if not len(score) or np.unique(label).size < 2:
        return {"claims": int(len(score)), "auroc": None, "average_precision": None}
    return {
        "claims": int(len(score)),
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
    }


def paired_bootstrap(
    label,
    first,
    second,
    source,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Compare two scores on the same claims while resampling whole sources."""

    label = np.asarray(label, dtype=bool)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    source = np.asarray(source).astype(str)
    valid = np.isfinite(first) & np.isfinite(second)
    label, first, second, source = (
        label[valid], first[valid], second[valid], source[valid]
    )
    if not len(label) or np.unique(label).size < 2:
        return {
            "claims": int(len(label)),
            "auroc_difference": None,
            "average_precision_difference": None,
            "auroc_difference_ci95": [None, None],
            "average_precision_difference_ci95": [None, None],
            "replicates": 0,
        }

    point = (
        float(roc_auc_score(label, first) - roc_auc_score(label, second)),
        float(
            average_precision_score(label, first)
            - average_precision_score(label, second)
        ),
    )
    groups = np.unique(source)
    rows = {group: np.flatnonzero(source == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(max(repeats, 0)):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(label[index], first[index])
                - roc_auc_score(label[index], second[index]),
                average_precision_score(label[index], first[index])
                - average_precision_score(label[index], second[index]),
            )
        )
    interval = (
        np.quantile(np.asarray(estimates), (0.025, 0.975), axis=0)
        if estimates
        else np.full((2, 2), np.nan)
    )
    return {
        "claims": int(len(label)),
        "auroc_difference": point[0],
        "average_precision_difference": point[1],
        "auroc_difference_ci95": (
            interval[:, 0].tolist() if estimates else [None, None]
        ),
        "average_precision_difference_ci95": (
            interval[:, 1].tolist() if estimates else [None, None]
        ),
        "replicates": len(estimates),
    }


def paired_effect(first, second) -> dict[str, float | int | None]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return {
            "samples": 0,
            "first_absolute_mean": None,
            "second_absolute_mean": None,
            "absolute_difference_mean": None,
        }
    first, second = np.abs(first[valid]), np.abs(second[valid])
    return {
        "samples": int(len(first)),
        "first_absolute_mean": float(first.mean()),
        "second_absolute_mean": float(second.mean()),
        "absolute_difference_mean": float((first - second).mean()),
    }
