"""Small claim-level metrics and source-cluster paired bootstrap."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metric(label: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(score)
    label = label[valid]
    score = score[valid]
    if len(label) == 0 or np.unique(label).size < 2:
        return {"claims": int(len(label)), "auroc": None, "average_precision": None}
    return {
        "claims": int(len(label)),
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": float(average_precision_score(label, score)),
    }


def paired_bootstrap(
    label: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    source: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    valid = np.isfinite(first) & np.isfinite(second)
    label, first, second, source = (
        label[valid], first[valid], second[valid], source[valid]
    )
    if repeats <= 0 or len(label) == 0 or np.unique(label).size < 2:
        return {"replicates": 0, "auroc_difference_ci95": [None, None]}
    groups = np.unique(source)
    rows = {group: np.flatnonzero(source == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size < 2:
            continue
        values.append(
            roc_auc_score(label[index], first[index])
            - roc_auc_score(label[index], second[index])
        )
    if not values:
        return {"replicates": 0, "auroc_difference_ci95": [None, None]}
    return {
        "replicates": len(values),
        "auroc_difference_ci95": np.quantile(values, (0.025, 0.975)).tolist(),
    }


def curve_mean(sum_: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.divide(sum_, count, out=np.full_like(sum_, np.nan), where=count > 0)
