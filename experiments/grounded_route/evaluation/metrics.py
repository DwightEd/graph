"""Token metrics and source-grouped uncertainty intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    label = np.asarray(label, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    prevalence = float(label.mean())
    auprc = float(average_precision_score(label, score))
    return {
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(label, score)),
        "auprc": auprc,
        "auprc_lift": float(auprc / prevalence),
    }


def source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    groups = np.unique(source_id.astype(str))
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                (
                    roc_auc_score(label[index], score[index]),
                    average_precision_score(label[index], score[index]),
                )
            )
    value = np.asarray(values)
    return {
        "replicates": int(len(value)),
        "auroc_low": float(np.quantile(value[:, 0], 0.025)),
        "auroc_high": float(np.quantile(value[:, 0], 0.975)),
        "auprc_low": float(np.quantile(value[:, 1], 0.025)),
        "auprc_high": float(np.quantile(value[:, 1], 0.975)),
    }


def paired_delta(
    label: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    point_left = binary_metrics(label, left)
    point_right = binary_metrics(label, right)
    groups = np.unique(source_id.astype(str))
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                (
                    roc_auc_score(label[index], left[index])
                    - roc_auc_score(label[index], right[index]),
                    average_precision_score(label[index], left[index])
                    - average_precision_score(label[index], right[index]),
                )
            )
    value = np.asarray(values)
    return {
        "auroc_delta": float(point_left["auroc"] - point_right["auroc"]),
        "auprc_delta": float(point_left["auprc"] - point_right["auprc"]),
        "replicates": int(len(value)),
        "auroc_delta_low": float(np.quantile(value[:, 0], 0.025)),
        "auroc_delta_high": float(np.quantile(value[:, 0], 0.975)),
        "auprc_delta_low": float(np.quantile(value[:, 1], 0.025)),
        "auprc_delta_high": float(np.quantile(value[:, 1], 0.975)),
    }
