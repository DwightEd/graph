"""Paired, sample-clustered comparison of matched head-model runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import write_json

IDENTITY_FIELDS = ("sample_id", "source_id", "task_type", "token_index", "token_label")


def _load_matched(reuse_path: Path, no_reuse_path: Path) -> tuple[dict, dict]:
    with np.load(reuse_path, allow_pickle=False) as artifact:
        reuse = {name: artifact[name] for name in artifact.files}
    with np.load(no_reuse_path, allow_pickle=False) as artifact:
        no_reuse = {name: artifact[name] for name in artifact.files}
    for name in IDENTITY_FIELDS:
        if name not in reuse or name not in no_reuse:
            raise ValueError(f"prediction artifact is missing {name}")
        if not np.array_equal(reuse[name], no_reuse[name]):
            raise ValueError(f"matched runs differ in {name}")
    return reuse, no_reuse


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return {
            "tokens": len(labels),
            "positives": positives,
            "prevalence": float(labels.mean()),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    return {
        "tokens": len(labels),
        "positives": positives,
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _paired_bootstrap(
    labels: np.ndarray,
    reuse_score: np.ndarray,
    no_reuse_score: np.ndarray,
    sample_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    unique_samples = np.unique(sample_id)
    clusters = [np.flatnonzero(sample_id == current) for current in unique_samples]
    rng = np.random.default_rng(seed)
    deltas = {"auroc": [], "auprc": []}
    for _ in range(replicates):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[index] for index in selected])
        if np.unique(labels[rows]).size < 2:
            continue
        reuse_metrics = _metrics(labels[rows], reuse_score[rows])
        no_reuse_metrics = _metrics(labels[rows], no_reuse_score[rows])
        for metric, values in deltas.items():
            values.append(reuse_metrics[metric] - no_reuse_metrics[metric])

    result = {}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            raise ValueError("bootstrap produced no samples containing both classes")
        result[metric] = {
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "valid_replicates": len(array),
        }
    return result


def _comparison(
    labels: np.ndarray,
    reuse_score: np.ndarray,
    no_reuse_score: np.ndarray,
    sample_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    reuse = _metrics(labels, reuse_score)
    no_reuse = _metrics(labels, no_reuse_score)
    intervals = _paired_bootstrap(
        labels,
        reuse_score,
        no_reuse_score,
        sample_id,
        replicates=replicates,
        seed=seed,
    )
    delta = {}
    for metric in ("auroc", "auprc"):
        delta[metric] = float(reuse[metric] - no_reuse[metric])
        delta[f"{metric}_ci95"] = intervals[metric]["ci95"]
    delta["valid_replicates"] = min(
        intervals["auroc"]["valid_replicates"],
        intervals["auprc"]["valid_replicates"],
    )
    return {
        "tokens": len(labels),
        "reuse": reuse,
        "no_reuse": no_reuse,
        "delta_reuse_minus_no_reuse": delta,
    }


def compare_head_model_runs(
    reuse_predictions: str | Path,
    no_reuse_predictions: str | Path,
    *,
    output: str | Path,
    bootstrap_replicates: int = 500,
    seed: int = 20260820,
) -> dict[str, object]:
    """Compare aligned runs without treating correlated tokens as independent."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    reuse, no_reuse = _load_matched(Path(reuse_predictions), Path(no_reuse_predictions))
    labels = reuse["token_label"].astype(np.int8)
    sample_id = reuse["sample_id"]
    current = _comparison(
        labels,
        reuse["current_probability"],
        no_reuse["current_probability"],
        sample_id,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    adjacent = (
        (sample_id[:-1] == sample_id[1:])
        & (reuse["token_index"][1:] == reuse["token_index"][:-1] + 1)
    )
    forecast = _comparison(
        labels[1:][adjacent],
        reuse["forecast_probability"][:-1][adjacent],
        no_reuse["forecast_probability"][:-1][adjacent],
        sample_id[:-1][adjacent],
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    result = {
        "schema": "head-model-paired-ablation-v1",
        "bootstrap_unit": "complete sample_id",
        "bootstrap_replicates": bootstrap_replicates,
        "reuse_predictions": str(Path(reuse_predictions).resolve()),
        "no_reuse_predictions": str(Path(no_reuse_predictions).resolve()),
        "current": current,
        "forecast_1": forecast,
    }
    write_json(Path(output), result)
    return result
