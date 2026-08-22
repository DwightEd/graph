"""Post-hoc label evaluation for frozen graph-structure audit artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, load_npz, write_json


def _open_dataset(split_root):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device="cpu", retain_embedded_labels=True)


def _labels_for_tokens(dataset, label_store, sample_id: np.ndarray) -> np.ndarray:
    labels = np.empty(len(sample_id), dtype=np.int8)
    start = 0
    while start < len(sample_id):
        current = str(sample_id[start])
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == current:
            stop += 1
        sample = dataset[current]
        current_labels = label_store.response_labels(sample).cpu().numpy().astype(np.int8)
        if len(current_labels) != stop - start:
            raise ValueError("graph-audit token count differs from response labels")
        labels[start:stop] = current_labels
        start = stop
    return labels


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _basic_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(score) == 0 or np.unique(labels).size < 2:
        return {
            "auroc": float("nan"),
            "auprc_high": float("nan"),
            "auprc_low": float("nan"),
        }
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc_high": float(average_precision_score(labels, score)),
        "auprc_low": float(average_precision_score(labels, -score)),
    }


def _group_bootstrap(
    labels: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    indices = {name: np.flatnonzero(groups == name) for name in names}
    auc_values: list[float] = []
    difference_values: list[float] = []
    for _ in range(replicates):
        chosen = rng.choice(names, size=len(names), replace=True)
        selected = np.concatenate([indices[name] for name in chosen])
        current_labels = labels[selected]
        current_score = score[selected]
        if np.unique(current_labels).size < 2:
            continue
        auc_values.append(float(roc_auc_score(current_labels, current_score)))
        difference_values.append(
            float(
                current_score[current_labels == 1].mean()
                - current_score[current_labels == 0].mean()
            )
        )
    if not auc_values:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "mean_difference_ci_low": float("nan"),
            "mean_difference_ci_high": float("nan"),
        }
    return {
        "auroc_ci_low": float(np.quantile(auc_values, 0.025)),
        "auroc_ci_high": float(np.quantile(auc_values, 0.975)),
        "mean_difference_ci_low": float(np.quantile(difference_values, 0.025)),
        "mean_difference_ci_high": float(np.quantile(difference_values, 0.975)),
    }


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    selected = np.flatnonzero(finite)
    if not len(selected):
        return result.tolist()
    order = selected[np.argsort(values[selected])]
    count = len(order)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, values[index] * count / rank)
        adjusted[rank - 1] = running
    for rank, index in enumerate(order):
        result[index] = min(adjusted[rank], 1.0)
    return result.tolist()


def _feature_rows(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    tasks: np.ndarray,
    names: list[str],
    values: np.ndarray,
    family: str,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column, name in enumerate(names):
        score = values[:, column].astype(np.float64)
        for task in ("__all__", *sorted(np.unique(tasks).tolist())):
            selected = np.isfinite(score)
            if task != "__all__":
                selected &= tasks == task
            current_labels = labels[selected]
            current_score = score[selected]
            current_groups = groups[selected]
            negative = current_score[current_labels == 0]
            positive = current_score[current_labels == 1]
            metrics = _basic_metrics(current_labels, current_score)
            pooled = float(current_score.std(ddof=1)) if len(current_score) > 1 else 0.0
            difference = (
                float(positive.mean() - negative.mean())
                if len(positive) and len(negative)
                else float("nan")
            )
            try:
                p_value = (
                    float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
                    if len(positive) and len(negative)
                    else float("nan")
                )
            except ValueError:
                p_value = float("nan")
            row = {
                "family": family,
                "feature": name,
                "task": task,
                "tokens": int(selected.sum()),
                "positives": int(current_labels.sum()) if len(current_labels) else 0,
                "prevalence": float(current_labels.mean()) if len(current_labels) else float("nan"),
                "correct_mean": float(negative.mean()) if len(negative) else float("nan"),
                "hallucination_mean": float(positive.mean()) if len(positive) else float("nan"),
                "correct_median": float(np.median(negative)) if len(negative) else float("nan"),
                "hallucination_median": float(np.median(positive)) if len(positive) else float("nan"),
                "mean_difference": difference,
                "standardized_difference": difference / pooled if pooled > 0 else 0.0,
                "mann_whitney_p": p_value,
                **metrics,
                "separability": (
                    max(metrics["auroc"], 1.0 - metrics["auroc"])
                    if np.isfinite(metrics["auroc"])
                    else float("nan")
                ),
                "diagnostic_direction": (
                    "high"
                    if np.isfinite(metrics["auroc"]) and metrics["auroc"] >= 0.5
                    else "low"
                ),
                "diagnostic_auprc": (
                    max(metrics["auprc_high"], metrics["auprc_low"])
                    if np.isfinite(metrics["auprc_high"])
                    else float("nan")
                ),
            }
            row.update(
                _group_bootstrap(
                    current_labels,
                    current_score,
                    current_groups,
                    replicates=bootstrap_replicates,
                    seed=seed + column,
                )
                if len(current_score) and np.unique(current_labels).size == 2
                else {
                    "auroc_ci_low": float("nan"),
                    "auroc_ci_high": float("nan"),
                    "mean_difference_ci_low": float("nan"),
                    "mean_difference_ci_high": float("nan"),
                }
            )
            rows.append(row)
    q_values = _benjamini_hochberg(
        [float(row["mann_whitney_p"]) for row in rows]
    )
    for row, q_value in zip(rows, q_values):
        row["mann_whitney_q"] = q_value
    return rows


def _matched_rows(
    *,
    sample_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    labels: np.ndarray,
    structural_names: list[str],
    structural: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
) -> list[dict[str, object]]:
    unique_sources = structural[:, structural_names.index("unique_sources")]
    retained_mass = structural[:, structural_names.index("retained_mass")]
    pairs: list[tuple[int, int]] = []
    start = 0
    while start < len(sample_id):
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == sample_id[start]:
            stop += 1
        local = np.arange(start, stop)
        positive = local[labels[local] == 1]
        negative = local[labels[local] == 0]
        for index in positive.tolist():
            if not len(negative):
                continue
            position_distance = np.abs(
                token_index[negative]
                / np.maximum(response_length[negative] - 1, 1)
                - token_index[index] / max(response_length[index] - 1, 1)
            )
            degree_distance = np.abs(
                np.log1p(unique_sources[negative]) - np.log1p(unique_sources[index])
            )
            mass_distance = np.abs(retained_mass[negative] - retained_mass[index])
            cost = position_distance + 0.25 * degree_distance + 0.25 * mass_distance
            pairs.append((index, int(negative[np.argmin(cost)])))
        start = stop

    rows: list[dict[str, object]] = []
    for column, name in enumerate(feature_names):
        differences = np.asarray(
            [
                feature_values[positive, column] - feature_values[negative, column]
                for positive, negative in pairs
                if np.isfinite(feature_values[positive, column])
                and np.isfinite(feature_values[negative, column])
            ],
            dtype=np.float64,
        )
        standard = differences.std(ddof=1) if len(differences) > 1 else 0.0
        rows.append(
            {
                "feature": name,
                "matched_pairs": len(differences),
                "hallucination_minus_correct": (
                    float(differences.mean()) if len(differences) else float("nan")
                ),
                "paired_dz": (
                    float(differences.mean() / standard) if standard > 0 else 0.0
                ),
                "median_difference": (
                    float(np.median(differences)) if len(differences) else float("nan")
                ),
            }
        )
    return rows


def _recoverability_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_features = {
        "endpoint_recovery_error",
        "channel_recovery_error",
        "channel_weight_mae",
        "endpoint_mrr",
        "channel_mrr",
    }
    rows: list[dict[str, object]] = []
    for row in metric_rows:
        if row["family"] != "recovery" or row["task"] != "__all__":
            continue
        if row["feature"] not in selected_features:
            continue
        difference = float(row["mean_difference"])
        low = float(row["mean_difference_ci_low"])
        high = float(row["mean_difference_ci_high"])
        error_metric = row["feature"] in {
            "endpoint_recovery_error",
            "channel_recovery_error",
            "channel_weight_mae",
        }
        signed_error_difference = difference if error_metric else -difference
        signed_low = low if error_metric else -high
        signed_high = high if error_metric else -low
        if signed_low > 0:
            conclusion = "correct_more_recoverable"
        elif signed_high < 0:
            conclusion = "hallucination_more_recoverable"
        else:
            conclusion = "inconclusive"
        rows.append(
            {
                "feature": row["feature"],
                "tokens": row["tokens"],
                "hallucination_minus_correct_error": signed_error_difference,
                "error_difference_ci_low": signed_low,
                "error_difference_ci_high": signed_high,
                "conclusion": conclusion,
            }
        )
    return rows


def evaluate_graph_audit(
    *,
    split_root,
    token_path,
    output_dir,
    bootstrap_replicates: int = 500,
    seed: int = 20260822,
) -> None:
    arrays = load_npz(token_path)
    if bool(arrays["labels_included"].item()):
        raise ValueError("graph-audit artifact unexpectedly contains labels")
    dataset = _open_dataset(split_root)
    label_store = dataset.prepare_evaluation_labels()
    sample_id = arrays["sample_id"].astype(str)
    labels = _labels_for_tokens(dataset, label_store, sample_id)
    source_id = arrays["source_id"].astype(str)
    tasks = arrays["task_type"].astype(str)
    structural_names = arrays["structural_names"].astype(str).tolist()
    recovery_names = arrays["recovery_names"].astype(str).tolist()
    structural = arrays["structural"].astype(np.float64)
    recovery = arrays["recovery"].astype(np.float64)

    structural_rows = _feature_rows(
        labels=labels,
        groups=source_id,
        tasks=tasks,
        names=structural_names,
        values=structural,
        family="structure",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    recovery_rows = _feature_rows(
        labels=labels,
        groups=source_id,
        tasks=tasks,
        names=recovery_names,
        values=recovery,
        family="recovery",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 10000,
    )
    metric_rows = structural_rows + recovery_rows
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "feature_metrics.csv", metric_rows)

    combined_names = [f"structure:{name}" for name in structural_names] + [
        f"recovery:{name}" for name in recovery_names
    ]
    combined_values = np.concatenate((structural, recovery), axis=1)
    matched = _matched_rows(
        sample_id=sample_id,
        token_index=arrays["token_index"].astype(np.int32),
        response_length=arrays["response_length"].astype(np.int32),
        labels=labels,
        structural_names=structural_names,
        structural=structural,
        feature_names=combined_names,
        feature_values=combined_values,
    )
    _write_csv(output_dir / "matched_effects.csv", matched)

    recoverability = _recoverability_rows(metric_rows)
    _write_csv(output_dir / "recoverability_hypotheses.csv", recoverability)
    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "token_path": str(Path(token_path).resolve()),
            "tokens": int(len(labels)),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "features": len(metric_rows),
            "recoverability_hypotheses": recoverability,
            "outputs": {
                "feature_metrics": "feature_metrics.csv",
                "matched_effects": "matched_effects.csv",
                "recoverability_hypotheses": "recoverability_hypotheses.csv",
            },
        },
    )
