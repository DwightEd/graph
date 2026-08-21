"""Post-hoc evaluation after source-predictability scores are frozen."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, load_npz, write_json


def _open_dataset(split_root):
    from research_dataset import open_research_dataset

    return open_research_dataset(
        split_root,
        device="cpu",
        retain_embedded_labels=True,
    )


def _metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _bootstrap_metrics(
    labels: np.ndarray,
    score: np.ndarray,
    group: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    names = np.unique(group)
    indices = {name: np.flatnonzero(group == name) for name in names}
    values = []
    for _ in range(replicates):
        chosen = rng.choice(names, size=len(names), replace=True)
        selected = np.concatenate([indices[name] for name in chosen])
        if np.unique(labels[selected]).size == 2:
            current = _metrics(labels[selected], score[selected])
            values.append((current["auroc"], current["auprc"]))
    if not values:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
        }
    array = np.asarray(values)
    return {
        "auroc_ci_low": float(np.quantile(array[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(array[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(array[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(array[:, 1], 0.975)),
    }


def _paired_delta(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    group: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    observed_first = _metrics(labels, first)
    observed_second = _metrics(labels, second)
    rng = np.random.default_rng(seed)
    names = np.unique(group)
    indices = {name: np.flatnonzero(group == name) for name in names}
    values = []
    for _ in range(replicates):
        chosen = rng.choice(names, size=len(names), replace=True)
        selected = np.concatenate([indices[name] for name in chosen])
        if np.unique(labels[selected]).size == 2:
            one = _metrics(labels[selected], first[selected])
            two = _metrics(labels[selected], second[selected])
            values.append((one["auroc"] - two["auroc"], one["auprc"] - two["auprc"]))
    array = np.asarray(values) if values else np.empty((0, 2))
    return {
        "auroc_delta": observed_first["auroc"] - observed_second["auroc"],
        "auprc_delta": observed_first["auprc"] - observed_second["auprc"],
        "auroc_delta_ci_low": float(np.quantile(array[:, 0], 0.025))
        if len(array)
        else float("nan"),
        "auroc_delta_ci_high": float(np.quantile(array[:, 0], 0.975))
        if len(array)
        else float("nan"),
        "auprc_delta_ci_low": float(np.quantile(array[:, 1], 0.025))
        if len(array)
        else float("nan"),
        "auprc_delta_ci_high": float(np.quantile(array[:, 1], 0.975))
        if len(array)
        else float("nan"),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _labels_for_scores(dataset, labels_store, sample_id: np.ndarray) -> np.ndarray:
    labels = np.empty(len(sample_id), dtype=np.int8)
    start = 0
    while start < len(sample_id):
        current = str(sample_id[start])
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == current:
            stop += 1
        sample = dataset[current]
        current_labels = labels_store.response_labels(sample).cpu().numpy().astype(np.int8)
        if len(current_labels) != stop - start:
            raise ValueError("score and label token counts differ")
        labels[start:stop] = current_labels
        start = stop
    return labels


def _onset_effect(
    sample_id: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    *,
    window: int,
) -> tuple[int, float, float]:
    values = []
    start = 0
    while start < len(sample_id):
        current = str(sample_id[start])
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == current:
            stop += 1
        local_labels = labels[start:stop]
        positive = np.flatnonzero(local_labels == 1)
        if len(positive):
            onset = int(positive[0])
            pre_start = max(0, onset - window)
            after_stop = min(stop - start, onset + window)
            if pre_start < onset < after_stop:
                local = score[start:stop]
                values.append(
                    float(local[onset:after_stop].mean() - local[pre_start:onset].mean())
                )
        start = stop
    array = np.asarray(values, dtype=np.float64)
    standard = array.std(ddof=1) if len(array) > 1 else 0.0
    return (
        len(array),
        float(array.mean()) if len(array) else float("nan"),
        float(array.mean() / standard) if standard > 0 else 0.0,
    )


def evaluate_scores(
    *,
    split_root,
    score_paths: dict[str, str | Path],
    output_dir,
    bootstrap_replicates: int = 500,
    onset_window: int = 4,
    seed: int = 20260820,
) -> None:
    """Compare current, birth, dynamic, and shuffled-memory scores."""

    artifacts = {name: load_npz(path) for name, path in score_paths.items()}
    if not artifacts:
        raise ValueError("at least one score artifact is required")
    first = next(iter(artifacts.values()))
    for name, arrays in artifacts.items():
        if bool(arrays["labels_included"].item()):
            raise ValueError(f"{name} score artifact unexpectedly contains labels")
        for field in ("sample_id", "token_index"):
            if not np.array_equal(arrays[field], first[field]):
                raise ValueError("score artifacts do not describe the same token order")

    dataset = _open_dataset(split_root)
    labels_store = dataset.prepare_evaluation_labels()
    labels = _labels_for_scores(dataset, labels_store, first["sample_id"].astype(str))
    tasks = first["task_type"].astype(str)
    source_ids = first["source_id"].astype(str)

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, arrays in artifacts.items():
        valid = arrays["valid_rounds"] > 0
        scores[f"{name}:nll"] = (arrays["endpoint_nll"].astype(np.float32), valid)
        scores[f"{name}:negative_margin"] = (
            -arrays["margin"].astype(np.float32),
            valid,
        )
        if name == "dynamic":
            scores["dynamic:shuffled_nll"] = (
                arrays["shuffled_nll"].astype(np.float32),
                valid,
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    onset_rows = []
    for name, (score, valid) in scores.items():
        for task in ("__all__", *sorted(np.unique(tasks).tolist())):
            selected = valid if task == "__all__" else valid & (tasks == task)
            current = {
                "score": name,
                "task": task,
                "tokens": int(selected.sum()),
                "coverage": float(selected.mean()),
                "positives": int(labels[selected].sum()),
                "prevalence": float(labels[selected].mean())
                if selected.any()
                else float("nan"),
                **_metrics(labels[selected], score[selected]),
            }
            current.update(
                _bootstrap_metrics(
                    labels[selected],
                    score[selected],
                    source_ids[selected],
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
            )
            metric_rows.append(current)
        responses, effect, paired_dz = _onset_effect(
            first["sample_id"].astype(str), labels, score, window=onset_window
        )
        onset_rows.append(
            {
                "score": name,
                "responses": responses,
                "onset_minus_pre": effect,
                "paired_dz": paired_dz,
            }
        )

    delta_rows = []
    comparisons = (
        ("dynamic:nll", "current:nll"),
        ("dynamic:nll", "birth:nll"),
        ("dynamic:nll", "dynamic:shuffled_nll"),
    )
    for first_name, second_name in comparisons:
        if first_name not in scores or second_name not in scores:
            continue
        first_score, first_valid = scores[first_name]
        second_score, second_valid = scores[second_name]
        selected = first_valid & second_valid
        delta_rows.append(
            {
                "first": first_name,
                "second": second_name,
                "tokens": int(selected.sum()),
                **_paired_delta(
                    labels[selected],
                    first_score[selected],
                    second_score[selected],
                    source_ids[selected],
                    replicates=bootstrap_replicates,
                    seed=seed,
                ),
            }
        )

    coverage_rows = []
    for name, arrays in artifacts.items():
        valid = arrays["valid_rounds"] > 0
        relative = np.minimum(
            9,
            arrays["token_index"] * 10 // np.maximum(arrays["response_length"], 1),
        )
        for position_bin in range(10):
            selected = relative == position_bin
            coverage_rows.append(
                {
                    "mode": name,
                    "position_bin": position_bin,
                    "tokens": int(selected.sum()),
                    "coverage": float(valid[selected].mean())
                    if selected.any()
                    else float("nan"),
                    "positive_coverage": float(valid[selected & (labels == 1)].mean())
                    if (selected & (labels == 1)).any()
                    else float("nan"),
                    "negative_coverage": float(valid[selected & (labels == 0)].mean())
                    if (selected & (labels == 0)).any()
                    else float("nan"),
                }
            )

    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "paired_deltas.csv", delta_rows)
    _write_csv(output_dir / "coverage.csv", coverage_rows)
    _write_csv(output_dir / "onset_effects.csv", onset_rows)
    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "score_paths": {
                name: str(Path(path).resolve()) for name, path in score_paths.items()
            },
            "tokens": int(len(labels)),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "metrics": metric_rows,
            "paired_deltas": delta_rows,
            "onset_effects": onset_rows,
        },
    )
