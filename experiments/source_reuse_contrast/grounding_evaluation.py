"""Post-hoc label evaluation for frozen grounding-sensitive scores."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import GROUNDING_EVALUATION_SCHEMA, load_npz, write_json


def _open_dataset(split_root):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device="cpu", retain_embedded_labels=True)


def _metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
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
    values: list[tuple[float, float]] = []
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


def _labels_for_scores(dataset, label_store, sample_id: np.ndarray) -> np.ndarray:
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
            raise ValueError("score and label token counts differ")
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


def _onset_rows(
    sample_id: np.ndarray,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    window: int,
) -> list[dict[str, object]]:
    effects: dict[str, list[float]] = {name: [] for name in scores}
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
            post_stop = min(stop - start, onset + window)
            if pre_start < onset:
                for name, value in scores.items():
                    local = value[start:stop]
                    effects[name].append(
                        float(local[onset:post_stop].mean() - local[pre_start:onset].mean())
                    )
        start = stop
    rows = []
    for name, values in effects.items():
        array = np.asarray(values, dtype=np.float64)
        standard = array.std(ddof=1) if len(array) > 1 else 0.0
        rows.append(
            {
                "score": name,
                "responses": len(array),
                "onset_minus_pre": float(array.mean()) if len(array) else float("nan"),
                "paired_dz": float(array.mean() / standard) if standard > 0 else 0.0,
            }
        )
    return rows


def evaluate_grounding_scores(
    *,
    split_root,
    score_path,
    output_dir,
    bootstrap_replicates: int = 500,
    onset_window: int = 4,
    seed: int = 20260821,
) -> None:
    arrays = load_npz(score_path)
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact unexpectedly contains labels")
    dataset = _open_dataset(split_root)
    label_store = dataset.prepare_evaluation_labels()
    labels = _labels_for_scores(dataset, label_store, arrays["sample_id"].astype(str))
    valid = arrays["valid_rounds"] > 0
    scores = {
        "reconstruction": arrays["reconstruction"].astype(np.float32),
        "raw_reconstruction": arrays["raw_reconstruction"].astype(np.float32),
        "prompt_independence": -arrays["prompt_gain"].astype(np.float32),
        "response_dependence": arrays["response_gain"].astype(np.float32),
        "closure": arrays["closure"].astype(np.float32),
        "fragility": arrays["fragility"].astype(np.float32),
        "refinement_gain": arrays["refinement_gain"].astype(np.float32),
        "state_gain": arrays["state_gain"].astype(np.float32),
        "memory_specificity": arrays["memory_specificity"].astype(np.float32),
        "endpoint_specificity": arrays["endpoint_specificity"].astype(np.float32),
        "sensitivity": arrays["sensitivity_mean"].astype(np.float32),
    }
    tasks = arrays["task_type"].astype(str)
    source_ids = arrays["source_id"].astype(str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    for name, score in scores.items():
        for task in ("__all__", *sorted(np.unique(tasks).tolist())):
            selected = valid if task == "__all__" else valid & (tasks == task)
            current = {
                "score": name,
                "task": task,
                "tokens": int(selected.sum()),
                "coverage": float(selected.sum() / max((tasks == task).sum() if task != "__all__" else len(tasks), 1)),
                "positives": int(labels[selected].sum()),
                "prevalence": float(labels[selected].mean()) if selected.any() else float("nan"),
                **_metrics(labels[selected], score[selected]),
            }
            current.update(
                _group_bootstrap(
                    labels[selected],
                    score[selected],
                    source_ids[selected],
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
            )
            metric_rows.append(current)
    _write_csv(output_dir / "metrics.csv", metric_rows)

    onset_rows = _onset_rows(
        arrays["sample_id"].astype(str),
        labels,
        scores,
        window=onset_window,
    )
    _write_csv(output_dir / "onset_effects.csv", onset_rows)

    position = arrays["token_index"].astype(np.int32)
    position_bucket = np.floor(np.log2(position + 1)).astype(np.int32)
    coverage_rows = []
    for bucket in sorted(np.unique(position_bucket).tolist()):
        selected = position_bucket == bucket
        coverage_rows.append(
            {
                "position_bucket": bucket,
                "tokens": int(selected.sum()),
                "valid_tokens": int((valid & selected).sum()),
                "coverage": float((valid & selected).sum() / max(selected.sum(), 1)),
                "positive_tokens": int(labels[selected].sum()),
            }
        )
    _write_csv(output_dir / "coverage.csv", coverage_rows)

    write_json(
        output_dir / "evaluation.json",
        {
            "schema": GROUNDING_EVALUATION_SCHEMA,
            "labels_read": True,
            "score_path": str(Path(score_path).resolve()),
            "tokens": int(len(labels)),
            "valid_tokens": int(valid.sum()),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "metrics": metric_rows,
            "onset_effects": onset_rows,
            "coverage": coverage_rows,
        },
    )
