"""Post-hoc evaluation of frozen typed route-grammar scores."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import (
    EVALUATION_SCHEMA,
    SCORE_SCHEMA,
    load_npz,
    sha256,
    write_json,
)

DIAGNOSTIC_DIRECTION = {
    "score": 1.0,
    "grammar_surprisal_mean": 1.0,
    "order1_surprisal_mean": 1.0,
    "order2_gain_mean": -1.0,
    "rupture_mean": 1.0,
    "closure_mean": 1.0,
    "rupture_closure_mean": 1.0,
}


def _labels(store, sample, count: int) -> np.ndarray:
    if hasattr(store, "response_labels"):
        return store.response_labels(sample).cpu().numpy().astype(np.int8)
    result = np.zeros(count, dtype=np.int8)
    for start, stop in store.positive_runs(
        sample.sample_id,
        response_count=count,
    ):
        result[start:stop] = 1
    return result


def _metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if np.unique(labels).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _cluster_bootstrap(
    labels: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    groups = list(dict.fromkeys(sample_id.astype(str).tolist()))
    index = {
        group: np.flatnonzero(sample_id.astype(str) == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        chosen = rng.choice(groups, len(groups), replace=True)
        selected = np.concatenate([index[group] for group in chosen])
        if np.unique(labels[selected]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(labels[selected], score[selected]),
                average_precision_score(labels[selected], score[selected]),
            )
        )
    if not estimates:
        return {
            "replicates_valid": 0,
            "auroc_ci_low": None,
            "auroc_ci_high": None,
            "auprc_ci_low": None,
            "auprc_ci_high": None,
        }
    values = np.asarray(estimates)
    return {
        "replicates_valid": len(values),
        "auroc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def evaluate_scores(
    dataset,
    score_path,
    output_dir,
    *,
    bootstrap_replicates: int = 500,
    seed: int = 20260825,
) -> dict[str, object]:
    arrays = load_npz(score_path)
    if str(arrays["schema"].item()) != SCORE_SCHEMA:
        raise ValueError("unsupported typed route score artifact")
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact must be frozen before labels are opened")
    if sha256(arrays["reference_path"].item()) != str(
        arrays["reference_sha256"].item()
    ):
        raise ValueError("reference bytes changed after scoring")

    labels_store = dataset.prepare_evaluation_labels()
    labels = np.empty(len(arrays["score"]), dtype=np.int8)
    sample_ids = arrays["sample_id"].astype(str)
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        selected = np.flatnonzero(sample_ids == sample_id)
        sample = dataset[sample_id]
        current = _labels(labels_store, sample, len(selected))
        if len(current) != len(selected):
            raise ValueError("score rows and token labels have different lengths")
        labels[selected] = current
        sample.release_attention()

    primary = arrays["score"].astype(np.float64)
    primary_metrics = _metrics(labels, primary)
    bootstrap = _cluster_bootstrap(
        labels,
        primary,
        sample_ids,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    diagnostic_rows = []
    for name, direction in DIAGNOSTIC_DIRECTION.items():
        score = arrays[name].astype(np.float64) * direction
        diagnostic_rows.append(
            {
                "score": name,
                "direction": direction,
                "tokens": len(labels),
                "positives": int(labels.sum()),
                "prevalence": float(labels.mean()),
                **_metrics(labels, score),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "diagnostic_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(diagnostic_rows[0]),
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "alignment": "post_token_state_t_to_label_t",
        "primary_detector": "score",
        "samples": len(set(sample_ids.tolist())),
        "tokens": len(labels),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "token_metrics": primary_metrics,
        "sample_cluster_bootstrap": bootstrap,
        "reference_path": str(arrays["reference_path"].item()),
        "reference_sha256": str(arrays["reference_sha256"].item()),
        "score_artifact_path": str(Path(score_path).resolve()),
        "score_artifact_sha256": sha256(score_path),
    }
    write_json(output_dir / "evaluation.json", report)
    return report
