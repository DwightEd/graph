"""Post-hoc evaluation of a frozen GroundedRoute token score artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import FrozenEvaluation, scalar_text, validate_source_audit

from .artifacts import load_scores


def metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if not len(label) or np.unique(label).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
    }


def source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    source_id = np.asarray(source_id).astype(str)
    groups = tuple(dict.fromkeys(source_id.tolist()))
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates: list[tuple[float, float]] = []
    for _ in range(replicates):
        selected_groups = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([rows[group] for group in selected_groups])
        if np.unique(label[selected]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(label[selected], score[selected]),
                average_precision_score(label[selected], score[selected]),
            )
        )
    if not estimates:
        return {"replicates_valid": 0}
    values = np.asarray(estimates)
    return {
        "replicates_valid": len(values),
        "auroc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def evaluate(
    dataset,
    score_path,
    output_path,
    *,
    bootstrap_replicates: int = 500,
    seed: int = 20260825,
) -> dict[str, object]:
    """Open labels only after score identity and source audit are frozen."""

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    scores = load_scores(frozen.artifact.path)
    validate_source_audit(
        reserved_source_ids=scores["reserved_source_ids"],
        test_source_ids=scores["test_source_ids"],
        test_sample_ids=scores["test_sample_ids"],
        row_sample_ids=scores["sample_id"],
        row_source_ids=scores["source_id"],
        audit_scope=scalar_text(scores, "audit_scope"),
    )
    labels = frozen.align_loaded(dataset, scores)

    label = labels.token_label.astype(np.int8)
    score = np.asarray(scores["score"], dtype=np.float64)
    result = metrics(label, score)
    variant = scalar_text(scores, "variant")
    report = {
        "schema": "grounded-route-evaluation",
        "version": 1,
        "model_type": "grounded_route",
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "variant": variant,
        "changed_fraction": float(np.asarray(scores["changed_fraction"]).item()),
        "calibration_changed_fraction": float(
            np.asarray(scores["calibration_changed_fraction"]).item()
        ),
        "samples": len(set(scores["sample_id"].astype(str).tolist())),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        **result,
        "source_bootstrap": source_bootstrap(
            label,
            score,
            labels.source_id,
            bootstrap_replicates,
            seed,
        ),
        "score_artifact": str(frozen.artifact.path),
        "score_sha256": frozen.artifact.sha256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "evaluation": str(output_path.resolve())}
