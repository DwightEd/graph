"""Post-hoc evaluation of frozen HoloRoute scores."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, SCORE_SCHEMA, load_npz, sha256, write_json


def _metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(label) == 0 or np.unique(label).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
    }


def _source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    groups = list(dict.fromkeys(source_id.astype(str).tolist()))
    index = {group: np.flatnonzero(source_id.astype(str) == group) for group in groups}
    rng = np.random.default_rng(seed)
    estimate: list[tuple[float, float]] = []
    for _ in range(replicates):
        chosen = rng.choice(groups, len(groups), replace=True)
        selected = np.concatenate([index[group] for group in chosen])
        if np.unique(label[selected]).size < 2:
            continue
        estimate.append(
            (
                roc_auc_score(label[selected], score[selected]),
                average_precision_score(label[selected], score[selected]),
            )
        )
    if not estimate:
        return {
            "replicates_valid": 0,
            "auroc_ci_low": None,
            "auroc_ci_high": None,
            "auprc_ci_low": None,
            "auprc_ci_high": None,
        }
    value = np.asarray(estimate)
    return {
        "replicates_valid": len(value),
        "auroc_ci_low": float(np.quantile(value[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(value[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(value[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(value[:, 1], 0.975)),
    }


def _labels(label_store, sample, count: int) -> np.ndarray:
    if hasattr(label_store, "response_labels"):
        return label_store.response_labels(sample).cpu().numpy().astype(np.int8)
    label = np.zeros(count, dtype=np.int8)
    for start, stop in label_store.positive_runs(sample.sample_id, response_count=count):
        label[start:stop] = 1
    return label


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        raise ValueError("unsupported HoloRoute score artifact")
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact must remain label-free")
    manifest = Path(dataset.root) / "manifest.json"
    if sha256(manifest) != str(arrays["dataset_manifest_sha256"].item()):
        raise ValueError("score artifact and test manifest differ")
    if sha256(arrays["checkpoint_path"].item()) != str(arrays["checkpoint_sha256"].item()):
        raise ValueError("checkpoint bytes changed after scoring")
    if sha256(arrays["density_path"].item()) != str(arrays["density_sha256"].item()):
        raise ValueError("density bytes changed after scoring")

    label_store = dataset.prepare_evaluation_labels()
    sample_id = arrays["sample_id"].astype(str)
    label = np.empty(len(sample_id), dtype=np.int8)
    for current_id in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == current_id)
        sample = dataset[current_id]
        try:
            attention = sample.attention()
            expected_token = attention.token_ids[attention.response_idx :].cpu().numpy().astype(np.int64)
            if not np.array_equal(expected_token, arrays["response_token_id"][selected]):
                raise ValueError("score rows and response token IDs differ")
            current = _labels(label_store, sample, len(selected))
            if len(current) != len(selected):
                raise ValueError("score rows and labels have different lengths")
            label[selected] = current
        finally:
            sample.release_attention()

    score = arrays["score"].astype(np.float64)
    same = _metrics(label, score)
    next_label: list[np.ndarray] = []
    next_score: list[np.ndarray] = []
    for current_id in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == current_id)
        if len(selected) > 1:
            next_label.append(label[selected][1:])
            next_score.append(score[selected][:-1])
    shifted_label = np.concatenate(next_label) if next_label else np.empty(0, dtype=np.int8)
    shifted_score = np.concatenate(next_score) if next_score else np.empty(0, dtype=np.float64)

    absolute_position = arrays["token_index"].astype(np.float64)
    relative_position = absolute_position / np.maximum(
        arrays["response_length"].astype(np.float64) - 1.0,
        1.0,
    )
    position_rows = []
    for name, value in (
        ("score", score),
        ("absolute_position", absolute_position),
        ("relative_position", relative_position),
    ):
        correlation = spearmanr(value, absolute_position).statistic
        position_rows.append(
            {
                "score": name,
                "spearman_with_absolute_position": float(correlation),
                **_metrics(label, value),
            }
        )

    feature_names = arrays["score_feature_names"].astype(str)
    feature = arrays["standardized_feature"].astype(np.float64)
    coverage = arrays["coverage"].astype(np.float64)
    feature_rows = []
    for index, name in enumerate(feature_names):
        available = coverage[:, index] > 0
        feature_rows.append(
            {
                "score": name,
                "tokens": int(available.sum()),
                "coverage": float(available.mean()),
                **_metrics(label[available], feature[available, index]),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "position_correlations.csv", position_rows)
    _write_csv(output_dir / "mechanism_metrics.csv", feature_rows)

    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_detector": "score",
        "samples": len(set(sample_id.tolist())),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "same_token_metrics": same,
        "next_token_metrics": _metrics(shifted_label, shifted_score),
        "position_baselines": {
            "absolute": _metrics(label, absolute_position),
            "relative": _metrics(label, relative_position),
            "score_spearman_absolute_position": float(
                spearmanr(score, absolute_position).statistic
            ),
        },
        "source_cluster_bootstrap": _source_bootstrap(
            label,
            score,
            arrays["source_id"],
            replicates=bootstrap_replicates,
            seed=seed,
        ),
        "score_artifact_path": str(Path(score_path).resolve()),
        "score_artifact_sha256": sha256(score_path),
    }
    write_json(output_dir / "evaluation.json", report)
    return report
