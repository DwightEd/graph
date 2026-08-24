"""Post-hoc evaluation for frozen attention holonomy audit scores."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, SCORE_SCHEMA, load_npz, save_npz, sha256


def _binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    selected = np.isfinite(score)
    labels = labels[selected]
    score = score[selected]
    result: dict[str, float | int | None] = {
        "tokens": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
    }
    if len(labels) == 0 or np.unique(labels).size < 2:
        result.update(auroc=None, auprc=None)
    else:
        result.update(
            auroc=float(roc_auc_score(labels, score)),
            auprc=float(average_precision_score(labels, score)),
        )
    return result


def _cluster_bootstrap(
    labels: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    selected = np.isfinite(score)
    labels = labels[selected]
    score = score[selected]
    source_id = source_id[selected].astype(str)
    groups = list(dict.fromkeys(source_id.tolist()))
    index = {group: np.flatnonzero(source_id == group) for group in groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        chosen = rng.choice(groups, len(groups), replace=True)
        rows = np.concatenate([index[group] for group in chosen])
        if np.unique(labels[rows]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(labels[rows], score[rows]),
                average_precision_score(labels[rows], score[rows]),
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


def _shifted_rows(
    labels: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_label = []
    shifted_score = []
    text = sample_id.astype(str)
    for sample in dict.fromkeys(text.tolist()):
        rows = np.flatnonzero(text == sample)
        if len(rows) < 2:
            continue
        shifted_label.append(labels[rows][1:])
        shifted_score.append(score[rows][:-1])
    if not shifted_label:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64)
    return np.concatenate(shifted_label), np.concatenate(shifted_score)


def _matched_effect(
    labels: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
    relative_position: np.ndarray,
    event_count: np.ndarray,
) -> tuple[int, float, float]:
    differences = []
    text = sample_id.astype(str)
    for sample in dict.fromkeys(text.tolist()):
        rows = np.flatnonzero(text == sample)
        positive = rows[labels[rows] == 1]
        negative = rows[labels[rows] == 0]
        if not len(positive) or not len(negative):
            continue
        available = set(map(int, negative.tolist()))
        for row in positive:
            candidates = np.asarray(sorted(available), dtype=np.int64)
            if not len(candidates) or not np.isfinite(score[row]):
                break
            candidates = candidates[np.isfinite(score[candidates])]
            if not len(candidates):
                break
            cost = np.abs(relative_position[candidates] - relative_position[row])
            cost += 0.25 * np.abs(
                np.log1p(event_count[candidates]) - np.log1p(event_count[row])
            )
            match = int(candidates[np.argmin(cost)])
            available.remove(match)
            differences.append(float(score[row] - score[match]))
    if not differences:
        return 0, float("nan"), float("nan")
    values = np.asarray(differences, dtype=np.float64)
    dz = float(values.mean() / max(values.std(ddof=1), 1e-12)) if len(values) > 1 else float("nan")
    return len(values), float(values.mean()), dz


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
        raise ValueError("unsupported attention holonomy score artifact")
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact must be frozen before evaluation")
    if sha256(arrays["reference_path"].item()) != str(arrays["reference_sha256"].item()):
        raise ValueError("reference changed after scoring")
    manifest = Path(dataset.root) / "manifest.json"
    if sha256(manifest) != str(arrays["dataset_manifest_sha256"].item()):
        raise ValueError("test dataset manifest changed after scoring")

    labels_store = dataset.prepare_evaluation_labels()
    sample_ids = arrays["sample_id"].astype(str)
    labels = np.empty(len(sample_ids), dtype=np.int8)
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        selected = np.flatnonzero(sample_ids == sample_id)
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            expected_tokens = attention.token_ids[attention.response_idx :].cpu().numpy()
            if not np.array_equal(expected_tokens, arrays["response_token_id"][selected]):
                raise ValueError("score rows and response token IDs disagree")
            current = labels_store.response_labels(sample).cpu().numpy().astype(np.int8)
            if len(current) != len(selected):
                raise ValueError("score rows and token labels have different lengths")
            labels[selected] = current
        finally:
            sample.release_attention()

    names = arrays["primary_feature_names"].astype(str).tolist()
    score_names = ["joint_score", *names]
    score_values = [arrays["joint_score"].astype(np.float64)]
    score_values.extend(
        arrays["standardized_primary"][:, index].astype(np.float64)
        for index in range(len(names))
    )
    score_names.extend(("absolute_position", "relative_position"))
    score_values.extend(
        (
            arrays["nuisance"][:, 0].astype(np.float64),
            arrays["nuisance"][:, 1].astype(np.float64),
        )
    )

    metric_rows = []
    correlation_rows = []
    matched_rows = []
    for index, (name, score) in enumerate(zip(score_names, score_values, strict=True)):
        same = _binary_metrics(labels, score)
        shifted_labels, shifted_score = _shifted_rows(labels, score, sample_ids)
        shifted = _binary_metrics(shifted_labels, shifted_score)
        bootstrap = _cluster_bootstrap(
            labels,
            score,
            arrays["source_id"],
            replicates=bootstrap_replicates,
            seed=seed + 101 * index,
        )
        metric_rows.append(
            {
                "score": name,
                "alignment": "same_token_posthoc",
                **same,
                **bootstrap,
            }
        )
        metric_rows.append(
            {
                "score": name,
                "alignment": "next_token_shifted",
                **shifted,
                "replicates_valid": None,
                "auroc_ci_low": None,
                "auroc_ci_high": None,
                "auprc_ci_low": None,
                "auprc_ci_high": None,
            }
        )
        finite = np.isfinite(score)
        correlation_rows.append(
            {
                "score": name,
                "spearman_absolute_position": float(
                    spearmanr(score[finite], arrays["nuisance"][finite, 0]).statistic
                )
                if finite.sum() > 2
                else None,
                "spearman_relative_position": float(
                    spearmanr(score[finite], arrays["nuisance"][finite, 1]).statistic
                )
                if finite.sum() > 2
                else None,
            }
        )
        pairs, difference, dz = _matched_effect(
            labels,
            score,
            sample_ids,
            arrays["nuisance"][:, 1],
            arrays["nuisance"][:, 3],
        )
        matched_rows.append(
            {
                "score": name,
                "pairs": pairs,
                "hallucination_minus_matched_correct": difference,
                "paired_dz": dz,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "position_correlations.csv", correlation_rows)
    _write_csv(output_dir / "matched_effects.csv", matched_rows)

    primary = metric_rows[0]
    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_detector": "joint_score",
        "samples": len(set(sample_ids.tolist())),
        "tokens": len(labels),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "token_metrics": {
            "auroc": primary["auroc"],
            "auprc": primary["auprc"],
        },
        "score_artifact_path": str(Path(score_path).resolve()),
        "score_artifact_sha256": sha256(score_path),
        "outputs": {
            "metrics": "metrics.csv",
            "position_correlations": "position_correlations.csv",
            "matched_effects": "matched_effects.csv",
        },
    }
    save_npz(output_dir / "aligned_labels.npz", labels=labels)
    (output_dir / "evaluation.json").write_text(
        __import__("json").dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
