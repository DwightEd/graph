"""Read labels only after P-Cut scores have been frozen."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, SCORE_SCHEMA, load_npz, sha256, write_json


SUPERVISED_MODEL_TYPE = "routing_fingerprint_linear_probe"


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
    source_id = source_id.astype(str)
    groups = list(dict.fromkeys(source_id.tolist()))
    index = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates: list[tuple[float, float]] = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([index[group] for group in chosen])
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
    value = np.asarray(estimates)
    return {
        "replicates_valid": len(value),
        "auroc_ci_low": float(np.quantile(value[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(value[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(value[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(value[:, 1], 0.975)),
    }


def response_labels(label_store, sample, count: int) -> np.ndarray:
    if hasattr(label_store, "positive_runs"):
        label = np.zeros(count, dtype=np.int8)
        for start, stop in label_store.positive_runs(
            sample.sample_id,
            response_count=count,
        ):
            label[start:stop] = 1
        return label
    return label_store.response_labels(sample).cpu().numpy().astype(np.int8)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_score_provenance(arrays: dict[str, np.ndarray]) -> None:
    if sha256(arrays["checkpoint_path"].item()) != str(arrays["checkpoint_sha256"].item()):
        raise ValueError("method file changed after scoring")
    if sha256(arrays["reference_path"].item()) != str(arrays["reference_sha256"].item()):
        raise ValueError("reference changed after scoring")


def validate_score_rows(dataset, arrays: dict[str, np.ndarray]) -> None:
    sample_id = arrays["sample_id"].astype(str)
    for current_id in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == current_id)
        sample = dataset[current_id]
        try:
            if not np.all(arrays["source_id"][selected].astype(str) == str(sample.source_id)):
                raise ValueError("score rows and source IDs differ")
            if not np.array_equal(
                arrays["token_index"][selected],
                np.arange(len(selected), dtype=arrays["token_index"].dtype),
            ):
                raise ValueError("score rows are not a complete response sequence")
            if not np.all(arrays["response_length"][selected] == len(selected)):
                raise ValueError("score rows and response lengths differ")
            attention = sample.attention()
            expected = (
                attention.token_ids[attention.response_idx :]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            if not np.array_equal(expected, arrays["response_token_id"][selected]):
                raise ValueError("score rows and response tokens differ")
        finally:
            sample.release_attention()


def evaluate(
    dataset,
    score_path,
    output_dir,
    bootstrap_replicates: int = 500,
    seed: int = 20260826,
) -> dict[str, object]:
    arrays = load_npz(score_path)
    if str(arrays["schema"].item()) != SCORE_SCHEMA:
        raise ValueError("unsupported P-Cut score artifact")
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact already contains labels")
    validate_score_provenance(arrays)
    validate_score_rows(dataset, arrays)

    sample_id = arrays["sample_id"].astype(str)
    labels = np.empty(len(sample_id), dtype=np.int8)
    label_store = dataset.prepare_evaluation_labels()
    for current_id in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == current_id)
        sample = dataset[current_id]
        try:
            current = response_labels(label_store, sample, len(selected))
            if len(current) != len(selected):
                raise ValueError("score rows and labels differ")
            labels[selected] = current
        finally:
            sample.release_attention()

    score = arrays["score"].astype(np.float64)
    absolute = arrays["token_index"].astype(np.float64)
    relative = absolute / np.maximum(arrays["response_length"].astype(np.float64) - 1.0, 1.0)

    next_labels: list[np.ndarray] = []
    next_scores: list[np.ndarray] = []
    for current_id in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == current_id)
        if len(selected) > 1:
            next_labels.append(labels[selected][1:])
            next_scores.append(score[selected][:-1])
    shifted_label = np.concatenate(next_labels) if next_labels else np.empty(0, dtype=np.int8)
    shifted_score = np.concatenate(next_scores) if next_scores else np.empty(0, dtype=np.float64)

    position_rows = []
    for name, value in (("score", score), ("absolute_position", absolute), ("relative_position", relative)):
        position_rows.append(
            {
                "score": name,
                "spearman_with_absolute_position": float(spearmanr(value, absolute).statistic),
                **metrics(labels, value),
            }
        )

    residual_rows = []
    residual_names = arrays["residual_names"].astype(str)
    standardized = arrays["standardized"].astype(np.float64)
    coverage = arrays["coverage"].astype(np.float64)
    for column, name in enumerate(residual_names):
        available = coverage[:, column] > 0
        residual_rows.append(
            {
                "residual": name,
                "tokens": int(available.sum()),
                "coverage": float(available.mean()),
                **metrics(labels[available], standardized[available, column]),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "position.csv", position_rows)
    write_csv(output_dir / "residuals.csv", residual_rows)

    labels_used_to_fit_model = bool(
        arrays.get(
            "labels_used_to_fit_model",
            np.asarray(str(arrays["model_type"].item()) == SUPERVISED_MODEL_TYPE),
        ).item()
    )
    report = {
        "schema": EVALUATION_SCHEMA,
        "model_type": str(arrays["model_type"].item()),
        "labels_read": True,
        "labels_used_to_fit_model": labels_used_to_fit_model,
        "labels_used_during": (
            "train_probe_fit_and_posthoc_test_evaluation"
            if labels_used_to_fit_model
            else "posthoc_evaluation_only"
        ),
        "samples": len(set(sample_id.tolist())),
        "tokens": len(labels),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "same_token": metrics(labels, score),
        "next_token": metrics(shifted_label, shifted_score),
        "absolute_position": metrics(labels, absolute),
        "relative_position": metrics(labels, relative),
        "score_position_spearman": float(spearmanr(score, absolute).statistic),
        "source_bootstrap": source_bootstrap(
            labels,
            score,
            arrays["source_id"],
            bootstrap_replicates,
            seed,
        ),
        "score_artifact": str(Path(score_path).resolve()),
        "score_sha256": sha256(score_path),
    }
    write_json(output_dir / "evaluation.json", report)
    return report
