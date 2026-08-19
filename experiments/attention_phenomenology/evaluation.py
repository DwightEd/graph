"""Post-hoc hypothesis tests for frozen attention phenomenology artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .artifacts import EVALUATION_SCHEMA, load_npz, read_json, write_json
from .hypotheses import (
    FAMILY_NAMES,
    FEATURE_NAMES,
    LOCKIN_DIRECTIONS,
    ONSET_DIRECTIONS,
)


def _metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _sample_bootstrap(
    labels_by_sample: list[np.ndarray],
    scores_by_sample: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    count = len(labels_by_sample)
    for _ in range(replicates):
        selected = rng.integers(count, size=count)
        labels = np.concatenate([labels_by_sample[index] for index in selected])
        score = np.concatenate([scores_by_sample[index] for index in selected])
        if np.unique(labels).size == 2:
            current = _metrics(labels, score)
            values.append((current["auroc"], current["auprc"]))
    values = np.asarray(values)
    return {
        "auroc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def _paired_bootstrap_delta(
    labels_by_sample: list[np.ndarray],
    real_by_sample: list[np.ndarray],
    null_by_sample: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = []
    count = len(labels_by_sample)
    for _ in range(replicates):
        selected = rng.integers(count, size=count)
        labels = np.concatenate([labels_by_sample[index] for index in selected])
        if np.unique(labels).size < 2:
            continue
        real = np.concatenate([real_by_sample[index] for index in selected])
        null = np.concatenate([null_by_sample[index] for index in selected])
        real_metrics = _metrics(labels, real)
        null_metrics = _metrics(labels, null)
        delta.append(
            (
                real_metrics["auroc"] - null_metrics["auroc"],
                real_metrics["auprc"] - null_metrics["auprc"],
            )
        )
    delta = np.asarray(delta)
    return {
        "auroc_drop_ci_low": float(np.quantile(delta[:, 0], 0.025)),
        "auroc_drop_ci_high": float(np.quantile(delta[:, 0], 0.975)),
        "auprc_drop_ci_low": float(np.quantile(delta[:, 1], 0.025)),
        "auprc_drop_ci_high": float(np.quantile(delta[:, 1], 0.975)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _first_positive_run(labels: np.ndarray) -> tuple[int, int] | None:
    positive = np.flatnonzero(labels == 1)
    if not len(positive):
        return None
    start = int(positive[0])
    end = start + 1
    while end < len(labels) and labels[end] == 1:
        end += 1
    return start, end


def _paired_effect(values: list[float]) -> tuple[int, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    dz = mean / standard_deviation if standard_deviation > 0 else 0.0
    return len(array), mean, dz


def evaluate_scores(
    *,
    split_root,
    score_dir,
    output_dir,
    onset_window: int = 4,
    bootstrap_replicates: int = 500,
    seed: int = 20260819,
) -> None:
    """Open labels after fit/score artifacts have been frozen."""

    score_dir = Path(score_dir)
    manifest = read_json(score_dir / "manifest.json")
    dataset = open_research_dataset(split_root, device="cpu")
    labels_store = dataset.prepare_evaluation_labels()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_sample = []
    real_family = {name: [] for name in FAMILY_NAMES}
    null_family = {name: [] for name in FAMILY_NAMES}
    real_family_layer = {name: [] for name in FAMILY_NAMES}
    null_family_layer = {name: [] for name in FAMILY_NAMES}
    all_labels = []
    all_raw_features = []
    all_standardized = []
    all_null_standardized = []
    onset_effects: dict[tuple[str, int], list[float]] = {}
    lockin_effects: dict[tuple[str, int], list[float]] = {}
    phase_scores: dict[tuple[str, int], list[float]] = {}
    role_errors = []
    changed_fractions = []

    for row in manifest["samples"]:
        sample = dataset[row["sample_id"]]
        labels = labels_store.response_labels(sample).cpu().numpy().astype(np.int8)
        sample.release_attention()
        arrays = load_npz(score_dir / row["score_path"])

        raw = arrays["layer_features"].astype(np.float32)
        standardized = arrays["standardized_features"].astype(np.float32)
        family = arrays["family_scores"].astype(np.float32)
        family_layer = arrays["family_layer_scores"].astype(np.float32)
        null_standardized = arrays.get("rewired_standardized_features")
        null_score = arrays.get("rewired_family_scores")
        null_layer_score = arrays.get("rewired_family_layer_scores")

        labels_by_sample.append(labels)
        all_labels.append(labels)
        all_raw_features.append(raw)
        all_standardized.append(standardized)
        if null_standardized is not None:
            all_null_standardized.append(null_standardized.astype(np.float32))
            role_errors.append(float(arrays["rewire_role_max_abs_error"].item()))
            changed_fractions.append(float(arrays["rewire_changed_fraction"].item()))

        for index, name in enumerate(FAMILY_NAMES):
            real_family[name].append(family[:, index])
            real_family_layer[name].append(family_layer[:, :, index])
            if null_score is not None:
                null_family[name].append(null_score[:, index].astype(np.float32))
                null_family_layer[name].append(
                    null_layer_score[:, :, index].astype(np.float32)
                )

        run = _first_positive_run(labels)
        if run is None:
            continue
        start, end = run
        pre_start = max(0, start - onset_window)
        onset_end = min(end, start + onset_window)
        if pre_start == start:
            continue

        pre = standardized[pre_start:start].mean(axis=0)
        onset = standardized[start:onset_end].mean(axis=0)
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            for layer in range(standardized.shape[1]):
                onset_effects.setdefault((feature_name, layer), []).append(
                    float(onset[layer, feature_index] - pre[layer, feature_index])
                )

        if end - start > onset_window:
            late_start = max(start + onset_window, end - onset_window)
            late = standardized[late_start:end].mean(axis=0)
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                for layer in range(standardized.shape[1]):
                    lockin_effects.setdefault((feature_name, layer), []).append(
                        float(late[layer, feature_index] - onset[layer, feature_index])
                    )

        for offset in range(-onset_window, onset_window + 1):
            token = start + offset
            if 0 <= token < len(labels):
                for family_index, name in enumerate(FAMILY_NAMES):
                    phase_scores.setdefault((name, offset), []).append(
                        float(family[token, family_index])
                    )

    labels = np.concatenate(all_labels)
    raw_features = np.concatenate(all_raw_features)
    standardized = np.concatenate(all_standardized)
    null_standardized = (
        np.concatenate(all_null_standardized) if all_null_standardized else None
    )

    family_rows = []
    for family in FAMILY_NAMES:
        score = np.concatenate(real_family[family])
        row = {"family": family, **_metrics(labels, score)}
        row.update(
            _sample_bootstrap(
                labels_by_sample,
                real_family[family],
                replicates=bootstrap_replicates,
                seed=seed,
            )
        )
        if null_family[family]:
            null_score = np.concatenate(null_family[family])
            null_metrics = _metrics(labels, null_score)
            row.update(
                rewired_auroc=null_metrics["auroc"],
                rewired_auprc=null_metrics["auprc"],
                auroc_drop=row["auroc"] - null_metrics["auroc"],
                auprc_drop=row["auprc"] - null_metrics["auprc"],
            )
            row.update(
                _paired_bootstrap_delta(
                    labels_by_sample,
                    real_family[family],
                    null_family[family],
                    replicates=bootstrap_replicates,
                    seed=seed,
                )
            )
        family_rows.append(row)
    _write_csv(output_dir / "family_metrics.csv", family_rows)

    family_layer_rows = []
    for family in FAMILY_NAMES:
        real = np.concatenate(real_family_layer[family])
        null = (
            np.concatenate(null_family_layer[family])
            if null_family_layer[family]
            else None
        )
        for layer in range(real.shape[1]):
            row = {"family": family, "layer": layer, **_metrics(labels, real[:, layer])}
            if null is not None:
                null_metrics = _metrics(labels, null[:, layer])
                row.update(
                    rewired_auroc=null_metrics["auroc"],
                    rewired_auprc=null_metrics["auprc"],
                    auroc_drop=row["auroc"] - null_metrics["auroc"],
                    auprc_drop=row["auprc"] - null_metrics["auprc"],
                )
            family_layer_rows.append(row)
    _write_csv(output_dir / "family_layer_metrics.csv", family_layer_rows)

    feature_rows = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        for layer in range(raw_features.shape[1]):
            anomaly = np.abs(standardized[:, layer, feature_index])
            raw_auc = _metrics(labels, raw_features[:, layer, feature_index])["auroc"]
            row = {
                "feature": feature_name,
                "layer": layer,
                "anomaly_auroc": _metrics(labels, anomaly)["auroc"],
                "anomaly_auprc": _metrics(labels, anomaly)["auprc"],
                "raw_auroc": raw_auc,
                "raw_separability": max(raw_auc, 1.0 - raw_auc),
            }
            if null_standardized is not None:
                null_anomaly = np.abs(null_standardized[:, layer, feature_index])
                null_auc = _metrics(labels, null_anomaly)["auroc"]
                row.update(
                    rewired_anomaly_auroc=null_auc,
                    real_minus_rewired_auroc=row["anomaly_auroc"] - null_auc,
                )
            feature_rows.append(row)
    _write_csv(output_dir / "layer_feature_metrics.csv", feature_rows)

    onset_rows = []
    for (feature_name, layer), values in onset_effects.items():
        responses, mean, dz = _paired_effect(values)
        direction = ONSET_DIRECTIONS.get(feature_name, 0)
        onset_rows.append(
            {
                "feature": feature_name,
                "layer": layer,
                "responses": responses,
                "standardized_onset_minus_pre": mean,
                "paired_dz": dz,
                "expected_direction": direction,
                "direction_supported": None
                if direction == 0
                else bool(mean * direction > 0),
            }
        )
    _write_csv(output_dir / "onset_layer_effects.csv", onset_rows)

    lockin_rows = []
    for (feature_name, layer), values in lockin_effects.items():
        responses, mean, dz = _paired_effect(values)
        direction = LOCKIN_DIRECTIONS.get(feature_name, 0)
        lockin_rows.append(
            {
                "feature": feature_name,
                "layer": layer,
                "responses": responses,
                "standardized_late_minus_onset": mean,
                "paired_dz": dz,
                "expected_direction": direction,
                "direction_supported": None
                if direction == 0
                else bool(mean * direction > 0),
            }
        )
    _write_csv(output_dir / "lockin_layer_effects.csv", lockin_rows)

    phase_rows = [
        {
            "family": family,
            "offset": offset,
            "tokens": len(values),
            "mean_score": float(np.mean(values)),
            "standard_error": float(np.std(values) / np.sqrt(len(values))),
        }
        for (family, offset), values in sorted(phase_scores.items())
    ]
    _write_csv(output_dir / "onset_phase_curves.csv", phase_rows)

    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "tokens": len(labels),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "samples": len(labels_by_sample),
            "family_metrics": family_rows,
            "rewire_role_max_abs_error": max(role_errors) if role_errors else None,
            "rewire_changed_fraction_mean": float(np.mean(changed_fractions))
            if changed_fractions
            else None,
            "hypothesis_outputs": {
                "routing_detection": "family_metrics.csv",
                "routing_fracture": "onset_layer_effects.csv",
                "integration_failure": "onset_layer_effects.csv",
                "fracture_to_lockin": "lockin_layer_effects.csv",
                "endpoint_topology": "real/rewired deltas in family_metrics.csv",
            },
        },
    )
