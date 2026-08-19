"""Post-hoc label evaluation for pre-frozen attention phenomenology fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .config import (
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


def _cluster_bootstrap(
    sample_labels: list[np.ndarray],
    sample_scores: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    auroc = []
    auprc = []
    samples = len(sample_labels)
    for _ in range(int(replicates)):
        selected = rng.integers(samples, size=samples)
        labels = np.concatenate([sample_labels[index] for index in selected])
        score = np.concatenate([sample_scores[index] for index in selected])
        if np.unique(labels).size < 2:
            continue
        current = _metrics(labels, score)
        auroc.append(current["auroc"])
        auprc.append(current["auprc"])
    if not auroc:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
        }
    return {
        "auroc_ci_low": float(np.quantile(auroc, 0.025)),
        "auroc_ci_high": float(np.quantile(auroc, 0.975)),
        "auprc_ci_low": float(np.quantile(auprc, 0.025)),
        "auprc_ci_high": float(np.quantile(auprc, 0.975)),
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
    if len(positive) == 0:
        return None
    start = int(positive[0])
    end = start + 1
    while end < len(labels) and labels[end] == 1:
        end += 1
    return start, end


def _paired_dz(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    dz = mean / std if std > 0 else 0.0
    return mean, dz


def evaluate_scores(
    *,
    split_root,
    score_dir,
    output_dir,
    onset_window: int = 4,
    bootstrap_replicates: int = 500,
    seed: int = 20260819,
) -> None:
    """Open labels only after all score artifacts have been frozen."""

    score_dir = Path(score_dir)
    manifest = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = open_research_dataset(split_root, device="cpu")
    labels_store = dataset.prepare_evaluation_labels()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_labels: list[np.ndarray] = []
    family_by_sample: dict[str, list[np.ndarray]] = {name: [] for name in FAMILY_NAMES}
    rewired_family_by_sample: dict[str, list[np.ndarray]] = {
        name: [] for name in FAMILY_NAMES
    }
    all_features = []
    all_standardized = []
    all_rewired_standardized = []
    rewire_role_errors = []
    all_labels = []
    onset_differences: dict[tuple[str, int], list[float]] = {}
    lockin_differences: dict[tuple[str, int], list[float]] = {}
    phase_family: dict[tuple[str, int], list[float]] = {}

    for row in manifest["samples"]:
        sample = dataset[row["sample_id"]]
        labels = labels_store.response_labels(sample).cpu().numpy().astype(np.int8)
        with np.load(score_dir / row["score_path"], allow_pickle=False) as arrays:
            features = arrays["layer_features"].astype(np.float32)
            standardized = arrays["standardized_features"].astype(np.float32)
            family_scores = arrays["family_scores"].astype(np.float32)
            rewired_standardized = (
                arrays["rewired_standardized_features"].astype(np.float32)
                if "rewired_standardized_features" in arrays
                else None
            )
            rewired_scores = (
                arrays["rewired_family_scores"].astype(np.float32)
                if "rewired_family_scores" in arrays
                else None
            )
            if "rewire_role_max_abs_error" in arrays:
                rewire_role_errors.append(float(arrays["rewire_role_max_abs_error"].item()))

        sample_labels.append(labels)
        all_labels.append(labels)
        all_features.append(features)
        all_standardized.append(standardized)
        if rewired_standardized is not None:
            all_rewired_standardized.append(rewired_standardized)
        for family_index, family in enumerate(FAMILY_NAMES):
            family_by_sample[family].append(family_scores[:, family_index])
            if rewired_scores is not None:
                rewired_family_by_sample[family].append(rewired_scores[:, family_index])

        run = _first_positive_run(labels)
        if run is None:
            continue
        start, end = run
        pre_start = max(0, start - int(onset_window))
        onset_end = min(end, start + int(onset_window))
        if pre_start == start or onset_end == start:
            continue
        pre = features[pre_start:start].mean(axis=0)
        onset = features[start:onset_end].mean(axis=0)
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            for layer in range(features.shape[1]):
                onset_differences.setdefault((feature_name, layer), []).append(
                    float(onset[layer, feature_index] - pre[layer, feature_index])
                )

        if end - start > onset_window:
            late_start = max(start + onset_window, end - onset_window)
            late = features[late_start:end].mean(axis=0)
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                for layer in range(features.shape[1]):
                    lockin_differences.setdefault((feature_name, layer), []).append(
                        float(late[layer, feature_index] - onset[layer, feature_index])
                    )

        for offset in range(-int(onset_window), int(onset_window) + 1):
            token = start + offset
            if not 0 <= token < len(labels):
                continue
            for family_index, family in enumerate(FAMILY_NAMES):
                phase_family.setdefault((family, offset), []).append(
                    float(family_scores[token, family_index])
                )

    labels = np.concatenate(all_labels)
    features = np.concatenate(all_features)
    standardized = np.concatenate(all_standardized)
    rewired_standardized = (
        np.concatenate(all_rewired_standardized)
        if all_rewired_standardized
        else None
    )

    family_rows = []
    for family in FAMILY_NAMES:
        score = np.concatenate(family_by_sample[family])
        current = _metrics(labels, score)
        current.update(
            _cluster_bootstrap(
                sample_labels,
                family_by_sample[family],
                replicates=bootstrap_replicates,
                seed=seed,
            )
        )
        row = {"family": family, **current}
        if rewired_family_by_sample[family]:
            rewired_score = np.concatenate(rewired_family_by_sample[family])
            rewired = _metrics(labels, rewired_score)
            row.update(
                rewired_auroc=rewired["auroc"],
                rewired_auprc=rewired["auprc"],
                auroc_drop=current["auroc"] - rewired["auroc"],
                auprc_drop=current["auprc"] - rewired["auprc"],
            )
        family_rows.append(row)
    _write_csv(output_dir / "family_metrics.csv", family_rows)

    feature_rows = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        for layer in range(features.shape[1]):
            anomaly = np.abs(standardized[:, layer, feature_index])
            current = _metrics(labels, anomaly)
            raw_auc = _metrics(labels, features[:, layer, feature_index])["auroc"]
            row = {
                "feature": feature_name,
                "layer": layer,
                "anomaly_auroc": current["auroc"],
                "anomaly_auprc": current["auprc"],
                "raw_auroc": raw_auc,
                "raw_separability": max(raw_auc, 1.0 - raw_auc),
            }
            if rewired_standardized is not None:
                rewired_anomaly = np.abs(
                    rewired_standardized[:, layer, feature_index]
                )
                row["rewired_anomaly_auroc"] = _metrics(
                    labels, rewired_anomaly
                )["auroc"]
                row["real_minus_rewired_auroc"] = (
                    row["anomaly_auroc"] - row["rewired_anomaly_auroc"]
                )
            feature_rows.append(row)
    _write_csv(output_dir / "layer_feature_metrics.csv", feature_rows)

    onset_rows = []
    for (feature_name, layer), values in onset_differences.items():
        values_array = np.asarray(values, dtype=np.float64)
        mean, dz = _paired_dz(values_array)
        direction = int(ONSET_DIRECTIONS.get(feature_name, 0))
        onset_rows.append(
            {
                "feature": feature_name,
                "layer": layer,
                "responses": len(values_array),
                "onset_minus_pre": mean,
                "paired_dz": dz,
                "expected_direction": direction,
                "direction_supported": bool(direction and mean * direction > 0),
            }
        )
    _write_csv(output_dir / "onset_layer_effects.csv", onset_rows)

    lockin_rows = []
    for (feature_name, layer), values in lockin_differences.items():
        values_array = np.asarray(values, dtype=np.float64)
        mean, dz = _paired_dz(values_array)
        direction = int(LOCKIN_DIRECTIONS.get(feature_name, 0))
        lockin_rows.append(
            {
                "feature": feature_name,
                "layer": layer,
                "responses": len(values_array),
                "late_minus_onset": mean,
                "paired_dz": dz,
                "expected_direction": direction,
                "direction_supported": bool(direction and mean * direction > 0),
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
        for (family, offset), values in sorted(phase_family.items())
    ]
    _write_csv(output_dir / "onset_phase_curves.csv", phase_rows)

    summary = {
        "schema": "attention-phenomenology-evaluation-v1",
        "labels_read": True,
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "samples": len(sample_labels),
        "family_metrics": family_rows,
        "rewire_role_max_abs_error": (
            max(rewire_role_errors) if rewire_role_errors else None
        ),
        "hypothesis_tests": {
            "routing_detection": "family_metrics.csv",
            "routing_fracture": "onset_layer_effects.csv",
            "integration_failure": "onset_layer_effects.csv",
            "fracture_to_lockin": "lockin_layer_effects.csv",
            "endpoint_topology": "real versus rewired columns in metrics CSVs",
        },
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
