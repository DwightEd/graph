"""Runnable fit, score, and post-hoc evaluation workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiment_protocol import (
    FrozenEvaluation,
    FrozenFile,
    dataset_manifest_sha256,
    file_sha256,
)

from .artifacts import (
    load_reference,
    load_scores,
    save_reference,
    save_scores,
)
from .detector import COMPONENT_NAMES, DetectorConfig, TokenRoutingDetector
from .routing import RoutingFeatureConfig


def fit_reference(
    dataset,
    output,
    *,
    detector_config: DetectorConfig | None = None,
    feature_config: RoutingFeatureConfig | None = None,
    limit: int | None = None,
) -> dict:
    manifest_digest = dataset_manifest_sha256(dataset)
    detector = TokenRoutingDetector(
        detector_config, feature_config=feature_config
    ).fit(dataset, limit=limit)
    if dataset_manifest_sha256(dataset) != manifest_digest:
        raise ValueError("training dataset manifest changed during fitting")
    path = save_reference(
        detector, output, train_manifest_sha256=manifest_digest
    )
    return {
        "reference": str(path),
        "reference_sha256": file_sha256(path),
        "labels_read": False,
        "fit_source_groups": len(detector.fit_source_ids),
        "calibration_source_groups": len(detector.calibration_source_ids),
        "features": len(detector.feature_names),
        "alignment": detector.alignment,
        "online_causal_score": True,
    }


def score_dataset(
    dataset,
    reference,
    output,
    *,
    limit: int | None = None,
) -> dict:
    frozen_reference = FrozenFile.capture(reference)
    detector = load_reference(frozen_reference.path)
    manifest_digest = dataset_manifest_sha256(dataset)
    table = detector.score(dataset, limit=limit)
    frozen_reference.verify(reference)
    if dataset_manifest_sha256(dataset) != manifest_digest:
        raise ValueError("score dataset manifest changed during scoring")
    test_sample_ids = tuple(dict.fromkeys(map(str, table.sample_id)))
    test_source_ids = tuple(sorted(set(map(str, table.source_id))))
    audit_scope = "selected_samples" if limit is not None else "complete_split"
    path = save_scores(
        table,
        output,
        dataset_manifest_sha256=manifest_digest,
        reference_sha256=frozen_reference.sha256,
        audit_scope=audit_scope,
        reserved_source_ids=detector.train_source_ids,
        test_source_ids=test_source_ids,
        test_sample_ids=test_sample_ids,
    )
    return {
        "scores": str(path),
        "scores_sha256": file_sha256(path),
        "labels_read": False,
        "samples": len(set(map(str, table.sample_id))),
        "tokens": len(table.score),
        "valid_tokens": int(table.valid.sum()),
        "invalid_tokens": int((~table.valid).sum()),
        "threshold": table.threshold,
        "alignment": table.alignment,
        "online_causal_score": table.online_causal_score,
    }


def evaluate_scores(dataset, scores, output_dir) -> dict:
    output = _new_directory(output_dir)
    frozen = FrozenEvaluation.capture(
        scores, expected_split=str(dataset.manifest.get("split"))
    )
    rows = load_scores(scores)
    aligned = frozen.align_loaded(dataset, rows)
    valid = np.asarray(rows["valid"], dtype=bool)
    all_labels = np.asarray(aligned.token_label, dtype=np.int8)
    all_scores = np.asarray(rows["score"], dtype=np.float32)
    labels = all_labels[valid]
    primary = all_scores[valid]
    threshold = float(np.asarray(rows["threshold"]).item())

    report = {
        "schema": "token-routing-basin-evaluation-v2",
        "analysis_status": "posthoc_labels_after_frozen_scores",
        "primary_detector": "calibrated_routing_basin_max",
        "labels_read": True,
        "online_causal_score": True,
        "alignment": _scalar_text(rows, "alignment"),
        "tokens": int(len(valid)),
        "valid_tokens": int(valid.sum()),
        "invalid_tokens": int((~valid).sum()),
        "positive_tokens": int(labels.sum()),
        "positive_tokens_total": int(all_labels.sum()),
        "coverage": _coverage_summary(all_labels, valid),
        "coverage_sensitivity": _coverage_sensitivity(
            all_labels, all_scores, valid
        ),
        "threshold": threshold,
        "primary": _ranking_metrics(labels, primary, threshold=threshold),
        "components": {},
        "features": {},
        "controls": {},
        "by_task": {},
        "by_data_source": {},
        "forecast": _forecast_summary(
            all_labels,
            all_scores,
            valid,
            np.asarray(rows["sample_id"]).astype(str),
            np.asarray(rows["token_index"], dtype=np.int64),
        ),
        "onset": _onset_summary(
            all_labels,
            all_scores,
            valid,
            np.asarray(rows["sample_id"]).astype(str),
            np.asarray(rows["token_index"], dtype=np.int64),
            threshold,
        ),
    }
    for name in COMPONENT_NAMES:
        values = np.asarray(rows[f"component_score/{name}"], dtype=np.float32)[
            valid
        ]
        report["components"][name] = _ranking_metrics(labels, values)
    feature_names = tuple(map(str, np.asarray(rows["feature_names"])))
    features = np.asarray(rows["features"], dtype=np.float32)[valid]
    for column, name in enumerate(feature_names):
        report["features"][name] = _ranking_metrics(labels, features[:, column])
    control_names = tuple(map(str, np.asarray(rows["control_names"])))
    controls = np.asarray(rows["controls"], dtype=np.float32)[valid]
    for column, name in enumerate(control_names):
        report["controls"][name] = _ranking_metrics(labels, controls[:, column])
    for field, destination in (
        ("task_type", "by_task"),
        ("data_source", "by_data_source"),
    ):
        groups = np.asarray(rows[field]).astype(str)[valid]
        for group in dict.fromkeys(groups.tolist()):
            selected = groups == group
            report[destination][group] = {
                "tokens": int(selected.sum()),
                "positive_tokens": int(labels[selected].sum()),
                **_ranking_metrics(labels[selected], primary[selected]),
            }

    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _ranking_metrics(labels, scores, *, threshold=None):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    available = np.isfinite(scores)
    unavailable = int((~available).sum())
    labels = labels[available]
    scores = scores[available]
    result = {
        "tokens": int(len(scores)),
        "unavailable_tokens": unavailable,
        "auroc": None,
        "auprc": None,
        "prevalence": float(labels.mean()) if len(labels) else None,
        "normal_median": float(np.median(scores[labels == 0]))
        if np.any(labels == 0)
        else None,
        "hallucination_median": float(np.median(scores[labels == 1]))
        if np.any(labels == 1)
        else None,
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["auprc"] = float(average_precision_score(labels, scores))
    if threshold is not None:
        prediction = scores >= float(threshold)
        result.update(
            {
                "precision": float(
                    precision_score(labels, prediction, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, prediction, zero_division=0)
                ),
                "f1": float(f1_score(labels, prediction, zero_division=0)),
                "predicted_positive_tokens": int(prediction.sum()),
                "false_alerts_per_1000_normal_tokens": float(
                    1000 * np.sum(prediction & (labels == 0)) / max(1, np.sum(labels == 0))
                ),
            }
        )
    return result


def _coverage_summary(labels, valid):
    labels = np.asarray(labels, dtype=np.int8)
    valid = np.asarray(valid, dtype=bool)
    positive = labels == 1
    negative = labels == 0
    return {
        "overall": float(valid.mean()) if len(valid) else None,
        "hallucination": float(valid[positive].mean()) if positive.any() else None,
        "normal": float(valid[negative].mean()) if negative.any() else None,
        "valid_hallucination_tokens": int(np.sum(valid & positive)),
        "invalid_hallucination_tokens": int(np.sum(~valid & positive)),
        "valid_normal_tokens": int(np.sum(valid & negative)),
        "invalid_normal_tokens": int(np.sum(~valid & negative)),
    }


def _coverage_sensitivity(labels, scores, valid):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return {"best_case": None, "worst_case": None}
    low = float(np.nanmin(scores[valid]) - 1.0)
    high = float(np.nanmax(scores[valid]) + 1.0)
    best = scores.copy()
    worst = scores.copy()
    best[~valid & (labels == 1)] = high
    best[~valid & (labels == 0)] = low
    worst[~valid & (labels == 1)] = low
    worst[~valid & (labels == 0)] = high
    return {
        "best_case": _ranking_metrics(labels, best),
        "worst_case": _ranking_metrics(labels, worst),
    }


def _forecast_summary(labels, scores, valid, sample_ids, token_indices):
    report = {}
    for horizon in (1, 2, 4):
        future_labels = []
        current_scores = []
        for sample_id in dict.fromkeys(sample_ids.tolist()):
            selected = sample_ids == sample_id
            order = np.argsort(token_indices[selected])
            sample_labels = labels[selected][order]
            sample_scores = scores[selected][order]
            sample_valid = valid[selected][order]
            if len(sample_labels) <= horizon:
                continue
            eligible = sample_valid[:-horizon]
            future_labels.extend(sample_labels[horizon:][eligible].tolist())
            current_scores.extend(sample_scores[:-horizon][eligible].tolist())
        report[str(horizon)] = {
            "tokens": len(future_labels),
            **_ranking_metrics(future_labels, current_scores),
        }
    return report


def _onset_summary(labels, scores, valid, sample_ids, token_indices, threshold):
    deltas = []
    delays = []
    total_responses = 0
    eligible_onsets = 0
    early = {1: [0, 0], 2: [0, 0], 4: [0, 0]}
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        selected = sample_ids == sample_id
        order = np.argsort(token_indices[selected])
        sample_labels = labels[selected][order]
        sample_scores = scores[selected][order]
        sample_valid = valid[selected][order]
        positive = np.flatnonzero(sample_labels == 1)
        if not len(positive):
            continue
        total_responses += 1
        onset = int(positive[0])
        if sample_valid[onset]:
            eligible_onsets += 1
            end = onset + 1
            while end < len(sample_labels) and sample_labels[end] == 1:
                end += 1
            detections = np.flatnonzero(
                sample_valid[onset:end] & (sample_scores[onset:end] >= threshold)
            )
            if len(detections):
                delays.append(int(detections[0]))
        if onset >= 1 and sample_valid[onset] and sample_valid[onset - 1]:
            deltas.append(float(sample_scores[onset] - sample_scores[onset - 1]))
        for horizon in early:
            token = onset - horizon
            if token < 0 or not sample_valid[token]:
                continue
            early[horizon][1] += 1
            early[horizon][0] += int(sample_scores[token] >= threshold)
    return {
        "responses_with_hallucination": total_responses,
        "eligible_onsets": eligible_onsets,
        "onset_delta_responses": len(deltas),
        "mean_onset_delta": float(np.mean(deltas)) if deltas else None,
        "median_onset_delta": float(np.median(deltas)) if deltas else None,
        "detected_spans": len(delays),
        "median_detection_delay": float(np.median(delays)) if delays else None,
        "pre_onset_alert": {
            str(horizon): {
                "alerts": counts[0],
                "eligible": counts[1],
                "recall": counts[0] / counts[1] if counts[1] else None,
            }
            for horizon, counts in early.items()
        },
    }


def _new_directory(path):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _scalar_text(mapping, name):
    value = np.asarray(mapping[name])
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)
