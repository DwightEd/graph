"""Train-calibrated, label-frozen evaluation of re-anchor failure scores."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import FrozenEvaluation, file_sha256
from experiments.common.ragtruth_alignment import TASK_TYPES, canonical_task_type

from .artifacts import CAPTURE_SCHEMA, load_result, save_result
from .detector import (
    DETECTOR_SCHEMA,
    RAW_FEATURES,
    SCORE_NAMES,
    ReanchorFailureDetector,
    raw_features,
)


SCORE_FIELDS = {
    **{name: f"score_{name}" for name in SCORE_NAMES},
    "context_opposition": "tail_context_opposition",
    "context_js": "tail_context_distribution_js",
    "route_demand": "tail_route_demand",
    "evidence_entry_deficit": "tail_evidence_entry_deficit",
    "evidence_reentry_strength": "tail_evidence_reentry_strength",
    "history_dominance": "tail_history_dominance",
    "adoption_deficit": "tail_adoption_deficit",
    "confidence_surprisal": "raw_confidence_surprisal",
    "relative_position": "raw_relative_position",
}
TOKEN_PRIMARY_SCORE = "online_failure"
ONSET_PRIMARY_SCORE = "onset_trigger"
DETECTION_THRESHOLD = 0.95


@dataclass(frozen=True)
class FeatureRecord:
    sample_id: str
    source_id: str
    task_type: str
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class _MetricOrder:
    label: np.ndarray
    source_index: np.ndarray
    group_start: np.ndarray

    @classmethod
    def build(
        cls,
        label: np.ndarray,
        score: np.ndarray,
        source_index: np.ndarray,
    ) -> "_MetricOrder":
        finite = np.isfinite(score)
        score = np.asarray(score, dtype=np.float64)[finite]
        order = np.argsort(score, kind="stable")
        ordered_score = score[order]
        start = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
        return cls(
            np.asarray(label, dtype=bool)[finite][order],
            np.asarray(source_index, dtype=np.int64)[finite][order],
            start,
        )

    def metric(self, source_weight: np.ndarray) -> tuple[float, float] | None:
        weight = source_weight[self.source_index]
        positive = np.add.reduceat(weight * self.label, self.group_start)
        negative = np.add.reduceat(weight * ~self.label, self.group_start)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive == 0 or total_negative == 0:
            return None
        negative_before = np.cumsum(negative) - negative
        auroc = np.sum(positive * (negative_before + 0.5 * negative)) / (
            total_positive * total_negative
        )
        positive_descending = positive[::-1]
        negative_descending = negative[::-1]
        cumulative_positive = np.cumsum(positive_descending)
        cumulative_total = cumulative_positive + np.cumsum(negative_descending)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive, dtype=np.float64),
            where=cumulative_total > 0,
        )
        auprc = np.sum(precision * positive_descending) / total_positive
        return float(auroc), float(auprc)


def _load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if not manifest.get("analysis_complete"):
        raise ValueError(f"capture is incomplete: {root}")
    if config.get("capture_schema") != CAPTURE_SCHEMA:
        raise ValueError(f"detection requires schema v{CAPTURE_SCHEMA}: {root}")
    if config.get("max_events") is not None:
        raise ValueError("detection requires complete response-token captures")
    return manifest


def _check_capture_pair(train: Mapping, test: Mapping) -> None:
    keys = (
        "capture_schema",
        "model",
        "model_id",
        "dtype",
        "source_info",
        "route_window",
        "future_horizon",
        "distance_scale",
        "functional_pass",
    )
    if any(train["config"].get(key) != test["config"].get(key) for key in keys):
        raise ValueError("train and test captures use different detector settings")


def _feature_records(root: Path, manifest: Mapping) -> list[FeatureRecord]:
    route_window = int(manifest["config"]["route_window"])
    records = []
    for entry in manifest["samples"]:
        result = load_result(root / entry["result"])
        schema = int(np.asarray(result["capture_schema"]).item())
        functional = bool(int(np.asarray(result["functional"]).item()))
        if schema != CAPTURE_SCHEMA or not functional:
            raise ValueError("detector input lacks the schema-v8 functional pass")
        task = canonical_task_type(np.asarray(result["task_type"]).item())
        records.append(
            FeatureRecord(
                sample_id=str(np.asarray(result["sample_id"]).item()),
                source_id=str(np.asarray(result["source_id"]).item()),
                task_type=task,
                features=raw_features(result, route_window),
            )
        )
    return records


def _fit_detectors(
    train: list[FeatureRecord], test: list[FeatureRecord]
) -> tuple[dict[str, ReanchorFailureDetector], dict]:
    test_sources = {record.source_id for record in test}
    calibration = [record for record in train if record.source_id not in test_sources]
    detectors = {}
    summary = {}
    for task in TASK_TYPES:
        selected = [record for record in calibration if record.task_type == task]
        if not selected:
            raise ValueError(f"no source-disjoint train calibration rows for {task}")
        detectors[task] = ReanchorFailureDetector.fit(
            [record.features for record in selected],
            [record.source_id for record in selected],
        )
        excluded = [
            record
            for record in train
            if record.task_type == task and record.source_id in test_sources
        ]
        summary[task] = {
            "samples": len(selected),
            "sources": len({record.source_id for record in selected}),
            "tokens": sum(len(record.features["relative_position"]) for record in selected),
            "overlap_samples_excluded": len(excluded),
            "overlap_sources_excluded": len({record.source_id for record in excluded}),
        }
    return detectors, summary


def _concatenate(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values) for name, values in parts.items()}


def _score_test(
    records: list[FeatureRecord],
    detectors: Mapping[str, ReanchorFailureDetector],
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "token_index": [],
        "response_length": [],
    }
    for name in (
        "relative_position",
        "baseline_entropy",
        "baseline_target_logprob",
        "confidence_surprisal",
        *RAW_FEATURES,
    ):
        parts[f"raw_{name}"] = []
    for name in RAW_FEATURES:
        parts[f"tail_{name}"] = []
    for name in SCORE_NAMES:
        parts[f"score_{name}"] = []

    for record in records:
        scored = detectors[record.task_type].score(record.features)
        count = len(record.features["relative_position"])
        parts["sample_id"].append(np.repeat(record.sample_id, count))
        parts["source_id"].append(np.repeat(record.source_id, count))
        parts["task_type"].append(np.repeat(record.task_type, count))
        parts["token_index"].append(np.arange(count, dtype=np.int32))
        parts["response_length"].append(np.full(count, count, dtype=np.int32))
        for name, value in scored.raw.items():
            parts[f"raw_{name}"].append(np.asarray(value, dtype=np.float32))
        for name, value in scored.tail.items():
            parts[f"tail_{name}"].append(np.asarray(value, dtype=np.float32))
        for name, value in scored.score.items():
            parts[f"score_{name}"].append(np.asarray(value, dtype=np.float32))
    return _concatenate(parts)


def _metric(label: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    label = np.asarray(label, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    selected = np.isfinite(score)
    label, score = label[selected], score[selected]
    prevalence = float(label.mean()) if len(label) else None
    if not len(label) or np.unique(label).size < 2:
        return {
            "tokens": len(label),
            "positives": int(label.sum()),
            "prevalence": prevalence,
            "auroc": None,
            "auprc": None,
            "auprc_lift": None,
        }
    auprc = float(average_precision_score(label, score))
    return {
        "tokens": len(label),
        "positives": int(label.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(label, score)),
        "auprc": auprc,
        "auprc_lift": auprc / prevalence if prevalence else None,
    }


def _threshold_metric(label: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    selected = np.isfinite(score)
    label, predicted = label[selected], score[selected] >= DETECTION_THRESHOLD
    true_positive = int(np.count_nonzero(label & predicted))
    predicted_positive = int(predicted.sum())
    positive = int(label.sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / positive if positive else 0.0
    return {
        "threshold": DETECTION_THRESHOLD,
        "predicted_positive": predicted_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _onset_labels(
    label: np.ndarray, sample_id: np.ndarray, token_index: np.ndarray
) -> np.ndarray:
    label = np.asarray(label, dtype=bool)
    sample_id = np.asarray(sample_id).astype(str, copy=False)
    token_index = np.asarray(token_index, dtype=np.int64)
    onset = np.zeros(len(label), dtype=bool)
    for sample in dict.fromkeys(sample_id.tolist()):
        selected = np.flatnonzero(sample_id == sample)
        order = selected[np.argsort(token_index[selected])]
        current = label[order]
        onset[order] = current & ~np.r_[False, current[:-1]]
    return onset


def _source_bootstrap(
    label: np.ndarray,
    scores: Mapping[str, np.ndarray],
    sources: np.ndarray,
    repeats: int,
    seed: int,
    primary_score: str,
) -> dict:
    if repeats <= 0:
        return {"replicates_valid": 0}
    sources = np.asarray(sources).astype(str, copy=False)
    groups, source_index = np.unique(sources, return_inverse=True)
    orders = {
        name: _MetricOrder.build(label, score, source_index)
        for name, score in scores.items()
    }
    random = np.random.default_rng(seed)
    estimates = {name: [] for name in scores}
    for _ in range(repeats):
        source_weight = random.multinomial(
            len(groups), np.full(len(groups), 1.0 / len(groups))
        )
        result = {name: order.metric(source_weight) for name, order in orders.items()}
        if any(value is None for value in result.values()):
            continue
        for name, value in result.items():
            estimates[name].append(value)
    valid = min((len(value) for value in estimates.values()), default=0)
    if not valid:
        return {"replicates_valid": 0}
    report = {"replicates_valid": valid, "scores": {}, "primary_deltas": {}}
    for name, values in estimates.items():
        array = np.asarray(values, dtype=np.float64)
        report["scores"][name] = {
            "auroc_ci95": np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
            "auprc_ci95": np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
        }
    primary = np.asarray(estimates[primary_score], dtype=np.float64)
    for name in scores:
        if name == primary_score:
            continue
        comparison = np.asarray(estimates[name], dtype=np.float64)
        delta = primary - comparison
        report["primary_deltas"][name] = {
            "auroc_ci95": np.quantile(delta[:, 0], [0.025, 0.975]).tolist(),
            "auprc_ci95": np.quantile(delta[:, 1], [0.025, 0.975]).tolist(),
        }
    return report


def _scope_report(
    rows: Mapping[str, np.ndarray],
    label: np.ndarray,
    selected: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
    primary_score: str,
) -> dict:
    outcome = {}
    scores = {
        name: np.asarray(rows[field], dtype=np.float64)[selected]
        for name, field in SCORE_FIELDS.items()
    }
    selected_label = label[selected]
    for name, score in scores.items():
        outcome[name] = _metric(selected_label, score)
    outcome[primary_score]["fixed_threshold"] = _threshold_metric(
        selected_label, scores[primary_score]
    )
    bootstrap_scores = {
        name: scores[name]
        for name in (
            primary_score,
            "adoption_deficit",
            "context_js",
            "confidence_surprisal",
        )
    }
    return {
        "primary_score": primary_score,
        "scores": outcome,
        "source_bootstrap": _source_bootstrap(
            selected_label,
            bootstrap_scores,
            np.asarray(rows["source_id"])[selected],
            bootstrap,
            seed,
            primary_score,
        ),
    }


def _evaluation_report(
    rows: Mapping[str, np.ndarray],
    token_label: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    task = np.asarray(rows["task_type"]).astype(str, copy=False)
    sample = np.asarray(rows["sample_id"]).astype(str, copy=False)
    source = np.asarray(rows["source_id"]).astype(str, copy=False)
    onset = _onset_labels(token_label, sample, rows["token_index"])
    reports = {}
    scopes = (("ALL", np.ones(len(task), dtype=bool)),) + tuple(
        (name, task == name) for name in TASK_TYPES
    )
    for index, (name, selected) in enumerate(scopes):
        reports[name] = {
            "samples": len(set(sample[selected].tolist())),
            "sources": len(set(source[selected].tolist())),
            "token": _scope_report(
                rows,
                token_label,
                selected,
                bootstrap=bootstrap,
                seed=seed + 100 * index,
                primary_score=TOKEN_PRIMARY_SCORE,
            ),
            "onset": _scope_report(
                rows,
                onset,
                selected,
                bootstrap=bootstrap,
                seed=seed + 100 * index + 1,
                primary_score=ONSET_PRIMARY_SCORE,
            ),
        }
    return reports


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_detection(
    output_root: str | Path,
    cache_root: str | Path,
    *,
    bootstrap: int = 1000,
    seed: int = 2026,
) -> dict:
    """Fit on unlabeled train, freeze test scores, then open test labels."""

    output = Path(output_root)
    cache = Path(cache_root)
    train_root, test_root = output / "train", output / "test"
    train_manifest = _load_manifest(train_root)
    test_manifest = _load_manifest(test_root)
    _check_capture_pair(train_manifest, test_manifest)
    train_records = _feature_records(train_root, train_manifest)
    test_records = _feature_records(test_root, test_manifest)
    detectors, calibration = _fit_detectors(train_records, test_records)
    rows = _score_test(test_records, detectors)

    test_sources = {record.source_id for record in test_records}
    calibration_sources = sorted(
        {
            record.source_id
            for record in train_records
            if record.source_id not in test_sources
        }
    )
    detector_spec = {
        "schema": DETECTOR_SCHEMA,
        "name": "grounded_reanchor_state",
        "calibration": "task_specific_source_balanced_conditional_ecdf",
        "nuisance": [
            "relative_position_8_bins",
            "baseline_entropy_4_train_quantile_bins",
            "baseline_target_logprob_4_train_quantile_bins",
        ],
        "primary_scores": {
            "token": TOKEN_PRIMARY_SCORE,
            "onset": ONSET_PRIMARY_SCORE,
        },
        "threshold": DETECTION_THRESHOLD,
        "online_causal_score": True,
        "state_reset": "evidence_reentry",
        "context_distribution_js_role": "diagnostic_control",
        "offline_future_features": ["predictor_reuse", "emitted_token_anchor"],
        "development_status": "test_informed_exploratory_v2",
    }
    rows.update(
        detector_schema=np.asarray(DETECTOR_SCHEMA, dtype=np.int16),
        audit_scope=np.asarray("selected_samples"),
        dataset_manifest_sha256=np.asarray(file_sha256(cache / "test" / "manifest.json")),
        train_capture_manifest_sha256=np.asarray(
            file_sha256(train_root / "run_manifest.json")
        ),
        test_capture_manifest_sha256=np.asarray(
            file_sha256(test_root / "run_manifest.json")
        ),
        calibration_source_id=np.asarray(calibration_sources, dtype=str),
        detector_spec=np.asarray(json.dumps(detector_spec, sort_keys=True)),
        calibration_summary=np.asarray(json.dumps(calibration, sort_keys=True)),
        flag_online_95=np.asarray(
            rows["score_online_failure"] >= DETECTION_THRESHOLD, dtype=np.int8
        ),
        flag_onset_95=np.asarray(
            rows["score_onset_trigger"] >= DETECTION_THRESHOLD, dtype=np.int8
        ),
    )
    detection_root = output / "detection"
    score_path = detection_root / "token_scores.npz"
    save_result(score_path, rows)

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    frozen_rows = load_result(score_path)
    from research_dataset import open_research_dataset

    dataset = open_research_dataset(
        cache / "test", device="cpu", retain_embedded_labels=True
    )
    labels = frozen.align_loaded(dataset, frozen_rows)
    report = {
        "schema": "reanchor-failure-evaluation",
        "version": DETECTOR_SCHEMA,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_scores": {
            "token": TOKEN_PRIMARY_SCORE,
            "onset": ONSET_PRIMARY_SCORE,
        },
        "development_status": "test_informed_exploratory_v2",
        "temporal_scope": {
            "onset_trigger": "current_prediction_event",
            "online_failure": "causal_prefix",
            "offline_failure": "uses_later_prediction_rows",
        },
        "calibration": calibration,
        "score_artifact": str(score_path.resolve()),
        "score_sha256": frozen.artifact.sha256,
        "tasks": _evaluation_report(
            frozen_rows,
            labels.token_label,
            bootstrap=bootstrap,
            seed=seed,
        ),
    }
    report = _json_ready(report)
    report_path = detection_root / "detection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report
