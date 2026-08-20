"""End-to-end validation for causal majorization and routing-state dynamics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from experiment_protocol import (
    FrozenEvaluation,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
    validate_complete_token_rows,
    validate_source_audit,
)
from research_dataset import open_research_dataset

from .artifacts import load_npz, save_npz, sha256_file, write_json
from .config import PhenomenologyConfig
from .majorization_detector import (
    CausalMajorizationDetector,
    MajorizationDetectorConfig,
)
from .majorization_dynamics import causal_route_trace_from_edges
from .majorization_nulls import (
    shuffle_prompt_source_identity,
    shuffle_prompt_time,
    uniform_prompt_excess,
)
from .routing import collect_routing_edges


REFERENCE_SCHEMA = "causal-majorization-reference-v1"
SCORE_SCHEMA = "causal-majorization-token-scores-v1"
EVALUATION_SCHEMA = "causal-majorization-evaluation-v1"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(score)
    labels = np.asarray(labels)[finite]
    score = np.asarray(score)[finite]
    if len(labels) == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
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
) -> dict[str, float]:
    samples = np.unique(sample_id)
    if len(samples) == 0:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(samples, size=len(samples), replace=True)
        index = np.concatenate([np.flatnonzero(sample_id == sample) for sample in selected])
        metrics = _binary_metrics(labels[index], score[index])
        if np.isfinite(metrics["auroc"]):
            values.append((metrics["auroc"], metrics["auprc"]))
    if not values:
        return {
            "auroc_ci_low": float("nan"),
            "auroc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
        }
    values = np.asarray(values)
    return {
        "auroc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def _forecast_rows(
    rows: dict[str, np.ndarray],
    labels: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_rows = []
    target_labels = []
    source_samples = []
    sample_ids = rows["sample_id"].astype(str)
    token_index = rows["token_index"]
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        index = np.flatnonzero(sample_ids == sample_id)
        index = index[np.argsort(token_index[index])]
        if len(index) <= horizon:
            continue
        source = index[:-horizon]
        target = index[horizon:]
        contiguous = token_index[target] == token_index[source] + horizon
        source = source[contiguous]
        target = target[contiguous]
        source_rows.append(rows["forecast_probability"][source])
        target_labels.append(labels[target])
        source_samples.append(sample_ids[source])
    if not source_rows:
        return np.empty(0), np.empty(0, dtype=np.int8), np.empty(0, dtype=str)
    return (
        np.concatenate(source_rows),
        np.concatenate(target_labels),
        np.concatenate(source_samples),
    )


def _mechanism_effects(
    rows: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, float | int]:
    sample_ids = rows["sample_id"].astype(str)
    token_index = rows["token_index"]
    onset_majorization = []
    onset_concentration = []
    onset_entry = []
    basin_residence = []
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        index = np.flatnonzero(sample_ids == sample_id)
        index = index[np.argsort(token_index[index])]
        sample_labels = labels[index]
        starts = np.flatnonzero(
            (sample_labels == 1)
            & np.concatenate(([True], sample_labels[:-1] == 0))
        )
        for start in starts:
            if start == 0:
                continue
            onset = index[start]
            before = index[max(0, start - 4) : start]
            if not rows["valid"][onset] or not rows["valid"][before].any():
                continue
            valid_before = before[rows["valid"][before]]
            onset_majorization.append(
                rows["majorization_evidence"][onset]
                - rows["majorization_evidence"][valid_before].mean()
            )
            onset_concentration.append(
                rows["concentration_level"][onset]
                - rows["concentration_level"][valid_before].mean()
            )
            onset_entry.append(
                rows["entry_probability"][onset]
                - rows["entry_probability"][valid_before].mean()
            )
            end = start + 1
            while end < len(index) and sample_labels[end] == 1:
                end += 1
            later = index[start + 1 : end]
            later = later[rows["valid"][later]]
            if len(later):
                basin_residence.append(
                    rows["basin_probability"][later].mean()
                    - rows["basin_probability"][onset]
                )

    def mean(values) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "onsets": len(onset_entry),
        "majorization_onset_minus_pre": mean(onset_majorization),
        "concentration_onset_minus_pre": mean(onset_concentration),
        "entry_probability_onset_minus_pre": mean(onset_entry),
        "spans_with_later_tokens": len(basin_residence),
        "basin_probability_later_minus_onset": mean(basin_residence),
    }


def evaluate_majorization_rows(
    rows: dict[str, np.ndarray],
    labels,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    """Evaluate frozen rows, keeping current detection and forecast distinct."""

    token_labels = np.asarray(labels.token_label, dtype=np.int8)
    valid = np.asarray(rows["valid"], dtype=bool)
    sample_id = rows["sample_id"].astype(str)
    current = _binary_metrics(
        token_labels[valid], rows["current_probability"][valid]
    )
    current.update(
        _cluster_bootstrap(
            token_labels[valid],
            rows["current_probability"][valid],
            sample_id[valid],
            replicates=bootstrap_replicates,
            seed=seed,
        )
    )

    feature_metrics = {}
    for name in (
        "majorization_evidence",
        "concentration_level",
        "hill_shape",
        "source_affinity",
        "entry_probability",
        "basin_probability",
    ):
        feature_metrics[name] = _binary_metrics(
            token_labels[valid], rows[name][valid]
        )

    forecasts = {}
    for horizon in (1, 2, 4):
        score, target, source_sample = _forecast_rows(rows, token_labels, horizon)
        metrics = _binary_metrics(target, score)
        metrics["source_tokens"] = int(np.isfinite(score).sum())
        metrics.update(
            _cluster_bootstrap(
                target,
                score,
                source_sample,
                replicates=bootstrap_replicates,
                seed=seed + horizon,
            )
            if len(score)
            else {}
        )
        forecasts[f"horizon_{horizon}"] = metrics

    control_metrics = {}
    for control in ("uniform", "source_shuffle", "time_shuffle"):
        field = f"{control}_current_probability"
        if field not in rows:
            continue
        metrics = _binary_metrics(token_labels[valid], rows[field][valid])
        metrics["real_minus_control_auroc"] = current["auroc"] - metrics["auroc"]
        metrics["real_minus_control_auprc"] = current["auprc"] - metrics["auprc"]
        control_metrics[control] = metrics

    return {
        "tokens": len(token_labels),
        "positive_tokens": int(token_labels.sum()),
        "prevalence": float(token_labels.mean()),
        "valid_tokens": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "current_detection": current,
        "forecast": forecasts,
        "control_metrics": control_metrics,
        "feature_metrics": feature_metrics,
        "mechanism_effects": _mechanism_effects(rows, token_labels),
    }


def _trace_edges(edges, detector_config: MajorizationDetectorConfig):
    return causal_route_trace_from_edges(
        edges,
        history_decay=detector_config.history_decay,
        majorization_tolerance=detector_config.majorization_tolerance,
        epsilon=detector_config.epsilon,
    )


def _sample_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "little")


def _score_arrays(scores) -> dict[str, np.ndarray]:
    return {
        "majorization_evidence": scores.majorization_evidence.cpu().numpy(),
        "concentration_level": scores.concentration_level.cpu().numpy(),
        "hill_shape": scores.hill_shape.cpu().numpy(),
        "source_affinity": scores.source_affinity.cpu().numpy(),
        "valid_channel_fraction": scores.valid_channel_fraction.cpu().numpy(),
        "standardized_observation": scores.standardized_observation.cpu().numpy(),
        "state_probability": scores.state_probability.cpu().numpy(),
        "entry_probability": scores.entry_probability.cpu().numpy(),
        "basin_probability": scores.basin_probability.cpu().numpy(),
        "current_probability": scores.current_probability.cpu().numpy(),
        "forecast_probability": scores.forecast_probability.cpu().numpy(),
        "valid": scores.valid.cpu().numpy(),
    }


def run_majorization_validation(
    *,
    train_split,
    test_split,
    output_dir,
    device: str = "cpu",
    detector_config: MajorizationDetectorConfig | None = None,
    block_rows: int = 8192,
    fit_limit: int | None = None,
    test_limit: int | None = None,
    bootstrap_replicates: int = 200,
    seed: int = 20260820,
) -> dict:
    """Fit label-free, freeze token scores, then open labels for evaluation."""

    detector_config = (
        MajorizationDetectorConfig() if detector_config is None else detector_config
    )
    if fit_limit is not None and fit_limit < 1:
        raise ValueError("fit_limit must be positive")
    if test_limit is not None and test_limit < 1:
        raise ValueError("test_limit must be positive")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = open_research_dataset(train_split, device=device)
    fit_sample_ids = train.sample_ids if fit_limit is None else train.sample_ids[:fit_limit]
    traces = []
    reserved_source_ids = set()
    for sample_id in tqdm(fit_sample_ids, desc="fit causal route reference", unit="sample"):
        sample = train[sample_id]
        try:
            edges = collect_routing_edges(
                sample,
                config=PhenomenologyConfig(block_rows=block_rows),
            )
            traces.append(_trace_edges(edges, detector_config))
            reserved_source_ids.add(canonical_source_group(sample))
        finally:
            sample.release_attention()
    detector = CausalMajorizationDetector.fit(traces, config=detector_config)

    reference_path = output / "reference.json"
    write_json(
        reference_path,
        {
            "schema": REFERENCE_SCHEMA,
            "labels_read": False,
            "fit_split": str(Path(train_split).resolve()),
            "fit_manifest_sha256": dataset_manifest_sha256(train),
            "fit_sample_ids": list(map(str, fit_sample_ids)),
            "reserved_source_ids": sorted(reserved_source_ids),
            "detector_config": detector_config.to_dict(),
            "center": detector.center.tolist(),
            "scale": detector.scale.tolist(),
        },
    )
    del traces

    test = open_research_dataset(test_split, device=device)
    test_sample_ids = test.sample_ids if test_limit is None else test.sample_ids[:test_limit]
    audit = HeldOutSourceAudit(
        test,
        selected_sample_ids=test_sample_ids,
        reserved_source_ids=reserved_source_ids,
        require_complete_split=test_limit is None,
    )
    columns: dict[str, list[np.ndarray]] = {}
    sample_column = []
    source_column = []
    token_column = []
    length_column = []
    for sample_id in tqdm(test_sample_ids, desc="score causal route states", unit="sample"):
        sample = test[sample_id]
        try:
            audit.observe(sample)
            edges = collect_routing_edges(
                sample,
                config=PhenomenologyConfig(block_rows=block_rows),
            )
            scores = detector.score(_trace_edges(edges, detector_config))
            arrays = _score_arrays(scores)
            sample_seed = _sample_seed(sample.sample_id, seed)
            controls = {
                "uniform": uniform_prompt_excess(edges),
                "source_shuffle": shuffle_prompt_source_identity(
                    edges, seed=sample_seed
                ),
                "time_shuffle": shuffle_prompt_time(edges, seed=sample_seed + 1),
            }
            for name, control_edges in controls.items():
                control_scores = detector.score(
                    _trace_edges(control_edges, detector_config)
                )
                arrays[f"{name}_current_probability"] = (
                    control_scores.current_probability.cpu().numpy()
                )
                if name == "uniform":
                    arrays["uniform_majorization_evidence"] = (
                        control_scores.majorization_evidence.cpu().numpy()
                    )
                    arrays["uniform_concentration_level"] = (
                        control_scores.concentration_level.cpu().numpy()
                    )
                elif name == "source_shuffle":
                    arrays["source_shuffle_source_affinity"] = (
                        control_scores.source_affinity.cpu().numpy()
                    )
            response_length = len(scores.valid)
            for name, values in arrays.items():
                columns.setdefault(name, []).append(values)
            sample_column.append(np.repeat(str(sample.sample_id), response_length))
            source_column.append(
                np.repeat(canonical_source_group(sample), response_length)
            )
            token_column.append(np.arange(response_length, dtype=np.int32))
            length_column.append(np.full(response_length, response_length, dtype=np.int32))
        finally:
            sample.release_attention()
    source_audit = audit.finish()

    score_path = output / "scores.npz"
    score_rows = {
        "schema": np.asarray(SCORE_SCHEMA),
        "labels_read": np.asarray(False),
        "audit_scope": np.asarray(source_audit.test_scope),
        "dataset_manifest_sha256": np.asarray(dataset_manifest_sha256(test)),
        "reference_sha256": np.asarray(sha256_file(reference_path)),
        "reserved_source_ids": np.asarray(sorted(reserved_source_ids), dtype=str),
        "test_source_ids": np.asarray(source_audit.test_source_ids, dtype=str),
        "test_sample_ids": np.asarray(source_audit.test_sample_ids, dtype=str),
        "sample_id": np.concatenate(sample_column),
        "source_id": np.concatenate(source_column),
        "token_index": np.concatenate(token_column),
        "response_length": np.concatenate(length_column),
        **{name: np.concatenate(values) for name, values in columns.items()},
    }
    validate_complete_token_rows(
        score_rows["sample_id"],
        score_rows["source_id"],
        score_rows["token_index"],
        score_rows["response_length"],
    )
    validate_source_audit(
        reserved_source_ids=score_rows["reserved_source_ids"],
        test_source_ids=score_rows["test_source_ids"],
        test_sample_ids=score_rows["test_sample_ids"],
        row_sample_ids=score_rows["sample_id"],
        row_source_ids=score_rows["source_id"],
        audit_scope=score_rows["audit_scope"],
    )
    save_npz(score_path, **score_rows)

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    loaded = load_npz(score_path)
    validate_source_audit(
        reserved_source_ids=loaded["reserved_source_ids"],
        test_source_ids=loaded["test_source_ids"],
        test_sample_ids=loaded["test_sample_ids"],
        row_sample_ids=loaded["sample_id"],
        row_source_ids=loaded["source_id"],
        audit_scope=loaded["audit_scope"],
    )
    labels = frozen.align_loaded(test, loaded)
    report = evaluate_majorization_rows(
        loaded,
        labels,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "score_sha256": sha256_file(score_path),
        "reference_sha256": sha256_file(reference_path),
        "detector_config": detector_config.to_dict(),
        **report,
    }
    write_json(output / "evaluation.json", _json_safe(evaluation))
    return evaluation
