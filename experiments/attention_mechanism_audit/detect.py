"""Label-sealed detection of prompt-carrier concentration and route narrowing."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCORE_NAMES = ("functional_route_collapse", "attention_route_collapse", "confidence")
SCORE_DEFINITIONS = {
    "functional_route_collapse": (
        "lower-tail collapse of prompt-carrier degrees of freedom measured "
        "from dynamic A*||W_O^h V_s|| messages"
    ),
    "attention_route_collapse": "the same collapse measured from attention mass",
    "confidence": "negative full-branch target-token log probability",
}
FACTORIAL_CHANNELS = ("evidence", "history", "interaction")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("artifact") is not None:
        return record["artifact"]
    try:
        import torch
        return torch.load(Path(record["path"]), map_location="cpu", weights_only=True)
    except TypeError:
        import torch
        return torch.load(Path(record["path"]), map_location="cpu")


def factorial_contrasts(artifact: Mapping[str, Any]) -> np.ndarray:
    """Symmetric evidence, history and interaction effects with shape [T, 3]."""
    inputs = artifact["score_inputs"]
    full = _array(inputs["full_logprob"]).astype(np.float64)
    no_e = _array(inputs["no_evidence_logprob"]).astype(np.float64)
    no_h = _array(inputs["no_history_logprob"]).astype(np.float64)
    neither = _array(inputs["no_evidence_history_logprob"]).astype(np.float64)
    if full.ndim != 1 or any(x.shape != full.shape for x in (no_e, no_h, neither)):
        raise ValueError("factorial branches must be aligned token vectors")
    if not all(np.isfinite(x).all() for x in (full, no_e, no_h, neither)):
        raise ValueError("factorial branches must be finite")
    evidence = 0.5 * ((full - no_e) + (no_h - neither))
    history = 0.5 * ((full - no_h) + (no_e - neither))
    interaction = full - no_e - no_h + neither
    return np.stack((evidence, history, interaction), axis=1)


def stable_source_hash(source_id: str, seed: int = 0) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()[:8], "big"
    )


def source_fold_assignments(
    source_ids: Sequence[str], *, folds: int, seed: int
) -> dict[str, int]:
    unique = sorted(set(map(str, source_ids)))
    effective = min(folds, len(unique))
    ordered = sorted(
        unique, key=lambda source: (stable_source_hash(source, seed), source)
    )
    return {
        source: index % effective for index, source in enumerate(ordered)
    } if effective else {}


def crossfit_partitions(
    source_ids: Sequence[str], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    present = sorted(set(assignment.values()))
    if len(present) < 3:
        return []
    result = []
    for offset, test_fold in enumerate(present):
        calibration_fold = present[(offset + 1) % len(present)]
        fit_folds = [f for f in present if f not in {test_fold, calibration_fold}]
        result.append({
            "test_fold": test_fold,
            "calibration_fold": calibration_fold,
            "fit_folds": fit_folds,
            "test_sources": [s for s, f in assignment.items() if f == test_fold],
            "calibration_sources": [
                s for s, f in assignment.items() if f == calibration_fold
            ],
            "fit_sources": [s for s, f in assignment.items() if f in fit_folds],
        })
    return result


def _temporal_anchor_support(anchor: np.ndarray, window: int = 4) -> np.ndarray:
    """Effective prompt-anchor count over heads and four recent tokens."""
    layers, tokens, _ = anchor.shape
    support = np.ones((layers, tokens), dtype=np.float64)
    for layer in range(layers):
        for token in range(tokens):
            values = anchor[layer, max(0, token - window + 1) : token + 1].ravel()
            values = values[values >= 0]
            if len(values):
                counts = np.unique(values, return_counts=True)[1].astype(np.float64)
                probability = counts / counts.sum()
                support[layer, token] = np.exp(
                    -np.sum(probability * np.log(probability))
                )
    return support


def carrier_log_volume(
    artifact: Mapping[str, Any], family: str, *, temporal_window: int = 4
) -> np.ndarray:
    """Return token-by-layer log prompt-routing degrees of freedom."""
    if family not in {"attention", "edge"}:
        raise ValueError("family must be attention or edge")
    trace = artifact["trace"]
    effective = _array(trace[f"prompt_{family}_effective_sources"]).astype(np.float64)
    rank = _array(trace[f"prompt_{family}_effective_rank"]).astype(np.float64)
    anchor = _array(trace[f"prompt_{family}_anchor_index"]).astype(np.int64)
    if effective.ndim != 2 or rank.shape != effective.shape:
        raise ValueError("prompt carrier fields must be [layer, token]")
    if anchor.shape[:2] != effective.shape:
        raise ValueError("prompt anchors must preserve layer and token axes")
    if (
        not np.isfinite(effective).all()
        or not np.isfinite(rank).all()
        or np.any(effective < 1)
        or np.any(rank < 1)
    ):
        raise ValueError("prompt carrier support and rank must be finite and >= 1")
    temporal = _temporal_anchor_support(anchor, temporal_window)
    volume = (
        np.log(np.maximum(effective, 1.0))
        + np.log(np.maximum(rank, 1.0))
        + np.log(np.maximum(temporal, 1.0))
    )
    return volume.T


def _design(artifact: Mapping[str, Any], tokens: int) -> np.ndarray:
    position = (np.arange(tokens, dtype=np.float64) + 0.5) / max(tokens, 1)
    length = np.full(
        tokens, np.log1p(float(artifact.get("response_start", 0)) + tokens)
    )
    return np.column_stack((np.ones(tokens), position, np.square(position), length))


@dataclass
class _VolumeModel:
    coefficients: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]


def _weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    return float(value[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def _fit_volume_model(records: Sequence[Mapping[str, Any]]) -> _VolumeModel:
    coefficients, scales = {}, {}
    source_counts: dict[str, int] = defaultdict(int)
    for record in records:
        source_counts[str(record["source_id"])] += 1
    artifacts = [(record, _load_artifact(record)) for record in records]
    for family in ("edge", "attention"):
        rows = []
        for record, artifact in artifacts:
            y = carrier_log_volume(artifact, family)
            source_weight = 1 / source_counts[str(record["source_id"])]
            rows.append(
                (
                    _design(artifact, len(y)),
                    y,
                    np.full(len(y), source_weight / len(y)),
                )
            )
        layers = rows[0][1].shape[1]
        beta, scale = np.zeros((layers, 4)), np.ones(layers)
        for layer in range(layers):
            x = np.concatenate([row[0] for row in rows])
            y = np.concatenate([row[1][:, layer] for row in rows])
            w = np.concatenate([row[2] for row in rows])
            root = np.sqrt(w)
            beta[layer] = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
            residual = y - x @ beta[layer]
            median = _weighted_median(residual, w)
            scale[layer] = max(
                1.4826 * _weighted_median(np.abs(residual - median), w), 1e-3
            )
        coefficients[family], scales[family] = beta, scale
    return _VolumeModel(coefficients, scales)


def _raw_collapse(
    artifact: Mapping[str, Any], model: _VolumeModel, family: str
) -> np.ndarray:
    volume = carrier_log_volume(artifact, family)
    expected = _design(artifact, len(volume)) @ model.coefficients[family].T
    standardized = (expected - volume) / model.scales[family][None]
    return np.maximum(standardized, 0).mean(axis=1)


def _position_bin(tokens: int) -> np.ndarray:
    return np.minimum((np.arange(tokens) * 10) // max(tokens, 1), 9)


def _calibration_tables(
    records: Sequence[Mapping[str, Any]], raw: Mapping[str, np.ndarray]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pieces: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    for record in records:
        source_counts[str(record["source_id"])] += 1
    for record in records:
        value = raw[str(record["sample_id"])]
        source_weight = 1 / source_counts[str(record["source_id"])]
        for bucket in range(10):
            selected = value[_position_bin(len(value)) == bucket]
            if len(selected):
                pieces[bucket].append(
                    (selected, np.full(len(selected), source_weight / len(value)))
                )
    pooled_values = np.concatenate(list(raw.values()))
    pooled_weights = np.concatenate(
        [
            np.full(
                len(raw[str(record["sample_id"])]),
                1
                / source_counts[str(record["source_id"])]
                / len(raw[str(record["sample_id"])]),
            )
            for record in records
        ]
    )

    def table(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        return values[order], cumulative / cumulative[-1]

    pooled = table(pooled_values, pooled_weights)
    result = {-1: pooled}
    for bucket in range(10):
        if pieces[bucket]:
            values = np.concatenate([part[0] for part in pieces[bucket]])
            weights = np.concatenate([part[1] for part in pieces[bucket]])
            result[bucket] = table(values, weights)
        else:
            result[bucket] = pooled
    return result


def _ecdf(
    value: np.ndarray, tables: Mapping[int, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    result = np.empty(len(value))
    for index, bucket in enumerate(_position_bin(len(value))):
        reference, cumulative = tables.get(int(bucket), tables[-1])
        location = np.searchsorted(reference, value[index], side="right")
        result[index] = cumulative[location - 1] if location else 0.0
    return result


def _confidence(record: Mapping[str, Any]) -> np.ndarray:
    artifact = _load_artifact(record)
    return -_array(artifact["score_inputs"]["full_logprob"]).astype(np.float64)


def score_records(
    records: Sequence[Mapping[str, Any]], *, seed: int = 0, folds: int = 5, **_: Any
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Freeze source-disjoint, label-free prompt-carrier collapse scores."""
    records = list(records)
    if not records:
        return {}, {"crossfit_complete": False, "mechanism_scores_available": False}
    sample_ids = [str(r["sample_id"]) for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")
    if len({str(r.get("task_type")) for r in records}) > 1:
        raise ValueError("score_records requires one task_type at a time")
    records.sort(key=lambda r: (str(r["source_id"]), str(r["sample_id"])))
    source_ids = [str(r["source_id"]) for r in records]
    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    partitions = crossfit_partitions(source_ids, folds=folds, seed=seed)
    output: dict[str, dict[str, np.ndarray]] = {}
    if not partitions:
        for record in records:
            confidence = _confidence(record).astype(np.float32)
            output[str(record["sample_id"])] = {
                SCORE_NAMES[0]: np.zeros_like(confidence),
                SCORE_NAMES[1]: np.zeros_like(confidence),
                SCORE_NAMES[2]: confidence,
            }
        return output, {
            "crossfit_complete": False,
            "mechanism_scores_available": False,
            "reason": "at least three distinct sources are required",
            "score_definitions": SCORE_DEFINITIONS,
        }
    metadata = []
    for partition in partitions:
        fit_sources = set(partition["fit_sources"])
        calibration_sources = set(partition["calibration_sources"])
        test_sources = set(partition["test_sources"])
        fit = [r for r in records if str(r["source_id"]) in fit_sources]
        calibration = [
            r for r in records if str(r["source_id"]) in calibration_sources
        ]
        test = [r for r in records if str(r["source_id"]) in test_sources]
        model = _fit_volume_model(fit)
        tables = {}
        for family in ("edge", "attention"):
            raw = {
                str(r["sample_id"]): _raw_collapse(
                    _load_artifact(r), model, family
                )
                for r in calibration
            }
            tables[family] = _calibration_tables(calibration, raw)
        for record in test:
            artifact = _load_artifact(record)
            output[str(record["sample_id"])] = {
                "functional_route_collapse": _ecdf(
                    _raw_collapse(artifact, model, "edge"), tables["edge"]
                ).astype(np.float32),
                "attention_route_collapse": _ecdf(
                    _raw_collapse(artifact, model, "attention"), tables["attention"]
                ).astype(np.float32),
                "confidence": _confidence(record).astype(np.float32),
            }
        metadata.append({
            "test_fold": partition["test_fold"],
            "calibration_fold": partition["calibration_fold"],
            "fit_folds": partition["fit_folds"],
            "fit_sources": len(fit_sources),
            "calibration_sources": len(calibration_sources),
            "test_sources": len(test_sources),
        })
    return output, {
        "crossfit_complete": True,
        "mechanism_scores_available": True,
        "seed": int(seed),
        "requested_folds": int(folds),
        "source_folds": assignment,
        "partitions": metadata,
        "score_definitions": SCORE_DEFINITIONS,
        "fit": "source-equal position/length nuisance regression; labels sealed",
        "carrier_volume": (
            "effective prompt sources * head route rank * temporal anchor support"
        ),
    }
