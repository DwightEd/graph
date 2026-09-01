"""Label-sealed detection from causal branches and head-resolved route graphs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .collect import validate_saved_artifact
from .graph import route_contraction

SCORE_NAMES = (
    "unsupported_history_takeover",
    "evidence_bypass",
    "evidence_route_contraction",
    "history_route_contraction",
    "confidence",
)
SCORE_DEFINITIONS = {
    "unsupported_history_takeover": (
        "out-of-fold percentile of strict-history support after direct "
        "evidence-token attention-value writes are cut at response predictor "
        "queries, minus direct evidence-write support with history present, "
        "conditioned on position and length"
    ),
    "evidence_bypass": (
        "out-of-fold percentile of target log-probability gain after "
        "direct evidence-token attention-value writes are cut at response "
        "predictor queries, conditioned on position and length"
    ),
    "evidence_route_contraction": (
        "out-of-fold percentile of recent head-resolved evidence-route contraction"
    ),
    "history_route_contraction": (
        "out-of-fold percentile of recent head-resolved strict-history contraction"
    ),
    "confidence": "negative full-branch target-token log probability",
}
NUISANCE_NAMES = (
    "intercept",
    "relative_position",
    "relative_position_squared",
    "log_prompt_length",
    "log_evidence_length",
    "log_response_length",
)
MECHANISM_SCORES = SCORE_NAMES[:-1]


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("artifact") is not None:
        artifact = record["artifact"]
    else:
        try:
            import torch

            artifact = torch.load(
                Path(record["path"]), map_location="cpu", weights_only=True
            )
        except TypeError:
            import torch

            artifact = torch.load(Path(record["path"]), map_location="cpu")
    validate_saved_artifact(artifact, record)
    return artifact


def _branches(artifact: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    inputs = artifact["score_inputs"]
    values = tuple(
        _array(inputs[name]).astype(np.float64)
        for name in (
            "full_logprob",
            "no_evidence_logprob",
            "no_history_logprob",
            "no_evidence_history_logprob",
        )
    )
    if values[0].ndim != 1 or any(value.shape != values[0].shape for value in values):
        raise ValueError("factorial branches must be aligned token vectors")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("factorial branches must be finite")
    return values


def factorial_contrasts(artifact: Mapping[str, Any]) -> np.ndarray:
    """Symmetric evidence, history, and interaction effects with shape [T, 3]."""

    full, no_evidence, no_history, neither = _branches(artifact)
    evidence = 0.5 * ((full - no_evidence) + (no_history - neither))
    history = 0.5 * ((full - no_history) + (no_evidence - neither))
    interaction = full - no_evidence - no_history + neither
    return np.stack((evidence, history, interaction), axis=1)


def raw_scores(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return fixed-direction scores before nuisance removal and calibration."""

    full, no_evidence, _, neither = _branches(artifact)
    return {
        "unsupported_history_takeover": 2 * no_evidence - full - neither,
        "evidence_bypass": no_evidence - full,
        "evidence_route_contraction": route_contraction(artifact, "evidence"),
        "history_route_contraction": route_contraction(artifact, "response_history"),
        "confidence": -full,
    }


def nuisance_design(artifact: Mapping[str, Any]) -> np.ndarray:
    """Build distinct response-position and three length controls."""

    full = _array(artifact["score_inputs"]["full_logprob"]).astype(np.float64)
    tokens = len(full)
    prompt_length = int(artifact["response_start"])
    evidence_mask = _array(artifact["evidence_mask"]).astype(bool)
    if full.ndim != 1:
        raise ValueError("full log probability must be a token vector")
    if evidence_mask.shape != (prompt_length,):
        raise ValueError("evidence_mask must align exactly with the prompt")
    position = (np.arange(tokens, dtype=np.float64) + 0.5) / max(tokens, 1)
    return np.column_stack(
        (
            np.ones(tokens),
            position,
            np.square(position),
            np.full(tokens, np.log1p(prompt_length)),
            np.full(tokens, np.log1p(evidence_mask.sum())),
            np.full(tokens, np.log1p(tokens)),
        )
    )


def stable_source_hash(source_id: str, seed: int = 0) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()[:8], "big"
    )


def source_fold_assignments(
    source_ids: Sequence[str], *, folds: int, seed: int
) -> dict[str, int]:
    if folds <= 0:
        raise ValueError("folds must be positive")
    unique = sorted(set(map(str, source_ids)))
    effective = min(folds, len(unique))
    ordered = sorted(
        unique, key=lambda source: (stable_source_hash(source, seed), source)
    )
    return (
        {source: index % effective for index, source in enumerate(ordered)}
        if effective
        else {}
    )


def crossfit_partitions(
    source_ids: Sequence[str], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    present = sorted(set(assignment.values()))
    if len(present) < 3:
        return []
    partitions = []
    for offset, test_fold in enumerate(present):
        calibration_fold = present[(offset + 1) % len(present)]
        fit_folds = [
            fold for fold in present if fold not in {test_fold, calibration_fold}
        ]
        partitions.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": fit_folds,
                "test_sources": [
                    source for source, fold in assignment.items() if fold == test_fold
                ],
                "calibration_sources": [
                    source
                    for source, fold in assignment.items()
                    if fold == calibration_fold
                ],
                "fit_sources": [
                    source for source, fold in assignment.items() if fold in fit_folds
                ],
            }
        )
    return partitions


@dataclass(frozen=True)
class NuisanceModel:
    coefficients: dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedSample:
    scores: dict[str, np.ndarray]
    design: np.ndarray
    valid: np.ndarray
    score_valid: dict[str, np.ndarray]


def prepare_record(record: Mapping[str, Any]) -> PreparedSample:
    """Load one large artifact, retain compact token arrays, then release it."""

    artifact = _load_artifact(record)
    scores = {
        name: np.asarray(value, dtype=np.float64).copy()
        for name, value in raw_scores(artifact).items()
    }
    design = nuisance_design(artifact).copy()
    valid = np.arange(len(design)) >= 2
    evidence_route_valid = route_contraction(artifact, "evidence", return_valid=True)[1]
    history_route_valid = route_contraction(
        artifact, "response_history", return_valid=True
    )[1]
    score_valid = {name: valid.copy() for name in MECHANISM_SCORES}
    score_valid["evidence_route_contraction"] &= evidence_route_valid
    score_valid["history_route_contraction"] &= history_route_valid
    if any(value.shape != (len(design),) for value in scores.values()):
        raise ValueError("raw scores and nuisance rows must be token aligned")
    return PreparedSample(
        scores=scores,
        design=design,
        valid=valid,
        score_valid=score_valid,
    )


def _source_token_weights(
    records: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, PreparedSample],
    name: str,
) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        sample_id = str(record["sample_id"])
        counts[str(record["source_id"])] += int(
            prepared[sample_id].score_valid[name].sum()
        )
    return {source: 1.0 / count for source, count in counts.items() if count}


def _fit_nuisance(
    records: Sequence[Mapping[str, Any]], prepared: Mapping[str, PreparedSample]
) -> tuple[NuisanceModel | None, dict[str, dict[str, int | bool]]]:
    coefficients = {}
    diagnostics = {}
    columns = len(NUISANCE_NAMES)
    for name in MECHANISM_SCORES:
        source_weight = _source_token_weights(records, prepared, name)
        design, weights, raw = [], [], []
        for record in records:
            sample_id = str(record["sample_id"])
            sample = prepared[sample_id]
            selected = sample.score_valid[name]
            if not selected.any():
                continue
            design.append(sample.design[selected])
            weights.append(
                np.full(selected.sum(), source_weight[str(record["source_id"])])
            )
            raw.append(sample.scores[name][selected])
        rows = sum(len(value) for value in raw)
        if rows:
            x = np.concatenate(design)
            root_weight = np.sqrt(np.concatenate(weights))
            weighted = x * root_weight[:, None]
            rank = int(np.linalg.matrix_rank(weighted))
        else:
            x = np.empty((0, columns))
            root_weight = np.empty(0)
            weighted = x
            rank = 0
        usable = rows >= columns and rank == columns
        diagnostics[name] = {
            "rows": rows,
            "rank": rank,
            "columns": columns,
            "sources": len(source_weight),
            "usable": usable,
        }
        if usable:
            coefficients[name] = np.linalg.lstsq(
                weighted,
                np.concatenate(raw) * root_weight,
                rcond=None,
            )[0]
    if len(coefficients) != len(MECHANISM_SCORES):
        return None, diagnostics
    return NuisanceModel(coefficients), diagnostics


def _residual_scores(
    sample: PreparedSample, model: NuisanceModel
) -> dict[str, np.ndarray]:
    return {
        name: sample.scores[name] - sample.design @ model.coefficients[name]
        for name in MECHANISM_SCORES
    }


def _calibration_table(
    records: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, PreparedSample],
    residuals: Mapping[str, Mapping[str, np.ndarray]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_weight = _source_token_weights(records, prepared, name)
    values, weights = [], []
    for record in records:
        sample_id = str(record["sample_id"])
        sample = prepared[sample_id]
        selected = sample.score_valid[name]
        if not selected.any():
            continue
        value = residuals[sample_id][name][selected]
        values.append(value)
        weights.append(np.full(len(value), source_weight[str(record["source_id"])]))
    value = np.concatenate(values)
    weight = np.concatenate(weights)
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    return value[order], cumulative / cumulative[-1]


def _ecdf(value: np.ndarray, table: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    reference, cumulative = table
    location = np.searchsorted(reference, value, side="right")
    result = np.zeros(len(value), dtype=np.float64)
    present = location > 0
    result[present] = cumulative[location[present] - 1]
    return result


def score_records(
    records: Sequence[Mapping[str, Any]], *, seed: int = 0, folds: int = 5
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Freeze source-disjoint scores without opening hallucination labels."""

    records = sorted(
        records,
        key=lambda record: (str(record["source_id"]), str(record["sample_id"])),
    )
    if not records:
        return {}, {"crossfit_complete": False, "mechanism_scores_available": False}
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")
    if len({str(record.get("task_type")) for record in records}) > 1:
        raise ValueError("score_records requires one task_type at a time")

    prepared = {}
    for record in records:
        prepared[str(record["sample_id"])] = prepare_record(record)
    source_ids = sorted(
        {
            str(record["source_id"])
            for record in records
            if prepared[str(record["sample_id"])].valid.any()
        }
    )
    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    partitions = crossfit_partitions(source_ids, folds=folds, seed=seed)
    output: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        sample = prepared[sample_id]
        confidence = sample.scores["confidence"].astype(np.float32)
        output[sample_id] = {
            **{name: np.zeros_like(confidence) for name in MECHANISM_SCORES},
            "confidence": confidence,
            "detection_valid": sample.valid.copy(),
        }
    score_coverage = {
        name: {
            "tokens": int(
                sum(sample.score_valid[name].sum() for sample in prepared.values())
            ),
            "sources": len(_source_token_weights(records, prepared, name)),
        }
        for name in MECHANISM_SCORES
    }
    if not partitions:
        return output, {
            "crossfit_complete": False,
            "mechanism_scores_available": False,
            "reason": (
                "at least three sources with strict-history-eligible tokens are "
                "required"
            ),
            "score_definitions": SCORE_DEFINITIONS,
            "score_coverage": score_coverage,
            "evaluated_tokens": int(
                sum(sample.valid.sum() for sample in prepared.values())
            ),
        }

    partition_metadata = []
    plans = []
    for partition in partitions:
        fit_sources = set(partition["fit_sources"])
        calibration_sources = set(partition["calibration_sources"])
        test_sources = set(partition["test_sources"])
        fit = [record for record in records if str(record["source_id"]) in fit_sources]
        calibration = [
            record
            for record in records
            if str(record["source_id"]) in calibration_sources
        ]
        test = [
            record for record in records if str(record["source_id"]) in test_sources
        ]
        model, nuisance = _fit_nuisance(fit, prepared)
        partition_metadata.append(
            {
                "test_fold": partition["test_fold"],
                "calibration_fold": partition["calibration_fold"],
                "fit_folds": partition["fit_folds"],
                "fit_sources": len(fit_sources),
                "calibration_sources": len(calibration_sources),
                "test_sources": len(test_sources),
                "nuisance": nuisance,
            }
        )
        if model is None:
            return output, {
                "crossfit_complete": False,
                "mechanism_scores_available": False,
                "reason": "a nuisance fit is empty or rank deficient",
                "source_folds": assignment,
                "partitions": partition_metadata,
                "score_definitions": SCORE_DEFINITIONS,
                "score_coverage": score_coverage,
                "nuisance_columns": list(NUISANCE_NAMES),
                "evaluated_tokens": int(
                    sum(sample.valid.sum() for sample in prepared.values())
                ),
            }
        calibration_rows = {
            name: int(
                sum(
                    prepared[str(record["sample_id"])].score_valid[name].sum()
                    for record in calibration
                )
            )
            for name in MECHANISM_SCORES
        }
        test_rows = {
            name: int(
                sum(
                    prepared[str(record["sample_id"])].score_valid[name].sum()
                    for record in test
                )
            )
            for name in MECHANISM_SCORES
        }
        partition_metadata[-1]["calibration_rows"] = calibration_rows
        partition_metadata[-1]["test_rows"] = test_rows
        if not all(
            calibration_rows[name] and test_rows[name] for name in MECHANISM_SCORES
        ):
            return output, {
                "crossfit_complete": False,
                "mechanism_scores_available": False,
                "reason": "a calibration or test partition has no comparable score rows",
                "source_folds": assignment,
                "partitions": partition_metadata,
                "score_definitions": SCORE_DEFINITIONS,
                "score_coverage": score_coverage,
                "nuisance_columns": list(NUISANCE_NAMES),
                "evaluated_tokens": int(
                    sum(sample.valid.sum() for sample in prepared.values())
                ),
            }
        calibration_residuals = {
            str(record["sample_id"]): _residual_scores(
                prepared[str(record["sample_id"])], model
            )
            for record in calibration
        }
        tables = {
            name: _calibration_table(calibration, prepared, calibration_residuals, name)
            for name in MECHANISM_SCORES
        }
        plans.append((test, model, tables))

    for sample_id, sample in prepared.items():
        for name in MECHANISM_SCORES:
            output[sample_id][name][sample.valid] = 0.5
    for test, model, tables in plans:
        for record in test:
            sample_id = str(record["sample_id"])
            sample = prepared[sample_id]
            residual = _residual_scores(sample, model)
            for name in MECHANISM_SCORES:
                selected = sample.score_valid[name]
                output[sample_id][name][selected] = _ecdf(
                    residual[name][selected], tables[name]
                ).astype(np.float32)

    return output, {
        "crossfit_complete": True,
        "mechanism_scores_available": True,
        "seed": int(seed),
        "requested_folds": int(folds),
        "source_folds": assignment,
        "partitions": partition_metadata,
        "score_definitions": SCORE_DEFINITIONS,
        "score_coverage": score_coverage,
        "nuisance_columns": list(NUISANCE_NAMES),
        "fit": (
            "source-equal position/length nuisance regression; calibration-source "
            "ECDF; strict-history-eligible tokens only; labels sealed"
        ),
        "score_scale": (
            "out-of-fold percentile after position/length conditioning; the fixed "
            "raw endpoint is reported separately"
        ),
        "evaluated_tokens": int(
            sum(sample.valid.sum() for sample in prepared.values())
        ),
    }
