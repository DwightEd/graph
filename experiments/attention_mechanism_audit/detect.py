"""Label-sealed raw scores from provenance registers and causal branches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import REGISTER_NAMES, REGISTER_STAGE_NAMES
SCORE_NAMES = (
    "evidence_bypass",
    "symmetric_route_capture",
    "unsupported_history_takeover",
    "provenance_takeover",
    "confidence",
)
SCORE_DEFINITIONS = {
    "evidence_bypass": "no_evidence_logprob - full_logprob",
    "symmetric_route_capture": "no_evidence_logprob - no_history_logprob",
    "unsupported_history_takeover": (
        "2 * no_evidence_logprob - full_logprob - no_evidence_history_logprob"
    ),
    "provenance_takeover": (
        "log ratio of the leading cross-layer Gram eigenvalue for autonomous "
        "history versus evidence adoption"
    ),
    "confidence": "-full_logprob",
}
_EPS = 1e-12


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("artifact") is not None:
        artifact = record["artifact"]
    else:
        import torch

        try:
            artifact = torch.load(
                Path(record["path"]), map_location="cpu", weights_only=True
            )
        except TypeError:
            artifact = torch.load(Path(record["path"]), map_location="cpu")
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
    """Return symmetric evidence, history, and interaction effects [T, 3]."""

    full, no_evidence, no_history, neither = _branches(artifact)
    evidence = 0.5 * ((full - no_evidence) + (no_history - neither))
    history = 0.5 * ((full - no_history) + (no_evidence - neither))
    interaction = full - no_evidence - no_history + neither
    return np.stack((evidence, history, interaction), axis=1)


def _provenance_takeover(artifact: Mapping[str, Any], tokens: int) -> np.ndarray:
    """Compare the dominant cross-layer step energy of the two registers."""

    gram = _array(artifact["trace"]["register_step_gram"]).astype(np.float64)
    if (
        gram.ndim != 4
        or gram.shape[1:3] != (tokens, len(REGISTER_NAMES))
        or gram.shape[0] != gram.shape[3]
    ):
        raise ValueError("register_step_gram must have shape [layer, token, 2, layer]")
    if not np.isfinite(gram).all():
        raise ValueError("register_step_gram must be finite")

    # A Gram matrix should be symmetric; averaging removes only replay roundoff.
    matrices = np.moveaxis(gram, 1, 0)  # [token, layer, register, layer]
    matrices = np.moveaxis(matrices, 2, 1)  # [token, register, layer, layer]
    matrices = 0.5 * (matrices + matrices.swapaxes(-1, -2))
    leading = np.linalg.eigvalsh(matrices)[..., -1]
    leading = np.maximum(leading, 0.0)
    score = np.log((leading[:, 1] + _EPS) / (leading[:, 0] + _EPS))
    score[:2] = 0.0  # Strict history excludes the predictor-self token.
    return score


def raw_scores(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute fixed raw equations without fitting, calibration, or labels."""

    full, no_evidence, no_history, neither = _branches(artifact)
    return {
        "evidence_bypass": no_evidence - full,
        "symmetric_route_capture": no_evidence - no_history,
        "unsupported_history_takeover": 2 * no_evidence - full - neither,
        "provenance_takeover": _provenance_takeover(artifact, len(full)),
        "confidence": -full,
    }


def _validity(tokens: int) -> dict[str, np.ndarray]:
    all_tokens = np.ones(tokens, dtype=bool)
    has_history = all_tokens.copy()
    has_history[:2] = False
    return {
        "evidence_bypass": all_tokens,
        "symmetric_route_capture": has_history,
        "unsupported_history_takeover": has_history,
        "provenance_takeover": has_history,
        "confidence": all_tokens,
    }


def score_records(
    records: Sequence[Mapping[str, Any]], *, seed: int = 0
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load and release each artifact while freezing label-free raw scores."""

    ordered = sorted(records, key=lambda row: str(row["sample_id"]))
    sample_ids = [str(record["sample_id"]) for record in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")

    output: dict[str, dict[str, np.ndarray]] = {}
    coverage = {name: {"tokens": 0, "sources": set()} for name in SCORE_NAMES}
    comparison = {"tokens": 0, "sources": set()}
    for record in ordered:
        artifact = _load_artifact(record)
        try:
            scores = raw_scores(artifact)
        finally:
            del artifact
        tokens = len(scores[SCORE_NAMES[0]])
        valid = _validity(tokens)
        sample_id = str(record["sample_id"])
        source_id = str(record["source_id"])
        frozen: dict[str, np.ndarray] = {}
        for name in SCORE_NAMES:
            value = np.asarray(scores[name], dtype=np.float32)
            value[~valid[name]] = 0.0
            frozen[name] = value
            frozen[f"{name}__valid"] = valid[name]
            coverage[name]["tokens"] += int(valid[name].sum())
            if valid[name].any():
                coverage[name]["sources"].add(source_id)
        comparison_valid = np.logical_and.reduce(tuple(valid.values()))
        frozen["detection_valid"] = comparison_valid
        comparison["tokens"] += int(comparison_valid.sum())
        if comparison_valid.any():
            comparison["sources"].add(source_id)
        output[sample_id] = frozen

    intrinsic_coverage = {
        name: {
            "tokens": values["tokens"],
            "sources": len(values["sources"]),
        }
        for name, values in coverage.items()
    }
    return output, {
        "mechanism_scores_available": bool(output),
        "crossfit": "not_applicable",
        "seed": int(seed),
        "score_definitions": SCORE_DEFINITIONS,
        "intrinsic_score_coverage": intrinsic_coverage,
        "comparison_coverage": {
            "tokens": comparison["tokens"],
            "sources": len(comparison["sources"]),
            "scope": "intersection of fixed score-validity masks",
        },
        "fit": "none",
        "calibration": "none",
        "score_scale": "raw fixed equations; no nuisance regression or ECDF",
        "labels_used": False,
        "evaluated_tokens": comparison["tokens"],
    }
