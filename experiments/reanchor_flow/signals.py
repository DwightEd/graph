"""Artifact validation and routing-signal extraction for re-anchor evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.common.ragtruth_alignment import canonical_task_type

from .capture import CAPTURE_SCHEMA
from .events import validate_coordinates

EPSILON = 1e-8


def scalar_text(values: dict, name: str, default: str = "") -> str:
    if name not in values:
        return default
    return str(np.asarray(values[name]).item())


def scalar_int(values: dict, name: str, default: int | None = None) -> int | None:
    if name not in values:
        return default
    return int(np.asarray(values[name]).item())


def same_model(first: str, second: str) -> bool:
    return bool(first and second) and (
        first == second or Path(first).name == Path(second).name
    )


def log_lift(observed, availability) -> np.ndarray:
    """Log observed share relative to a source-availability null."""

    observed = np.asarray(observed, dtype=np.float64)
    availability = np.asarray(availability, dtype=np.float64)
    if observed.shape != availability.shape:
        raise ValueError("observed routes and availability null must align")
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("routing traces must have shape [layer,event,3]")
    if np.any(observed < 0) or np.any(availability < 0):
        raise ValueError("routing shares cannot be negative")
    lift = np.log((observed + EPSILON) / (availability + EPSILON))
    # No history source exists for the first response event. Treating 0/0 as
    # evidence of neutral routing would bias the early-response baseline.
    lift[availability <= 0] = np.nan
    return lift


def finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def layer_mean(values) -> np.ndarray:
    """Mean across layers without all-NaN warnings."""

    values = np.asarray(values, dtype=np.float64)
    count = np.isfinite(values).sum(axis=0)
    total = np.nansum(values, axis=0)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def validate_artifact_identity(
    entry: dict, result: dict, sample, task: str, config: dict
) -> None:
    expected = {
        "sample_id": str(sample.sample_id),
        "source_id": str(sample.source_id),
        "task_type": task,
    }
    for name, value in expected.items():
        recorded = scalar_text(result, name)
        manifest = str(entry.get(name, value))
        if recorded != value or manifest != value:
            raise ValueError(
                f"artifact identity mismatch for {name}: result={recorded!r}, "
                f"manifest={manifest!r}, dataset={value!r}"
            )
    schema = scalar_int(result, "capture_schema")
    if schema != CAPTURE_SCHEMA:
        raise ValueError(
            f"stale capture schema {schema}; expected {CAPTURE_SCHEMA}. Rerun analyze in a new output directory."
        )
    recorded_model = scalar_text(result, "model_id")
    if (
        recorded_model != str(config.get("model_id", ""))
        or str(entry.get("model_id", "")) != recorded_model
    ):
        raise ValueError("artifact model_id differs from the run manifest")
    observer = scalar_text(result, "observer_model")
    expected_observer = str(Path(config.get("model", "")).resolve())
    if observer != expected_observer or str(entry.get("observer_model", "")) != observer:
        raise ValueError("artifact observer model differs from the run manifest")
    generator = scalar_text(result, "generator_model")
    expected_generator = str(getattr(sample, "generator_model", "") or "")
    if (
        str(entry.get("generator_model", "")) != generator
        or generator != expected_generator
    ):
        raise ValueError("artifact generator model differs from the run manifest")
    recorded_dtype = scalar_text(result, "dtype")
    if (
        recorded_dtype != str(config.get("dtype", ""))
        or str(entry.get("dtype", "")) != recorded_dtype
    ):
        raise ValueError("artifact dtype differs from the run manifest")
    cached_observer = scalar_text(result, "cached_observer_model")
    if str(entry.get("cached_observer_model", "")) != cached_observer:
        raise ValueError("cached observer metadata differs from the run manifest")
    sample_observer = str(getattr(sample, "observer_model", "") or "")
    if sample_observer and not same_model(sample_observer, cached_observer):
        raise ValueError("cached observer metadata differs from the dataset")
    if scalar_int(result, "query_chunk") != int(entry.get("query_chunk", -1)):
        raise ValueError("query-chunk metadata differs from the run manifest")


def compact_row(entry: dict, result: dict, sample, label_store, config: dict) -> dict:
    cached = sample.attention()
    response_start = int(cached.response_idx)
    token_ids = np.asarray(cached.token_ids.detach().cpu(), dtype=np.int64)
    task = canonical_task_type(sample.task_type)
    validate_artifact_identity(entry, result, sample, task, config)
    if scalar_int(result, "response_start") != response_start:
        raise ValueError(f"response boundary mismatch: {sample.sample_id}")
    count = validate_coordinates(result, token_ids, response_start, str(sample.sample_id))
    label = np.asarray(label_store.response_labels(sample), dtype=bool)[:count]
    if len(label) != count:
        raise ValueError(f"response labels do not cover all prediction events: {sample.sample_id}")

    functional = np.asarray(result["functional_role_share"], dtype=np.float64)
    functional_null = np.asarray(
        result["functional_availability_null"], dtype=np.float64
    )
    attention = np.asarray(result["attention_role_share"], dtype=np.float64)
    attention_null = np.asarray(
        result["attention_availability_null"], dtype=np.float64
    )
    expected_shape = (functional.shape[0], count, 3)
    for name, trace in (
        ("functional", functional),
        ("functional null", functional_null),
        ("attention", attention),
        ("attention null", attention_null),
    ):
        if trace.shape != expected_shape:
            raise ValueError(f"invalid {name} routing trace: {sample.sample_id}")

    functional_lift = log_lift(functional, functional_null)
    attention_lift = log_lift(attention, attention_null)
    mean_functional = layer_mean(functional_lift)
    mean_attention = layer_mean(attention_lift)
    raw = functional.mean(axis=0)
    causal_cuts = bool(scalar_int(result, "causal_cuts", 0))
    cut_names = ("direct_evidence_cut_delta", "global_evidence_cut_delta")
    if causal_cuts and any(name not in result for name in cut_names):
        raise ValueError(f"causal-cut artifact is incomplete: {sample.sample_id}")
    if not causal_cuts and any(name in result for name in cut_names):
        raise ValueError(f"unexpected causal-cut arrays: {sample.sample_id}")

    row = {
        "sample_id": str(sample.sample_id),
        "source_id": str(sample.source_id),
        "generator_model": scalar_text(
            result, "generator_model", str(getattr(sample, "generator_model", "") or "")
        ),
        "observer_model": scalar_text(result, "observer_model"),
        "response_start": response_start,
        "target_token_id": np.asarray(result["target_token_id"], dtype=np.int64),
        "claim_start": np.asarray(result["claim_start"], dtype=np.int64),
        "claim_stop": np.asarray(result["claim_stop"], dtype=np.int64),
        "claim_boundary_kind": np.asarray(result["claim_boundary_kind"], dtype=np.int8),
        "label": label,
        "evidence_enrichment": mean_functional[:, 0],
        "other_prompt_enrichment": mean_functional[:, 1],
        "history_enrichment": mean_functional[:, 2],
        "evidence_specificity": mean_functional[:, 0] - mean_functional[:, 1],
        "attention_evidence_specificity": mean_attention[:, 0] - mean_attention[:, 1],
        "attention_history_enrichment": mean_attention[:, 2],
        "raw_evidence_share": raw[:, 0],
        "raw_history_share": raw[:, 2],
        "functional_log_lift_trace": functional_lift,
        "causal_cuts": causal_cuts,
    }
    if row["claim_boundary_kind"].shape != row["claim_start"].shape:
        raise ValueError(f"claim boundary kinds do not align: {sample.sample_id}")
    for name in cut_names:
        if name in result:
            values = np.asarray(result[name], dtype=np.float64)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {name}: {sample.sample_id}")
            row[name] = values
    return row

