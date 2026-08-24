"""Strict, label-free artifacts for causal typed-path De Bruijn scoring.

There is intentionally no learned-model sidecar.  A reference is a single NPZ
file containing probabilities, robust train statistics, and empirical
calibration distributions.  A score artifact contains one and only one primary
detector column named ``score`` plus mechanism diagnostics whose names cannot be
mistaken for alternative detectors by automatic benchmark adapters.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiment_protocol import (
    FrozenFile,
    TemporalScope,
    scalar_text,
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)


REFERENCE_SCHEMA = "causal-typed-path-debruijn-reference-v1"
SCORE_SCHEMA = "causal-typed-path-debruijn-score-v1"
EVALUATION_SCHEMA = "causal-typed-path-debruijn-evaluation-v1"
_ROUTE_STATE_NAMES = ("P0", "P_PLUS", "R0", "R_PLUS", "U")


REFERENCE_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "train_dataset_manifest_sha256",
        "num_layers",
        "num_heads",
        "graph_config_json",
        "debruijn_config_json",
        "change_config_json",
        "calibration_config_json",
        "fit_group_id",
        "channel_calibration_group_id",
        "fusion_calibration_group_id",
        "calibration_group_id",
        "route_state_names",
        "prior_probability",
        "transition_probability",
        "debruijn_token_count",
        "debruijn_transition_window_count",
        "surprisal_median",
        "surprisal_scale",
        "prompt_lineage_drop_median",
        "prompt_lineage_drop_scale",
        "calibration_channel_score",
        "calibration_fusion_stat",
        "calibration_independent_fusion",
        "topology_gate_mean_gap",
        "topology_gate_median_gap",
        "topology_gate_positive_fraction",
        "topology_gate_changed_edge_fraction",
        "topology_gate_pass",
        "labels_included",
    }
)

SCORE_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "source_id",
        "token_index",
        "response_length",
        "task_type",
        "data_source",
        "generator_model",
        "score",
        "fusion_stat",
        "channel_score_mean",
        "transition_surprisal_mean",
        "prompt_lineage_mean",
        "response_survival_mean",
        "rupture_mean",
        "lockin_mean",
        "conservation_error_max",
        "top_channel_index",
        "top_channel_value",
    }
)

SCORE_REQUIRED_FIELDS = SCORE_ROW_FIELDS | frozenset(
    {
        "schema",
        "reference_path",
        "reference_sha256",
        "dataset_manifest_sha256",
        "fit_group_id",
        "channel_calibration_group_id",
        "fusion_calibration_group_id",
        "calibration_group_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
        "channel_sidecar_path",
        "labels_included",
    }
)

_CONFIG_FIELDS = (
    "graph_config_json",
    "debruijn_config_json",
    "change_config_json",
    "calibration_config_json",
)
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "label",
        "labels",
        "token_label",
        "token_labels",
        "response_label",
        "response_labels",
        "hallucination_label",
        "hallucination_labels",
    }
)


def score_temporal_scope() -> TemporalScope:
    """Typed-path scores use only the current causal response prefix."""

    return TemporalScope(online_causal_score=True)


def _scalar_integer(mapping, name: str, *, minimum: int | None = None) -> int:
    """Read one required scalar integer with an optional lower bound."""

    if name not in mapping:
        raise ValueError(f"artifact is missing field {name!r}")
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"artifact field {name!r} must be a scalar integer")
    result = int(value.item())
    if minimum is not None and result < int(minimum):
        raise ValueError(f"artifact field {name!r} is below its minimum")
    return result


def _scalar_float(mapping, name: str) -> float:
    """Read one required finite scalar floating-point value."""

    if name not in mapping:
        raise ValueError(f"artifact is missing field {name!r}")
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"artifact field {name!r} must be a scalar float")
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f"artifact field {name!r} must be finite")
    return result


def _scalar_boolean(mapping, name: str) -> bool:
    """Read one required scalar NumPy boolean."""

    if name not in mapping:
        raise ValueError(f"artifact is missing field {name!r}")
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.bool_):
        raise ValueError(f"artifact field {name!r} must be a scalar boolean")
    return bool(value.item())


def _json_object(mapping, name: str) -> dict[str, Any]:
    """Parse one scalar JSON field and require an object at its root."""

    raw = scalar_text(mapping, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"artifact field {name!r} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact field {name!r} must encode a JSON object")
    return value


def _source_groups(mapping, name: str) -> tuple[str, ...]:
    """Validate one non-empty, unique vector of source-group identifiers."""

    if name not in mapping:
        raise ValueError(f"artifact is missing field {name!r}")
    values = np.asarray(mapping[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"artifact field {name!r} must be a text vector")
    groups = tuple(map(str, values.tolist()))
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(
            not group.strip()
            or group != group.strip()
            or group.lower() in {"none", "null", "nan"}
            for group in groups
        )
    ):
        raise ValueError(f"artifact field {name!r} has invalid source groups")
    return groups


def _validate_label_free(mapping) -> None:
    """Reject explicit or covert canonical label payloads."""

    if _scalar_boolean(mapping, "labels_included"):
        raise ValueError("typed-path artifacts must set labels_included=false")
    present = _FORBIDDEN_LABEL_FIELDS.intersection(mapping)
    if present:
        raise ValueError(
            "typed-path artifacts must not contain label fields: "
            f"{sorted(present)}"
        )


def _validate_three_stream_groups(
    mapping,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the three mandatory, pairwise-disjoint source streams."""

    fit = _source_groups(mapping, "fit_group_id")
    channel = _source_groups(mapping, "channel_calibration_group_id")
    fusion = _source_groups(mapping, "fusion_calibration_group_id")
    calibration = _source_groups(mapping, "calibration_group_id")
    streams = (set(fit), set(channel), set(fusion))
    if any(
        streams[left] & streams[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("fit/channel/fusion source groups must be pairwise disjoint")
    if set(calibration) != set(channel) | set(fusion):
        raise ValueError(
            "calibration_group_id must equal the channel/fusion group union"
        )
    return fit, channel, fusion


def _validate_exact_fields(mapping, expected: frozenset[str], description: str) -> None:
    """Require an exact schema so label-like or unaudited payloads cannot hide."""

    present = set(mapping)
    missing = expected.difference(present)
    if missing:
        raise ValueError(f"{description} misses fields: {sorted(missing)}")
    unknown = present.difference(expected)
    if unknown:
        raise ValueError(f"{description} contains unsupported fields: {sorted(unknown)}")


def _validate_probability_reference(reference, *, channels: int, config: dict) -> None:
    """Validate the flattened-context representation used by ``debruijn.py``."""

    prior = np.asarray(reference["prior_probability"])
    transition = np.asarray(reference["transition_probability"])
    if (
        prior.ndim != 2
        or transition.ndim != 3
        or not np.issubdtype(prior.dtype, np.floating)
        or not np.issubdtype(transition.dtype, np.floating)
    ):
        raise ValueError("De Bruijn prior/transition must be floating tensors")
    if prior.shape[0] != channels or transition.shape[0] != channels:
        raise ValueError("De Bruijn tensors disagree with layer/head channels")

    state_count = int(prior.shape[1])
    configured_order = config.get("order")
    if configured_order is None:
        raise ValueError("De Bruijn config must declare its Markov order")
    order = int(configured_order)
    if order < 1:
        raise ValueError("De Bruijn order must be positive")
    expected_transition = (channels, state_count**order, state_count)
    if state_count < 2 or transition.shape != expected_transition:
        raise ValueError(
            "transition_probability must have shape [C, M**order, M]"
        )
    if not bool(np.isfinite(prior).all()) or not bool(np.isfinite(transition).all()):
        raise ValueError("De Bruijn probabilities contain non-finite values")
    if bool(((prior < 0.0) | (prior > 1.0)).any()) or bool(
        ((transition < 0.0) | (transition > 1.0)).any()
    ):
        raise ValueError("De Bruijn probabilities must lie in [0, 1]")

    prior_sum = prior.sum(axis=-1, dtype=np.float64)
    transition_sum = transition.sum(axis=-1, dtype=np.float64)
    if not bool(np.allclose(prior_sum, 1.0, rtol=5e-4, atol=5e-4)):
        raise ValueError("prior_probability is not normalized per channel")
    if not bool(np.allclose(transition_sum, 1.0, rtol=5e-4, atol=5e-4)):
        raise ValueError("transition_probability is not normalized by next state")

    configured_states = config.get(
        "num_states",
        config.get("num_route_states"),
    )
    if configured_states is not None and int(configured_states) != state_count:
        raise ValueError("De Bruijn config state count disagrees with tensors")


def _load_npz(path, *, schema: str, description: str) -> dict[str, np.ndarray]:
    """Load one pickle-free NPZ and copy arrays away from the file handle."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != schema:
            raise ValueError(f"unsupported {description} schema")
        return {name: arrays[name].copy() for name in arrays.files}


def load_reference(path) -> dict[str, np.ndarray]:
    """Load and strictly validate one frozen typed-path reference."""

    reference = _load_npz(
        path,
        schema=REFERENCE_SCHEMA,
        description="causal typed-path De Bruijn reference",
    )
    _validate_exact_fields(
        reference,
        REFERENCE_REQUIRED_FIELDS,
        "typed-path reference",
    )
    _validate_label_free(reference)
    sha256_text(reference, "train_dataset_manifest_sha256")

    layers = _scalar_integer(reference, "num_layers", minimum=1)
    heads = _scalar_integer(reference, "num_heads", minimum=1)
    channels = layers * heads
    parsed_config = {name: _json_object(reference, name) for name in _CONFIG_FIELDS}
    _validate_three_stream_groups(reference)
    if not _scalar_boolean(reference, "calibration_independent_fusion"):
        raise ValueError(
            "v1 references require an independent fusion-calibration stream"
        )
    _validate_probability_reference(
        reference,
        channels=channels,
        config=parsed_config["debruijn_config_json"],
    )
    route_names = np.asarray(reference["route_state_names"])
    if (
        route_names.ndim != 1
        or route_names.dtype.kind not in {"U", "S"}
        or tuple(route_names.astype(str).tolist()) != _ROUTE_STATE_NAMES
    ):
        raise ValueError("route_state_names disagrees with the v1 automaton")
    token_count = _scalar_integer(reference, "debruijn_token_count", minimum=1)
    window_count = _scalar_integer(
        reference,
        "debruijn_transition_window_count",
        minimum=0,
    )
    if window_count > token_count:
        raise ValueError("De Bruijn transition windows exceed fitted tokens")

    statistic_names = (
        "surprisal_median",
        "surprisal_scale",
        "prompt_lineage_drop_median",
        "prompt_lineage_drop_scale",
    )
    for name in statistic_names:
        values = np.asarray(reference[name])
        if (
            values.shape != (channels,)
            or not np.issubdtype(values.dtype, np.floating)
            or not bool(np.isfinite(values).all())
        ):
            raise ValueError(f"reference field {name!r} must have shape [C]")
    for name in ("surprisal_scale", "prompt_lineage_drop_scale"):
        if bool((np.asarray(reference[name]) <= 0.0).any()):
            raise ValueError(f"reference field {name!r} must be strictly positive")

    channel_score = np.asarray(reference["calibration_channel_score"])
    fusion_stat = np.asarray(reference["calibration_fusion_stat"])
    if (
        channel_score.ndim != 2
        or channel_score.shape[0] < 2
        or channel_score.shape[1] != channels
        or not np.issubdtype(channel_score.dtype, np.floating)
        or not bool(np.isfinite(channel_score).all())
    ):
        raise ValueError(
            "calibration_channel_score must have finite shape [K>=2, C]"
        )
    if (
        fusion_stat.ndim != 1
        or len(fusion_stat) < 2
        or not np.issubdtype(fusion_stat.dtype, np.floating)
        or not bool(np.isfinite(fusion_stat).all())
    ):
        raise ValueError("calibration_fusion_stat must be a finite K>=2 vector")

    mean_gap = _scalar_float(reference, "topology_gate_mean_gap")
    _scalar_float(reference, "topology_gate_median_gap")
    positive_fraction = _scalar_float(
        reference,
        "topology_gate_positive_fraction",
    )
    changed_fraction = _scalar_float(
        reference,
        "topology_gate_changed_edge_fraction",
    )
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("topology gate positive fraction must lie in [0,1]")
    if not 0.0 <= changed_fraction <= 1.0:
        raise ValueError("topology changed-edge fraction must lie in [0,1]")
    gate_pass = _scalar_boolean(reference, "topology_gate_pass")
    if gate_pass != bool(mean_gap > 0.0):
        raise ValueError("topology gate pass flag disagrees with its mean gap")
    return reference


def load_score_artifact(path) -> dict[str, np.ndarray]:
    """Load a complete held-out token score artifact without reading labels."""

    artifact = _load_npz(
        path,
        schema=SCORE_SCHEMA,
        description="causal typed-path De Bruijn score",
    )
    _validate_exact_fields(
        artifact,
        SCORE_REQUIRED_FIELDS,
        "typed-path score artifact",
    )
    _validate_label_free(artifact)
    alternative_score_fields = sorted(
        name for name in artifact if name != "score" and name.startswith("score")
    )
    if alternative_score_fields:
        raise ValueError(
            "only the primary detector may use the score prefix: "
            f"{alternative_score_fields}"
        )

    primary_score = np.asarray(artifact["score"])
    if primary_score.ndim != 1 or len(primary_score) < 1:
        raise ValueError("typed-path score artifact must contain at least one row")
    row_count = len(primary_score)
    for name in SCORE_ROW_FIELDS:
        values = np.asarray(artifact[name])
        expected_dimensions = 2 if name in {
            "top_channel_index",
            "top_channel_value",
        } else 1
        if values.ndim != expected_dimensions or len(values) != row_count:
            raise ValueError("typed-path score row columns are inconsistent")
    if (
        np.asarray(artifact["top_channel_index"]).shape
        != np.asarray(artifact["top_channel_value"]).shape
        or np.asarray(artifact["top_channel_index"]).shape[1] < 1
    ):
        raise ValueError("top channel IDs and values must share non-empty [N,K]")

    text_rows = {
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
    }
    integer_rows = {"token_index", "response_length", "top_channel_index"}
    float_rows = SCORE_ROW_FIELDS.difference(text_rows | integer_rows)
    for name in text_rows:
        if np.asarray(artifact[name]).dtype.kind not in {"U", "S"}:
            raise ValueError(f"score field {name!r} must contain text rows")
    for name in integer_rows:
        values = np.asarray(artifact[name])
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"score field {name!r} must contain integer rows")
    for name in float_rows:
        values = np.asarray(artifact[name])
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"score field {name!r} must contain floating rows")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"score field {name!r} contains non-finite values")

    if bool((np.asarray(artifact["score"]) < 0.0).any()):
        raise ValueError("primary score must be non-negative")
    if bool((np.asarray(artifact["conservation_error_max"]) < 0.0).any()):
        raise ValueError("conservation_error_max must be non-negative")
    if bool((np.asarray(artifact["top_channel_index"]) < 0).any()):
        raise ValueError("top_channel_index must be non-negative")

    reference_path = scalar_text(artifact, "reference_path")
    if not reference_path.strip():
        raise ValueError("score artifact reference_path must be non-empty")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    fit, channel, fusion = _validate_three_stream_groups(artifact)
    validate_source_audit(
        reserved_source_ids=fit + channel + fusion,
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=scalar_text(artifact, "audit_scope"),
    )
    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )
    sidecars = np.asarray(artifact["channel_sidecar_path"])
    sample_count = len(np.asarray(artifact["test_sample_id"]))
    if (
        sidecars.ndim != 1
        or sidecars.dtype.kind not in {"U", "S"}
        or len(sidecars) not in {0, sample_count}
    ):
        raise ValueError(
            "channel_sidecar_path must be empty or contain one path per sample"
        )
    sidecar_text = tuple(sidecars.astype(str).tolist())
    if any(not value.strip() for value in sidecar_text) or len(set(sidecar_text)) != len(
        sidecar_text
    ):
        raise ValueError("channel_sidecar_path contains empty or duplicate paths")
    return artifact


def verify_score_provenance(artifact) -> dict[str, np.ndarray]:
    """Verify the exact reference bytes and frozen source-group binding.

    :class:`~experiment_protocol.FrozenFile` is captured before the reference is
    parsed and reverified afterwards, closing the time-of-check/time-of-use gap.
    """

    reference_file = FrozenFile.capture(scalar_text(artifact, "reference_path"))
    if reference_file.sha256 != sha256_text(artifact, "reference_sha256"):
        raise ValueError("score reference digest differs from its artifact")
    reference = load_reference(reference_file.path)
    reference_file.verify(reference_file.path)

    group_fields = (
        "fit_group_id",
        "calibration_group_id",
        "channel_calibration_group_id",
        "fusion_calibration_group_id",
    )
    for name in group_fields:
        if name not in reference:
            if name in {"channel_calibration_group_id", "fusion_calibration_group_id"}:
                raise ValueError(
                    "production score reference lacks separate calibration groups"
                )
            raise ValueError(f"score reference lacks source group field {name!r}")
        if not np.array_equal(
            np.asarray(artifact[name], dtype=str),
            np.asarray(reference[name], dtype=str),
        ):
            raise ValueError(f"score source groups differ from reference field {name!r}")

    channels = int(reference["num_layers"]) * int(reference["num_heads"])
    if bool((np.asarray(artifact["top_channel_index"]) >= channels).any()):
        raise ValueError("score top_channel_index exceeds reference geometry")
    return reference


def _normalise_payload(
    payload: Mapping[str, Any] | None,
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Merge a mapping and keyword arrays into pickle-free NumPy values."""

    merged: dict[str, Any] = {} if payload is None else dict(payload)
    overlap = set(merged).intersection(arrays)
    if overlap:
        raise ValueError(f"duplicate artifact fields: {sorted(overlap)}")
    merged.update(arrays)
    normalised = {name: np.asarray(value) for name, value in merged.items()}
    object_fields = [name for name, value in normalised.items() if value.dtype.kind == "O"]
    if object_fields:
        raise ValueError(
            "artifact fields must not require pickle: " f"{sorted(object_fields)}"
        )
    return normalised


def _atomic_npz(
    path,
    payload: Mapping[str, np.ndarray],
    *,
    validator: Callable[[Path], object],
) -> Path:
    """Validate a temporary NPZ and atomically replace its final path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        validator(temporary)
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def save_reference(
    path,
    payload: Mapping[str, Any] | None = None,
    **arrays,
) -> Path:
    """Atomically save a validated, explicitly label-free reference NPZ."""

    values = _normalise_payload(payload, arrays)
    if "schema" in values and scalar_text(values, "schema") != REFERENCE_SCHEMA:
        raise ValueError("cannot save a reference with a foreign schema")
    if "labels_included" in values and _scalar_boolean(values, "labels_included"):
        raise ValueError("cannot save labels in a typed-path reference")
    values["schema"] = np.asarray(REFERENCE_SCHEMA)
    values["labels_included"] = np.asarray(False, dtype=np.bool_)
    _validate_exact_fields(
        values,
        REFERENCE_REQUIRED_FIELDS,
        "typed-path reference",
    )
    return _atomic_npz(path, values, validator=load_reference)


def save_score_artifact(
    path,
    payload: Mapping[str, Any] | None = None,
    **arrays,
) -> Path:
    """Atomically save a validated held-out score NPZ without labels."""

    values = _normalise_payload(payload, arrays)
    if "schema" in values and scalar_text(values, "schema") != SCORE_SCHEMA:
        raise ValueError("cannot save a score artifact with a foreign schema")
    if "labels_included" in values and _scalar_boolean(values, "labels_included"):
        raise ValueError("cannot save labels in a typed-path score artifact")
    values["schema"] = np.asarray(SCORE_SCHEMA)
    values["labels_included"] = np.asarray(False, dtype=np.bool_)
    _validate_exact_fields(
        values,
        SCORE_REQUIRED_FIELDS,
        "typed-path score artifact",
    )
    return _atomic_npz(path, values, validator=load_score_artifact)


def atomic_write_json(path, value: Mapping[str, Any]) -> Path:
    """Write strict JSON through a same-directory temporary file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".json",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "EVALUATION_SCHEMA",
    "REFERENCE_SCHEMA",
    "SCORE_ROW_FIELDS",
    "SCORE_SCHEMA",
    "atomic_write_json",
    "load_reference",
    "load_score_artifact",
    "save_reference",
    "save_score_artifact",
    "score_temporal_scope",
    "verify_score_provenance",
]
