"""Strict schemas for RR topology-dynamics experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_protocol import validate_complete_token_rows, validate_source_audit


REFERENCE_SCHEMA = "rr-topology-dynamics-reference-v2"
SCORE_SCHEMA = "rr-topology-dynamics-features-v2"
EVALUATION_SCHEMA = "rr-topology-dynamics-evaluation-v2"


def _scalar_text(artifact, name: str) -> str:
    value = np.asarray(artifact[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"RR topology field {name} must be scalar text")
    return str(value.item())


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _source_groups(artifact, name: str) -> tuple[str, ...]:
    values = np.asarray(artifact[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError("RR topology reference source groups are invalid")
    groups = tuple(map(str, values.tolist()))
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(
            group != group.strip()
            or not group
            or group.lower() in {"none", "null", "nan"}
            for group in groups
        )
    ):
        raise ValueError("RR topology reference source groups are invalid")
    return groups


def load_topology_reference(path):
    """Load and validate one frozen topology reference."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        if (
            "schema" not in arrays
            or np.asarray(arrays["schema"]).ndim != 0
            or str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA
        ):
            raise ValueError("unsupported RR topology-dynamics reference schema")
        reference = {name: arrays[name].copy() for name in arrays.files}

    required = {
        "schema",
        "spectral_reference_path",
        "spectral_reference_sha256",
        "reference_source_id",
        "feature_names",
        "lag_bins",
        "spectral_top_k",
        "block_rows",
        "position_bins",
        "top_source_count",
        "recent_lag_max",
        "mid_lag_max",
        "far_lag_fraction",
        "epsilon",
        "reference_per_sample",
        "min_task_bin_rows",
        "phase_bins",
        "onset_window",
        "bootstrap_replicates",
        "seed",
        "reference_features",
        "reference_position_bin",
        "reference_task",
        "reference_sample_id",
        "reference_token_index",
        "global_center",
        "global_scale",
        "position_center",
        "position_scale",
        "task_names",
        "task_center",
        "task_scale",
        "task_count",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"RR topology reference misses fields: {sorted(missing)}")

    spectral_path = Path(_scalar_text(reference, "spectral_reference_path"))
    if not spectral_path.is_absolute():
        raise ValueError("RR topology spectral reference path must be absolute")
    if not _valid_digest(_scalar_text(reference, "spectral_reference_sha256")):
        raise ValueError("RR topology reference has an invalid spectral digest")
    _source_groups(reference, "reference_source_id")

    feature_names = np.asarray(reference["feature_names"])
    features = np.asarray(reference["reference_features"])
    if (
        feature_names.ndim != 1
        or feature_names.dtype.kind not in {"U", "S"}
        or not len(feature_names)
        or len(set(map(str, feature_names.tolist()))) != len(feature_names)
    ):
        raise ValueError("RR topology feature names are invalid")
    if features.ndim != 2 or features.shape[1] != len(feature_names) or not len(features):
        raise ValueError("RR topology reference feature geometry is inconsistent")
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("RR topology reference features must use a floating dtype")

    row_count = len(features)
    row_vectors = {
        "reference_position_bin": np.integer,
        "reference_task": np.str_,
        "reference_sample_id": np.str_,
        "reference_token_index": np.integer,
    }
    for name, dtype in row_vectors.items():
        values = np.asarray(reference[name])
        if values.ndim != 1 or len(values) != row_count:
            raise ValueError("RR topology reference row columns are inconsistent")
        if dtype is np.str_ and values.dtype.kind not in {"U", "S"}:
            raise ValueError(f"RR topology reference field {name} must be text")
        if dtype is np.integer and not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"RR topology reference field {name} must be integer")

    position_bins = int(reference["position_bins"])
    task_names = np.asarray(reference["task_names"])
    feature_dim = len(feature_names)
    if task_names.ndim != 1 or task_names.dtype.kind not in {"U", "S"}:
        raise ValueError("RR topology task names are invalid")
    expected = {
        "global_center": (feature_dim,),
        "global_scale": (feature_dim,),
        "position_center": (position_bins, feature_dim),
        "position_scale": (position_bins, feature_dim),
        "task_center": (len(task_names), position_bins, feature_dim),
        "task_scale": (len(task_names), position_bins, feature_dim),
        "task_count": (len(task_names), position_bins),
    }
    for name, shape in expected.items():
        if np.asarray(reference[name]).shape != shape:
            raise ValueError("RR topology reference model geometry is inconsistent")

    float_fields = {
        "reference_features",
        "global_center",
        "global_scale",
        "position_center",
        "position_scale",
        "task_center",
        "task_scale",
    }
    if any(
        not np.issubdtype(np.asarray(reference[name]).dtype, np.floating)
        for name in float_fields
    ):
        raise ValueError("RR topology reference model must use floating dtypes")
    if any(not bool(np.isfinite(reference[name]).all()) for name in float_fields):
        raise ValueError("RR topology reference contains non-finite values")
    if any(
        bool((np.asarray(reference[name]) <= 0).any())
        for name in ("global_scale", "position_scale", "task_scale")
    ):
        raise ValueError("RR topology reference contains a non-positive scale")
    return reference


def load_topology_artifact(path):
    """Load and validate one frozen full-split topology feature artifact."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        if (
            "schema" not in arrays
            or np.asarray(arrays["schema"]).ndim != 0
            or str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA
        ):
            raise ValueError("unsupported RR topology-dynamics feature schema")
        artifact = {name: arrays[name].copy() for name in arrays.files}

    text_rows = {
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
    }
    integer_rows = {"token_index", "response_length", "position_bin"}
    float_rows = {"relative_position"}
    matrix_rows = {
        "features_raw",
        "features_z",
        "layer_route_effective_rank",
        "layer_route_consensus",
        "layer_residual_energy",
        "spectral_rank_residual_energy",
        "rr_embedding",
    }
    row_columns = text_rows | integer_rows | float_rows | matrix_rows
    required = row_columns | {
        "schema",
        "spectral_reference_path",
        "spectral_reference_sha256",
        "topology_reference_path",
        "topology_reference_sha256",
        "dataset_manifest_sha256",
        "reference_source_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
        "feature_names",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"RR topology artifact misses fields: {sorted(missing)}")

    for name in ("spectral_reference_path", "topology_reference_path"):
        if not Path(_scalar_text(artifact, name)).is_absolute():
            raise ValueError(f"RR topology artifact field {name} must be absolute")
    for name in ("spectral_reference_sha256", "topology_reference_sha256"):
        if not _valid_digest(_scalar_text(artifact, name)):
            raise ValueError(f"RR topology artifact field {name} is not a digest")
    if not _valid_digest(_scalar_text(artifact, "dataset_manifest_sha256")):
        raise ValueError("RR topology artifact dataset manifest is not a digest")

    feature_names = np.asarray(artifact["feature_names"])
    if (
        feature_names.ndim != 1
        or feature_names.dtype.kind not in {"U", "S"}
        or not len(feature_names)
        or len(set(map(str, feature_names.tolist()))) != len(feature_names)
    ):
        raise ValueError("RR topology artifact feature names are invalid")
    raw = np.asarray(artifact["features_raw"])
    if raw.ndim != 2 or not len(raw):
        raise ValueError("RR topology artifact requires a non-empty feature matrix")
    row_count = len(raw)
    for name in row_columns:
        values = np.asarray(artifact[name])
        if values.ndim < 1 or len(values) != row_count:
            raise ValueError("RR topology artifact row columns are inconsistent")

    for name in text_rows:
        values = np.asarray(artifact[name])
        if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
            raise ValueError(f"RR topology artifact field {name} must be text rows")
    for name in integer_rows:
        values = np.asarray(artifact[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"RR topology artifact field {name} must be integer rows")
        if bool((values < 0).any()):
            raise ValueError(f"RR topology artifact field {name} must be non-negative")
    for name in float_rows:
        values = np.asarray(artifact[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"RR topology artifact field {name} must be floating rows")
    for name in matrix_rows:
        values = np.asarray(artifact[name])
        if (
            values.ndim != 2
            or values.shape[1] < 1
            or not np.issubdtype(values.dtype, np.floating)
        ):
            raise ValueError(f"RR topology artifact field {name} must be a float matrix")

    if raw.shape != np.asarray(artifact["features_z"]).shape or raw.shape[1] != len(
        feature_names
    ):
        raise ValueError("RR topology raw/z feature geometry is inconsistent")
    layer_shape = np.asarray(artifact["layer_route_effective_rank"]).shape
    if any(
        np.asarray(artifact[name]).shape != layer_shape
        for name in ("layer_route_consensus", "layer_residual_energy")
    ):
        raise ValueError("RR topology layer feature geometry is inconsistent")
    numeric = float_rows | matrix_rows
    if any(not bool(np.isfinite(artifact[name]).all()) for name in numeric):
        raise ValueError("RR topology artifact contains non-finite values")
    validate_source_audit(
        reserved_source_ids=artifact["reference_source_id"],
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=_scalar_text(artifact, "audit_scope"),
    )
    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )
    return artifact
