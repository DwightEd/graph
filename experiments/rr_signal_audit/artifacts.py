"""Strict frozen artifacts for the RR signal-decomposition audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_protocol import (
    FrozenFile,
    TemporalScope,
    scalar_text,
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)

REFERENCE_SCHEMA = "rr-signal-audit-reference-v1"
SCORE_SCHEMA = "rr-signal-audit-score-v1"
EVALUATION_SCHEMA = "rr-signal-audit-evaluation-v1"


def score_temporal_scope() -> TemporalScope:
    """The artifact mixes online-causal and offline-relative audit scores."""

    return TemporalScope(
        online_causal_score=False,
        future_length_conditioned_fields=(
            "relative_position",
            "relative_position_conditioned_scores",
        ),
    )


def load_reference(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != REFERENCE_SCHEMA:
            raise ValueError("unsupported RR signal-audit reference schema")
        reference = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "schema",
        "train_dataset_manifest_sha256",
        "signal_config_json",
        "geometry_config_json",
        "fit_group_id",
        "calibration_group_id",
        "num_layers",
        "num_heads",
        "task_names",
        "block_names",
        "calibration_collapse_values",
        "calibration_collapse_relative_conditions",
        "calibration_collapse_causal_conditions",
        "collapse_feature_names",
        "collapse_directions",
        "score_names",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"RR signal-audit reference misses fields: {sorted(missing)}")
    sha256_text(reference, "train_dataset_manifest_sha256")
    fit_groups = set(map(str, reference["fit_group_id"].tolist()))
    calibration_groups = set(map(str, reference["calibration_group_id"].tolist()))
    if not fit_groups or not calibration_groups or fit_groups & calibration_groups:
        raise ValueError("RR signal-audit fit/calibration groups are not disjoint")
    if int(reference["num_layers"]) < 1 or int(reference["num_heads"]) < 1:
        raise ValueError("RR signal-audit attention geometry is invalid")
    collapse = np.asarray(reference["calibration_collapse_values"])
    if collapse.ndim != 2 or len(collapse) < 2:
        raise ValueError("RR signal-audit collapse calibration is invalid")
    if any(
        not bool(np.isfinite(value).all())
        for name, value in reference.items()
        if np.asarray(value).dtype.kind in {"f", "c"}
        and "__coordination_" not in name
    ):
        raise ValueError("RR signal-audit reference contains non-finite model values")
    return reference


def load_score_artifact(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != SCORE_SCHEMA:
            raise ValueError("unsupported RR signal-audit score schema")
        artifact = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "schema",
        "reference_path",
        "reference_sha256",
        "dataset_manifest_sha256",
        "fit_group_id",
        "calibration_group_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
        "sample_id",
        "source_id",
        "token_index",
        "response_length",
        "relative_position",
        "causal_position_bucket",
        "task_type",
        "data_source",
        "generator_model",
        "score_names",
        "scores",
        "collapse_feature_names",
        "collapse_raw",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"RR signal-audit score misses fields: {sorted(missing)}")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    rows = len(artifact["sample_id"])
    row_columns = (
        "source_id",
        "token_index",
        "response_length",
        "relative_position",
        "causal_position_bucket",
        "task_type",
        "data_source",
        "generator_model",
    )
    if rows < 1 or any(len(artifact[name]) != rows for name in row_columns):
        raise ValueError("RR signal-audit score row columns are inconsistent")
    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )
    validate_source_audit(
        reserved_source_ids=np.concatenate(
            (artifact["fit_group_id"], artifact["calibration_group_id"])
        ),
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=scalar_text(artifact, "audit_scope"),
    )
    score_names = np.asarray(artifact["score_names"], dtype=str)
    scores = np.asarray(artifact["scores"])
    collapse_names = np.asarray(artifact["collapse_feature_names"], dtype=str)
    collapse = np.asarray(artifact["collapse_raw"])
    if (
        score_names.ndim != 1
        or scores.ndim != 2
        or scores.shape != (rows, len(score_names))
        or len(set(score_names.tolist())) != len(score_names)
    ):
        raise ValueError("RR signal-audit score matrix geometry is invalid")
    if (
        collapse_names.ndim != 1
        or collapse.ndim != 2
        or collapse.shape != (rows, len(collapse_names))
    ):
        raise ValueError("RR signal-audit collapse geometry is invalid")
    if not bool(np.isfinite(scores).all()) or not bool(np.isfinite(collapse).all()):
        raise ValueError("RR signal-audit score artifact contains non-finite values")
    return artifact


def verify_score_provenance(artifact):
    reference_path = Path(scalar_text(artifact, "reference_path")).resolve()
    FrozenFile(
        reference_path,
        sha256_text(artifact, "reference_sha256"),
    ).verify(reference_path)
    reference = load_reference(reference_path)
    if not np.array_equal(
        np.asarray(artifact["fit_group_id"], dtype=str),
        np.asarray(reference["fit_group_id"], dtype=str),
    ) or not np.array_equal(
        np.asarray(artifact["calibration_group_id"], dtype=str),
        np.asarray(reference["calibration_group_id"], dtype=str),
    ):
        raise ValueError("RR signal-audit source groups differ from reference")
    if not np.array_equal(
        np.asarray(artifact["score_names"], dtype=str),
        np.asarray(reference["score_names"], dtype=str),
    ):
        raise ValueError("RR signal-audit score contract differs from reference")
    return reference
