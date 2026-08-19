"""Strict frozen artifacts for the causal attention signal audit."""

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
from .components import EVIDENCE_DIRECTIONS, EVIDENCE_FEATURE_NAMES, SIGNAL_BLOCKS

REFERENCE_SCHEMA = "causal-attention-signal-audit-reference"
SCORE_SCHEMA = "causal-attention-signal-audit-score"
EVALUATION_SCHEMA = "causal-attention-signal-audit-evaluation"


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
        "evidence_feature_names",
        "evidence_directions",
        "evidence_registry_json",
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
    block_names = tuple(
        map(str, np.asarray(reference["block_names"], dtype=str).tolist())
    )
    if block_names != SIGNAL_BLOCKS:
        raise ValueError("attention signal-audit block contract changed")
    collapse = np.asarray(reference["calibration_collapse_values"])
    if collapse.ndim != 2 or len(collapse) < 2:
        raise ValueError("RR signal-audit collapse calibration is invalid")
    evidence_names = tuple(
        map(str, np.asarray(reference["evidence_feature_names"], dtype=str).tolist())
    )
    evidence_directions = np.asarray(reference["evidence_directions"], dtype=np.int8)
    if evidence_names != EVIDENCE_FEATURE_NAMES:
        raise ValueError("attention signal-audit evidence contract changed")
    if not np.array_equal(evidence_directions, EVIDENCE_DIRECTIONS):
        raise ValueError("attention signal-audit evidence directions changed")
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
        "evidence_feature_names",
        "evidence_directions",
        "evidence_raw",
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
    evidence_names = np.asarray(artifact["evidence_feature_names"], dtype=str)
    evidence_directions = np.asarray(artifact["evidence_directions"], dtype=np.int8)
    evidence = np.asarray(artifact["evidence_raw"])
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
    if (
        evidence_names.ndim != 1
        or evidence_directions.shape != evidence_names.shape
        or evidence.shape != (rows, len(evidence_names))
    ):
        raise ValueError("attention signal-audit evidence geometry is invalid")
    if not set(map(int, evidence_directions.tolist())).issubset({-1, 1}):
        raise ValueError("attention signal-audit evidence directions are invalid")
    if tuple(map(str, evidence_names.tolist())) != EVIDENCE_FEATURE_NAMES:
        raise ValueError("attention signal-audit evidence names changed")
    if not np.array_equal(evidence_directions, EVIDENCE_DIRECTIONS):
        raise ValueError("attention signal-audit evidence directions changed")
    if (
        not bool(np.isfinite(scores).all())
        or not bool(np.isfinite(collapse).all())
        or not bool(np.isfinite(evidence).all())
    ):
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
    if not np.array_equal(
        np.asarray(artifact["evidence_feature_names"], dtype=str),
        np.asarray(reference["evidence_feature_names"], dtype=str),
    ) or not np.array_equal(
        np.asarray(artifact["evidence_directions"], dtype=np.int8),
        np.asarray(reference["evidence_directions"], dtype=np.int8),
    ):
        raise ValueError("attention signal-audit evidence contract differs from reference")
    return reference
