"""Strict artifact contracts for the causal attention SetWalk experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_protocol import (
    scalar_text,
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)
from .model import MODEL_FIELDS
from .representation import DIAGNOSTIC_NAMES, LAYER_PROFILE_NAMES, VIEW_NAMES


REFERENCE_SCHEMA = "causal-attention-setwalk-reference-v1"
SCORE_SCHEMA = "causal-attention-setwalk-nodes-v1"
EVALUATION_SCHEMA = "causal-attention-setwalk-evaluation-v1"


def _load(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _text_vector(artifact, name, *, nonempty=True):
    values = np.asarray(artifact[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a text vector")
    if nonempty and not len(values):
        raise ValueError(f"{name} must not be empty")
    return values.astype(str, copy=False)


def load_reference(path):
    artifact = _load(path)
    if scalar_text(artifact, "schema") != REFERENCE_SCHEMA:
        raise ValueError("unsupported causal SetWalk reference schema")
    required = {
        "schema",
        "train_dataset_manifest_sha256",
        "reference_source_id",
        "view_names",
        "fourier_features",
        "dct_components",
        "recent_lag_max",
        "block_rows",
        "seed",
        "epsilon",
        "reference_per_sample",
        "position_bins",
        "min_task_bin_rows",
        "trim_fraction",
        "reference_sample_id",
        "reference_token_index",
        "reference_position_bin",
        "reference_task",
    }
    for view in VIEW_NAMES:
        required.update(f"model_{view}_{field}" for field in MODEL_FIELDS)
        required.add(f"reference_embedding_{view}")
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"causal SetWalk reference misses fields: {sorted(missing)}")
    sha256_text(artifact, "train_dataset_manifest_sha256")
    views = tuple(_text_vector(artifact, "view_names").tolist())
    if views != VIEW_NAMES:
        raise ValueError("causal SetWalk reference view order is invalid")
    _text_vector(artifact, "reference_source_id")
    sample = _text_vector(artifact, "reference_sample_id")
    task = _text_vector(artifact, "reference_task")
    token = np.asarray(artifact["reference_token_index"])
    position = np.asarray(artifact["reference_position_bin"])
    if len({len(sample), len(task), len(token), len(position)}) != 1 or not len(sample):
        raise ValueError("causal SetWalk reference rows are inconsistent")
    for view in VIEW_NAMES:
        embedding = np.asarray(artifact[f"reference_embedding_{view}"])
        precision = np.asarray(artifact[f"model_{view}_precision"])
        if embedding.ndim != 2 or len(embedding) != len(sample):
            raise ValueError("causal SetWalk reference embedding rows are inconsistent")
        if precision.shape != (embedding.shape[1], embedding.shape[1]):
            raise ValueError("causal SetWalk precision geometry is inconsistent")
    float_fields = [
        name
        for name, value in artifact.items()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    ]
    if any(not bool(np.isfinite(artifact[name]).all()) for name in float_fields):
        raise ValueError("causal SetWalk reference contains non-finite values")
    return artifact


def load_score_artifact(path):
    artifact = _load(path)
    if scalar_text(artifact, "schema") != SCORE_SCHEMA:
        raise ValueError("unsupported causal SetWalk score schema")
    required = {
        "schema",
        "reference_path",
        "reference_sha256",
        "dataset_manifest_sha256",
        "reference_source_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
        "view_names",
        "diagnostic_names",
        "diagnostic_directions",
        "layer_profile_names",
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
        "token_index",
        "response_length",
        "relative_position",
        "position_bin",
        "diagnostics",
        "layer_profiles",
        "true_layer_order",
        "shuffled_layer_order",
    }
    for view in VIEW_NAMES:
        required.update((f"embedding_{view}", f"score_{view}"))
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"causal SetWalk score artifact misses: {sorted(missing)}")
    reference_path = Path(scalar_text(artifact, "reference_path"))
    if not reference_path.is_absolute():
        raise ValueError("causal SetWalk reference path must be absolute")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    if tuple(_text_vector(artifact, "view_names").tolist()) != VIEW_NAMES:
        raise ValueError("causal SetWalk score view order is invalid")
    if tuple(_text_vector(artifact, "diagnostic_names").tolist()) != DIAGNOSTIC_NAMES:
        raise ValueError("causal SetWalk diagnostics are invalid")
    if tuple(_text_vector(artifact, "layer_profile_names").tolist()) != LAYER_PROFILE_NAMES:
        raise ValueError("causal SetWalk layer profiles are invalid")

    sample_id = _text_vector(artifact, "sample_id")
    source_id = _text_vector(artifact, "source_id")
    validate_complete_token_rows(
        sample_id,
        source_id,
        artifact["token_index"],
        artifact["response_length"],
    )
    validate_source_audit(
        reserved_source_ids=artifact["reference_source_id"],
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=sample_id,
        row_source_ids=source_id,
        audit_scope=scalar_text(artifact, "audit_scope"),
    )
    rows = len(sample_id)
    row_vectors = (
        "task_type",
        "data_source",
        "generator_model",
        "token_index",
        "response_length",
        "relative_position",
        "position_bin",
    )
    if any(np.asarray(artifact[name]).shape != (rows,) for name in row_vectors):
        raise ValueError("causal SetWalk row columns are inconsistent")
    if np.asarray(artifact["diagnostics"]).shape != (rows, len(DIAGNOSTIC_NAMES)):
        raise ValueError("causal SetWalk diagnostic geometry is inconsistent")
    profiles = np.asarray(artifact["layer_profiles"])
    if profiles.ndim != 3 or profiles.shape[0] != rows or profiles.shape[2] != len(
        LAYER_PROFILE_NAMES
    ):
        raise ValueError("causal SetWalk layer profile geometry is inconsistent")
    for view in VIEW_NAMES:
        embedding = np.asarray(artifact[f"embedding_{view}"])
        score = np.asarray(artifact[f"score_{view}"])
        if embedding.ndim != 2 or len(embedding) != rows or score.shape != (rows,):
            raise ValueError("causal SetWalk view geometry is inconsistent")
    numeric = [
        name
        for name, value in artifact.items()
        if np.issubdtype(np.asarray(value).dtype, np.number)
    ]
    if any(not bool(np.isfinite(artifact[name]).all()) for name in numeric):
        raise ValueError("causal SetWalk score artifact contains non-finite values")
    return artifact

