"""Strict frozen artifacts for the causal attention Set-Flow experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiment_protocol import (
    scalar_text,
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)

from .calibration import COMPONENT_NAMES


REFERENCE_SCHEMA = "causal-setflow-reference-v1"
SCORE_SCHEMA = "causal-setflow-score-v1"
EVALUATION_SCHEMA = "causal-setflow-evaluation-v1"
CHECKPOINT_SCHEMA = "causal-setflow-checkpoint-v1"


def save_checkpoint(path, payload) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save({"schema": CHECKPOINT_SCHEMA, **payload}, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported causal Set-Flow checkpoint schema")
    required = {
        "state_dict",
        "num_layers",
        "num_heads",
        "source_config",
        "model_config",
        "training_config",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"causal Set-Flow checkpoint misses: {sorted(missing)}")
    return checkpoint


def _load_npz(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def load_reference(path):
    artifact = _load_npz(path)
    if scalar_text(artifact, "schema") != REFERENCE_SCHEMA:
        raise ValueError("unsupported causal Set-Flow reference schema")
    required = {
        "schema",
        "model_path",
        "model_sha256",
        "train_dataset_manifest_sha256",
        "source_config_json",
        "model_config_json",
        "training_config_json",
        "calibration_config_json",
        "fit_group_id",
        "calibration_group_id",
        "component_names",
        "calibration_components",
        "calibration_conditions",
        "calibration_sample_id",
        "calibration_token_index",
        "latent_center",
        "latent_scale",
        "latent_precision_center",
        "latent_precision",
        "latent_trim_threshold",
        "latent_retained_rows",
        "training_history_json",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"causal Set-Flow reference misses: {sorted(missing)}")
    model_path = Path(scalar_text(artifact, "model_path"))
    if not model_path.is_absolute():
        raise ValueError("causal Set-Flow model path must be absolute")
    sha256_text(artifact, "model_sha256")
    sha256_text(artifact, "train_dataset_manifest_sha256")
    if tuple(np.asarray(artifact["component_names"], dtype=str).tolist()) != COMPONENT_NAMES:
        raise ValueError("causal Set-Flow component order changed")
    calibration = np.asarray(artifact["calibration_components"])
    conditions = np.asarray(artifact["calibration_conditions"])
    sample = np.asarray(artifact["calibration_sample_id"])
    token = np.asarray(artifact["calibration_token_index"])
    if calibration.ndim != 2 or calibration.shape[1] != len(COMPONENT_NAMES):
        raise ValueError("causal Set-Flow calibration geometry is invalid")
    if len({len(calibration), len(conditions), len(sample), len(token)}) != 1:
        raise ValueError("causal Set-Flow calibration rows are not aligned")
    precision = np.asarray(artifact["latent_precision"])
    center = np.asarray(artifact["latent_center"])
    if precision.shape != (len(center), len(center)):
        raise ValueError("causal Set-Flow latent precision geometry is invalid")
    fit = set(map(str, np.asarray(artifact["fit_group_id"], dtype=str).tolist()))
    calibration_groups = set(
        map(str, np.asarray(artifact["calibration_group_id"], dtype=str).tolist())
    )
    if not fit or not calibration_groups or fit & calibration_groups:
        raise ValueError("causal Set-Flow fit/calibration groups are invalid")
    numeric = [
        value
        for value in artifact.values()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    ]
    if any(not bool(np.isfinite(value).all()) for value in numeric):
        raise FloatingPointError("causal Set-Flow reference has non-finite values")
    return artifact


def load_score_artifact(path):
    artifact = _load_npz(path)
    if scalar_text(artifact, "schema") != SCORE_SCHEMA:
        raise ValueError("unsupported causal Set-Flow score schema")
    required = {
        "schema",
        "reference_path",
        "reference_sha256",
        "model_path",
        "model_sha256",
        "dataset_manifest_sha256",
        "fit_group_id",
        "calibration_group_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
        "component_names",
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
        "token_index",
        "response_length",
        "embedding",
        "components_raw",
        "components_tail",
        "score",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"causal Set-Flow score artifact misses: {sorted(missing)}")
    reference_path = Path(scalar_text(artifact, "reference_path"))
    model_path = Path(scalar_text(artifact, "model_path"))
    if not reference_path.is_absolute() or not model_path.is_absolute():
        raise ValueError("causal Set-Flow frozen paths must be absolute")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "model_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    if tuple(np.asarray(artifact["component_names"], dtype=str).tolist()) != COMPONENT_NAMES:
        raise ValueError("causal Set-Flow score component order changed")
    sample = np.asarray(artifact["sample_id"])
    source = np.asarray(artifact["source_id"])
    validate_complete_token_rows(
        sample, source, artifact["token_index"], artifact["response_length"]
    )
    validate_source_audit(
        reserved_source_ids=np.concatenate(
            (artifact["fit_group_id"], artifact["calibration_group_id"])
        ),
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=sample,
        row_source_ids=source,
        audit_scope=scalar_text(artifact, "audit_scope"),
    )
    rows = len(sample)
    for name in (
        "task_type",
        "data_source",
        "generator_model",
        "token_index",
        "response_length",
        "score",
    ):
        if np.asarray(artifact[name]).shape != (rows,):
            raise ValueError(f"causal Set-Flow row field {name} is inconsistent")
    embedding = np.asarray(artifact["embedding"])
    raw = np.asarray(artifact["components_raw"])
    tail = np.asarray(artifact["components_tail"])
    if embedding.ndim != 2 or len(embedding) != rows:
        raise ValueError("causal Set-Flow embedding rows are invalid")
    expected = (rows, len(COMPONENT_NAMES))
    if raw.shape != expected or tail.shape != expected:
        raise ValueError("causal Set-Flow component rows are invalid")
    if any(
        not bool(np.isfinite(value).all())
        for value in (embedding, raw, tail, np.asarray(artifact["score"]))
    ):
        raise FloatingPointError("causal Set-Flow score has non-finite values")
    return artifact
