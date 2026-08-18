"""Strict frozen artifacts for Mechanism-Guided Causal Attention Set-Flow."""

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

from .calibration import ENERGY_NAMES
from .config import CORRUPTION_NAMES


REFERENCE_SCHEMA = "mechanism-guided-causal-setflow-reference-v2"
SCORE_SCHEMA = "mechanism-guided-causal-setflow-score-v2"
EVALUATION_SCHEMA = "mechanism-guided-causal-setflow-evaluation-v2"
CHECKPOINT_SCHEMA = "mechanism-guided-causal-setflow-checkpoint-v2"


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
        raise ValueError("unsupported MG-CASF checkpoint schema")
    required = {
        "state_dict",
        "num_layers",
        "num_heads",
        "source_config",
        "model_config",
        "corruption_config",
        "training_config",
        "corruption_names",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"MG-CASF checkpoint misses: {sorted(missing)}")
    if tuple(checkpoint["corruption_names"]) != CORRUPTION_NAMES:
        raise ValueError("MG-CASF checkpoint corruption order changed")
    return checkpoint


def _load_npz(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def load_reference(path):
    artifact = _load_npz(path)
    if scalar_text(artifact, "schema") != REFERENCE_SCHEMA:
        raise ValueError("unsupported MG-CASF reference schema")
    required = {
        "schema",
        "model_path",
        "model_sha256",
        "train_dataset_manifest_sha256",
        "source_config_json",
        "model_config_json",
        "corruption_config_json",
        "training_config_json",
        "calibration_config_json",
        "fit_group_id",
        "calibration_group_id",
        "energy_names",
        "corruption_names",
        "calibration_energy",
        "calibration_conditions",
        "calibration_sample_id",
        "calibration_token_index",
        "training_history_json",
        "fit_samples",
        "calibration_samples",
        "calibration_tokens",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"MG-CASF reference misses: {sorted(missing)}")
    model_path = Path(scalar_text(artifact, "model_path"))
    if not model_path.is_absolute():
        raise ValueError("MG-CASF model path must be absolute")
    sha256_text(artifact, "model_sha256")
    sha256_text(artifact, "train_dataset_manifest_sha256")
    if tuple(np.asarray(artifact["energy_names"], dtype=str).tolist()) != ENERGY_NAMES:
        raise ValueError("MG-CASF energy order changed")
    if tuple(np.asarray(artifact["corruption_names"], dtype=str).tolist()) != CORRUPTION_NAMES:
        raise ValueError("MG-CASF corruption order changed")
    energy = np.asarray(artifact["calibration_energy"])
    conditions = np.asarray(artifact["calibration_conditions"])
    sample = np.asarray(artifact["calibration_sample_id"])
    token = np.asarray(artifact["calibration_token_index"])
    if energy.ndim != 2 or energy.shape[1] != len(ENERGY_NAMES):
        raise ValueError("MG-CASF calibration energy geometry is invalid")
    if len({len(energy), len(conditions), len(sample), len(token)}) != 1:
        raise ValueError("MG-CASF calibration rows are not aligned")
    fit = set(map(str, np.asarray(artifact["fit_group_id"], dtype=str).tolist()))
    calibration = set(
        map(str, np.asarray(artifact["calibration_group_id"], dtype=str).tolist())
    )
    if not fit or not calibration or fit & calibration:
        raise ValueError("MG-CASF fit/calibration groups are invalid")
    numeric = [
        np.asarray(value)
        for value in artifact.values()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    ]
    if any(not bool(np.isfinite(value).all()) for value in numeric):
        raise FloatingPointError("MG-CASF reference contains non-finite values")
    return artifact


def load_score_artifact(path):
    artifact = _load_npz(path)
    if scalar_text(artifact, "schema") != SCORE_SCHEMA:
        raise ValueError("unsupported MG-CASF score schema")
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
        "energy_names",
        "corruption_names",
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
        "token_index",
        "response_length",
        "embedding",
        "energy_raw",
        "energy_tail",
        "score",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"MG-CASF score artifact misses: {sorted(missing)}")
    reference_path = Path(scalar_text(artifact, "reference_path"))
    model_path = Path(scalar_text(artifact, "model_path"))
    if not reference_path.is_absolute() or not model_path.is_absolute():
        raise ValueError("MG-CASF frozen paths must be absolute")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "model_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    if tuple(np.asarray(artifact["energy_names"], dtype=str).tolist()) != ENERGY_NAMES:
        raise ValueError("MG-CASF score energy order changed")
    if tuple(np.asarray(artifact["corruption_names"], dtype=str).tolist()) != CORRUPTION_NAMES:
        raise ValueError("MG-CASF score corruption order changed")

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
            raise ValueError(f"MG-CASF row field {name} is inconsistent")
    embedding = np.asarray(artifact["embedding"])
    raw = np.asarray(artifact["energy_raw"])
    tail = np.asarray(artifact["energy_tail"])
    if embedding.ndim != 2 or len(embedding) != rows:
        raise ValueError("MG-CASF embedding rows are invalid")
    expected = (rows, len(ENERGY_NAMES))
    if raw.shape != expected or tail.shape != expected:
        raise ValueError("MG-CASF energy rows are invalid")
    if any(
        not bool(np.isfinite(value).all())
        for value in (embedding, raw, tail, np.asarray(artifact["score"]))
    ):
        raise FloatingPointError("MG-CASF score contains non-finite values")
    return artifact