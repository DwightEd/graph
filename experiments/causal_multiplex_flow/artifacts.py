"""Strict artifact schemas for Causal Multiplex Routing Prediction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REFERENCE_SCHEMA = "cmrp-reference-v1"
SCORE_SCHEMA = "cmrp-score-v1"
EVALUATION_SCHEMA = "cmrp-evaluation-v1"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(arrays, name: str) -> str:
    value = np.asarray(arrays[name])
    if value.ndim != 0:
        raise ValueError(f"artifact field {name} must be scalar text")
    return str(value.item())


def load_reference(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if _scalar_text(arrays, "schema") != REFERENCE_SCHEMA:
            raise ValueError("unsupported CMRP reference schema")
        reference = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "schema",
        "model_file",
        "model_sha256",
        "num_layers",
        "num_heads",
        "event_config_json",
        "model_config_json",
        "train_config_json",
        "fit_group_id",
        "calibration_group_id",
        "calibration_raw_route_surprise",
        "topology_gate_mean_gap",
        "topology_gate_median_gap",
        "topology_gate_positive_fraction",
        "topology_gate_count",
        "topology_gate_pass",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"CMRP reference misses fields: {sorted(missing)}")
    model_sha = _scalar_text(reference, "model_sha256")
    if len(model_sha) != 64:
        raise ValueError("CMRP reference has an invalid model digest")
    if int(reference["num_layers"]) < 1 or int(reference["num_heads"]) < 1:
        raise ValueError("CMRP reference has invalid attention geometry")
    fit_groups = set(map(str, reference["fit_group_id"].tolist()))
    calibration_groups = set(map(str, reference["calibration_group_id"].tolist()))
    if not fit_groups or not calibration_groups or fit_groups & calibration_groups:
        raise ValueError("CMRP fit/calibration groups are not disjoint")
    calibration = np.asarray(
        reference["calibration_raw_route_surprise"], dtype=np.float64
    )
    if calibration.ndim != 1 or len(calibration) < 2:
        raise ValueError("CMRP calibration score distribution is invalid")
    if not bool(np.isfinite(calibration).all()):
        raise ValueError("CMRP calibration score distribution is non-finite")
    return reference


def load_score_artifact(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if _scalar_text(arrays, "schema") != SCORE_SCHEMA:
            raise ValueError("unsupported CMRP score schema")
        artifact = {name: arrays[name].copy() for name in arrays.files}
    # Only ``score`` is exposed to the automatic conditioned-benchmark adapter.
    # Raw mechanism diagnostics deliberately avoid the ``score_`` prefix so
    # they cannot become test-selected detectors by accident.
    row_fields = {
        "sample_id",
        "source_id",
        "token_index",
        "task_type",
        "data_source",
        "generator_model",
        "score",
        "raw_route_surprise",
        "presence_nll",
        "source_nll",
        "weight_error",
        "rewired_source_nll",
        "rewire_gap",
        "selected_rr_edges",
    }
    required = row_fields | {
        "schema",
        "reference_sha256",
        "model_sha256",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"CMRP score artifact misses fields: {sorted(missing)}")
    row_count = len(artifact["score"])
    if row_count < 1 or any(len(artifact[name]) != row_count for name in row_fields):
        raise ValueError("CMRP score artifact row columns are inconsistent")
    if not bool(np.isfinite(artifact["score"]).all()):
        raise ValueError("CMRP primary score contains non-finite values")
    finite_required = (
        "raw_route_surprise",
        "presence_nll",
        "source_nll",
        "weight_error",
    )
    if any(not bool(np.isfinite(artifact[name]).all()) for name in finite_required):
        raise ValueError("CMRP required raw diagnostics contain non-finite values")
    for digest_name in ("reference_sha256", "model_sha256"):
        if len(_scalar_text(artifact, digest_name)) != 64:
            raise ValueError(f"CMRP score artifact has invalid {digest_name}")
    return artifact
