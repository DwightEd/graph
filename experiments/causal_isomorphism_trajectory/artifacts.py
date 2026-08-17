"""Strict frozen artifacts for Causal Isomorphism Trajectory Geometry."""

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
from .geometry import VARIANTS, variant_artifact


REFERENCE_SCHEMA = "citg-reference-v1"
SCORE_SCHEMA = "citg-score-v1"
EVALUATION_SCHEMA = "citg-evaluation-v1"


def score_temporal_scope() -> TemporalScope:
    """CITG uses only the current causal response prefix."""
    return TemporalScope(online_causal_score=True)


def load_reference(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != REFERENCE_SCHEMA:
            raise ValueError("unsupported CITG reference schema")
        reference = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "schema",
        "train_dataset_manifest_sha256",
        "event_config_json",
        "signature_config_json",
        "geometry_config_json",
        "fit_group_id",
        "calibration_group_id",
        "topology_gate_token_count",
        "topology_gate_evaluated_tokens",
        "topology_gate_coverage",
        "topology_gate_source_groups",
        "topology_gate_mean_gap",
        "topology_gate_median_gap",
        "topology_gate_positive_group_fraction",
        "topology_gate_ci_low",
        "topology_gate_ci_high",
        "topology_gate_pass",
    }
    for variant in VARIANTS:
        required.add(f"calibration_energy_{variant}")
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"CITG reference misses fields: {sorted(missing)}")
    sha256_text(reference, "train_dataset_manifest_sha256")
    fit = set(map(str, reference["fit_group_id"].tolist()))
    calibration = set(map(str, reference["calibration_group_id"].tolist()))
    if not fit or not calibration or fit & calibration:
        raise ValueError("CITG fit/calibration source groups are invalid")
    for variant in VARIANTS:
        values = np.asarray(reference[f"calibration_energy_{variant}"], dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not bool(np.isfinite(values).all()):
            raise ValueError(f"CITG {variant} calibration energy is invalid")
        variant_artifact(reference, variant)
    return reference


def load_score_artifact(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != SCORE_SCHEMA:
            raise ValueError("unsupported CITG score schema")
        artifact = {name: arrays[name].copy() for name in arrays.files}
    row_fields = {
        "sample_id", "source_id", "token_index", "response_length",
        "task_type", "data_source", "generator_model", "score",
        "score_static", "score_topology", "score_mass", "energy_full",
        "energy_static", "energy_topology", "energy_mass",
        "rewired_energy_full", "rewire_energy_gap", "rewire_valid",
    }
    required = row_fields | {
        "schema", "reference_path", "reference_sha256",
        "dataset_manifest_sha256", "fit_group_id", "calibration_group_id",
        "test_group_id", "test_sample_id", "audit_scope",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"CITG score artifact misses fields: {sorted(missing)}")
    row_count = len(artifact["score"])
    if row_count < 1 or any(len(artifact[name]) != row_count for name in row_fields):
        raise ValueError("CITG score rows are inconsistent")
    for name in (
        "score", "score_static", "score_topology", "score_mass",
        "energy_full", "energy_static", "energy_topology", "energy_mass",
    ):
        if not bool(np.isfinite(artifact[name]).all()):
            raise ValueError(f"CITG field {name} contains non-finite values")
    valid = np.asarray(artifact["rewire_valid"], dtype=bool)
    rewired = np.asarray(artifact["rewired_energy_full"], dtype=np.float64)
    gap = np.asarray(artifact["rewire_energy_gap"], dtype=np.float64)
    if bool(np.isinf(rewired).any()) or bool(np.isinf(gap).any()):
        raise ValueError("CITG rewire diagnostics contain infinite values")
    if bool((~np.isfinite(rewired[valid])).any()) or bool((~np.isfinite(gap[valid])).any()):
        raise ValueError("CITG valid rewires must have finite diagnostics")
    sha256_text(artifact, "reference_sha256")
    sha256_text(artifact, "dataset_manifest_sha256")
    if not scalar_text(artifact, "reference_path"):
        raise ValueError("CITG reference_path must be non-empty")
    validate_source_audit(
        reserved_source_ids=np.concatenate((artifact["fit_group_id"], artifact["calibration_group_id"])),
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=scalar_text(artifact, "audit_scope"),
    )
    validate_complete_token_rows(
        artifact["sample_id"], artifact["source_id"],
        artifact["token_index"], artifact["response_length"],
    )
    return artifact


def verify_score_provenance(artifact):
    """Verify the exact fitted reference bound to a score artifact."""
    reference_file = FrozenFile.capture(scalar_text(artifact, "reference_path"))
    if reference_file.sha256 != sha256_text(artifact, "reference_sha256"):
        raise ValueError("CITG score reference digest differs from its artifact")
    reference = load_reference(reference_file.path)
    reference_file.verify(reference_file.path)
    for field in ("fit_group_id", "calibration_group_id"):
        if not np.array_equal(
            np.asarray(artifact[field], dtype=str),
            np.asarray(reference[field], dtype=str),
        ):
            raise ValueError(
                f"CITG score source groups differ from its reference ({field})"
            )
    return reference
