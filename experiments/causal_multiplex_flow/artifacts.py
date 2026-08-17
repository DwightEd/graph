"""Strict artifact schemas for Causal Multiplex Routing Prediction."""

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

REFERENCE_SCHEMA = "cmrp-reference-v2"
SCORE_SCHEMA = "cmrp-score-v2"
EVALUATION_SCHEMA = "cmrp-evaluation-v2"


def score_temporal_scope() -> TemporalScope:
    """CMRP scores use only the current causal response prefix."""

    return TemporalScope(online_causal_score=True)


def load_reference(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != REFERENCE_SCHEMA:
            raise ValueError("unsupported CMRP reference schema")
        reference = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "schema",
        "model_file",
        "model_sha256",
        "train_dataset_manifest_sha256",
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
        "topology_gate_evaluated_edge_count",
        "topology_gate_selected_edge_count",
        "topology_gate_coverage",
        "topology_gate_positive_fraction",
        "topology_gate_pass",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"CMRP reference misses fields: {sorted(missing)}")
    sha256_text(reference, "model_sha256")
    sha256_text(reference, "train_dataset_manifest_sha256")
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
    evaluated = int(reference["topology_gate_evaluated_edge_count"])
    selected = int(reference["topology_gate_selected_edge_count"])
    coverage = float(reference["topology_gate_coverage"])
    mean_gap = float(reference["topology_gate_mean_gap"])
    median_gap = float(reference["topology_gate_median_gap"])
    positive_fraction = float(reference["topology_gate_positive_fraction"])
    if evaluated < 0 or selected < evaluated:
        raise ValueError("CMRP topology gate edge counts are inconsistent")
    if evaluated == 0:
        if (
            coverage != 0.0
            or np.isfinite(mean_gap)
            or np.isfinite(median_gap)
            or np.isfinite(positive_fraction)
            or bool(reference["topology_gate_pass"])
        ):
            raise ValueError("CMRP empty topology gate is inconsistent")
    elif (
        not np.isfinite(mean_gap)
        or not np.isfinite(median_gap)
        or not np.isclose(coverage, evaluated / selected)
        or not 0.0 <= positive_fraction <= 1.0
        or bool(reference["topology_gate_pass"]) != bool(mean_gap > 0.0)
    ):
        raise ValueError("CMRP topology gate summary is inconsistent")
    return reference


def load_score_artifact(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != SCORE_SCHEMA:
            raise ValueError("unsupported CMRP score schema")
        artifact = {name: arrays[name].copy() for name in arrays.files}
    # Only ``score`` is exposed to the automatic conditioned-benchmark adapter.
    # Raw mechanism diagnostics deliberately avoid the ``score_`` prefix so
    # they cannot become test-selected detectors by accident.
    row_fields = {
        "sample_id",
        "source_id",
        "token_index",
        "response_length",
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
    for digest_name in (
        "reference_sha256",
        "model_sha256",
        "dataset_manifest_sha256",
    ):
        sha256_text(artifact, digest_name)
    for path_name in ("reference_path", "model_path"):
        if not scalar_text(artifact, path_name):
            raise ValueError(
                f"CMRP score artifact field {path_name} must be scalar non-empty text"
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
    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )
    return artifact


def verify_score_provenance(artifact):
    """Verify the exact CMRP reference, model, and held-out group binding."""

    reference_file = FrozenFile.capture(scalar_text(artifact, "reference_path"))
    if reference_file.sha256 != sha256_text(artifact, "reference_sha256"):
        raise ValueError("CMRP score reference digest differs from its artifact")
    reference = load_reference(reference_file.path)
    reference_file.verify(reference_file.path)

    if not np.array_equal(
        np.asarray(artifact["fit_group_id"], dtype=str),
        np.asarray(reference["fit_group_id"], dtype=str),
    ) or not np.array_equal(
        np.asarray(artifact["calibration_group_id"], dtype=str),
        np.asarray(reference["calibration_group_id"], dtype=str),
    ):
        raise ValueError("CMRP score source groups differ from its reference")

    model_file = FrozenFile.capture(scalar_text(artifact, "model_path"))
    expected_model_path = (reference_file.path.parent / scalar_text(reference, "model_file")).resolve()
    if model_file.path != expected_model_path:
        raise ValueError("CMRP score model identity differs from its reference")
    expected_model_sha256 = sha256_text(reference, "model_sha256")
    if (
        model_file.sha256 != sha256_text(artifact, "model_sha256")
        or model_file.sha256 != expected_model_sha256
    ):
        raise ValueError("CMRP score model digest differs from its reference")
    model_file.verify(model_file.path)
    return reference
