"""Schemas and strict loaders for RR spectral experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_protocol import validate_complete_token_rows, validate_source_audit

from .representations import rr_spectral_dimension


REFERENCE_SCHEMA = "rr-spectral-reference-v2"
SCORE_SCHEMA = "rr-spectral-score-v2"
EVALUATION_SCHEMA = "rr-spectral-evaluation-v2"


def load_spectral_reference(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA:
            raise ValueError(
                "unsupported RR spectral reference; rerun the current fit command"
            )
        reference = {name: arrays[name].copy() for name in arrays.files}
    required = {
        "num_layers",
        "num_heads",
        "top_k",
        "block_rows",
        "position_bins",
        "subspace_dim",
        "reference_per_sample",
        "trim_fraction",
        "calibration_fraction",
        "split_seed",
        "channel_tail_fraction",
        "attribution_topk",
        "epsilon",
        "rr_center",
        "rr_scale",
        "channel_center",
        "channel_scale",
        "rr_pca_mean",
        "rr_pca_components",
        "rr_pca_explained_variance",
        "rr_pca_noise_variance",
        "calibration_rr_residual",
        "calibration_rr_latent",
        "calibration_rr_ppca",
        "calibration_rr_localized",
        "fit_group_id",
        "calibration_group_id",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"RR spectral reference misses fields: {sorted(missing)}")

    layers = int(reference["num_layers"])
    heads = int(reference["num_heads"])
    input_dim = rr_spectral_dimension(layers, heads, int(reference["top_k"]))
    subspace_dim = int(reference["subspace_dim"])
    if reference["rr_pca_components"].shape != (subspace_dim, input_dim):
        raise ValueError("RR PCA component geometry is inconsistent")
    if reference["rr_pca_mean"].shape != (input_dim,) or reference[
        "rr_pca_explained_variance"
    ].shape != (subspace_dim,):
        raise ValueError("RR PCA parameter geometry is inconsistent")
    if reference["rr_center"].shape != (
        int(reference["position_bins"]),
        input_dim,
    ):
        raise ValueError("RR position-control geometry is inconsistent")
    if reference["channel_center"].shape != (layers * heads,) or reference[
        "channel_scale"
    ].shape != (layers * heads,):
        raise ValueError("RR channel-control geometry is inconsistent")

    fit_groups = set(map(str, reference["fit_group_id"].tolist()))
    calibration_groups = set(
        map(str, reference["calibration_group_id"].tolist())
    )
    if not fit_groups or not calibration_groups or fit_groups & calibration_groups:
        raise ValueError("RR fit/calibration source groups are not disjoint")
    numeric = (
        "rr_center",
        "rr_scale",
        "channel_center",
        "channel_scale",
        "rr_pca_mean",
        "rr_pca_components",
        "rr_pca_explained_variance",
        "rr_pca_noise_variance",
        "calibration_rr_residual",
        "calibration_rr_latent",
        "calibration_rr_ppca",
        "calibration_rr_localized",
    )
    if any(not bool(np.isfinite(reference[name]).all()) for name in numeric):
        raise ValueError("RR spectral reference contains non-finite model values")
    if (
        bool((reference["rr_scale"] <= 0).any())
        or bool((reference["channel_scale"] <= 0).any())
        or bool((reference["rr_pca_explained_variance"] <= 0).any())
        or float(reference["rr_pca_noise_variance"]) <= 0
    ):
        raise ValueError("RR spectral reference contains a non-positive scale")
    calibration_lengths = {
        len(reference[name])
        for name in (
            "calibration_rr_residual",
            "calibration_rr_latent",
            "calibration_rr_ppca",
            "calibration_rr_localized",
        )
    }
    if len(calibration_lengths) != 1 or next(iter(calibration_lengths)) < 2:
        raise ValueError("RR calibration arrays have inconsistent lengths")
    return reference


def load_score_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if (
            "schema" not in arrays
            or np.asarray(arrays["schema"]).ndim != 0
            or str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA
        ):
            raise ValueError("unsupported RR spectral score artifact")
        artifact = {name: arrays[name].copy() for name in arrays.files}

    text_rows = {
        "sample_id",
        "source_id",
        "task_type",
        "data_source",
        "generator_model",
    }
    float_rows = {
        "score",
        "score_rr_residual",
        "score_rr_latent",
        "score_rr_ppca",
        "score_rr_localized",
        "rr_residual_energy",
        "rr_latent_energy",
        "rr_ppca_energy",
        "rr_localized_residual",
    }
    matrix_rows = {
        "rr_embedding",
        "top_channel_index",
        "top_channel_score",
    }
    row_columns = text_rows | float_rows | matrix_rows | {
        "token_index",
        "response_length",
    }
    required = row_columns | {
        "schema",
        "reference_path",
        "reference_sha256",
        "dataset_manifest_sha256",
        "fit_group_id",
        "calibration_group_id",
        "test_group_id",
        "test_sample_id",
        "audit_scope",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"RR score artifact misses fields: {sorted(missing)}")

    score = np.asarray(artifact["score"])
    if score.ndim != 1 or len(score) < 1:
        raise ValueError("RR score artifact requires a non-empty score vector")
    row_count = len(score)
    for name in row_columns:
        values = np.asarray(artifact[name])
        if values.ndim < 1 or len(values) != row_count:
            raise ValueError("RR score artifact row columns have inconsistent lengths")

    for name in text_rows:
        values = np.asarray(artifact[name])
        if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
            raise ValueError(f"RR score artifact field {name} must be text rows")
    token_index = np.asarray(artifact["token_index"])
    if token_index.ndim != 1 or not np.issubdtype(token_index.dtype, np.integer):
        raise ValueError("RR score artifact token_index must use an integer dtype")
    if bool((token_index < 0).any()):
        raise ValueError("RR score artifact token_index must be non-negative")
    response_length = np.asarray(artifact["response_length"])
    if response_length.ndim != 1 or not np.issubdtype(
        response_length.dtype, np.integer
    ):
        raise ValueError("RR score artifact response_length must use an integer dtype")

    for name in float_rows:
        values = np.asarray(artifact[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"RR score artifact field {name} must be floating rows")
    rr_embedding = np.asarray(artifact["rr_embedding"])
    if rr_embedding.ndim != 2 or rr_embedding.shape[1] < 1:
        raise ValueError("RR score artifact embedding must be a non-empty matrix")
    if not np.issubdtype(rr_embedding.dtype, np.floating):
        raise ValueError("RR score artifact embedding must use a floating dtype")
    top_channel_index = np.asarray(artifact["top_channel_index"])
    top_channel_score = np.asarray(artifact["top_channel_score"])
    if top_channel_index.ndim != 2 or top_channel_score.ndim != 2:
        raise ValueError("RR score artifact attribution fields must be matrices")
    if (
        top_channel_index.shape != top_channel_score.shape
        or top_channel_index.shape[1] < 1
    ):
        raise ValueError("RR score artifact attribution matrix geometry is inconsistent")
    if not np.issubdtype(top_channel_index.dtype, np.integer):
        raise ValueError("RR score artifact channel indices must use an integer dtype")
    if not np.issubdtype(top_channel_score.dtype, np.floating):
        raise ValueError("RR score artifact channel scores must use a floating dtype")
    if bool((top_channel_index < 0).any()):
        raise ValueError("RR score artifact channel indices must be non-negative")

    numeric = float_rows | matrix_rows
    if any(not bool(np.isfinite(artifact[name]).all()) for name in numeric):
        raise ValueError("RR score artifact contains non-finite values")
    if (
        artifact["score"].dtype != artifact["score_rr_residual"].dtype
        or not np.array_equal(artifact["score"], artifact["score_rr_residual"])
    ):
        raise ValueError("RR primary score must exactly equal score_rr_residual")

    reference_path = _scalar_artifact_text(artifact, "reference_path")
    if not reference_path:
        raise ValueError("RR score artifact has an empty reference path")
    reference_sha256 = _scalar_artifact_text(artifact, "reference_sha256")
    if len(reference_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in reference_sha256.lower()
    ):
        raise ValueError("RR score artifact has an invalid reference digest")
    dataset_sha256 = _scalar_artifact_text(artifact, "dataset_manifest_sha256")
    if len(dataset_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_sha256.lower()
    ):
        raise ValueError("RR score artifact has an invalid dataset manifest digest")
    _validate_source_audit(artifact)
    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )
    return artifact


def _scalar_artifact_text(artifact, name: str) -> str:
    value = np.asarray(artifact[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"RR score artifact field {name} must be scalar text")
    return str(value.item())


def _source_group_values(artifact, name: str) -> tuple[str, ...]:
    values = np.asarray(artifact[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError("RR score artifact has an invalid source-group audit")
    groups = tuple(map(str, values.tolist()))
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(
            not group.strip() or group.strip().lower() in {"none", "null", "nan"}
            for group in groups
        )
    ):
        raise ValueError("RR score artifact has an invalid source-group audit")
    return groups


def _validate_source_audit(artifact) -> None:
    fit_groups = _source_group_values(artifact, "fit_group_id")
    calibration_groups = _source_group_values(artifact, "calibration_group_id")
    if set(fit_groups) & set(calibration_groups):
        raise ValueError("RR score artifact has an invalid source-group audit")
    validate_source_audit(
        reserved_source_ids=fit_groups + calibration_groups,
        test_source_ids=artifact["test_group_id"],
        test_sample_ids=artifact["test_sample_id"],
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=_scalar_artifact_text(artifact, "audit_scope"),
    )
