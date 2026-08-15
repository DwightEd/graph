"""Schemas and strict loaders for RR spectral experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .representations import rr_spectral_dimension


REFERENCE_SCHEMA = "rr-spectral-reference"
SCORE_SCHEMA = "rr-spectral-score"
EVALUATION_SCHEMA = "rr-spectral-evaluation"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA:
            raise ValueError("unsupported RR spectral score artifact")
        artifact = {name: arrays[name].copy() for name in arrays.files}
    row_columns = {
        "sample_id",
        "token_index",
        "score",
        "score_rr_residual",
        "score_rr_latent",
        "score_rr_ppca",
        "score_rr_localized",
        "rr_residual_energy",
        "top_channel_score",
    }
    required = row_columns | {"reference_sha256"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"RR score artifact misses fields: {sorted(missing)}")
    row_count = len(artifact["score"])
    if any(len(artifact[name]) != row_count for name in row_columns):
        raise ValueError("RR score artifact columns have inconsistent lengths")
    reference_sha256 = str(np.asarray(artifact["reference_sha256"]).item())
    if len(reference_sha256) != 64:
        raise ValueError("RR score artifact has an invalid reference digest")
    return artifact
