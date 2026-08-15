"""Train-only robust PCA geometry and independent score calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class SubspaceProjection:
    """Token coordinates and complementary distances to a fitted subspace."""

    embedding: np.ndarray
    residual_energy: np.ndarray
    latent_energy: np.ndarray
    ppca_energy: np.ndarray
    residual_vector: np.ndarray


def robust_location_scale(values: np.ndarray, *, epsilon: float = 1e-6):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("robust statistics require a non-empty matrix")
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    std = np.std(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(std > epsilon, std, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def position_location_scale(
    values: np.ndarray,
    bins: np.ndarray,
    count: int,
):
    """Fit position controls, falling back to the global train distribution."""
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    if len(values) != len(bins):
        raise ValueError("values and position bins must have equal length")
    global_center, global_scale = robust_location_scale(values)
    centers = np.empty((count, values.shape[1]), dtype=np.float32)
    scales = np.empty_like(centers)
    for position_bin in range(count):
        selected = bins == position_bin
        if int(selected.sum()) >= 2:
            centers[position_bin], scales[position_bin] = robust_location_scale(
                values[selected]
            )
        else:
            centers[position_bin] = global_center
            scales[position_bin] = global_scale
    return centers, scales


def standardize_by_position(values, bins, center, scale):
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    return ((values - np.asarray(center)[bins]) / np.asarray(scale)[bins]).astype(
        np.float32,
        copy=False,
    )


def _fit_pca(values: np.ndarray, requested_dim: int, *, seed: int) -> PCA:
    values = np.asarray(values, dtype=np.float32)
    # Leave at least one sample and one feature outside the fitted subspace so
    # orthogonal residual and PPCA noise are both defined.
    dimension = min(int(requested_dim), len(values) - 2, values.shape[1] - 1)
    if dimension < 1:
        raise ValueError(
            "subspace fitting needs at least three fit rows and two coordinates"
        )
    model = PCA(
        n_components=dimension,
        svd_solver="randomized",
        random_state=int(seed),
    )
    model.fit(values)
    return model


def fit_robust_pca(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    requested_dim: int,
    trim_fraction: float,
    seed: int,
    epsilon: float = 1e-8,
):
    """Fit a two-pass PCA after fixed per-position residual trimming.

    Trimming is an unlabeled contamination guard.  It is applied only to the
    fit stream; the disjoint calibration stream is never filtered.
    """
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    provisional = _fit_pca(
        values,
        min(int(requested_dim), 16),
        seed=seed,
    )
    provisional_reference = pca_artifact(provisional, epsilon=epsilon)
    residual = project_subspace(values, provisional_reference).residual_energy
    keep = np.ones(len(values), dtype=bool)
    if float(trim_fraction) < 1.0:
        global_threshold = float(np.quantile(residual, trim_fraction))
        for position_bin in np.unique(bins):
            selected = np.flatnonzero(bins == position_bin)
            threshold = (
                float(np.quantile(residual[selected], trim_fraction))
                if len(selected) >= 8
                else global_threshold
            )
            keep[selected] = residual[selected] <= threshold
    if int(keep.sum()) < 3:
        raise ValueError("robust trimming left too few fit rows")
    return _fit_pca(values[keep], requested_dim, seed=seed), keep, residual


def pca_artifact(model: PCA, *, epsilon: float):
    explained = np.maximum(
        model.explained_variance_.astype(np.float64), epsilon
    ).astype(np.float32)
    noise = max(float(model.noise_variance_), float(epsilon))
    return {
        "rr_pca_mean": model.mean_.astype(np.float32),
        "rr_pca_components": model.components_.astype(np.float32),
        "rr_pca_explained_variance": explained,
        "rr_pca_noise_variance": np.asarray(noise, dtype=np.float32),
        "rr_pca_whiten_scale": np.sqrt(explained).astype(np.float32),
    }


def project_subspace(values: np.ndarray, reference) -> SubspaceProjection:
    """Project tokens and retain orthogonal and in-subspace geometry.

    ``residual_energy`` detects escape orthogonal to the dominant train
    subspace. ``latent_energy`` detects extreme coordinates within it.
    ``ppca_energy`` is their probabilistic-PCA quadratic form, normalized by
    input dimension; it is diagnostic and is not fused into the primary score.
    """
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(reference["rr_pca_mean"], dtype=np.float32)
    components = np.asarray(reference["rr_pca_components"], dtype=np.float32)
    explained = np.asarray(
        reference["rr_pca_explained_variance"], dtype=np.float32
    )
    noise = max(float(reference["rr_pca_noise_variance"]), 1e-12)
    centered = values - mean
    scores = centered @ components.T
    residual_vector = centered - scores @ components
    residual_sum = np.square(residual_vector).sum(axis=1)
    latent_terms = np.square(scores) / explained
    input_dim = values.shape[1]
    return SubspaceProjection(
        embedding=(scores / np.sqrt(explained)).astype(np.float32),
        residual_energy=(residual_sum / input_dim).astype(np.float32),
        latent_energy=latent_terms.mean(axis=1).astype(np.float32),
        ppca_energy=(
            (latent_terms.sum(axis=1) + residual_sum / noise) / input_dim
        ).astype(np.float32),
        residual_vector=residual_vector.astype(np.float32),
    )


def empirical_upper_tail(reference_values, values, *, epsilon: float = 1e-12):
    """Map a magnitude to a monotone finite-sample ``-log(p)`` score."""
    reference = np.asarray(reference_values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) < 2:
        raise ValueError("calibration requires at least two finite values")
    result = np.full(len(values), np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    count_ge = len(reference) - np.searchsorted(
        reference,
        values[finite],
        side="left",
    )
    probability = (count_ge + 1.0) / (len(reference) + 1.0)
    result[finite] = -np.log(np.maximum(probability, epsilon)).astype(np.float32)
    return result
