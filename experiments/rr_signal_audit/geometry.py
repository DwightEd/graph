"""Conditioned one-class geometry and channel-alignment controls for RR signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experiments.spectral_feasibility.subspace import (
    empirical_upper_tail,
    fit_robust_pca,
    pca_artifact,
    project_subspace,
    robust_location_scale,
)


CONDITION_MODES = ("relative", "causal")
SCORE_KINDS = ("residual", "independent_nll", "ppca_nll")


@dataclass(frozen=True)
class RRGeometryConfig:
    """Reference sampling, robust factor geometry, and audit controls."""

    relative_position_bins: int = 4
    reservoir_rows: int = 4096
    pca_dim: int = 32
    min_condition_rows: int = 32
    trim_fraction: float = 0.90
    calibration_fraction: float = 0.25
    bootstrap_replicates: int = 500
    seed: int = 20260818
    epsilon: float = 1e-8

    def validate(self) -> None:
        integer_fields = (
            self.relative_position_bins,
            self.reservoir_rows,
            self.pca_dim,
            self.min_condition_rows,
        )
        if min(map(int, integer_fields)) < 1:
            raise ValueError("RR geometry integer settings must be positive")
        if not 0.5 <= float(self.trim_fraction) <= 1.0:
            raise ValueError("trim_fraction must be in [0.5,1]")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if int(self.bootstrap_replicates) < 0:
            raise ValueError("bootstrap_replicates must be non-negative")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


def condition_keys(task_type, position_bin) -> np.ndarray:
    task = np.asarray(task_type, dtype=str)
    position = np.asarray(position_bin, dtype=np.int64)
    if task.ndim != 1 or position.ndim != 1 or len(task) != len(position):
        raise ValueError("condition inputs are not aligned")
    return np.asarray(
        [
            f"{name}\x1f{int(value)}"
            for name, value in zip(task, position, strict=True)
        ],
        dtype=str,
    )


def relative_position_bins(relative_position, bins: int) -> np.ndarray:
    values = np.asarray(relative_position, dtype=np.float64)
    if values.ndim != 1 or bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("relative position must be a vector in [0,1]")
    return np.minimum(
        (values * int(bins)).astype(np.int64),
        int(bins) - 1,
    ).astype(np.int16)


def _fit_conditioned_scaler(
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    min_rows: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=str)
    if values.ndim != 2 or len(values) != len(conditions) or len(values) < 3:
        raise ValueError("conditioned scaling needs an aligned non-empty matrix")
    global_center, global_scale = robust_location_scale(values)
    names = np.asarray(sorted(set(conditions.tolist())), dtype=str)
    centers = np.empty((len(names), values.shape[1]), dtype=np.float32)
    scales = np.empty_like(centers)
    counts = np.empty(len(names), dtype=np.int32)
    for index, name in enumerate(names):
        selected = conditions == name
        counts[index] = int(selected.sum())
        if counts[index] >= int(min_rows):
            centers[index], scales[index] = robust_location_scale(values[selected])
        else:
            centers[index] = global_center
            scales[index] = global_scale
    return {
        "condition_names": names,
        "condition_center": centers,
        "condition_scale": scales,
        "condition_count": counts,
        "global_center": global_center,
        "global_scale": global_scale,
    }


def _standardize(
    values: np.ndarray,
    conditions: np.ndarray,
    scaler: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=str)
    if values.ndim != 2 or len(values) != len(conditions):
        raise ValueError("standardization inputs are not aligned")
    lookup = {
        str(name): index
        for index, name in enumerate(
            np.asarray(scaler["condition_names"], dtype=str).tolist()
        )
    }
    centers = np.asarray(scaler["condition_center"], dtype=np.float32)
    scales = np.asarray(scaler["condition_scale"], dtype=np.float32)
    global_center = np.asarray(scaler["global_center"], dtype=np.float32)
    global_scale = np.asarray(scaler["global_scale"], dtype=np.float32)
    result = np.empty_like(values)
    for name in np.unique(conditions):
        selected = conditions == name
        index = lookup.get(str(name))
        center = global_center if index is None else centers[index]
        scale = global_scale if index is None else scales[index]
        result[selected] = (values[selected] - center) / scale
    return result.astype(np.float32, copy=False)


def _independent_parameters(
    standardized: np.ndarray,
    keep: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.asarray(standardized, dtype=np.float64)[np.asarray(keep, dtype=bool)]
    mean = retained.mean(axis=0)
    variance = np.maximum(retained.var(axis=0), float(epsilon))
    return mean.astype(np.float32), variance.astype(np.float32)


def independent_nll(
    standardized: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(standardized, dtype=np.float64)
    mean = np.asarray(model["independent_mean"], dtype=np.float64)
    variance = np.maximum(
        np.asarray(model["independent_variance"], dtype=np.float64),
        1e-12,
    )
    terms = (
        np.square(values - mean) / variance
        + np.log(variance)
        + np.log(2.0 * np.pi)
    )
    return (0.5 * terms.mean(axis=1)).astype(np.float32)


def ppca_nll(
    standardized: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(standardized, dtype=np.float64)
    mean = np.asarray(model["rr_pca_mean"], dtype=np.float64)
    components = np.asarray(model["rr_pca_components"], dtype=np.float64)
    explained = np.maximum(
        np.asarray(model["rr_pca_explained_variance"], dtype=np.float64),
        1e-12,
    )
    noise = max(float(model["rr_pca_noise_variance"]), 1e-12)
    centered = values - mean
    scores = centered @ components.T
    residual = centered - scores @ components
    quadratic = (np.square(scores) / explained).sum(axis=1)
    quadratic = quadratic + np.square(residual).sum(axis=1) / noise
    dimension = values.shape[1]
    latent = explained.shape[0]
    log_determinant = (
        np.log(explained).sum()
        + max(dimension - latent, 0) * np.log(noise)
    )
    return (
        0.5
        * (
            quadratic
            + log_determinant
            + dimension * np.log(2.0 * np.pi)
        )
        / float(dimension)
    ).astype(np.float32)


def fit_geometry(
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    config: RRGeometryConfig,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    scaler = _fit_conditioned_scaler(
        values,
        conditions,
        min_rows=config.min_condition_rows,
    )
    standardized = _standardize(values, conditions, scaler)
    condition_lookup = {
        name: index
        for index, name in enumerate(
            np.asarray(scaler["condition_names"], dtype=str).tolist()
        )
    }
    condition_index = np.asarray(
        [condition_lookup.get(str(value), -1) for value in conditions],
        dtype=np.int16,
    )
    model, keep, provisional = fit_robust_pca(
        standardized,
        condition_index,
        requested_dim=config.pca_dim,
        trim_fraction=config.trim_fraction,
        seed=config.seed + int(seed_offset),
        epsilon=config.epsilon,
    )
    independent_mean, independent_variance = _independent_parameters(
        standardized,
        keep,
        epsilon=config.epsilon,
    )
    return {
        **scaler,
        **pca_artifact(model, epsilon=config.epsilon),
        "independent_mean": independent_mean,
        "independent_variance": independent_variance,
        "fit_rows": np.asarray(len(values), dtype=np.int32),
        "retained_fit_rows": np.asarray(int(keep.sum()), dtype=np.int32),
        "provisional_residual_median": np.asarray(
            float(np.median(provisional)),
            dtype=np.float32,
        ),
    }


def project_geometry(
    values: np.ndarray,
    conditions: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    standardized = _standardize(values, conditions, model)
    projection = project_subspace(standardized, model)
    return {
        "standardized": standardized,
        "residual": projection.residual_energy.astype(np.float32, copy=False),
        "independent_nll": independent_nll(standardized, model),
        "ppca_nll": ppca_nll(standardized, model),
    }


def flatten_model(prefix: str, model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}": np.asarray(value)
        for name, value in model.items()
    }


def unflatten_model(reference: Mapping[str, np.ndarray], prefix: str):
    marker = f"{prefix}__"
    result = {
        name[len(marker):]: np.asarray(value)
        for name, value in reference.items()
        if name.startswith(marker)
    }
    required = {
        "condition_names",
        "condition_center",
        "condition_scale",
        "global_center",
        "global_scale",
        "rr_pca_mean",
        "rr_pca_components",
        "rr_pca_explained_variance",
        "rr_pca_noise_variance",
        "independent_mean",
        "independent_variance",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"geometry {prefix} misses fields: {sorted(missing)}")
    return result


def calibration_fields(
    prefix: str,
    projected: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__calibration_{kind}": np.asarray(projected[kind])
        for kind in SCORE_KINDS
    }


def calibrated_scores(
    prefix: str,
    projected: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    score_prefix = prefix.replace("__", ".")
    return {
        f"{score_prefix}.{kind}_tail": empirical_upper_tail(
            np.asarray(reference[f"{prefix}__calibration_{kind}"]),
            np.asarray(projected[kind]),
        )
        for kind in SCORE_KINDS
    }


def shuffle_channel_blocks(
    values: np.ndarray,
    *,
    num_channels: int,
    features_per_channel: int,
    conditions: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Preserve every conditional channel marginal while breaking row alignment."""

    values = np.asarray(values, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=str)
    expected = int(num_channels) * int(features_per_channel)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError("channel shuffle input does not match block geometry")
    if len(values) != len(conditions):
        raise ValueError("channel shuffle conditions are not aligned")
    tensor = values.reshape(len(values), int(num_channels), int(features_per_channel))
    shuffled = np.empty_like(tensor)
    rng = np.random.default_rng(int(seed))
    for condition in np.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        for channel in range(int(num_channels)):
            if len(rows) < 2:
                shuffled[rows, channel] = tensor[rows, channel]
                continue
            order = rng.permutation(rows)
            shuffled[rows, channel] = tensor[order, channel]
    return shuffled.reshape(values.shape)


def cluster_gap_summary(
    gap: np.ndarray,
    group_id: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    gap = np.asarray(gap, dtype=np.float64)
    group_id = np.asarray(group_id, dtype=str)
    finite = np.isfinite(gap)
    gap = gap[finite]
    groups = group_id[finite]
    if len(gap) == 0:
        return {
            "rows": 0,
            "groups": 0,
            "mean_gap": None,
            "median_gap": None,
            "positive_group_fraction": None,
            "ci_low": None,
            "ci_high": None,
            "pass": False,
        }
    unique = np.unique(groups)
    group_means = np.asarray(
        [float(gap[groups == group].mean()) for group in unique],
        dtype=np.float64,
    )
    ci_low = ci_high = None
    if int(bootstrap_replicates) > 0 and len(group_means) >= 2:
        rng = np.random.default_rng(int(seed))
        estimates = np.empty(int(bootstrap_replicates), dtype=np.float64)
        for index in range(int(bootstrap_replicates)):
            selected = rng.integers(0, len(group_means), size=len(group_means))
            estimates[index] = float(group_means[selected].mean())
        ci_low = float(np.quantile(estimates, 0.025))
        ci_high = float(np.quantile(estimates, 0.975))
    mean_gap = float(group_means.mean())
    return {
        "rows": int(len(gap)),
        "groups": int(len(group_means)),
        "mean_gap": mean_gap,
        "median_gap": float(np.median(group_means)),
        "positive_group_fraction": float(np.mean(group_means > 0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass": bool(ci_low > 0 if ci_low is not None else mean_gap > 0),
    }
