"""Phase-conditioned robust PPCA geometry for attention trajectories."""

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


VARIANTS = ("full", "static", "topology", "mass")


@dataclass(frozen=True)
class GeometryConfig:
    pca_dim: int = 32
    reference_per_sample: int = 16
    min_condition_rows: int = 32
    trim_fraction: float = 1.0
    calibration_fraction: float = 0.25
    bootstrap_replicates: int = 1000
    topology_gate_min_coverage: float = 0.25
    seed: int = 20260817
    epsilon: float = 1e-8

    def validate(self) -> None:
        if min(
            int(self.pca_dim),
            int(self.reference_per_sample),
            int(self.min_condition_rows),
        ) < 1:
            raise ValueError("geometry integer settings must be positive")
        if not 0.5 <= float(self.trim_fraction) <= 1.0:
            raise ValueError("trim_fraction must be in [0.5,1]")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if int(self.bootstrap_replicates) < 0:
            raise ValueError("bootstrap_replicates must be non-negative")
        if not 0.0 <= float(self.topology_gate_min_coverage) <= 1.0:
            raise ValueError("topology_gate_min_coverage must be in [0,1]")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


def reference_positions(response_count: int, count: int) -> np.ndarray:
    if response_count < 1:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), response_count)
    if count < 1:
        raise ValueError("reference position count must be positive")
    if count == response_count:
        return np.arange(response_count, dtype=np.int64)
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    positions = np.rint(quantiles * (response_count - 1)).astype(np.int64)
    return np.unique(np.clip(positions, 0, response_count - 1))


def condition_keys(task_type, position_bucket) -> np.ndarray:
    task = np.asarray(task_type, dtype=str)
    bucket = np.asarray(position_bucket, dtype=np.int64)
    if task.ndim != 1 or bucket.ndim != 1 or len(task) != len(bucket):
        raise ValueError("task and position buckets must be aligned vectors")
    return np.asarray(
        [
            f"{name}\x1f{int(value)}"
            for name, value in zip(task, bucket, strict=True)
        ],
        dtype=str,
    )


def _fit_conditioned_scaler(
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    min_rows: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=str)
    if values.ndim != 2 or len(values) != len(conditions) or len(values) < 3:
        raise ValueError("conditioned scaling needs an aligned matrix")
    global_center, global_scale = robust_location_scale(values)
    names = np.asarray(sorted(set(conditions.tolist())), dtype=str)
    centers = np.empty((len(names), values.shape[1]), dtype=np.float32)
    scales = np.empty_like(centers)
    counts = np.empty(len(names), dtype=np.int32)
    for index, name in enumerate(names):
        selected = conditions == name
        counts[index] = int(selected.sum())
        if counts[index] >= int(min_rows):
            centers[index], scales[index] = robust_location_scale(
                values[selected]
            )
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
    names = np.asarray(scaler["condition_names"], dtype=str)
    lookup = {name: index for index, name in enumerate(names.tolist())}
    result = np.empty_like(values)
    global_center = np.asarray(scaler["global_center"], dtype=np.float32)
    global_scale = np.asarray(scaler["global_scale"], dtype=np.float32)
    centers = np.asarray(scaler["condition_center"], dtype=np.float32)
    scales = np.asarray(scaler["condition_scale"], dtype=np.float32)
    for name in np.unique(conditions):
        selected = conditions == name
        index = lookup.get(str(name))
        center = global_center if index is None else centers[index]
        scale = global_scale if index is None else scales[index]
        result[selected] = (values[selected] - center) / scale
    return result.astype(np.float32, copy=False)


def fit_variant_geometry(
    values: np.ndarray,
    conditions: np.ndarray,
    feature_names: np.ndarray,
    *,
    config: GeometryConfig,
    seed_offset: int,
) -> dict[str, np.ndarray]:
    scaler = _fit_conditioned_scaler(
        values,
        conditions,
        min_rows=config.min_condition_rows,
    )
    standardized = _standardize(values, conditions, scaler)
    model, keep, provisional = fit_robust_pca(
        standardized,
        np.zeros(len(standardized), dtype=np.int16),
        requested_dim=config.pca_dim,
        trim_fraction=config.trim_fraction,
        seed=config.seed + int(seed_offset),
        epsilon=config.epsilon,
    )
    return {
        **scaler,
        **pca_artifact(model, epsilon=config.epsilon),
        "feature_names": np.asarray(feature_names, dtype=str),
        "fit_rows": np.asarray(len(values), dtype=np.int32),
        "retained_fit_rows": np.asarray(int(keep.sum()), dtype=np.int32),
        "provisional_residual_median": np.asarray(
            float(np.median(provisional)), dtype=np.float32
        ),
    }


def project_variant_geometry(
    values: np.ndarray,
    conditions: np.ndarray,
    model: Mapping[str, np.ndarray],
):
    standardized = _standardize(values, conditions, model)
    return project_subspace(standardized, model)


def flatten_variant_artifact(
    variant: str,
    artifact: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown trajectory variant: {variant}")
    return {
        f"{variant}_{name}": np.asarray(value)
        for name, value in artifact.items()
    }


def variant_artifact(
    reference: Mapping[str, np.ndarray],
    variant: str,
) -> dict[str, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown trajectory variant: {variant}")
    prefix = f"{variant}_"
    result = {
        name[len(prefix) :]: np.asarray(value)
        for name, value in reference.items()
        if name.startswith(prefix)
    }
    required = {
        "condition_names",
        "condition_center",
        "condition_scale",
        "condition_count",
        "global_center",
        "global_scale",
        "rr_pca_mean",
        "rr_pca_components",
        "rr_pca_explained_variance",
        "rr_pca_noise_variance",
        "feature_names",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{variant} geometry misses fields: {sorted(missing)}")
    return result


def fit_all_geometries(
    values_by_variant: Mapping[str, np.ndarray],
    conditions: np.ndarray,
    feature_names_by_variant: Mapping[str, np.ndarray],
    *,
    config: GeometryConfig,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for offset, variant in enumerate(VARIANTS):
        artifact = fit_variant_geometry(
            values_by_variant[variant],
            conditions,
            feature_names_by_variant[variant],
            config=config,
            seed_offset=100 * offset,
        )
        result.update(flatten_variant_artifact(variant, artifact))
    return result


def energy_by_variant(
    values_by_variant: Mapping[str, np.ndarray],
    conditions: np.ndarray,
    reference: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {}
    for variant in VARIANTS:
        projection = project_variant_geometry(
            values_by_variant[variant],
            conditions,
            variant_artifact(reference, variant),
        )
        result[variant] = projection.ppca_energy.astype(
            np.float32, copy=False
        )
    return result


def calibrate_energies(
    energies: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        variant: empirical_upper_tail(
            np.asarray(reference[f"calibration_energy_{variant}"]),
            np.asarray(energies[variant]),
        )
        for variant in VARIANTS
    }


def topology_gate_summary(
    true_energy: np.ndarray,
    rewired_energy: np.ndarray,
    valid: np.ndarray,
    source_id: np.ndarray,
    *,
    config: GeometryConfig,
) -> dict[str, object]:
    """Cluster-bootstrap the rewired-minus-true energy gap."""
    true_energy = np.asarray(true_energy, dtype=np.float64)
    rewired_energy = np.asarray(rewired_energy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    source_id = np.asarray(source_id, dtype=str)
    if not (
        true_energy.shape
        == rewired_energy.shape
        == valid.shape
        == source_id.shape
    ):
        raise ValueError("topology gate arrays are not aligned")
    finite = valid & np.isfinite(true_energy) & np.isfinite(rewired_energy)
    coverage = float(finite.mean()) if len(finite) else 0.0
    gap = rewired_energy[finite] - true_energy[finite]
    groups = source_id[finite]
    group_means = np.asarray(
        [
            float(gap[groups == group].mean())
            for group in np.unique(groups)
        ],
        dtype=np.float64,
    )
    if len(gap) == 0 or len(group_means) == 0:
        return {
            "token_count": int(len(valid)),
            "evaluated_tokens": 0,
            "coverage": coverage,
            "source_groups": 0,
            "mean_gap": None,
            "median_gap": None,
            "positive_group_fraction": None,
            "ci_low": None,
            "ci_high": None,
            "pass": False,
        }

    ci_low = ci_high = None
    if config.bootstrap_replicates > 0 and len(group_means) >= 2:
        rng = np.random.default_rng(config.seed + 909)
        estimates = np.empty(
            config.bootstrap_replicates, dtype=np.float64
        )
        for index in range(config.bootstrap_replicates):
            selected = rng.integers(
                0, len(group_means), size=len(group_means)
            )
            estimates[index] = group_means[selected].mean()
        ci_low = float(np.quantile(estimates, 0.025))
        ci_high = float(np.quantile(estimates, 0.975))
    mean_gap = float(group_means.mean())
    gate_pass = bool(
        coverage >= config.topology_gate_min_coverage
        and (
            ci_low > 0.0
            if ci_low is not None
            else mean_gap > 0.0
        )
    )
    return {
        "token_count": int(len(valid)),
        "evaluated_tokens": int(finite.sum()),
        "coverage": coverage,
        "source_groups": int(len(group_means)),
        "mean_gap": mean_gap,
        "median_gap": float(np.median(group_means)),
        "positive_group_fraction": float(np.mean(group_means > 0.0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass": gate_pass,
    }
