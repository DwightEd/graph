"""Position-conditioned robust density for neural mechanism residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_SCALE = 1.482602218505602


class AlignedReservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.rows = 0
        self.blocks: dict[str, np.ndarray] | None = None

    def add(self, **blocks) -> None:
        arrays = {name: np.asarray(value) for name, value in blocks.items()}
        count = len(next(iter(arrays.values())))
        if any(len(value) != count for value in arrays.values()):
            raise ValueError("reservoir blocks must have aligned rows")
        if self.blocks is None:
            self.blocks = {
                name: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for name, value in arrays.items()
            }
        for row in range(count):
            self.seen += 1
            if self.rows < self.capacity:
                index = self.rows
                self.rows += 1
            else:
                index = int(self.rng.integers(self.seen))
                if index >= self.capacity:
                    continue
            for name, value in arrays.items():
                self.blocks[name][index] = value[row]

    def values(self) -> dict[str, np.ndarray]:
        if self.blocks is None or self.rows < 2:
            raise ValueError("density reservoir contains fewer than two rows")
        return {name: value[: self.rows].copy() for name, value in self.blocks.items()}


def _finite_column_median(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        if len(finite):
            result[column] = np.median(finite)
    return result


def _fill_missing(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    row, column = np.where(~np.isfinite(output))
    if len(row):
        output[row, column] = fill[column]
    return output


def _design(nuisance: np.ndarray, task: np.ndarray, task_names: tuple[str, ...]) -> np.ndarray:
    nuisance = np.asarray(nuisance, dtype=np.float64)
    task = np.asarray(task).astype(str)
    one_hot = np.column_stack([task == name for name in task_names]).astype(np.float64)
    return np.column_stack((np.ones(len(nuisance)), nuisance, one_hot))


@dataclass(frozen=True)
class ConditionalDensity:
    task_names: tuple[str, ...]
    feature_fill: np.ndarray
    coefficient: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    energy_reference: np.ndarray
    active_feature: np.ndarray

    @classmethod
    def fit(
        cls,
        feature,
        nuisance,
        task,
        *,
        ridge_alpha: float,
        covariance_shrinkage: float,
        scale_floor: float,
    ) -> "ConditionalDensity":
        raw = np.asarray(feature, dtype=np.float64)
        active = np.mean(np.isfinite(raw), axis=0) >= 0.05
        fill = _finite_column_median(raw)
        values = _fill_missing(raw, fill)
        tasks = tuple(sorted(set(np.asarray(task).astype(str).tolist())))
        design = _design(nuisance, task, tasks)
        penalty = np.eye(design.shape[1]) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ values)
        residual = values - design @ coefficient
        median = np.median(residual, axis=0)
        mad = np.median(np.abs(residual - median), axis=0)
        scale = np.maximum(MAD_SCALE * mad, float(scale_floor))
        standardized = (residual - median) / scale
        standardized[:, ~active] = 0.0
        covariance = np.cov(standardized, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]])
        diagonal = np.diag(np.diag(covariance))
        shrinkage = float(covariance_shrinkage)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-4
        precision = np.linalg.pinv(covariance)
        positive = np.maximum(standardized, 0.0)
        energy = np.einsum("nf,fg,ng->n", positive, precision, positive)
        return cls(
            task_names=tasks,
            feature_fill=fill.astype(np.float32),
            coefficient=coefficient.astype(np.float32),
            median=median.astype(np.float32),
            scale=scale.astype(np.float32),
            precision=precision.astype(np.float32),
            energy_reference=np.sort(energy.astype(np.float32)),
            active_feature=active.astype(np.bool_),
        )

    def residual(self, feature, nuisance, task) -> np.ndarray:
        values = _fill_missing(feature, self.feature_fill)
        design = _design(nuisance, task, self.task_names)
        raw = values - design @ self.coefficient
        standardized = ((raw - self.median) / self.scale).astype(np.float32)
        standardized[:, ~self.active_feature] = 0.0
        return standardized

    def score(self, feature, nuisance, task) -> tuple[np.ndarray, np.ndarray]:
        standardized = self.residual(feature, nuisance, task)
        positive = np.maximum(standardized, 0.0)
        energy = np.einsum(
            "nf,fg,ng->n",
            positive,
            self.precision,
            positive,
        )
        first = np.searchsorted(self.energy_reference, energy, side="left")
        probability = (
            len(self.energy_reference) - first + 1
        ) / float(len(self.energy_reference) + 1)
        score = -np.log10(np.clip(probability, 1e-12, 1.0)).astype(np.float32)
        return score, standardized

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "density_task_names": np.asarray(self.task_names),
            "density_feature_fill": self.feature_fill,
            "density_coefficient": self.coefficient,
            "density_median": self.median,
            "density_scale": self.scale,
            "density_precision": self.precision,
            "density_energy_reference": self.energy_reference,
            "density_active_feature": self.active_feature,
        }

    @classmethod
    def from_arrays(cls, arrays) -> "ConditionalDensity":
        return cls(
            task_names=tuple(np.asarray(arrays["density_task_names"]).astype(str).tolist()),
            feature_fill=np.asarray(arrays["density_feature_fill"]),
            coefficient=np.asarray(arrays["density_coefficient"]),
            median=np.asarray(arrays["density_median"]),
            scale=np.asarray(arrays["density_scale"]),
            precision=np.asarray(arrays["density_precision"]),
            energy_reference=np.asarray(arrays["density_energy_reference"]),
            active_feature=np.asarray(arrays["density_active_feature"]).astype(bool),
        )
