"""Position-conditioned, label-free references for mechanism residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ReferenceConfig

MAD_SCALE = 1.482602218505602


class AlignedReservoir:
    """Uniform reservoir that keeps mechanism, nuisance and task rows aligned."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.seen = 0
        self.rows: dict[str, np.ndarray] | None = None
        self.size = 0

    def add(self, **blocks) -> None:
        values = {name: np.asarray(value) for name, value in blocks.items()}
        count = len(next(iter(values.values())))
        if any(len(value) != count for value in values.values()):
            raise ValueError("reservoir blocks must share rows")
        if self.rows is None:
            self.rows = {
                name: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for name, value in values.items()
            }
        for row in range(count):
            self.seen += 1
            if self.size < self.capacity:
                target = self.size
                self.size += 1
            else:
                target = int(self.rng.integers(self.seen))
                if target >= self.capacity:
                    continue
            assert self.rows is not None
            for name, value in values.items():
                self.rows[name][target] = value[row]

    def values(self) -> dict[str, np.ndarray]:
        if self.rows is None or self.size < 2:
            raise ValueError("reference reservoir is empty")
        return {name: value[: self.size].copy() for name, value in self.rows.items()}


@dataclass(frozen=True)
class NuisanceReference:
    task_names: tuple[str, ...]
    coefficient: np.ndarray
    residual_median: np.ndarray
    residual_scale: np.ndarray
    position_degree: int

    def design(self, nuisance: np.ndarray, task_type) -> np.ndarray:
        nuisance = np.asarray(nuisance, dtype=np.float64)
        task = np.asarray(task_type).astype(str)
        relative = nuisance[:, 1]
        columns = [relative ** power for power in range(1, self.position_degree + 1)]
        columns.extend(
            (
                np.log1p(nuisance[:, 2]),
                np.log1p(nuisance[:, 3]),
                np.log1p(nuisance[:, 4]),
                np.log1p(nuisance[:, 5]),
                np.log1p(np.clip(nuisance[:, 6], 0, None)),
                nuisance[:, 7],
                nuisance[:, 8],
            )
        )
        for name in self.task_names:
            columns.append((task == name).astype(np.float64))
        return np.stack(columns, axis=1)

    def transform(
        self,
        primary: np.ndarray,
        nuisance: np.ndarray,
        task_type,
    ) -> tuple[np.ndarray, np.ndarray]:
        primary = np.asarray(primary, dtype=np.float64)
        design = self.design(nuisance, task_type)
        augmented = np.concatenate((design, np.ones((len(design), 1))), axis=1)
        predicted = augmented @ self.coefficient.T
        residual = primary - predicted
        standardized = (residual - self.residual_median) / self.residual_scale
        standardized[~np.isfinite(primary)] = np.nan
        positive = np.square(np.maximum(standardized, 0.0))
        available = np.isfinite(positive)
        joint = np.nansum(positive, axis=1) / np.maximum(available.sum(axis=1), 1)
        joint[available.sum(axis=1) == 0] = np.nan
        return standardized.astype(np.float32), joint.astype(np.float32)


def fit_nuisance_reference(
    primary: np.ndarray,
    nuisance: np.ndarray,
    task_type,
    *,
    config: ReferenceConfig,
) -> NuisanceReference:
    task_names = tuple(sorted(set(np.asarray(task_type).astype(str).tolist())))
    template = NuisanceReference(
        task_names=task_names,
        coefficient=np.empty((0, 0)),
        residual_median=np.empty(0),
        residual_scale=np.empty(0),
        position_degree=config.position_degree,
    )
    design = template.design(nuisance, task_type)
    augmented = np.concatenate((design, np.ones((len(design), 1))), axis=1)
    feature_count = primary.shape[1]
    coefficient = np.zeros((feature_count, augmented.shape[1]), dtype=np.float64)
    median = np.zeros(feature_count, dtype=np.float64)
    scale = np.ones(feature_count, dtype=np.float64)

    penalty = np.eye(augmented.shape[1], dtype=np.float64) * float(
        config.nuisance_ridge_alpha
    )
    penalty[-1, -1] = 0.0
    for feature in range(feature_count):
        selected = np.isfinite(primary[:, feature])
        if selected.sum() < augmented.shape[1] + 2:
            continue
        x = augmented[selected]
        y = primary[selected, feature].astype(np.float64)
        coefficient[feature] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residual = y - x @ coefficient[feature]
        median[feature] = np.median(residual)
        mad = np.median(np.abs(residual - median[feature]))
        scale[feature] = max(MAD_SCALE * mad, 1e-4)

    return NuisanceReference(
        task_names=task_names,
        coefficient=coefficient.astype(np.float32),
        residual_median=median.astype(np.float32),
        residual_scale=scale.astype(np.float32),
        position_degree=config.position_degree,
    )
