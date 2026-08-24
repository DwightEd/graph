"""Bounded, label-free Markov-order prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RowReservoir:
    """Uniform row reservoir shared by nested feature matrices."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.rows = 0
        self.arrays: dict[str, np.ndarray] = {}

    def add(self, **batches: np.ndarray) -> None:
        count = len(next(iter(batches.values())))
        if not self.arrays:
            self.arrays = {
                name: np.empty((self.capacity, *np.asarray(value).shape[1:]), dtype=np.float32)
                for name, value in batches.items()
            }
        values = {name: np.asarray(value, dtype=np.float32) for name, value in batches.items()}
        for row in range(count):
            self.seen += 1
            if self.rows < self.capacity:
                index = self.rows
                self.rows += 1
            else:
                index = int(self.rng.integers(self.seen))
                if index >= self.capacity:
                    continue
            for name, value in values.items():
                self.arrays[name][index] = value[row]

    def matrix(self, name: str) -> np.ndarray:
        return self.arrays[name][: self.rows]


@dataclass(frozen=True)
class RidgePredictor:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> "RidgePredictor":
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
        return cls(
            mean=scaler.mean_.astype(np.float32),
            scale=np.maximum(scaler.scale_, 1e-6).astype(np.float32),
            coefficient=model.coef_.astype(np.float32),
            intercept=np.asarray(model.intercept_, dtype=np.float32),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(x, dtype=np.float32) - self.mean) / self.scale
        return standardized @ self.coefficient.T + self.intercept

    def error(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.square(self.predict(x) - np.asarray(y, dtype=np.float32)).mean(axis=1)

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_scale": self.scale,
            f"{prefix}_coefficient": self.coefficient,
            f"{prefix}_intercept": self.intercept,
        }

    @classmethod
    def from_arrays(cls, arrays, prefix: str) -> "RidgePredictor":
        return cls(
            mean=arrays[f"{prefix}_mean"],
            scale=arrays[f"{prefix}_scale"],
            coefficient=arrays[f"{prefix}_coefficient"],
            intercept=arrays[f"{prefix}_intercept"],
        )


@dataclass(frozen=True)
class NestedMarkovModel:
    order1: RidgePredictor
    order2: RidgePredictor
    order3: RidgePredictor
    order2_null: RidgePredictor
    order3_null: RidgePredictor
    order1_dim: int
    order2_dim: int

    @staticmethod
    def _permutation(order1: np.ndarray, seed: int) -> np.ndarray:
        count = len(order1)
        rng = np.random.default_rng(seed)
        if order1.shape[1] < 2 or np.any(order1[:, -2:] < 0) or np.any(order1[:, -2:] > 1):
            return rng.permutation(count)
        position = np.minimum((order1[:, -2] * 4).astype(np.int16), 3)
        depth = np.minimum((order1[:, -1] * 8).astype(np.int16), 7)
        group = position * 8 + depth
        permutation = np.arange(count)
        for name in np.unique(group):
            selected = np.flatnonzero(group == name)
            permutation[selected] = rng.permutation(selected)
        return permutation

    @classmethod
    def fit(
        cls,
        order1: np.ndarray,
        order2: np.ndarray,
        order3: np.ndarray,
        target: np.ndarray,
        *,
        alpha: float,
        seed: int,
    ) -> "NestedMarkovModel":
        order1_dim = order1.shape[1]
        order2_dim = order2.shape[1]
        permutation2 = cls._permutation(order1, seed)
        permutation3 = cls._permutation(order1, seed + 1)
        order2_null = np.concatenate(
            (order1, order2[permutation2, order1_dim:]), axis=1
        )
        order3_null = np.concatenate(
            (order2, order3[permutation3, order2_dim:]), axis=1
        )
        return cls(
            order1=RidgePredictor.fit(order1, target, alpha),
            order2=RidgePredictor.fit(order2, target, alpha),
            order3=RidgePredictor.fit(order3, target, alpha),
            order2_null=RidgePredictor.fit(order2_null, target, alpha),
            order3_null=RidgePredictor.fit(order3_null, target, alpha),
            order1_dim=order1_dim,
            order2_dim=order2_dim,
        )

    def errors(
        self,
        order1: np.ndarray,
        order2: np.ndarray,
        order3: np.ndarray,
        target: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, np.ndarray]:
        permutation2 = self._permutation(order1, seed)
        permutation3 = self._permutation(order1, seed + 1)
        order2_null = np.concatenate(
            (order1, order2[permutation2, self.order1_dim :]), axis=1
        )
        order3_null = np.concatenate(
            (order2, order3[permutation3, self.order2_dim :]), axis=1
        )
        error1 = self.order1.error(order1, target)
        error2 = self.order2.error(order2, target)
        error3 = self.order3.error(order3, target)
        null2 = self.order2_null.error(order2_null, target)
        null3 = self.order3_null.error(order3_null, target)
        return {
            "order1_error": error1,
            "order2_error": error2,
            "order3_error": error3,
            "order2_gain": error1 - error2,
            "order3_gain": error2 - error3,
            "order2_path_gain": null2 - error2,
            "order3_path_gain": null3 - error3,
        }

    def validation_summary(
        self,
        order1: np.ndarray,
        order2: np.ndarray,
        order3: np.ndarray,
        target: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, float]:
        errors = self.errors(order1, order2, order3, target, seed=seed)
        return {name: float(value.mean()) for name, value in errors.items()}

    def save(self, path: str | Path, metadata: dict[str, object]) -> None:
        arrays = {
            "schema": np.asarray("causal-walk-markov-model-v1"),
            "order1_dim": np.asarray(self.order1_dim, dtype=np.int32),
            "order2_dim": np.asarray(self.order2_dim, dtype=np.int32),
            "metadata_json": np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
        }
        for prefix, model in (
            ("order1", self.order1),
            ("order2", self.order2),
            ("order3", self.order3),
            ("order2_null", self.order2_null),
            ("order3_null", self.order3_null),
        ):
            arrays.update(model.arrays(prefix))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> tuple["NestedMarkovModel", dict[str, object]]:
        with np.load(path, allow_pickle=False) as arrays:
            model = cls(
                order1=RidgePredictor.from_arrays(arrays, "order1"),
                order2=RidgePredictor.from_arrays(arrays, "order2"),
                order3=RidgePredictor.from_arrays(arrays, "order3"),
                order2_null=RidgePredictor.from_arrays(arrays, "order2_null"),
                order3_null=RidgePredictor.from_arrays(arrays, "order3_null"),
                order1_dim=int(arrays["order1_dim"]),
                order2_dim=int(arrays["order2_dim"]),
            )
            metadata = __import__("json").loads(str(arrays["metadata_json"].item()))
        return model, metadata
