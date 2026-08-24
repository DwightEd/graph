"""Small streaming affine ridge regressors used only for mechanism auditing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineMap:
    weight: np.ndarray
    bias: np.ndarray
    target_mean: np.ndarray
    count: int

    def predict(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values @ self.weight + self.bias


class RidgeAccumulator:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.xtx = np.zeros((self.input_dim + 1, self.input_dim + 1), dtype=np.float64)
        self.xty = np.zeros((self.input_dim + 1, self.output_dim), dtype=np.float64)
        self.target_sum = np.zeros(self.output_dim, dtype=np.float64)
        self.count = 0

    def add(self, inputs, targets) -> None:
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        if y.ndim == 1:
            y = y[None]
        if not len(x):
            return
        augmented = np.concatenate((x, np.ones((len(x), 1))), axis=1)
        self.xtx += augmented.T @ augmented
        self.xty += augmented.T @ y
        self.target_sum += y.sum(axis=0)
        self.count += len(x)

    def freeze(self, alpha: float, minimum_pairs: int) -> AffineMap:
        target_mean = self.target_sum / max(self.count, 1)
        if self.count < int(minimum_pairs):
            return AffineMap(
                weight=np.zeros((self.input_dim, self.output_dim), dtype=np.float32),
                bias=target_mean.astype(np.float32),
                target_mean=target_mean.astype(np.float32),
                count=self.count,
            )
        penalty = np.eye(self.input_dim + 1, dtype=np.float64) * float(alpha)
        penalty[-1, -1] = 0.0
        coefficient = np.linalg.solve(self.xtx + penalty, self.xty)
        return AffineMap(
            weight=coefficient[:-1].astype(np.float32),
            bias=coefficient[-1].astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            count=self.count,
        )
