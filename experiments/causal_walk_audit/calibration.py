"""Label-free hierarchical calibration across heads and layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RowReservoir:
    """Uniform bounded reservoir for aligned float32 channel rows."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.rows = 0
        self.values: np.ndarray | None = None

    def add(self, batch) -> None:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError("reservoir values must have shape [row, channel]")
        if self.values is None:
            self.values = np.empty(
                (self.capacity, batch.shape[1]), dtype=np.float32
            )
        for row in batch:
            self.seen += 1
            if self.rows < self.capacity:
                index = self.rows
                self.rows += 1
            else:
                index = int(self.rng.integers(self.seen))
                if index >= self.capacity:
                    continue
            self.values[index] = row

    def matrix(self) -> np.ndarray:
        if self.values is None or self.rows < 2:
            raise ValueError("reservoir contains fewer than two rows")
        return self.values[: self.rows].copy()


def _upper_tail(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    rows, columns = sorted_reference.shape
    probability = np.empty_like(values)
    for column in range(columns):
        first = np.searchsorted(
            sorted_reference[:, column], values[:, column], side="left"
        )
        probability[:, column] = (rows - first + 1) / float(rows + 1)
    return probability


def _leave_one_out_upper_tail(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    rows, columns = values.shape
    probability = np.empty_like(values)
    for column in range(columns):
        ordered = np.sort(values[:, column])
        first = np.searchsorted(ordered, values[:, column], side="left")
        probability[:, column] = (rows - first) / float(rows)
    return probability.clip(1.0 / rows, 1.0)


def _cauchy_statistic(probability: np.ndarray, axis: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.mean(
        np.tan(np.pi * (0.5 - clipped)),
        axis=axis,
        dtype=np.float32,
    )


@dataclass(frozen=True)
class HierarchicalCalibration:
    channel_reference: np.ndarray
    layer_reference: np.ndarray
    global_reference: np.ndarray
    num_layers: int
    num_heads: int

    @classmethod
    def fit(
        cls,
        channel_values,
        fusion_values,
        *,
        num_layers: int,
        num_heads: int,
    ) -> "HierarchicalCalibration":
        channel_values = np.asarray(channel_values, dtype=np.float32)
        fusion_values = np.asarray(fusion_values, dtype=np.float32)
        channels = num_layers * num_heads
        if channel_values.shape[1] != channels or fusion_values.shape[1] != channels:
            raise ValueError("calibration rows do not match layer/head geometry")

        channel_reference = np.sort(channel_values, axis=0)
        channel_p = _upper_tail(channel_reference, fusion_values)
        layer_stat = _cauchy_statistic(
            channel_p.reshape(len(channel_p), num_layers, num_heads),
            axis=2,
        )
        layer_reference = np.sort(layer_stat, axis=0)
        layer_p_loo = _leave_one_out_upper_tail(layer_stat)
        global_stat = _cauchy_statistic(layer_p_loo, axis=1)
        global_reference = np.sort(global_stat)
        return cls(
            channel_reference=channel_reference,
            layer_reference=layer_reference,
            global_reference=global_reference,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def score(self, channel_values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(channel_values, dtype=np.float32)
        channel_p = _upper_tail(self.channel_reference, values)
        layer_stat = _cauchy_statistic(
            channel_p.reshape(
                len(values), self.num_layers, self.num_heads
            ),
            axis=2,
        )
        layer_p = _upper_tail(self.layer_reference, layer_stat)
        global_stat = _cauchy_statistic(layer_p, axis=1)
        first = np.searchsorted(self.global_reference, global_stat, side="left")
        global_p = (
            len(self.global_reference) - first + 1
        ) / float(len(self.global_reference) + 1)
        score = -np.log10(np.clip(global_p, 1e-12, 1.0)).astype(np.float32)
        return score, layer_stat.astype(np.float32), channel_p.astype(np.float32)
