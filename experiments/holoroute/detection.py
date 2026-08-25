"""Position-conditioned calibration for the single P-Cut closure score."""

from dataclasses import dataclass

import numpy as np

from .config import DetectionConfig
from .graph import AttentionGraph
from .pcut import PCutResult

RESIDUAL_NAMES = ("closure",)
CONDITION_NAMES = (
    "log_position",
    "relative_position",
    "relative_position_squared",
    "log_response_length",
    "unresolved_mass",
    "provenance_interval_width",
    "cut_fallback_fraction",
)
MAD_SCALE = 1.482602218505602


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.random = np.random.default_rng(seed)
        self.seen = 0
        self.size = 0
        self.blocks: dict[str, np.ndarray] | None = None

    def add(self, **blocks) -> None:
        arrays = {name: np.asarray(value) for name, value in blocks.items()}
        rows = len(next(iter(arrays.values())))
        if self.blocks is None:
            self.blocks = {
                name: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for name, value in arrays.items()
            }
        for row in range(rows):
            self.seen += 1
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                index = int(self.random.integers(self.seen))
                if index >= self.capacity:
                    continue
            for name, value in arrays.items():
                self.blocks[name][index] = value[row]

    def values(self) -> dict[str, np.ndarray]:
        if self.blocks is None or self.size < 2:
            raise RuntimeError("calibration needs at least two token rows")
        return {name: value[: self.size].copy() for name, value in self.blocks.items()}


def design_matrix(
    condition: np.ndarray,
    task: np.ndarray,
    task_names: tuple[str, ...],
) -> np.ndarray:
    condition = np.asarray(condition, dtype=np.float64)
    task = np.asarray(task).astype(str)
    task_indicator = np.column_stack([task == name for name in task_names]).astype(np.float64)
    return np.column_stack((np.ones(len(condition)), condition, task_indicator))


@dataclass(frozen=True)
class ConditionalReference:
    task_names: tuple[str, ...]
    coefficient: np.ndarray
    median: float
    scale: float
    tail_reference: np.ndarray

    @classmethod
    def fit(
        cls,
        residual: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
        config: DetectionConfig,
    ) -> "ConditionalReference":
        value = np.asarray(residual, dtype=np.float64).reshape(-1)
        task_names = tuple(sorted(set(np.asarray(task).astype(str).tolist())))
        design = design_matrix(condition, task, task_names)
        penalty = np.eye(design.shape[1], dtype=np.float64) * config.ridge_alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ value)
        centered = value - design @ coefficient
        median = float(np.median(centered))
        mad = float(np.median(np.abs(centered - median)))
        scale = max(MAD_SCALE * mad, config.scale_floor)
        standardized = (centered - median) / scale
        return cls(
            task_names=task_names,
            coefficient=coefficient.astype(np.float32),
            median=median,
            scale=scale,
            tail_reference=np.sort(standardized.astype(np.float32)),
        )

    def transform(
        self,
        residual: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(residual, dtype=np.float64).reshape(-1)
        design = design_matrix(condition, task, self.task_names)
        standardized = ((value - design @ self.coefficient - self.median) / self.scale).astype(np.float32)
        rank = np.searchsorted(self.tail_reference, standardized, side="left")
        probability = (len(self.tail_reference) - rank + 1) / float(len(self.tail_reference) + 1)
        score = -np.log10(np.clip(probability, 1e-12, 1.0)).astype(np.float32)
        return score, standardized[:, None]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "reference_task_names": np.asarray(self.task_names),
            "reference_coefficient": self.coefficient,
            "reference_median": np.asarray(self.median, dtype=np.float32),
            "reference_scale": np.asarray(self.scale, dtype=np.float32),
            "reference_tail": self.tail_reference,
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "ConditionalReference":
        return cls(
            task_names=tuple(np.asarray(arrays["reference_task_names"]).astype(str).tolist()),
            coefficient=np.asarray(arrays["reference_coefficient"]),
            median=float(np.asarray(arrays["reference_median"]).item()),
            scale=float(np.asarray(arrays["reference_scale"]).item()),
            tail_reference=np.asarray(arrays["reference_tail"]),
        )


def token_conditions(graph: AttentionGraph, result: PCutResult) -> np.ndarray:
    tokens = graph.response_count
    position = np.arange(tokens, dtype=np.float32)
    relative = position / max(tokens - 1, 1)
    tail = min(8, graph.layer_count)
    unresolved = graph.unresolved[:, -tail:].mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
    return np.column_stack(
        (
            np.log1p(position),
            relative,
            relative**2,
            np.full(tokens, np.log1p(tokens), dtype=np.float32),
            unresolved,
            result.uncertainty_width.cpu().numpy().astype(np.float32),
            result.cut_fallback_fraction.cpu().numpy().astype(np.float32),
        )
    ).astype(np.float32)
