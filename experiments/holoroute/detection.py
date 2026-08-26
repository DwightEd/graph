"""Unsupervised node detection in the structural-feature space."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .config import DetectionConfig
from .graph import AttentionGraph

RESIDUAL_NAMES = ("subspace_residual",)
CONDITION_NAMES = (
    "log_position",
    "relative_position",
    "relative_position_squared",
    "log_response_length",
    "unresolved_mass",
    "log_incoming_edges",
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
            raise RuntimeError("reservoir needs at least two token rows")
        return {name: value[: self.size].copy() for name, value in self.blocks.items()}


def design_matrix(
    condition: np.ndarray,
    task: np.ndarray,
    task_names: tuple[str, ...],
) -> np.ndarray:
    condition = np.asarray(condition, dtype=np.float32)
    task = np.asarray(task).astype(str)
    task_indicator = np.column_stack(
        [task == name for name in task_names]
    ).astype(np.float32)
    return np.column_stack(
        (
            np.ones(len(condition), dtype=np.float32),
            condition,
            task_indicator,
        )
    )


@dataclass(frozen=True)
class SubspaceReference:
    task_names: tuple[str, ...]
    coefficient: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    tail_reference: np.ndarray
    standardized_clip: float

    @classmethod
    def fit(
        cls,
        feature: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
        config: DetectionConfig,
    ) -> "SubspaceReference":
        value = np.asarray(feature, dtype=np.float32)
        task_names = tuple(sorted(set(np.asarray(task).astype(str).tolist())))
        design = design_matrix(condition, task, task_names)
        penalty = np.eye(design.shape[1], dtype=np.float32) * config.ridge_alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ value,
        ).astype(np.float32)

        centered = value - design @ coefficient
        center = np.median(centered, axis=0).astype(np.float32)
        mad = np.median(np.abs(centered - center), axis=0).astype(np.float32)
        scale = np.maximum(MAD_SCALE * mad, config.scale_floor).astype(np.float32)
        standardized = np.clip(
            (centered - center) / scale,
            -config.standardized_clip,
            config.standardized_clip,
        )

        component_count = min(
            config.pca_components,
            len(standardized) - 1,
            standardized.shape[1],
        )
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=config.seed,
        )
        pca.fit(standardized)
        return cls(
            task_names=task_names,
            coefficient=coefficient,
            center=center,
            scale=scale,
            components=pca.components_.astype(np.float32),
            tail_reference=np.empty(0, dtype=np.float32),
            standardized_clip=float(config.standardized_clip),
        )

    def standardize(
        self,
        feature: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float32)
        design = design_matrix(condition, task, self.task_names)
        return np.clip(
            (value - design @ self.coefficient - self.center) / self.scale,
            -self.standardized_clip,
            self.standardized_clip,
        ).astype(np.float32)

    def energy(
        self,
        feature: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
    ) -> np.ndarray:
        standardized = self.standardize(feature, condition, task)
        coordinate = standardized @ self.components.T
        residual = standardized - coordinate @ self.components
        return np.mean(np.square(residual), axis=1).astype(np.float32)

    def calibrate(
        self,
        feature: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
    ) -> "SubspaceReference":
        return SubspaceReference(
            task_names=self.task_names,
            coefficient=self.coefficient,
            center=self.center,
            scale=self.scale,
            components=self.components,
            tail_reference=np.sort(self.energy(feature, condition, task)),
            standardized_clip=self.standardized_clip,
        )

    def transform(
        self,
        feature: np.ndarray,
        condition: np.ndarray,
        task: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        energy = self.energy(feature, condition, task)
        if not len(self.tail_reference):
            raise RuntimeError("reference has not been calibrated")
        rank = np.searchsorted(self.tail_reference, energy, side="left")
        probability = (
            len(self.tail_reference) - rank + 1
        ) / float(len(self.tail_reference) + 1)
        score = -np.log10(np.clip(probability, 1e-12, 1.0)).astype(np.float32)
        return score, energy[:, None]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "reference_task_names": np.asarray(self.task_names),
            "reference_coefficient": self.coefficient,
            "reference_center": self.center,
            "reference_scale": self.scale,
            "reference_components": self.components,
            "reference_tail": self.tail_reference,
            "reference_clip": np.asarray(self.standardized_clip, dtype=np.float32),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "SubspaceReference":
        return cls(
            task_names=tuple(
                np.asarray(arrays["reference_task_names"]).astype(str).tolist()
            ),
            coefficient=np.asarray(arrays["reference_coefficient"]),
            center=np.asarray(arrays["reference_center"]),
            scale=np.asarray(arrays["reference_scale"]),
            components=np.asarray(arrays["reference_components"]),
            tail_reference=np.asarray(arrays["reference_tail"]),
            standardized_clip=float(np.asarray(arrays["reference_clip"]).item()),
        )


def token_conditions(graph: AttentionGraph) -> np.ndarray:
    tokens = graph.response_count
    position = np.arange(tokens, dtype=np.float32)
    relative = position / max(tokens - 1, 1)
    unresolved = graph.unresolved.mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
    target = (
        graph.edges.target - graph.response_start
    ).cpu().numpy().astype(np.int64)
    incoming = np.bincount(target, minlength=tokens).astype(np.float32)
    return np.column_stack(
        (
            np.log1p(position),
            relative,
            relative**2,
            np.full(tokens, np.log1p(tokens), dtype=np.float32),
            unresolved,
            np.log1p(incoming),
        )
    ).astype(np.float32)
