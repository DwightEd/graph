"""A fixed product metric over complete register-graph state tensors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .graph import GraphSequence

BLOCK_NAMES = (
    "node_embedding",
    "residual_gram",
    "head_write_gram",
    "route_topology",
    "mlp_relation",
    "margin_contribution",
)


Frame = tuple[GraphSequence, int]


def as_array(value: object) -> np.ndarray:
    """Move a captured tensor to NumPy without retaining an autograd graph."""

    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value)


def block_distances(left: Frame, right: Frame) -> np.ndarray:
    """Return one RMS distance for every complete physical state block.

    Layer, head, origin and topology-channel axes remain aligned.  The RMS is
    only the final reduction after corresponding tensor entries are compared.
    """

    left_sequence, left_index = left
    right_sequence, right_index = right
    distance = []
    for name in BLOCK_NAMES:
        difference = np.asarray(
            as_array(getattr(left_sequence, name)[left_index]), dtype=np.float64
        ) - np.asarray(
            as_array(getattr(right_sequence, name)[right_index]), dtype=np.float64
        )
        distance.append(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
    return np.asarray(distance, dtype=np.float64)


@dataclass(frozen=True)
class RouteMetric:
    """Equal-weight product metric with label-free block normalization."""

    scale: np.ndarray

    @classmethod
    def fit(cls, pairs: Iterable[tuple[Frame, Frame]]) -> RouteMetric:
        """Set each block scale to its median nonzero reference distance."""

        observed = [block_distances(left, right) for left, right in pairs]
        if not observed:
            return cls(np.ones(len(BLOCK_NAMES), dtype=np.float64))

        values = np.stack(observed)
        scale = np.ones(values.shape[1], dtype=np.float64)
        for block in range(values.shape[1]):
            positive = values[:, block][values[:, block] > 1e-12]
            if len(positive):
                scale[block] = np.median(positive)
        return cls(scale)

    def block_distance(self, left: Frame, right: Frame) -> np.ndarray:
        return block_distances(left, right) / self.scale

    def distance(self, left: Frame, right: Frame) -> float:
        """Compare every tensor block, then give the six blocks equal weight."""

        return float(self.block_distance(left, right).mean())

    def distances_to_batch(
        self,
        sequence: GraphSequence,
        index: int,
        batch: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Vectorized distance from one frame to a batch of prototype frames."""

        return self.distances_from_indices_to_batch(sequence, [index], batch)[0]

    def distances_from_indices_to_batch(
        self,
        sequence: GraphSequence,
        indices: Iterable[int],
        batch: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Exact all-coordinate distances for many frames and prototypes."""

        selected = np.fromiter(indices, dtype=np.int64)
        blocks = []
        for block, name in enumerate(BLOCK_NAMES):
            reference = np.asarray(batch[name], dtype=np.float64).reshape(
                len(batch[name]), -1
            )
            value = np.asarray(
                as_array(getattr(sequence, name))[selected], dtype=np.float64
            ).reshape(len(selected), -1)
            squared = (
                np.square(value).sum(1)[:, None]
                + np.square(reference).sum(1)[None]
                - 2.0 * value @ reference.T
            ) / value.shape[1]
            rms = np.sqrt(np.maximum(squared, 0.0))
            blocks.append(rms / self.scale[block])
        return np.stack(blocks).mean(axis=0)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"metric_scale": np.asarray(self.scale, dtype=np.float64)}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> RouteMetric:
        return cls(np.asarray(arrays["metric_scale"], dtype=np.float64))
