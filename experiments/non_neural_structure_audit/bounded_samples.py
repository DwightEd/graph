"""Disk-backed compact sample arrays used by multiple evaluation audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DiskBackedSamples:
    """Store compact token-by-relation matrices outside the Python heap."""

    def __init__(self, path, *, capacity: int, relations: int):
        path = Path(path)
        self.labels = self._array(path, "labels", np.int8, (capacity,))
        self.eligible = self._array(path, "eligible", np.bool_, (capacity,))
        shape = (capacity, relations)
        self.relation = self._array(path, "relation", np.float32, shape)
        self.final_relation = self._array(path, "final", np.float32, shape)
        self.endpoint_null = self._array(path, "endpoint", np.float32, shape)
        self.layer_shuffle = self._array(path, "layer", np.float32, shape)
        self.rows = 0

    @staticmethod
    def _array(path: Path, name: str, dtype, shape) -> np.memmap:
        return np.memmap(
            path.with_name(f"{path.name}.{name}.dat"),
            mode="w+",
            dtype=dtype,
            shape=shape,
        )

    def add(
        self,
        *,
        labels: np.ndarray,
        eligible: np.ndarray,
        relation: np.ndarray,
        final_relation: np.ndarray,
        endpoint_null: np.ndarray,
        layer_shuffle: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        end = self.rows + len(labels)
        selected = slice(self.rows, end)
        self.labels[selected] = labels
        self.eligible[selected] = eligible
        self.relation[selected] = relation
        self.final_relation[selected] = final_relation
        self.endpoint_null[selected] = endpoint_null
        self.layer_shuffle[selected] = layer_shuffle
        self.rows = end
        return tuple(
            array[selected]
            for array in (
                self.labels,
                self.eligible,
                self.relation,
                self.final_relation,
                self.endpoint_null,
                self.layer_shuffle,
            )
        )

    def close(self) -> None:
        for array in (
            self.labels,
            self.eligible,
            self.relation,
            self.final_relation,
            self.endpoint_null,
            self.layer_shuffle,
        ):
            array.flush()
            array._mmap.close()
