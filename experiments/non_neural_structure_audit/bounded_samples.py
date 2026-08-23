"""Disk-backed compact sample arrays used by multiple evaluation audits."""

from __future__ import annotations

import numpy as np

from experiments.disk_row_store import DiskRowStore, FieldSpec


class DiskBackedSamples:
    """Store compact token-by-relation matrices outside the Python heap."""

    def __init__(self, path, *, capacity: int, relations: int):
        vector = (relations,)
        self.store = DiskRowStore(
            path,
            capacity=capacity,
            fields={
                "labels": FieldSpec(np.dtype(np.int8)),
                "eligible": FieldSpec(np.dtype(np.bool_)),
                "relation": FieldSpec(np.dtype(np.float32), vector),
                "final": FieldSpec(np.dtype(np.float32), vector),
                "endpoint": FieldSpec(np.dtype(np.float32), vector),
                "layer": FieldSpec(np.dtype(np.float32), vector),
            },
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
        fields = {
            "labels": labels,
            "eligible": eligible,
            "relation": relation,
            "final": final_relation,
            "endpoint": endpoint_null,
            "layer": layer_shuffle,
        }
        selected = self.store.append(fields)
        return tuple(
            self.store.view(name, selected)
            for name in ("labels", "eligible", "relation", "final", "endpoint", "layer")
        )

    def close(self) -> None:
        self.store.close()
