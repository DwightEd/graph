"""Disk-backed exact ensemble metrics with bounded resident memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.disk_row_store import DiskRowStore, FieldSpec

from .statistics import binary_metrics


@dataclass(frozen=True)
class EnsembleAUPRC:
    real: np.ndarray
    null: np.ndarray


class DiskBackedAUPRC:
    """Accumulate token scores on disk and evaluate one replicate at a time."""

    def __init__(self, path, *, capacity: int, replicates: int, relations: int):
        self.store = DiskRowStore(
            path,
            capacity=capacity,
            fields={
                "labels": FieldSpec(np.dtype(np.int8)),
                "real": FieldSpec(np.dtype(np.float32), (relations,)),
                "null": FieldSpec(
                    np.dtype(np.float32),
                    (replicates, relations),
                ),
            },
        )
        self.replicates = replicates
        self.relations = relations

    @property
    def rows(self) -> int:
        return self.store.rows

    @property
    def labels(self) -> np.ndarray:
        return self.store.view("labels")

    @property
    def real(self) -> np.ndarray:
        return self.store.view("real")

    @property
    def null(self) -> np.ndarray:
        return np.moveaxis(self.store.view("null"), 1, 0)

    def add(self, labels: np.ndarray, real: np.ndarray, null: np.ndarray) -> None:
        self.store.append(
            {
                "labels": np.asarray(labels, dtype=np.int8),
                "real": np.asarray(real, dtype=np.float32),
                "null": np.moveaxis(
                    np.asarray(null, dtype=np.float32),
                    0,
                    1,
                ),
            }
        )

    def add_masked(
        self,
        labels: np.ndarray,
        real: np.ndarray,
        null: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Write selected rows without copying a full replicate ensemble."""

        mask = np.asarray(mask, dtype=bool)
        selected_labels = np.asarray(labels[mask], dtype=np.int8)
        selected_real = np.asarray(real[mask], dtype=np.float32)
        end = self.rows + len(selected_labels)
        if end > self.store.capacity:
            raise ValueError("row-store capacity exceeded")

        selected = slice(self.rows, end)
        selected_null = self.store.view("null", selected)
        for replicate in range(self.replicates):
            np.copyto(
                selected_null[:, replicate],
                np.asarray(null[replicate], dtype=np.float32)[mask],
                casting="no",
            )
        self.store.append(
            {
                "labels": selected_labels,
                "real": selected_real,
                "null": selected_null,
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def finish(self) -> EnsembleAUPRC:
        labels = self.labels
        observed = np.asarray(
            [
                binary_metrics(labels, self.real[:, relation])["auprc"]
                for relation in range(self.relations)
            ],
            dtype=np.float64,
        )
        null = np.empty((self.replicates, self.relations), dtype=np.float64)
        for replicate in range(self.replicates):
            for relation in range(self.relations):
                null[replicate, relation] = binary_metrics(
                    labels, self.null[replicate, :, relation]
                )["auprc"]
        return EnsembleAUPRC(real=observed, null=null)

    def close(self) -> None:
        self.store.close()
