"""Disk-backed exact ensemble metrics with bounded resident memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .statistics import binary_metrics


@dataclass(frozen=True)
class EnsembleAUPRC:
    real: np.ndarray
    null: np.ndarray


class DiskBackedAUPRC:
    """Accumulate token scores on disk and evaluate one replicate at a time."""

    def __init__(self, path, *, capacity: int, replicates: int, relations: int):
        path = Path(path)
        self.labels = np.memmap(
            path.with_suffix(".labels.dat"),
            mode="w+",
            dtype=np.int8,
            shape=(capacity,),
        )
        self.real = np.memmap(
            path.with_suffix(".real.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(capacity, relations),
        )
        self.null = np.memmap(
            path.with_suffix(".null.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(replicates, capacity, relations),
        )
        self.rows = 0
        self.replicates = replicates
        self.relations = relations

    def add(self, labels: np.ndarray, real: np.ndarray, null: np.ndarray) -> None:
        end = self.rows + len(labels)
        selected = slice(self.rows, end)
        self.labels[selected] = labels
        self.real[selected] = real
        self.null[:, selected] = null
        self.rows = end

    def add_masked(
        self,
        labels: np.ndarray,
        real: np.ndarray,
        null: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Write selected rows without copying a full replicate ensemble."""

        mask = np.asarray(mask, dtype=bool)
        end = self.rows + int(mask.sum())
        selected = slice(self.rows, end)
        self.labels[selected] = labels[mask]
        self.real[selected] = real[mask]
        for replicate in range(self.replicates):
            self.null[replicate, selected] = null[replicate][mask]
        self.rows = end

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def finish(self) -> EnsembleAUPRC:
        labels = self.labels[: self.rows]
        observed = np.asarray(
            [
                binary_metrics(labels, self.real[: self.rows, relation])["auprc"]
                for relation in range(self.relations)
            ],
            dtype=np.float64,
        )
        null = np.empty((self.replicates, self.relations), dtype=np.float64)
        for replicate in range(self.replicates):
            for relation in range(self.relations):
                null[replicate, relation] = binary_metrics(
                    labels, self.null[replicate, : self.rows, relation]
                )["auprc"]
        return EnsembleAUPRC(real=observed, null=null)

    def close(self) -> None:
        for array in (self.labels, self.real, self.null):
            array.flush()
            array._mmap.close()
