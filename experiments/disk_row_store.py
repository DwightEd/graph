"""Fixed-schema row arrays backed by independently closable NumPy mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    dtype: np.dtype
    tail_shape: tuple[int, ...] = ()


class DiskRowStore:
    """Append one sample at a time without retaining earlier arrays in RAM."""

    def __init__(
        self,
        root,
        *,
        capacity: int,
        fields: Mapping[str, FieldSpec],
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.capacity = int(capacity)
        self.fields = dict(fields)
        self.rows = 0
        self._arrays = {
            name: np.lib.format.open_memmap(
                self.root / f"{name}.npy",
                mode="w+",
                dtype=spec.dtype,
                shape=(self.capacity, *spec.tail_shape),
            )
            for name, spec in self.fields.items()
        }

    def append(self, arrays: Mapping[str, np.ndarray]) -> slice:
        if arrays.keys() != self.fields.keys():
            raise ValueError("sample fields differ from the row-store schema")

        first = next(iter(arrays.values()))
        rows = len(first)
        for name, spec in self.fields.items():
            value = np.asarray(arrays[name])
            if value.dtype != spec.dtype:
                raise TypeError(f"{name} dtype differs from the row-store schema")
            if value.shape != (rows, *spec.tail_shape):
                raise ValueError(f"{name} shape differs from the row-store schema")

        end = self.rows + rows
        if end > self.capacity:
            raise ValueError("row-store capacity exceeded")
        selected = slice(self.rows, end)
        for name, value in arrays.items():
            np.copyto(self._arrays[name][selected], value, casting="no")
        self.rows = end
        return selected

    def view(self, field: str, rows: slice | None = None) -> np.ndarray:
        selected = slice(0, self.rows) if rows is None else rows
        return self._arrays[field][selected]

    def close(self) -> None:
        for array in self._arrays.values():
            array.flush()
            array._mmap.close()
        self._arrays.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
