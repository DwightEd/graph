"""Fixed-schema row arrays backed by independently closable NumPy mappings."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
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
        self._arrays = {}
        with ExitStack() as cleanup:
            for name, spec in self.fields.items():
                array = np.lib.format.open_memmap(
                    self.root / f"{name}.npy",
                    mode="w+",
                    dtype=spec.dtype,
                    shape=(self.capacity, *spec.tail_shape),
                )
                self._arrays[name] = array
                cleanup.callback(array._mmap.close)
            cleanup.pop_all()

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
        arrays = tuple(self._arrays.values())
        self._arrays.clear()
        with ExitStack() as closing:
            for array in arrays:
                closing.callback(array._mmap.close)
            for array in arrays:
                array.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
