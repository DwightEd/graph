"""Aligned train reservoirs for independent one-class fitting and calibration."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

import numpy as np


class AlignedReservoir:
    """Keep the same bottom-k token rows for every atomic feature block.

    ``fit`` and ``cal`` are independent streams: fitting a position reference
    cannot consume the rows reserved to calibrate its anomaly score.  Within a
    stream, every block is replaced at the same reservoir slot, so their rows
    remain exactly aligned.
    """

    def __init__(self, *, position_bins: int, size: int, seed: int,
                 groups: tuple[str, ...] = ("fit", "cal")):
        self.position_bins = int(position_bins)
        self.capacity = max(1, int(math.ceil(int(size) / self.position_bins)))
        self.groups = tuple(groups)
        self._rng = {
            group: np.random.default_rng(np.random.SeedSequence([seed, index]))
            for index, group in enumerate(self.groups)
        }
        self._block_names: tuple[str, ...] | None = None
        self._widths: dict[str, int] | None = None
        self._values: dict[str, dict[str, np.ndarray]] | None = None
        self._priorities: dict[str, np.ndarray] | None = None
        self._filled: dict[str, np.ndarray] | None = None

    @property
    def maximum_rows(self) -> int:
        return self.position_bins * self.capacity

    @property
    def block_names(self) -> tuple[str, ...]:
        if self._block_names is None:
            raise ValueError("reservoir has no token rows")
        return self._block_names

    @property
    def block_widths(self) -> dict[str, int]:
        if self._widths is None:
            raise ValueError("reservoir has no token rows")
        return dict(self._widths)

    def add(self, group: str, blocks: Mapping[str, np.ndarray],
            position: np.ndarray) -> None:
        """Add one token batch to one stream using shared bottom-k priorities."""
        values = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in blocks.items()
        }
        self._initialize(values)
        assert self._block_names is not None
        assert self._widths is not None
        assert self._values is not None
        assert self._priorities is not None
        assert self._filled is not None
        if group not in self.groups:
            raise ValueError(f"unknown reservoir group: {group}")
        if tuple(values) != self._block_names or any(
            value.ndim != 2 or value.shape[1] != self._widths[name]
            for name, value in values.items()
        ):
            raise ValueError("atomic block contract changed after first add")
        rows = len(next(iter(values.values())))
        if any(len(value) != rows for value in values.values()):
            raise ValueError("atomic blocks must have the same token rows")
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (rows,):
            raise ValueError("position must have one value per token row")
        bins = np.minimum(
            (position * self.position_bins).astype(np.int64), self.position_bins - 1
        )
        priorities = self._rng[group].random(rows)
        for bin_id in range(self.position_bins):
            selected = np.flatnonzero(bins == bin_id)
            if not len(selected):
                continue
            filled = int(self._filled[group][bin_id])
            candidate_priorities = np.concatenate((
                self._priorities[group][bin_id, :filled], priorities[selected]
            ))
            keep = np.argsort(candidate_priorities, kind="stable")[:self.capacity]
            retained = len(keep)
            for name in self._block_names:
                candidates = np.concatenate((
                    self._values[group][name][bin_id, :filled], values[name][selected]
                ))
                self._values[group][name][bin_id, :retained] = candidates[keep].astype(
                    np.float16
                )
            self._priorities[group][bin_id, :retained] = candidate_priorities[keep]
            self._filled[group][bin_id] = retained

    def matrix(self, group: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return aligned sampled blocks and their position-bin labels."""
        if self._values is None or self._filled is None or self._block_names is None:
            raise ValueError("reservoir has no token rows")
        if group not in self.groups:
            raise ValueError(f"unknown reservoir group: {group}")
        counts = self._filled[group]
        if not int(counts.sum()):
            raise ValueError(f"reservoir group {group} has no token rows")
        blocks = {name: self.block(group, name) for name in self._block_names}
        return blocks, self.bins(group)

    def block(self, group: str, name: str) -> np.ndarray:
        """Materialize one atomic block, keeping peak host memory bounded."""
        if self._values is None or self._filled is None:
            raise ValueError("reservoir has no token rows")
        if group not in self.groups or name not in self.block_names:
            raise ValueError("unknown reservoir group or atomic block")
        return np.concatenate([
            self._values[group][name][bin_id, :count]
            for bin_id, count in enumerate(self._filled[group]) if count
        ]).astype(np.float32)

    def bins(self, group: str) -> np.ndarray:
        """Return the position-bin labels shared by every block in a group."""
        if self._filled is None or group not in self.groups:
            raise ValueError("unknown or empty reservoir group")
        if not int(self._filled[group].sum()):
            raise ValueError(f"reservoir group {group} has no token rows")
        return np.concatenate([
            np.full(count, bin_id, dtype=np.int16)
            for bin_id, count in enumerate(self._filled[group]) if count
        ])

    def snapshot(self, *, copy_arrays: bool = True) -> dict:
        """Return the minimal state required to continue sampling bit-for-bit."""
        copy = (lambda value: value.copy()) if copy_arrays else (lambda value: value)
        return {
            "block_names": self._block_names,
            "widths": dict(self._widths) if self._widths is not None else None,
            "values": None if self._values is None else {
                group: {name: copy(values) for name, values in block_values.items()}
                for group, block_values in self._values.items()
            },
            "priorities": None if self._priorities is None else {
                group: copy(values) for group, values in self._priorities.items()
            },
            "filled": None if self._filled is None else {
                group: copy(values) for group, values in self._filled.items()
            },
            "rng_state": {
                group: deepcopy(rng.bit_generator.state)
                for group, rng in self._rng.items()
            },
        }

    def restore(self, state: dict, *, copy_arrays: bool = True) -> "AlignedReservoir":
        """Restore a snapshot produced by an equivalent reservoir."""
        copy = (lambda value: value.copy()) if copy_arrays else (lambda value: value)
        self._block_names = tuple(state["block_names"])
        self._widths = {name: int(width) for name, width in state["widths"].items()}
        self._values = {
            group: {
                name: copy(np.asarray(values, dtype=np.float16))
                for name, values in block_values.items()
            }
            for group, block_values in state["values"].items()
        }
        self._priorities = {
            group: copy(np.asarray(values, dtype=np.float64))
            for group, values in state["priorities"].items()
        }
        self._filled = {
            group: copy(np.asarray(values, dtype=np.int64))
            for group, values in state["filled"].items()
        }
        for group, rng_state in state["rng_state"].items():
            self._rng[group].bit_generator.state = deepcopy(rng_state)
        return self

    def _initialize(self, values: dict[str, np.ndarray]) -> None:
        if self._block_names is not None:
            return
        if not values or any(value.ndim != 2 for value in values.values()):
            raise ValueError("atomic blocks must be non-empty matrices")
        self._block_names = tuple(values)
        self._widths = {name: int(value.shape[1]) for name, value in values.items()}
        self._values = {
            group: {
                name: np.zeros(
                    (self.position_bins, self.capacity, width), dtype=np.float16
                )
                for name, width in self._widths.items()
            }
            for group in self.groups
        }
        self._priorities = {
            group: np.full((self.position_bins, self.capacity), np.inf, dtype=np.float64)
            for group in self.groups
        }
        self._filled = {
            group: np.zeros(self.position_bins, dtype=np.int64) for group in self.groups
        }
