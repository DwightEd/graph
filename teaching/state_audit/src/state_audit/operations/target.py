"""Absolute input positions, native head numbers, Cartesian axis selection."""

from dataclasses import dataclass

import torch

from ..model.sites import AXES, GLOBAL_SITES


@dataclass(frozen=True)
class Target:
    representation: str
    layers: tuple[int, ...] = ()
    positions: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None
    keys: tuple[int, ...] | None = None
    features: tuple[int, ...] | None = None

    def __post_init__(self):
        axes = AXES[self.representation]
        selections = {"head": self.heads, "key": self.keys, "feature": self.features}
        for axis, selection in selections.items():
            if selection is not None and axis not in axes:
                raise ValueError(f"{self.representation} has no {axis} axis")
        if (self.representation in GLOBAL_SITES) == bool(self.layers):
            raise ValueError("Use layers for block states; omit layers for global states")
        selections = (self.layers, self.positions, self.heads, self.keys, self.features)
        for selection in selections:
            if selection is not None and (
                len(set(selection)) != len(selection) or any(index < 0 for index in selection)
            ):
                raise ValueError("Indices are absolute, nonnegative and unique")

    def indices(self, tensor):
        """Broadcast index vectors, without building dense index grids."""
        selections = {
            "position": self.positions,
            "head": self.heads,
            "key": self.keys,
            "feature": self.features,
        }
        indices = []
        for dimension, axis in enumerate(AXES[self.representation]):
            selected = selections[axis]
            if selected is None:
                index = torch.arange(tensor.shape[dimension], device=tensor.device)
            else:
                index = torch.tensor(selected, device=tensor.device, dtype=torch.long)
            shape = [1] * tensor.ndim
            shape[dimension] = -1
            indices.append(index.reshape(shape))
        return tuple(indices)
