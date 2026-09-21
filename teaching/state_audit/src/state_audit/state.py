"""Shared, lazy state container. One layer is loaded at a time, without the model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model.sites import AXES, FULL_SEQUENCE_SITES
from .storage import read_arrays, read_json, write_arrays


@dataclass
class LayerState:
    layer: int
    positions: np.ndarray
    tensors: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.tensors[name]

    def at(self, name: str, positions) -> np.ndarray:
        """Select absolute input positions; fail rather than substitute missing rows."""
        saved = self.positions
        axis = AXES[name].index("position")
        if name in FULL_SEQUENCE_SITES:
            saved = np.arange(self[name].shape[axis])
        indices = {int(position): index for index, position in enumerate(saved)}
        rows = [indices[position] for position in positions]
        return np.take(self[name], rows, axis=axis)

    def save(self, path: Path) -> None:
        write_arrays(path, queries=self.positions, **self.tensors)

    def select(self, positions):
        """Slice query-side states, keeping complete source K/V and RoPE metadata."""
        indices = {int(position): index for index, position in enumerate(self.positions)}
        rows = [indices[position] for position in positions]
        tensors = {}
        for name, value in self.tensors.items():
            if name in AXES and name not in FULL_SEQUENCE_SITES:
                value = self.at(name, positions)
            elif name == "attention_bias":
                value = value[rows]
            tensors[name] = value
        return LayerState(self.layer, np.asarray(positions), tensors)

    @classmethod
    def load(cls, path: Path, layer: int):
        arrays = read_arrays(path)
        return cls(layer, arrays.pop("queries"), arrays)


@dataclass
class ModelState:
    directory: Path
    metadata: dict

    @classmethod
    def open(cls, directory: Path):
        return cls(directory, read_json(directory / "complete.json"))

    @property
    def layers(self) -> list[int]:
        return self.metadata["layers"]

    @property
    def readout(self) -> dict[str, np.ndarray]:
        return read_arrays(self.directory / "readout.npz")

    def layer(self, index: int) -> LayerState:
        return LayerState.load(self.directory / f"layer_{index:03d}.npz", index)

    def at(self, name: str, positions, layer: int | None = None) -> np.ndarray:
        """Read global or layer states using the same absolute positions as Target."""
        if layer is not None:
            return self.layer(layer).at(name, positions)
        arrays = self.readout
        # v1 stored only response queries; v2 can also capture prompt states.
        saved = arrays.get("state_positions", arrays["queries"])
        indices = {int(position): index for index, position in enumerate(saved)}
        return arrays[name][[indices[position] for position in positions]]

    def iter_layers(self):
        for index in self.layers:
            yield self.layer(index)
