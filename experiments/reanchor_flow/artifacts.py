"""Compact NPZ persistence for frozen re-anchor flow results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def save_result(path: str | Path, values: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary, **{name: as_array(value) for name, value in values.items()}
    )
    temporary.replace(destination)


def load_result(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as stored:
        return {name: np.array(stored[name], copy=True) for name in stored.files}
