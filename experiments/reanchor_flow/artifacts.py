"""Atomic persistence for frozen mechanism-audit results."""

from __future__ import annotations

import json
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


def save_json(path: str | Path, value: object) -> None:
    """Atomically write strict, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
