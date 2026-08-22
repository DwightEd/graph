"""Small file helpers for graph audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


AUDIT_SCHEMA = "attention-graph-structure-audit-v1"
EVALUATION_SCHEMA = "attention-graph-structure-evaluation-v1"


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}
