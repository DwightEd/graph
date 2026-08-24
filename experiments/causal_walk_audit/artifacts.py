"""Small, pickle-free artifacts for the typed route-grammar method."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REFERENCE_SCHEMA = "typed-route-grammar-reference-v2"
SCORE_SCHEMA = "typed-route-grammar-score-v2"
EVALUATION_SCHEMA = "typed-route-grammar-evaluation-v2"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def write_json(path, value: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
