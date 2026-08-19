"""Small, explicit artifact helpers for the attention phenomenology workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_SCHEMA = "attention-phenomenology-reference-v2"
SCORE_SCHEMA = "attention-phenomenology-score-v2"
MANIFEST_SCHEMA = "attention-phenomenology-manifest-v2"
EVALUATION_SCHEMA = "attention-phenomenology-evaluation-v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
