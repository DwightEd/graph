"""Artifact I/O owned by the non-neural structure audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path, names=None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        selected = arrays.files if names is None else names
        return {name: arrays[name] for name in selected}


def npz_shapes(path, names) -> dict[str, tuple[int, ...]]:
    """Read NPY headers inside an NPZ without decompressing array payloads."""

    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    shapes = {}
    with ZipFile(path) as archive:
        for name in names:
            with archive.open(f"{name}.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version not in readers:
                    raise ValueError(f"unsupported NPY header version {version}")
                shape, _, _ = readers[version](handle)
                shapes[name] = tuple(shape)
    return shapes


def write_json(path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(current) for current in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
