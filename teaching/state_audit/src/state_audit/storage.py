"""The disk boundary: readable JSON/CSV and numeric NPZ, never pickle."""

import csv
import json
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_arrays(path: Path, *, compressed: bool = True, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    save = np.savez_compressed if compressed else np.savez
    save(temporary, **arrays)
    temporary.replace(path)


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def start_stage(path: Path, settings: dict, resume: bool) -> None:
    """Resume only an identical stage; successful sample markers are written last."""
    settings = json.loads(json.dumps(settings))
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; use --resume or a new output directory")
        if read_json(path) != settings:
            raise ValueError(f"{path}: settings changed; use a new output directory")
    else:
        write_json(path, settings)
