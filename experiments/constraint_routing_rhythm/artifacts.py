"""Thin NPZ persistence for frozen per-token intervention results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def check_result(result: Mapping[str, np.ndarray]) -> None:
    """Check the event axis and the teacher-forcing offset at the file boundary."""

    missing = {"query_position", "prediction_position"} - result.keys()
    if missing:
        raise ValueError(f"result is missing: {', '.join(sorted(missing))}")

    query = result["query_position"]
    prediction = result["prediction_position"]
    if query.ndim != 1 or prediction.shape != query.shape:
        raise ValueError("query and prediction positions must be aligned vectors")
    if not np.array_equal(prediction, query + 1):
        raise ValueError("prediction_position must equal query_position + 1")

    event_count = len(query)
    for name, value in result.items():
        if value.ndim == 1 and len(value) != event_count:
            raise ValueError(f"event array has the wrong length: {name}")


def save_result(path: str | Path, result: Mapping[str, object]) -> None:
    """Atomically save uncompressed numeric arrays without identity machinery."""

    arrays = {name: as_array(value) for name, value in result.items()}
    check_result(arrays)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_result(path: str | Path) -> dict[str, np.ndarray]:
    """Load one result and validate only its common event coordinates."""

    with np.load(Path(path), allow_pickle=False) as stored:
        result = {name: np.array(stored[name], copy=True) for name in stored.files}
    check_result(result)
    return result
