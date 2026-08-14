"""Strict, label-blind access to the existing sparse attention cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # NPZ-only analysis and unit tests remain available.
    torch = None  # type: ignore[assignment]


REQUIRED_FIELDS = {
    "response_idx",
    "token_ids",
    "attention_diagonal",
    "response_row_ptr",
    "response_column_indices",
    "response_values",
    "attention_floor",
}


def _numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class SparseAttentionSample:
    path: Path
    sample_id: str
    response_idx: int
    token_ids: np.ndarray
    diagonal: np.ndarray
    row_ptr: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    attention_floor: float

    @property
    def layers(self) -> int:
        return int(self.diagonal.shape[0])

    @property
    def heads(self) -> int:
        return int(self.diagonal.shape[1])

    @property
    def token_count(self) -> int:
        return int(self.token_ids.size)

    @property
    def response_tokens(self) -> int:
        return self.token_count - self.response_idx

    def validate(self) -> None:
        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be a vector")
        if self.diagonal.ndim != 3:
            raise ValueError("attention_diagonal must have shape [L,H,N]")
        if self.diagonal.shape[2] != self.token_count:
            raise ValueError("attention_diagonal and token_ids disagree")
        if not 0 < self.response_idx < self.token_count:
            raise ValueError("response_idx must split prompt and response")
        rows = self.layers * self.heads * self.response_tokens
        if self.row_ptr.shape != (rows + 1,):
            raise ValueError("response_row_ptr has the wrong length")
        if self.columns.ndim != 1 or self.values.ndim != 1:
            raise ValueError("sparse columns and values must be vectors")
        if self.columns.size != self.values.size:
            raise ValueError("sparse columns and values do not align")
        if self.row_ptr[0] != 0 or self.row_ptr[-1] != self.values.size:
            raise ValueError("response_row_ptr does not span response_values")
        if np.any(np.diff(self.row_ptr) < 0):
            raise ValueError("response_row_ptr must be monotone")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("response_values must be finite and nonnegative")
        if not np.all(np.isfinite(self.diagonal)) or np.any(self.diagonal < 0):
            raise ValueError("attention_diagonal must be finite and nonnegative")
        if not 0 < self.attention_floor <= 1:
            raise ValueError("attention_floor must be in (0,1]")
        lengths = np.diff(self.row_ptr)
        row = np.repeat(np.arange(rows, dtype=np.int64), lengths)
        target = self.response_idx + row % self.response_tokens
        if np.any(self.columns < 0) or np.any(self.columns >= target):
            raise ValueError("sparse attention must point to earlier tokens")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}
    if torch is None:
        raise RuntimeError("PyTorch is required to read formal .pt attention caches")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("attention cache must contain a dictionary")
    return value


def load_attention_sample(path: str | Path) -> SparseAttentionSample:
    """Load only label-blind fields; label arrays are deliberately ignored."""
    resolved = Path(path).expanduser().resolve()
    payload = _load_payload(resolved)
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"attention cache is missing fields: {sorted(missing)}")
    sample_id = payload.get("response_id", payload.get("sample_id", resolved.stem))
    sample = SparseAttentionSample(
        path=resolved,
        sample_id=str(sample_id),
        response_idx=int(np.asarray(_numpy(payload["response_idx"])).item()),
        token_ids=_numpy(payload["token_ids"]).astype(np.int64, copy=False),
        diagonal=_numpy(payload["attention_diagonal"]).astype(np.float64, copy=False),
        row_ptr=_numpy(payload["response_row_ptr"]).astype(np.int64, copy=False),
        columns=_numpy(payload["response_column_indices"]).astype(np.int64, copy=False),
        values=_numpy(payload["response_values"]).astype(np.float64, copy=False),
        attention_floor=float(np.asarray(_numpy(payload["attention_floor"])).item()),
    )
    sample.validate()
    return sample


def discover_attention_files(root: str | Path, split: str | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [root]
    selected = root / split if split and (root / split).is_dir() else root
    files = sorted(selected.rglob("attention_*.pt"))
    files.extend(sorted(selected.rglob("attention_*.npz")))
    if not files:
        raise FileNotFoundError(f"no attention_*.pt or attention_*.npz under {selected}")
    return files
