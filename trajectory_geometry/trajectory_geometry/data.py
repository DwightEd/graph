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

DEFAULT_FORMAL_ROOT = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)


def _numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class SparseRowBlock:
    """A bounded block of nonzero response-row entries.

    Missing entries are not materialized. ``row`` uses the global CSR row
    index; the other coordinates are decoded views of the same rows.
    """

    row: np.ndarray
    layer: np.ndarray
    head: np.ndarray
    query: np.ndarray
    target: np.ndarray
    source: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class DenseAttentionRow:
    """One thresholded attention row with absent CSR entries filled by zero."""

    layer: int
    head: int
    query: int
    target: int
    values: np.ndarray


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

    @property
    def response_rows(self) -> int:
        return self.layers * self.heads * self.response_tokens

    def iter_sparse_row_blocks(self, block_rows: int = 4096):
        """Yield bounded CSR blocks without reconstructing a dense matrix."""
        if block_rows < 1:
            raise ValueError("block_rows must be positive")
        rows_per_layer = self.heads * self.response_tokens
        for row_start in range(0, self.response_rows, block_rows):
            row_stop = min(row_start + block_rows, self.response_rows)
            pointer = self.row_ptr[row_start : row_stop + 1]
            lengths = np.diff(pointer)
            row = np.repeat(
                np.arange(row_start, row_stop, dtype=np.int64), lengths
            )
            value_start, value_stop = int(pointer[0]), int(pointer[-1])
            query = row % self.response_tokens
            yield SparseRowBlock(
                row=row,
                layer=row // rows_per_layer,
                head=(row % rows_per_layer) // self.response_tokens,
                query=query,
                target=self.response_idx + query,
                source=self.columns[value_start:value_stop],
                weight=self.values[value_start:value_stop],
            )

    def iter_dense_rows(self, dtype=np.float32):
        """Recover the thresholded matrix one response row at a time.

        Every unretained entry is returned as zero, as requested. This recovers
        the cache-censored matrix, not the unknown original values below
        ``attention_floor``. The full ``[L,H,R,N]`` tensor is never allocated.
        """
        rows_per_layer = self.heads * self.response_tokens
        for row in range(self.response_rows):
            layer = row // rows_per_layer
            within_layer = row % rows_per_layer
            head = within_layer // self.response_tokens
            query = within_layer % self.response_tokens
            target = self.response_idx + query
            values = np.zeros(self.token_count, dtype=dtype)
            start, stop = int(self.row_ptr[row]), int(self.row_ptr[row + 1])
            values[self.columns[start:stop]] = self.values[start:stop]
            values[target] = self.diagonal[layer, head, target]
            yield DenseAttentionRow(layer, head, query, target, values)

    def validate(self) -> None:
        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be a vector")
        if self.diagonal.ndim != 3:
            raise ValueError("attention_diagonal must have shape [L,H,N]")
        if self.diagonal.shape[2] != self.token_count:
            raise ValueError("attention_diagonal and token_ids disagree")
        if not 0 < self.response_idx < self.token_count:
            raise ValueError("response_idx must split prompt and response")
        rows = self.response_rows
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
        for block in self.iter_sparse_row_blocks():
            if np.any(block.source < 0) or np.any(block.source >= block.target):
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
        diagonal=_numpy(payload["attention_diagonal"]).astype(np.float32, copy=False),
        row_ptr=_numpy(payload["response_row_ptr"]).astype(np.int64, copy=False),
        columns=_numpy(payload["response_column_indices"]).astype(np.int64, copy=False),
        values=_numpy(payload["response_values"]).astype(np.float32, copy=False),
        attention_floor=float(np.asarray(_numpy(payload["attention_floor"])).item()),
    )
    sample.validate()
    return sample


def discover_attention_files(root: str | Path, split: str | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if str(root).startswith("/path/to/"):
        raise FileNotFoundError(
            f"{root} is a documentation placeholder, not an attention cache. "
            f"Use the existing cache root: {DEFAULT_FORMAL_ROOT}"
        )
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(
            f"attention cache root does not exist: {root}. "
            f"Expected existing cache: {DEFAULT_FORMAL_ROOT}"
        )
    selected = root / split if split and (root / split).is_dir() else root
    files = sorted(selected.rglob("attention_*.pt"))
    files.extend(sorted(selected.rglob("attention_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"no attention_*.pt or attention_*.npz under {selected}. "
            "Pass the directory containing the formal train/test cache."
        )
    return files
