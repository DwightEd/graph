"""Read one attention cache without reading hallucination annotations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResponseRecord:
    response_id: str
    source_id: str
    attention: np.ndarray | None
    response_idx: int
    prompt_length: int | None = None
    token_ids: np.ndarray | None = None
    offsets: np.ndarray | None = None
    hidden: np.ndarray | None = None
    metadata: dict | None = None
    sparse: dict | None = None
    query_positions: np.ndarray | None = None


class ResponseCache:
    """Canonical CSR uses q=P+r; compact legacy rows use q=P-1+r.

    A compact cache may override the legacy convention with query_positions.
    Only the explicitly listed metadata fields are read. Hidden states are not
    needed by the source-flow method and are not materialized by this loader.
    """

    def __init__(self, attention_key="attention"):
        self.attention_key = attention_key

    def load(self, path):
        path = Path(path)
        fields = ("response_row_ptr", "response_column_indices", "response_values", "attention_diagonal")
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            sparse = {k: data[k] for k in fields} if set(fields) <= names else None
            attention = None if sparse is not None else data[self.attention_key if self.attention_key in names else "data"]
            p = int(data["prompt_length"]) if "prompt_length" in names else int(data["response_idx"])
            response_idx = int(data["response_idx"]) if "response_idx" in names else p
            if response_idx != p:
                raise ValueError("response_idx and prompt_length must identify the same prompt boundary")
            token_ids = data["token_ids"] if "token_ids" in names else None
            offsets = data["offsets"] if "offsets" in names else None
            source_id = str(data["source_id"].item()) if "source_id" in names else path.stem
            metadata = {k: data[k] for k in (
                "cache_version", "sample_id", "dataset_split", "official_split", "task", "generator",
                "response_sha256", "attention_floor",
            ) if k in names}
            queries = data["query_positions"] if "query_positions" in names else None
        return ResponseRecord(path.stem, source_id, attention, response_idx, p,
                              token_ids, offsets, None, metadata, sparse, queries)


class CacheDataset:
    """Iterate files, not a list of all decoded attention tensors."""

    def __init__(self, root, pattern="*.npz", cache=None):
        self.root = Path(root)
        self.files = sorted(self.root.glob(pattern))
        self.cache = cache or ResponseCache()

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for path in self.files:
            yield self.cache.load(path)
