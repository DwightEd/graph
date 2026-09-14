"""Load label-free response attention caches."""

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


class ResponseCache:
    """Load one formal sparse-CSR or legacy dense response cache."""

    def __init__(self, attention_key="attention"):
        self.attention_key = attention_key

    def load(self, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as loaded:
            names = set(loaded.files)
            sparse = None
            if {"response_row_ptr", "response_column_indices", "response_values", "attention_diagonal"}.issubset(names):
                attention = None
                sparse = {name: loaded[name].copy() for name in ("response_row_ptr", "response_column_indices", "response_values", "attention_diagonal")}
            else:
                attention = loaded[self.attention_key] if self.attention_key in names else loaded["data"]
            response_idx = int(loaded["response_idx"]) if "response_idx" in names else 0
            prompt_length = int(loaded["prompt_length"]) if "prompt_length" in names else None
            token_ids = loaded["token_ids"] if "token_ids" in names else None
            offsets = loaded["offsets"] if "offsets" in names else None
            source_id = str(loaded["source_id"].item()) if "source_id" in names else path.stem
            hidden = loaded["hidden"] if "hidden" in names else None
            # NPZ members are lazy: never deserialize unknown annotations.
            metadata = {name: loaded[name] for name in (
                "cache_version", "num_attention_layers", "num_attention_heads",
                "sample_id", "dataset_split", "original_idx", "hidden_schema",
            ) if name in names}
        return ResponseRecord(path.stem, source_id, None if attention is None else np.asarray(attention), response_idx,
                      prompt_length, token_ids, offsets, hidden, metadata, sparse)


class CacheDataset:
    """Iterate response caches without opening RAGTruth annotations."""

    def __init__(self, root, pattern="*.npz", cache=None):
        self.root = Path(root)
        self.files = sorted(self.root.glob(pattern))
        self.cache = cache or ResponseCache()

    def __iter__(self):
        return (self.cache.load(path) for path in self.files)
