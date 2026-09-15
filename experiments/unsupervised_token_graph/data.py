"""Read existing attention NPZs and reuse their original token identities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache_index import identity_fields


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
    """Canonical CSR, legacy dense arrays, or AttentionAdjacency.save_layer_head.

    CSR/exported rows start at q=P. Compact legacy arrays retain their existing
    q=P-1 convention unless query_positions are supplied. No labels are read.
    """

    def __init__(self, attention_key="attention", index=None):
        self.attention_key = attention_key
        self.index = index

    def load(self, path):
        path = Path(path)
        fields = ("response_row_ptr", "response_column_indices", "response_values", "attention_diagonal")
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            native = identity_fields(data)
            sparse = {k: data[k] for k in fields} if set(fields) <= names else None
            queries = data["query_positions"] if "query_positions" in names else None
            channel_ids = {}
            if sparse is not None:
                attention = None
            elif self.attention_key in names or "data" in names:
                attention = data[self.attention_key if self.attention_key in names else "data"]
            elif "adjacency" in names:
                attention = data["adjacency"][None, None]
                queries = int(native["prompt_length"]) + np.arange(attention.shape[2])
                channel_ids = {k: int(data[k]) for k in ("layer", "head") if k in names}
            else:
                raise ValueError("expected canonical CSR, attention/data, or exported adjacency; "
                                 "local_attention and feature-only NPZs are not full attention graphs")
        identity, files = self.index.resolve(path, native) if self.index else (native, [])
        p = int(identity["prompt_length"])
        token_ids = np.asarray(identity["token_ids"]) if "token_ids" in identity else None
        offsets = np.asarray(identity["offsets"]) if "offsets" in identity else None
        if sparse is not None and token_ids is not None:
            if sparse["attention_diagonal"].shape[-1] != len(token_ids):
                raise ValueError("canonical token_ids and attention length disagree")
        if offsets is not None:
            if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.issubdtype(offsets.dtype, np.integer):
                raise ValueError("original integer response-relative offsets [R,2] are required")
            if token_ids is not None and len(offsets) != len(token_ids) - p:
                raise ValueError("offsets and original response token count disagree")
        metadata = {k: v for k, v in identity.items() if k not in ("token_ids", "offsets", "prompt_length")}
        metadata.update(channel_ids, identity_files=files)
        return ResponseRecord(identity.get("id", path.stem), identity.get("source_id", ""),
                              attention, p, p, token_ids, offsets, None, metadata, sparse, queries)


class CacheDataset:
    """Iterate files, not all decoded tensors; a single NPZ path is also valid."""

    def __init__(self, root, pattern="*.npz", cache=None):
        root = Path(root)
        self.root = root.parent if root.is_file() else root
        self.files = [root] if root.is_file() else sorted(root.glob(pattern))
        self.cache = cache or ResponseCache()

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for path in self.files:
            yield self.cache.load(path)
