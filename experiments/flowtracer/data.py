"""Compressed attention loading for FlowTracer."""

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class AttentionSample:

    attention: np.ndarray
    token_ids: list | None = None
    metadata: dict | None = None

    @property
    def shape(self):
        return self.attention.shape

    @property
    def num_tokens(self):
        return self.attention.shape[-1]

    def aggregate(self, layers=None, heads=None):
        from .attention import aggregate_attention
        return aggregate_attention(self.attention, layers=layers, heads=heads)


class AttentionDataset:

    def __init__(self, root, pattern="*.npz"):
        self.root = Path(root)
        self.files = sorted(self.root.glob(pattern)) if self.root.is_dir() else [self.root]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return (self.load(path) for path in self.files)

    @staticmethod
    def load(path, key="attention"):
        path = Path(path)
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            attention = loaded
            metadata = {}
        else:
            names = set(loaded.files)
            if key in names:
                attention = loaded[key]
            elif {"layer", "head", "source", "target", "weight", "shape"}.issubset(names):
                attention = AttentionDataset._from_coo(loaded)
            elif "data" in names:
                attention = loaded["data"]
            else:
                attention = loaded[loaded.files[0]]
            metadata = {name: loaded[name] for name in names if name not in {key, "data"}}
        return AttentionSample(np.asarray(attention, dtype=np.float32), metadata=metadata)

    @staticmethod
    def _from_coo(data):
        attention = np.zeros(tuple(data["shape"].tolist()), dtype=np.float32)
        attention[tuple(data[name].astype(np.int64) for name in ("layer", "head", "target", "source"))] = data["weight"]
        return attention


def iter_samples(path):
    if Path(path).is_dir():
        yield from AttentionDataset(path)
        return
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        yield from (value if isinstance(value, list) else [value])


def get_token_regions(sample):
    if "target_tokens" in sample:
        target = sorted({int(i) for i in sample["target_tokens"]})
    elif "target" in sample:
        value = sample["target"]
        start, end = (value["start"], value["end"]) if isinstance(value, dict) else value
        target = list(range(int(start), int(end)))
    elif "target_span" in sample:
        start, end = sample["target_span"]
        target = list(range(int(start), int(end)))
    else:
        raise ValueError("sample requires target_tokens, target, or target_span")
    return target
