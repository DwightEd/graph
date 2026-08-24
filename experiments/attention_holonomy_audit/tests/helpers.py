from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import torch


@dataclass
class FakeAttention:
    response_idx: int
    num_layers: int
    num_heads: int
    num_tokens: int
    attention_floor: float
    token_ids: torch.Tensor
    attention_diagonal: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    weight: torch.Tensor

    @property
    def num_response_tokens(self):
        return self.num_tokens - self.response_idx

    @property
    def response_values(self):
        return self.weight


class FakeSample:
    def __init__(self, sample_id: str, source_id: str, attention: FakeAttention, task_type="QA"):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = task_type
        self._attention = attention

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        del block_rows
        query = self._attention.target - self._attention.response_idx
        yield SimpleNamespace(
            source=self._attention.source,
            target=self._attention.target,
            layer=self._attention.layer,
            head=self._attention.head,
            query=query,
            weight=self._attention.weight,
        )

    def release_attention(self):
        pass


class FakeLabels:
    def __init__(self, labels):
        self.labels = labels

    def response_labels(self, sample):
        return self.labels[sample.sample_id]


class FakeDataset:
    def __init__(self, root: Path, split: str, samples, labels=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = {sample.sample_id: sample for sample in samples}
        first = samples[0].attention()
        self.manifest = {
            "split": split,
            "num_layers": first.num_layers,
            "num_heads": first.num_heads,
        }
        (root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.device = "cpu"
        self._labels = labels or {}

    @property
    def sample_ids(self):
        return list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return FakeLabels(self._labels)


def make_sample(sample_id: str, source_id: str, shift: float = 0.0) -> FakeSample:
    response_idx = 2
    layers = 3
    heads = 2
    num_tokens = 6
    token_ids = torch.arange(100, 100 + num_tokens, dtype=torch.long)
    diagonal = torch.full((layers, heads, num_tokens), 0.12, dtype=torch.float32)
    event_pairs = [
        (0, 2),
        (1, 2),
        (2, 3),
        (0, 3),
        (3, 4),
        (2, 4),
        (4, 5),
        (3, 5),
    ]
    source = []
    target = []
    layer = []
    head = []
    weight = []
    for depth in range(layers):
        for pair_index, (left, right) in enumerate(event_pairs):
            for current_head in range(heads):
                source.append(left)
                target.append(right)
                layer.append(depth)
                head.append(current_head)
                base = 0.025 + 0.008 * pair_index + 0.004 * depth
                weight.append(base * (1.0 + 0.45 * current_head + shift))

    attention = FakeAttention(
        response_idx=response_idx,
        num_layers=layers,
        num_heads=heads,
        num_tokens=num_tokens,
        attention_floor=0.01,
        token_ids=token_ids,
        attention_diagonal=diagonal,
        source=torch.tensor(source, dtype=torch.long),
        target=torch.tensor(target, dtype=torch.long),
        layer=torch.tensor(layer, dtype=torch.long),
        head=torch.tensor(head, dtype=torch.long),
        weight=torch.tensor(weight, dtype=torch.float32),
    )
    return FakeSample(sample_id, source_id, attention)
