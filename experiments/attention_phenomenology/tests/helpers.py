from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class SyntheticAttention:
    num_layers: int
    num_heads: int
    num_response_tokens: int
    response_idx: int
    attention_diagonal: torch.Tensor
    attention_floor: float = 0.01
    response_value_count: int = 0

    @property
    def num_tokens(self):
        return self.response_idx + self.num_response_tokens

    @property
    def num_channels(self):
        return self.num_layers * self.num_heads

    @property
    def response_values(self):
        return torch.empty(self.response_value_count)


class SyntheticSample:
    def __init__(self, attention, edges):
        self._attention = attention
        self.edges = edges
        self._attention.response_value_count = len(edges[-1])

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        layer, head, query, source, weight = self.edges
        yield SimpleNamespace(
            layer=torch.tensor(layer, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            query=torch.tensor(query, dtype=torch.long),
            source=torch.tensor(source, dtype=torch.long),
            weight=torch.tensor(weight, dtype=torch.float32),
        )
