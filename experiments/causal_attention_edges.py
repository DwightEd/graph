"""Preallocated decoding of retained causal off-diagonal attention edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CausalAttentionEdges:
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.weight.numel())


def collect_causal_attention_edges(
    sample,
    *,
    block_rows: int,
) -> CausalAttentionEdges:
    """Decode blocks into one bounded buffer instead of list-plus-concatenate."""

    attention = sample.attention()
    capacity = int(attention.response_values.numel())
    device = attention.response_values.device
    buffers = {
        name: torch.empty(capacity, dtype=torch.long, device=device)
        for name in ("layer", "head", "query", "source")
    }
    buffers["weight"] = torch.empty(capacity, dtype=torch.float32, device=device)

    rows = 0
    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        selected = block.source < int(attention.response_idx) + block.query
        count = int(selected.sum())
        if count == 0:
            continue
        current = slice(rows, rows + count)
        for name in ("layer", "head", "query", "source"):
            buffers[name][current].copy_(getattr(block, name)[selected])
        buffers["weight"][current].copy_(block.weight[selected].float().clamp_min(0.0))
        rows += count

    return CausalAttentionEdges(
        **{name: value[:rows] for name, value in buffers.items()}
    )
