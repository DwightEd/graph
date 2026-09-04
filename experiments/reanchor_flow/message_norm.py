"""Exact per-head Value norms after the matching W_O block."""

from __future__ import annotations

import torch
from torch import Tensor

HEAD_CHUNK = 4
CACHE_ATTRIBUTE = "_routing_rhythm_output_gram_cpu"


def output_gram(output_weight: Tensor, heads: int, head_dim: int) -> Tensor:
    hidden = output_weight.shape[0]
    if output_weight.shape != (hidden, heads * head_dim):
        raise ValueError("W_O input width does not match query heads")
    blocks = output_weight.view(hidden, heads, head_dim).permute(1, 2, 0)
    gram = torch.empty(
        (heads, head_dim, head_dim),
        device=output_weight.device,
        dtype=torch.float32,
    )
    for begin in range(0, heads, HEAD_CHUNK):
        end = min(begin + HEAD_CHUNK, heads)
        block = blocks[begin:end].float()
        gram[begin:end] = torch.bmm(block, block.transpose(1, 2))
    return gram.cpu()


def source_norm(value: Tensor, output_weight: Tensor, gram: Tensor | None = None) -> Tensor:
    """Return exact ``||W_O[h] V[h,s]||`` for every head and source."""

    heads, sources, head_dim = value.shape
    if gram is None:
        gram = output_gram(output_weight, heads, head_dim)
    gram = gram.to(value.device)
    result = torch.empty((heads, sources), device=value.device, dtype=torch.float32)
    for begin in range(0, heads, HEAD_CHUNK):
        end = min(begin + HEAD_CHUNK, heads)
        current = value[begin:end].float()
        squared = torch.einsum(
            "hsd,hde,hse->hs",
            current,
            gram[begin:end],
            current,
        )
        result[begin:end] = squared.clamp_min(0).sqrt()
    return result


def model_gram_cache(model) -> dict[int, Tensor]:
    cache = getattr(model, CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        setattr(model, CACHE_ATTRIBUTE, cache)
    return cache
