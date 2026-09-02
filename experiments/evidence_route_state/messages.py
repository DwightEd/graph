"""Exact local attention writes with head and source identity intact."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PromptCarriers:
    """The locked all-prompt route-collapse measurements."""

    effective_sources: torch.Tensor
    effective_rank: torch.Tensor
    anchor_source: torch.Tensor


def gqa_head_index(
    query_heads: int,
    key_value_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Map every query head to the KV head used by grouped-query attention."""

    if query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return torch.arange(query_heads, device=device) // (query_heads // key_value_heads)


def output_projection_blocks(
    output_weight: torch.Tensor,
    query_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return the matching FP32 ``W_O`` block for every query head."""

    hidden = output_weight.shape[0]
    return output_weight.float().reshape(hidden, query_heads, head_dim).permute(1, 2, 0)


def output_projection_gram(
    output_weight: torch.Tensor,
    query_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return the per-head FP32 Gram used by every source-value norm."""

    blocks = output_projection_blocks(output_weight, query_heads, head_dim)
    return blocks @ blocks.transpose(1, 2)


def attention_messages(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``A[h,q,s] W_O[h] V[s,g(h)]`` for small exact checks.

    ``attention`` is ``[head, query, source]`` and ``values`` is
    ``[source, kv_head, head_dim]``.  The result is
    ``[query, head, source, hidden]``.  Full capture uses
    :func:`route_register_values` consumes the same equation without this
    impractical tensor during full capture.
    """

    query_heads, _, sources = attention.shape
    values = values[:sources]
    head_dim = values.shape[-1]
    head_to_kv = gqa_head_index(query_heads, values.shape[1], values.device)
    values_by_head = values.float()[:, head_to_kv]
    blocks = output_projection_blocks(output_weight, query_heads, head_dim)
    source_write = torch.einsum("shd,hdk->hsk", values_by_head, blocks)
    return attention.float().permute(1, 0, 2)[..., None] * source_write[None]


def reconstruct_attention_write(messages: torch.Tensor) -> torch.Tensor:
    """Sum exact head/source messages into the attention residual write."""

    return messages.float().sum(dim=(1, 2))


def source_write_norms(
    values: torch.Tensor,
    output_gram: torch.Tensor,
) -> torch.Tensor:
    """Compute ``||W_O[h] V[s,g(h)]||`` without a ``[H,S,D]`` tensor."""

    query_heads = output_gram.shape[0]
    head_to_kv = gqa_head_index(query_heads, values.shape[1], values.device)
    values_by_head = values.float()[:, head_to_kv]
    squared = torch.einsum(
        "shd,hde,she->sh", values_by_head, output_gram, values_by_head
    )
    return squared.clamp_min(0).sqrt().transpose(0, 1)


def prompt_carriers(
    mass: torch.Tensor,
    query_position: torch.Tensor,
    response_start: int,
) -> PromptCarriers:
    """Retain the established head-resolved collapse audit on prompt sources.

    ``mass`` is ``[query, head, source]`` and may be raw attention or functional
    message capacity.  Predictor self is excluded even when it is the final
    prompt token.
    """

    sources = torch.arange(mass.shape[-1], device=mass.device)
    prompt = (sources[None] < response_start) & (
        sources[None] != query_position[:, None]
    )
    selected = mass.float() * prompt[:, None]
    total = selected.sum(-1)
    valid = total > 1e-12
    probability = selected / total.clamp_min(1e-12)[..., None]
    active_heads = valid.sum(1).clamp_min(1)

    mixture = (probability * valid[..., None]).sum(1) / active_heads[:, None]
    entropy = -(mixture * mixture.clamp_min(1e-12).log()).sum(-1)
    gram = probability @ probability.transpose(1, 2)
    trace = gram.diagonal(dim1=1, dim2=2).sum(1)
    rank = trace.square() / gram.square().sum((1, 2)).clamp_min(1e-12)
    anchor = selected.argmax(-1).masked_fill(~valid, -1)
    return PromptCarriers(entropy.exp(), rank, anchor)
