"""Exact local attention writes with head and source identity intact."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MessageStatistics:
    """Dense scalar accounts and per-head totals for one query chunk."""

    capacity: torch.Tensor
    support: torch.Tensor
    head_write: torch.Tensor
    attention_write: torch.Tensor


@dataclass(frozen=True)
class MLPDiagnostics:
    """Same-token MLP update geometry; no cross-token route is implied."""

    write_norm: torch.Tensor
    relative_norm: torch.Tensor
    state_cosine: torch.Tensor


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
    :func:`message_statistics` because this dense vector tensor is too large
    for real sequences.
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


def message_statistics(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_blocks: torch.Tensor,
    post_attention_state: torch.Tensor,
    source_norm: torch.Tensor | None = None,
) -> MessageStatistics:
    """Compute exact edge accounts without materializing edge vectors.

    All derived products are FP32.  ``capacity`` and ``support`` have shape
    ``[query, head, source]``; ``head_write`` keeps the additive total written
    by each head for exact sparse-tail accounting.
    """

    query_heads, _, sources = attention.shape
    values = values[:sources]
    head_to_kv = gqa_head_index(query_heads, values.shape[1], values.device)
    values_by_head = values.float()[:, head_to_kv]
    attention_qhs = attention.float().permute(1, 0, 2)
    state = post_attention_state.float()

    if source_norm is None:
        gram = output_blocks @ output_blocks.transpose(1, 2)
        source_norm = source_write_norms(values, gram)
    capacity = attention_qhs * source_norm[None, :, :sources].float()

    state_in_head_space = torch.einsum("hdk,qk->qhd", output_blocks, state)
    source_alignment = torch.einsum("shd,qhd->qhs", values_by_head, state_in_head_space)
    denominator = state.square().sum(-1).clamp_min(1e-12)
    support = attention_qhs * source_alignment / denominator[:, None, None]

    head_context = torch.einsum("hqs,shd->qhd", attention.float(), values_by_head)
    head_write = torch.einsum("qhd,hdk->qhk", head_context, output_blocks)
    return MessageStatistics(
        capacity=capacity,
        support=support,
        head_write=head_write,
        attention_write=head_write.sum(1),
    )


def selected_messages(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_blocks: torch.Tensor,
    head: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Materialize only selected head/source message vectors for one query."""

    query = torch.zeros_like(head)
    return selected_chunk_messages(
        attention[:, None], values, output_blocks, query, head, source
    )


def selected_chunk_messages(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_blocks: torch.Tensor,
    query: torch.Tensor,
    head: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Materialize head-major selected vectors from one query chunk."""

    if len(head) == 0:
        return output_blocks.new_empty((0, output_blocks.shape[-1]))
    head_to_kv = gqa_head_index(output_blocks.shape[0], values.shape[1], values.device)
    selected_value = values[source, head_to_kv[head]].float()
    selected_write = output_blocks.new_empty((len(head), output_blocks.shape[-1]))

    # Advanced-indexing W_O by every edge would replicate a 2 MB head block
    # thousands of times.  The graph selector groups edges by head so each
    # block projects one contiguous batch of selected values.
    counts = torch.bincount(head, minlength=output_blocks.shape[0]).cpu().tolist()
    start = 0
    for current_head, count in enumerate(counts):
        stop = start + count
        if count:
            selected_write[start:stop] = (
                selected_value[start:stop] @ output_blocks[current_head]
            )
        start = stop
    coefficient = attention[head, query, source].float()
    return coefficient[:, None] * selected_write


def reconstruction_error(
    reconstructed: torch.Tensor,
    native_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-query maximum and relative-L2 reconstruction errors."""

    native = native_write.float()
    difference = reconstructed.float() - native
    maximum = difference.abs().amax(-1)
    relative = difference.norm(dim=-1) / native.norm(dim=-1).clamp_min(1e-12)
    return maximum, relative


def mlp_diagnostics(
    post_attention_state: torch.Tensor,
    mlp_write: torch.Tensor,
) -> MLPDiagnostics:
    """Describe the local MLP update without treating it as an information edge."""

    state = post_attention_state.float()
    write = mlp_write.float()
    state_norm = state.norm(dim=-1)
    write_norm = write.norm(dim=-1)
    denominator = (state_norm * write_norm).clamp_min(1e-12)
    return MLPDiagnostics(
        write_norm=write_norm,
        relative_norm=write_norm / state_norm.clamp_min(1e-12),
        state_cosine=(state * write).sum(-1) / denominator,
    )


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
