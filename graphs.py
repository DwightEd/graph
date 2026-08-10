"""Convert retained-attention CSR caches into token graphs."""

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class TokenGraph:
    sample_id: Any
    source_id: Any
    response_idx: int
    token_ids: torch.Tensor
    node_attr: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    trace_ptr: torch.Tensor | None = None
    trace_channel: torch.Tensor | None = None
    trace_value: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if value is not None}


def _cache_entries(sample: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return each retained CSR entry as (source, target, channel, value)."""
    token_count = sample.token_ids.numel()
    response_count = token_count - sample.response_idx
    channel_count = sample.attention_diagonal.shape[0] * sample.attention_diagonal.shape[1]
    counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=sample.token_ids.device), counts
    )
    channels = row_ids // response_count
    targets = sample.response_idx + row_ids % response_count
    sources = sample.response_column_indices.to(torch.int64)
    return sources, targets, channels, channel_count, token_count


def _validated_tau(attention_floor: float, tau: float) -> float:
    """Validate a graph threshold and return its Python float value."""
    try:
        threshold = float(tau)
    except (TypeError, ValueError) as error:
        raise ValueError("tau must be a finite number in [0, 1]") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("tau must be finite and within [0, 1]")
    if threshold < attention_floor:
        raise ValueError("tau must be at least the retained cache attention_floor")
    return threshold


def _base_graph(sample: Any) -> tuple[dict[str, Any], int]:
    channel_count = sample.attention_diagonal.shape[0] * sample.attention_diagonal.shape[1]
    node_attr = sample.attention_diagonal.reshape(channel_count, -1).transpose(0, 1).contiguous()
    return {
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "response_idx": sample.response_idx,
        "token_ids": sample.token_ids,
        "node_attr": node_attr,
    }, channel_count


def _select_relation_topk(
    candidate_indices: torch.Tensor,
    pair_targets: torch.Tensor,
    retained_mean: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    if limit <= 0:
        return candidate_indices[:0]

    score_order = torch.argsort(retained_mean[candidate_indices], descending=True, stable=True)
    score_sorted = candidate_indices[score_order]
    target_order = torch.argsort(pair_targets[score_sorted], stable=True)
    ordered_indices = score_sorted[target_order]
    ordered_targets = pair_targets[ordered_indices]
    positions = torch.arange(ordered_indices.numel(), device=ordered_indices.device)
    starts = torch.ones(ordered_indices.numel(), dtype=torch.bool, device=ordered_indices.device)
    starts[1:] = ordered_targets[1:] != ordered_targets[:-1]
    group_starts = torch.cummax(
        torch.where(starts, positions, torch.zeros_like(positions)), dim=0
    ).values
    return ordered_indices[(positions - group_starts) < limit]


def build_original_graph(sample: Any, tau: float) -> TokenGraph:
    """Build the compatibility graph from cache entries retained above ``tau``."""
    tau = _validated_tau(sample.attention_floor, tau)

    base, channel_count = _base_graph(sample)
    sources, targets, channels, _, token_count = _cache_entries(sample)
    values = sample.response_values
    entry_pairs = targets * token_count + sources
    selection = values.to(torch.float32) > float(tau)
    selected_pairs = torch.unique(entry_pairs[selection], sorted=True)
    edge_count = selected_pairs.numel()
    edge_sources = selected_pairs % token_count
    edge_targets = selected_pairs // token_count
    edge_index = torch.stack((edge_sources, edge_targets))
    edge_type = (edge_sources >= sample.response_idx).to(torch.int8)
    edge_attr = torch.zeros(
        (edge_count, channel_count), dtype=values.dtype, device=values.device
    )

    if edge_count:
        positions = torch.searchsorted(selected_pairs, entry_pairs)
        matches = positions < edge_count
        matches &= selected_pairs[positions.clamp(max=edge_count - 1)] == entry_pairs
        matches &= selection
        edge_attr[positions[matches], channels[matches]] = values[matches]

    return TokenGraph(
        **base,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
    )


def build_relation_topk_graph(
    sample: Any,
    k_prompt: int,
    k_history: int,
    with_channels: bool = False,
) -> TokenGraph:
    """Select relation top-k edges within the retained cache, not full attention.

    Scores are the mean over channels of values retained in the CSR cache;
    missing channel entries contribute zero.  Consequently this function cannot
    claim an exact top-k over attention values absent from that cache.
    """
    if k_prompt < 0 or k_history < 0:
        raise ValueError("k_prompt and k_history must be non-negative")
    base, channel_count = _base_graph(sample)
    sources, targets, channels, _, token_count = _cache_entries(sample)
    values = sample.response_values
    entry_pairs = targets * token_count + sources
    unique_pairs, inverse = torch.unique(entry_pairs, sorted=True, return_inverse=True)
    retained_sums = torch.zeros(
        unique_pairs.numel(), dtype=torch.float32, device=values.device
    )
    retained_sums.index_add_(0, inverse, values.to(torch.float32))
    retained_mean = retained_sums / channel_count
    pair_sources = unique_pairs % token_count
    pair_targets = unique_pairs // token_count

    pair_indices = torch.arange(unique_pairs.numel(), device=values.device)
    prompt_indices = _select_relation_topk(
        pair_indices[pair_sources < sample.response_idx], pair_targets, retained_mean, k_prompt
    )
    history_indices = _select_relation_topk(
        pair_indices[pair_sources >= sample.response_idx], pair_targets, retained_mean, k_history
    )
    chosen_indices = torch.cat((prompt_indices, history_indices))
    edge_type = torch.cat(
        (
            torch.zeros(prompt_indices.numel(), dtype=torch.int8, device=values.device),
            torch.ones(history_indices.numel(), dtype=torch.int8, device=values.device),
        )
    )
    group_order = torch.argsort(pair_targets[chosen_indices] * 2 + edge_type, stable=True)
    chosen_indices = chosen_indices[group_order]
    edge_type = edge_type[group_order]

    chosen_pairs = unique_pairs[chosen_indices]
    edge_sources = chosen_pairs % token_count
    edge_targets = chosen_pairs // token_count
    edge_index = torch.stack((edge_sources, edge_targets))
    graph = TokenGraph(
        **base,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=retained_mean[chosen_indices],
    )

    if not with_channels:
        return graph

    edge_for_pair = torch.full(
        (unique_pairs.numel(),), -1, dtype=torch.int64, device=values.device
    )
    edge_for_pair[chosen_indices] = torch.arange(chosen_indices.numel(), device=values.device)
    traced_edges = edge_for_pair[inverse]
    traced_mask = traced_edges >= 0
    traced_edges = traced_edges[traced_mask]
    trace_order = torch.argsort(traced_edges, stable=True)
    traced_edges = traced_edges[trace_order]
    trace_channel = channels[traced_mask][trace_order].to(torch.int32)
    trace_value = values[traced_mask][trace_order]
    trace_counts = torch.bincount(traced_edges, minlength=chosen_indices.numel())
    trace_ptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=values.device),
            trace_counts.cumsum(0),
        )
    )
    graph.trace_ptr = trace_ptr
    graph.trace_channel = trace_channel
    graph.trace_value = trace_value
    return graph
