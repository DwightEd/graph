"""Token graph views built from canonical sparse attention."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import torch


@dataclass
class TokenGraph:
    num_nodes: int
    response_idx: int
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor | None = None
    edge_ptr: torch.Tensor | None = None
    edge_channel: torch.Tensor | None = None
    edge_value: torch.Tensor | None = None

    def to_dict(self):
        return {name: value for name, value in self.__dict__.items() if value is not None}


def _entries(sample):
    """Decode retained CSR entries into source, target and channel vectors."""
    R = sample.num_response_tokens
    counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
    rows = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)
    channels = rows // R
    targets = sample.response_idx + rows % R
    sources = sample.response_column_indices.to(torch.int64)
    return sources, targets, channels


def _validate_tau(sample, tau) -> None:
    if (
        isinstance(tau, bool)
        or not isinstance(tau, Real)
        or not isfinite(tau)
        or not sample.attention_floor <= tau <= 1
    ):
        raise ValueError("tau must be a finite real number in [attention_floor, 1]")


def _validate_topk_limits(k_prompt, k_history) -> None:
    for name, value in (("k_prompt", k_prompt), ("k_history", k_history)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _sparse_edge_channels(pair, channel, value, selected_pairs, keep):
    """Store per-edge attention channels sparsely instead of dense [E,L*H]."""
    E = len(selected_pairs)
    if E == 0:
        device = value.device
        return (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            value[:0],
        )

    edge_id = torch.searchsorted(selected_pairs, pair)
    valid = keep & (edge_id < E)
    valid &= selected_pairs[edge_id.clamp(max=E - 1)] == pair
    edge_id = edge_id[valid]
    order = torch.argsort(edge_id, stable=True)
    edge_id = edge_id[order]
    counts = torch.bincount(edge_id, minlength=E)
    edge_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=value.device), counts.cumsum(0))
    )
    return edge_ptr, channel[valid][order].to(torch.int32), value[valid][order]


def build_original_graph(sample, tau: float = 0.05) -> TokenGraph:
    """Reproduce old threshold-union topology with sparse channel attributes."""
    _validate_tau(sample, tau)

    source, target, channel = _entries(sample)
    value = sample.response_values
    pair = target * sample.num_tokens + source
    keep = value.float() > float(tau)
    selected = torch.unique(pair[keep], sorted=True)

    edge_source = selected % sample.num_tokens
    edge_target = selected // sample.num_tokens
    edge_ptr, edge_channel, edge_value = _sparse_edge_channels(
        pair, channel, value, selected, keep
    )
    return TokenGraph(
        sample.num_tokens,
        sample.response_idx,
        edge_index=torch.stack((edge_source, edge_target)),
        edge_type=(edge_source >= sample.response_idx).to(torch.int8),
        edge_ptr=edge_ptr,
        edge_channel=edge_channel,
        edge_value=edge_value,
    )


def dense_edge_attr(graph: TokenGraph, num_channels: int) -> torch.Tensor:
    """Materialize legacy dense [E,C] edge_attr only when old code needs it."""
    E = graph.edge_index.shape[1]
    dtype = graph.edge_value.dtype if graph.edge_value is not None else torch.float32
    device = graph.edge_index.device
    dense = torch.zeros((E, num_channels), dtype=dtype, device=device)
    if graph.edge_value is None or not E:
        return dense
    counts = graph.edge_ptr[1:] - graph.edge_ptr[:-1]
    edge_id = torch.repeat_interleave(torch.arange(E, device=device), counts)
    dense[edge_id, graph.edge_channel.long()] = graph.edge_value
    return dense


def _topk(indices, targets, score, k):
    """Select top-k edges per target with source-order tie breaking."""
    if not len(indices) or k == 0:
        return indices[:0]
    ranked = indices[torch.argsort(score[indices], descending=True, stable=True)]
    ranked = ranked[torch.argsort(targets[ranked], stable=True)]
    starts = torch.where(
        torch.cat((torch.ones(1, dtype=torch.bool, device=ranked.device), targets[ranked][1:] != targets[ranked][:-1])),
        torch.arange(len(ranked), device=ranked.device),
        0,
    ).cummax(0).values
    return ranked[torch.arange(len(ranked), device=ranked.device) - starts < k]


def build_relation_topk_graph(sample, k_prompt=8, k_history=8, with_channels=False) -> TokenGraph:
    """Keep strongest prompt and response-history token pairs for each target."""
    _validate_topk_limits(k_prompt, k_history)
    source, target, channel = _entries(sample)
    value = sample.response_values
    pair = target * sample.num_tokens + source
    unique_pair, inverse = torch.unique(pair, sorted=True, return_inverse=True)

    score = torch.zeros(len(unique_pair), dtype=torch.float32, device=value.device)
    score.index_add_(0, inverse, value.float())
    score /= sample.num_channels

    pair_source = unique_pair % sample.num_tokens
    pair_target = unique_pair // sample.num_tokens
    ids = torch.arange(len(unique_pair), device=value.device)
    prompt = _topk(ids[pair_source < sample.response_idx], pair_target, score, k_prompt)
    history = _topk(ids[pair_source >= sample.response_idx], pair_target, score, k_history)
    chosen = torch.cat((prompt, history))
    group_order = pair_target[chosen] * 2 + (pair_source[chosen] >= sample.response_idx)
    chosen = chosen[torch.argsort(group_order, stable=True)]

    chosen_pair = unique_pair[chosen]
    edge_source = chosen_pair % sample.num_tokens
    edge_target = chosen_pair // sample.num_tokens
    graph = TokenGraph(
        sample.num_tokens,
        sample.response_idx,
        edge_index=torch.stack((edge_source, edge_target)),
        edge_type=(edge_source >= sample.response_idx).to(torch.int8),
        edge_weight=score[chosen],
    )
    if not with_channels:
        return graph

    pair_to_edge = torch.full((len(unique_pair),), -1, dtype=torch.long, device=value.device)
    pair_to_edge[chosen] = torch.arange(len(chosen), device=value.device)
    traced_edge = pair_to_edge[inverse]
    mask = traced_edge >= 0
    order = torch.argsort(traced_edge[mask], stable=True)
    traced_edge = traced_edge[mask][order]
    counts = torch.bincount(traced_edge, minlength=len(chosen))
    graph.edge_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=value.device), counts.cumsum(0))
    )
    graph.edge_channel = channel[mask][order].to(torch.int32)
    graph.edge_value = value[mask][order]
    return graph
