"""Token graph views built from the canonical sparse attention cache."""

from dataclasses import dataclass
import torch


@dataclass
class TokenGraph:
    response_idx: int
    token_ids: torch.Tensor
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    trace_ptr: torch.Tensor | None = None
    trace_channel: torch.Tensor | None = None
    trace_value: torch.Tensor | None = None

    def to_dict(self):
        return {name: value for name, value in self.__dict__.items() if value is not None}


def _entries(sample):
    R = sample.num_response_tokens
    counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
    rows = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)
    channels = rows // R
    targets = sample.response_idx + rows % R
    sources = sample.response_column_indices.to(torch.int64)
    return sources, targets, channels


def _base(sample):
    x = sample.attention_diagonal.reshape(sample.num_channels, sample.num_tokens).T.contiguous()
    return {
        "response_idx": sample.response_idx,
        "token_ids": sample.token_ids,
        "x": x,
    }


def build_original_graph(sample, tau: float = 0.05) -> TokenGraph:
    """Reproduce the old threshold-union attributed graph."""
    if tau < sample.attention_floor:
        raise ValueError("tau cannot be lower than attention_floor")

    source, target, channel = _entries(sample)
    value = sample.response_values
    pair = target * sample.num_tokens + source
    keep = value.float() > float(tau)
    selected = torch.unique(pair[keep], sorted=True)

    edge_source = selected % sample.num_tokens
    edge_target = selected // sample.num_tokens
    edge_index = torch.stack((edge_source, edge_target))
    edge_type = (edge_source >= sample.response_idx).to(torch.int8)
    edge_attr = torch.zeros(
        (len(selected), sample.num_channels), dtype=value.dtype, device=value.device
    )

    if len(selected):
        edge_id = torch.searchsorted(selected, pair)
        valid = keep & (edge_id < len(selected))
        valid &= selected[edge_id.clamp(max=len(selected) - 1)] == pair
        edge_attr[edge_id[valid], channel[valid]] = value[valid]

    return TokenGraph(
        **_base(sample), edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
    )


def _topk(indices, targets, score, k):
    chosen = []
    for target in torch.unique(targets[indices]):
        local = indices[targets[indices] == target]
        order = torch.argsort(score[local], descending=True)
        chosen.append(local[order[:k]])
    return torch.cat(chosen) if chosen else indices[:0]


def build_relation_topk_graph(
    sample, k_prompt=8, k_history=8, with_channels=False
) -> TokenGraph:
    """Keep the strongest prompt and response-history pairs for each target."""
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
    history = _topk(
        ids[pair_source >= sample.response_idx], pair_target, score, k_history
    )
    chosen = torch.cat((prompt, history))

    chosen_pair = unique_pair[chosen]
    edge_source = chosen_pair % sample.num_tokens
    edge_target = chosen_pair // sample.num_tokens
    graph = TokenGraph(
        **_base(sample),
        edge_index=torch.stack((edge_source, edge_target)),
        edge_type=(edge_source >= sample.response_idx).to(torch.int8),
        edge_weight=score[chosen],
    )
    if not with_channels:
        return graph

    pair_to_edge = torch.full(
        (len(unique_pair),), -1, dtype=torch.long, device=value.device
    )
    pair_to_edge[chosen] = torch.arange(len(chosen), device=value.device)
    traced_edge = pair_to_edge[inverse]
    mask = traced_edge >= 0
    traced_edge = traced_edge[mask]
    order = torch.argsort(traced_edge)
    traced_edge = traced_edge[order]
    counts = torch.bincount(traced_edge, minlength=len(chosen))
    graph.trace_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=value.device), counts.cumsum(0))
    )
    graph.trace_channel = channel[mask][order].to(torch.int32)
    graph.trace_value = value[mask][order]
    return graph
