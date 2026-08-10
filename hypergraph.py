"""Typed attention hypergraph view."""

from dataclasses import dataclass
import torch

from graphs import attention_node_features


@dataclass
class AttentionHypergraph:
    response_idx: int
    x: torch.Tensor
    incidence_index: torch.Tensor
    incidence_weight: torch.Tensor
    hyperedge_target: torch.Tensor
    hyperedge_channel: torch.Tensor
    hyperedge_type: torch.Tensor

    def to_dict(self):
        return self.__dict__.copy()


def build_attention_hypergraph(sample, tau: float = 0.05, x=None) -> AttentionHypergraph:
    if tau < sample.attention_floor:
        raise ValueError("tau cannot be lower than attention_floor")
    if x is None:
        x = attention_node_features(sample)

    C, N, R = sample.num_channels, sample.num_tokens, sample.num_response_tokens
    counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
    rows = torch.repeat_interleave(torch.arange(C * R, device=counts.device), counts)
    source = sample.response_column_indices.to(torch.int64)
    weight = sample.response_values
    keep = weight.float() > float(tau)
    rows, source, weight = rows[keep], source[keep], weight[keep]

    if not len(source):
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return AttentionHypergraph(
            sample.response_idx,
            x,
            torch.empty((2, 0), dtype=torch.long, device=source.device),
            weight,
            empty,
            empty.to(torch.int32),
            empty.to(torch.int8),
        )

    relation = (source >= sample.response_idx).long()
    group = rows * 2 + relation
    unique_group, hyperedge_id = torch.unique(group, sorted=True, return_inverse=True)
    channel = (unique_group // 2) // R
    target = sample.response_idx + (unique_group // 2) % R
    edge_type = (unique_group % 2).to(torch.int8)

    target_weight = sample.attention_diagonal.reshape(C, N)[channel, target].to(weight.dtype)
    node = torch.cat((source, target))
    hedge = torch.cat((hyperedge_id, torch.arange(len(unique_group), device=source.device)))
    incidence_weight = torch.cat((weight, target_weight))
    order = torch.argsort(hedge)

    return AttentionHypergraph(
        sample.response_idx,
        x,
        torch.stack((node[order], hedge[order])),
        incidence_weight[order],
        target,
        channel.to(torch.int32),
        edge_type,
    )
