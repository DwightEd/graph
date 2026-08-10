"""Thresholded, typed attention hypergraphs built from sparse response attention."""

from dataclasses import dataclass

import torch

from cache import AttentionSample


@dataclass
class AttentionHypergraph:
    sample_id: str
    source_id: str
    response_idx: int
    token_ids: torch.Tensor
    node_attr: torch.Tensor
    incidence_index: torch.Tensor
    incidence_weight: torch.Tensor
    hyperedge_target: torch.Tensor
    hyperedge_channel: torch.Tensor
    hyperedge_type: torch.Tensor

    def to_dict(self) -> dict[str, str | int | torch.Tensor]:
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "response_idx": self.response_idx,
            "token_ids": self.token_ids,
            "node_attr": self.node_attr,
            "incidence_index": self.incidence_index,
            "incidence_weight": self.incidence_weight,
            "hyperedge_target": self.hyperedge_target,
            "hyperedge_channel": self.hyperedge_channel,
            "hyperedge_type": self.hyperedge_type,
        }


def build_attention_hypergraph(sample: AttentionSample, tau: float) -> AttentionHypergraph:
    """Build typed prompt/response-history hyperedges from retained attention entries."""
    if tau < sample.attention_floor:
        raise ValueError("tau must be at least attention_floor for exact thresholding")
    sample.validate()

    num_tokens = sample.num_tokens
    num_response_tokens = sample.num_response_tokens
    num_channels = sample.num_channels
    node_attr = sample.attention_diagonal.reshape(num_channels, num_tokens).transpose(0, 1)

    row_counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
    rows = torch.arange(num_channels * num_response_tokens, device=row_counts.device)
    entry_rows = torch.repeat_interleave(rows, row_counts)
    source_tokens = sample.response_column_indices.to(torch.int64)
    retained = sample.response_values > tau
    entry_rows = entry_rows[retained]
    source_tokens = source_tokens[retained]
    source_weights = sample.response_values[retained]

    if source_tokens.numel() == 0:
        return AttentionHypergraph(
            sample_id=sample.sample_id,
            source_id=sample.source_id,
            response_idx=sample.response_idx,
            token_ids=sample.token_ids,
            node_attr=node_attr,
            incidence_index=torch.empty((2, 0), dtype=torch.int64, device=source_tokens.device),
            incidence_weight=source_weights,
            hyperedge_target=torch.empty(0, dtype=torch.int64, device=source_tokens.device),
            hyperedge_channel=torch.empty(0, dtype=torch.int32, device=source_tokens.device),
            hyperedge_type=torch.empty(0, dtype=torch.int8, device=source_tokens.device),
        )

    relation_type = (source_tokens >= sample.response_idx).to(torch.int64)
    pair_ids = entry_rows * 2 + relation_type
    source_order = torch.argsort(source_tokens, stable=True)
    pair_order = torch.argsort(pair_ids[source_order], stable=True)
    order = source_order[pair_order]
    pair_ids = pair_ids[order]
    source_tokens = source_tokens[order]
    source_weights = source_weights[order]

    starts = torch.ones(pair_ids.numel(), dtype=torch.bool, device=pair_ids.device)
    starts[1:] = pair_ids[1:] != pair_ids[:-1]
    source_hyperedges = starts.cumsum(0, dtype=torch.int64) - 1
    edge_pairs = pair_ids[starts]
    hyperedge_ids = torch.arange(edge_pairs.numel(), device=edge_pairs.device, dtype=torch.int64)
    hyperedge_type = (edge_pairs % 2).to(torch.int8)
    hyperedge_rows = torch.div(edge_pairs, 2, rounding_mode="floor")
    hyperedge_channel = torch.div(
        hyperedge_rows, num_response_tokens, rounding_mode="floor"
    ).to(torch.int32)
    hyperedge_target = (
        sample.response_idx + hyperedge_rows.remainder(num_response_tokens)
    ).to(torch.int64)
    target_weights = sample.attention_diagonal.reshape(num_channels, num_tokens)[
        hyperedge_channel.to(torch.int64), hyperedge_target
    ].to(dtype=source_weights.dtype)

    member_tokens = torch.cat((source_tokens, hyperedge_target))
    member_hyperedges = torch.cat((source_hyperedges, hyperedge_ids))
    member_weights = torch.cat((source_weights, target_weights))
    member_order = torch.argsort(member_hyperedges, stable=True)
    incidence_index = torch.stack(
        (member_tokens[member_order], member_hyperedges[member_order])
    )
    incidence_weight = member_weights[member_order]

    return AttentionHypergraph(
        sample_id=sample.sample_id,
        source_id=sample.source_id,
        response_idx=sample.response_idx,
        token_ids=sample.token_ids,
        node_attr=node_attr,
        incidence_index=incidence_index,
        incidence_weight=incidence_weight,
        hyperedge_target=hyperedge_target,
        hyperedge_channel=hyperedge_channel,
        hyperedge_type=hyperedge_type,
    )
