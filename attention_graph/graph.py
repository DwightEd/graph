"""Sparse attention graph construction.

One token is one node.  A response-query pair ``source -> target`` becomes one
RP (prompt-to-response) or RR (response-history-to-response) edge.  Every
retained layer/head attention value stays as a sparse trace attached to that
pair edge; layer/head channels are never averaged before the learned encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

RP = 0
RR = 1


@dataclass(frozen=True)
class GraphBuildConfig:
    selection: str = "threshold"
    threshold: float | None = None
    top_k: int = 8
    mass_cover: float = 0.80
    max_edges_per_target: int | None = None
    query_block: int = 64

    def validate(self, *, attention_floor: float):
        if self.selection not in {"threshold", "global_topk", "typed_topk", "typed_mass_cover"}:
            raise ValueError("unknown graph support selection")
        if self.top_k < 1 or self.query_block < 1:
            raise ValueError("top_k and query_block must be positive")
        if not 0.0 < float(self.mass_cover) <= 1.0:
            raise ValueError("mass_cover must be in (0,1]")
        if self.max_edges_per_target is not None and self.max_edges_per_target < 1:
            raise ValueError("max_edges_per_target must be positive when provided")
        threshold = attention_floor if self.threshold is None else float(self.threshold)
        if not math.isfinite(threshold) or not attention_floor <= threshold <= 1.0:
            raise ValueError("threshold must be finite and at least attention_floor")


@dataclass(frozen=True)
class AttentionGraph:
    sample_id: str
    source_id: str
    response_idx: int
    num_layers: int
    num_heads: int
    attention_floor: float
    token_ids: torch.Tensor
    node_attr: torch.Tensor
    node_context: torch.Tensor
    response_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_score: torch.Tensor
    trace_edge_id: torch.Tensor
    trace_channel: torch.Tensor
    trace_value: torch.Tensor
    build_config: GraphBuildConfig

    @property
    def num_nodes(self):
        return int(self.node_attr.shape[0])

    @property
    def num_edges(self):
        return int(self.edge_index.shape[1])

    @property
    def num_channels(self):
        return self.num_layers * self.num_heads

    def to(self, device):
        device = torch.device(device)
        return AttentionGraph(**{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        })


def _segmented_topk(score: torch.Tensor, group: torch.Tensor, top_k: int):
    if score.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    score_order = torch.argsort(score, descending=True, stable=True)
    order = score_order[torch.argsort(group[score_order], stable=True)]
    ordered_group = group[order]
    position = torch.arange(len(order), device=score.device)
    starts = torch.where(
        torch.cat((
            torch.ones(1, dtype=torch.bool, device=score.device),
            ordered_group[1:] != ordered_group[:-1],
        )),
        position,
        torch.zeros_like(position),
    )
    rank = position - torch.cummax(starts, dim=0).values
    return order[rank < int(top_k)]


def _selected_pairs(score, maximum, target, relation, config, attention_floor):
    if config.selection == "threshold":
        threshold = attention_floor if config.threshold is None else float(config.threshold)
        selected = torch.nonzero(maximum >= threshold, as_tuple=False).flatten()
        if config.max_edges_per_target is not None and selected.numel():
            selected = selected[
                _segmented_topk(score[selected], target[selected], config.max_edges_per_target)
            ]
        return selected
    if config.selection == "typed_mass_cover":
        group = target * 2 + relation
        selected = []
        for group_id in torch.unique(group, sorted=True):
            ids = torch.nonzero(group == group_id, as_tuple=False).flatten()
            ranked = ids[torch.argsort(score[ids], descending=True, stable=True)]
            mass = score[ranked]
            total = mass.sum()
            if not bool(total > 0):
                continue
            reached = torch.nonzero(
                mass.cumsum(0) >= float(config.mass_cover) * total, as_tuple=False
            )
            count = int(reached[0]) + 1 if reached.numel() else len(ranked)
            selected.append(ranked[:count])
        return (
            torch.cat(selected)
            if selected
            else torch.empty(0, dtype=torch.long, device=score.device)
        )
    group = target if config.selection == "global_topk" else target * 2 + relation
    return _segmented_topk(score, group, config.top_k)


def build_attention_graph(sample, config: GraphBuildConfig | None = None):
    """Build one label-free graph directly from canonical sparse attention CSR."""
    config = GraphBuildConfig() if config is None else config
    config.validate(attention_floor=float(sample.attention_floor))
    device = sample.response_values.device
    response_tokens = sample.num_response_tokens
    channels = sample.num_channels
    token_count = sample.num_tokens

    row_ptr = sample.response_row_ptr.long()
    columns = sample.response_column_indices.long()
    values = sample.response_values.float()
    channel_ids = torch.arange(channels, device=device)

    edge_sources, edge_targets, edge_types, edge_scores = [], [], [], []
    trace_edges, trace_channels, trace_values = [], [], []
    edge_offset = 0

    for start in range(0, response_tokens, config.query_block):
        end = min(start + config.query_block, response_tokens)
        query = torch.arange(start, end, device=device)
        row_ids = (channel_ids[:, None] * response_tokens + query).reshape(-1)
        row_starts = row_ptr[row_ids]
        lengths = row_ptr[row_ids + 1] - row_starts
        entry_count = int(lengths.sum())
        if entry_count == 0:
            continue

        repeated_starts = torch.repeat_interleave(row_starts, lengths)
        prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        positions = repeated_starts + torch.arange(entry_count, device=device) - prefix
        entry_rows = torch.repeat_interleave(row_ids, lengths)
        source = columns[positions]
        observed = values[positions]
        target = sample.response_idx + entry_rows.remainder(response_tokens)
        channel = entry_rows // response_tokens
        if bool(((source < 0) | (source >= target)).any()):
            raise ValueError("canonical attention violates causal source ordering")

        # ``threshold`` has CHARM-style semantics: prune individual channel
        # traces first, then materialise the union support of the surviving
        # channels. Other selection policies operate on every trace retained by
        # the canonical floor.
        if config.selection == "threshold":
            threshold = (
                float(sample.attention_floor)
                if config.threshold is None
                else float(config.threshold)
            )
            keep_trace = observed >= threshold
            source = source[keep_trace]
            target = target[keep_trace]
            channel = channel[keep_trace]
            observed = observed[keep_trace]
            if observed.numel() == 0:
                continue

        pair_key, inverse = torch.unique(
            target * token_count + source, sorted=True, return_inverse=True
        )
        pair_count = len(pair_key)
        pair_sum = torch.zeros(pair_count, dtype=torch.float32, device=device)
        pair_sum.index_add_(0, inverse, observed)
        pair_max = torch.full((pair_count,), -torch.inf, device=device)
        pair_max.scatter_reduce_(0, inverse, observed, reduce="amax", include_self=True)
        pair_source = pair_key.remainder(token_count)
        pair_target = torch.div(pair_key, token_count, rounding_mode="floor")
        relation = (pair_source >= sample.response_idx).long()
        pair_score = pair_sum / float(channels)
        selected = _selected_pairs(
            pair_score, pair_max, pair_target, relation, config, sample.attention_floor
        )
        if selected.numel() == 0:
            continue

        local_to_edge = torch.full((pair_count,), -1, dtype=torch.long, device=device)
        local_to_edge[selected] = torch.arange(
            edge_offset, edge_offset + len(selected), device=device
        )
        entry_edge = local_to_edge[inverse]
        keep = entry_edge >= 0
        edge_sources.append(pair_source[selected])
        edge_targets.append(pair_target[selected])
        edge_types.append(relation[selected])
        edge_scores.append(pair_score[selected])
        trace_edges.append(entry_edge[keep])
        trace_channels.append(channel[keep])
        trace_values.append(observed[keep])
        edge_offset += len(selected)

    if edge_sources:
        edge_index = torch.stack((torch.cat(edge_sources), torch.cat(edge_targets)))
        edge_type = torch.cat(edge_types)
        edge_score = torch.cat(edge_scores)
        trace_edge_id = torch.cat(trace_edges)
        trace_channel = torch.cat(trace_channels)
        trace_value = torch.cat(trace_values)
        trace_key = trace_edge_id * channels + trace_channel
        order = torch.argsort(trace_key, stable=True)
        trace_edge_id, trace_channel, trace_value = (
            trace_edge_id[order], trace_channel[order], trace_value[order]
        )
        sorted_key = trace_key[order]
        if sorted_key.numel() > 1 and bool((sorted_key[1:] == sorted_key[:-1]).any()):
            raise ValueError("duplicate edge/channel entries in canonical attention")
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty(0, dtype=torch.long, device=device)
        edge_score = torch.empty(0, dtype=torch.float32, device=device)
        trace_edge_id = torch.empty(0, dtype=torch.long, device=device)
        trace_channel = torch.empty(0, dtype=torch.long, device=device)
        trace_value = torch.empty(0, dtype=torch.float32, device=device)

    diagonal = sample.attention_diagonal.float().permute(2, 0, 1).reshape(token_count, channels)
    response_mask = torch.arange(token_count, device=device) >= sample.response_idx
    absolute = torch.arange(token_count, dtype=torch.float32, device=device)
    response_position = (absolute - sample.response_idx).clamp_min(0)
    node_context = torch.stack((
        absolute / max(token_count - 1, 1),
        response_position / max(response_tokens - 1, 1),
        (~response_mask).float(),
        response_mask.float(),
    ), dim=1)

    return AttentionGraph(
        sample_id=str(sample.sample_id),
        source_id=str(sample.source_id),
        response_idx=int(sample.response_idx),
        num_layers=int(sample.num_layers),
        num_heads=int(sample.num_heads),
        attention_floor=float(sample.attention_floor),
        token_ids=sample.token_ids.long(),
        node_attr=diagonal,
        node_context=node_context,
        response_mask=response_mask,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_score=edge_score,
        trace_edge_id=trace_edge_id,
        trace_channel=trace_channel,
        trace_value=trace_value,
        build_config=config,
    )
