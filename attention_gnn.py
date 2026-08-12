"""Sparse layer/head-aware message passing over canonical attention graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionGraph:
    num_nodes: int
    response_idx: int
    num_channels: int
    attention_floor: float
    node_attr: torch.Tensor
    node_context: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    edge_ptr: torch.Tensor
    edge_channel: torch.Tensor
    edge_value: torch.Tensor


@dataclass(frozen=True)
class MaskedGraphView:
    visible_edges: torch.Tensor
    visible_channels: torch.Tensor


def build_attention_graph(sample) -> AttentionGraph:
    """Decode canonical response CSR into pair edges and sparse channel traces."""
    response_count = sample.num_response_tokens
    lengths = sample.response_row_ptr[1:].long() - sample.response_row_ptr[:-1].long()
    row = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
    channel = row // response_count
    target = sample.response_idx + row.remainder(response_count)
    source = sample.response_column_indices.long()
    value = sample.response_values

    pair = target * sample.num_tokens + source
    unique_pair, pair_inverse = torch.unique(pair, sorted=True, return_inverse=True)
    edge_source = unique_pair.remainder(sample.num_tokens)
    edge_target = torch.div(unique_pair, sample.num_tokens, rounding_mode="floor")

    order = torch.argsort(pair_inverse, stable=True)
    ordered_edge = pair_inverse[order]
    trace_count = torch.bincount(ordered_edge, minlength=len(unique_pair))
    edge_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=value.device), trace_count.cumsum(0))
    )
    edge_weight = torch.zeros(len(unique_pair), dtype=torch.float32, device=value.device)
    edge_weight.index_add_(0, pair_inverse, value.float())
    edge_weight /= float(sample.num_channels)

    diagonal = sample.attention_diagonal.permute(2, 0, 1).reshape(sample.num_tokens, -1)
    response_mask = torch.arange(sample.num_tokens, device=value.device) >= sample.response_idx
    position = torch.arange(sample.num_tokens, dtype=torch.float32, device=value.device)
    node_context = torch.stack(
        (torch.log1p(position), (~response_mask).float(), response_mask.float()), dim=1
    )
    return AttentionGraph(
        num_nodes=sample.num_tokens,
        response_idx=sample.response_idx,
        num_channels=sample.num_channels,
        attention_floor=sample.attention_floor,
        node_attr=diagonal,
        node_context=node_context,
        edge_index=torch.stack((edge_source, edge_target)),
        edge_type=(edge_source >= sample.response_idx).long(),
        edge_weight=edge_weight,
        edge_ptr=edge_ptr,
        edge_channel=channel[order].long(),
        edge_value=value[order],
    )


def masked_view(
    graph: AttentionGraph,
    *,
    masked_edges: torch.Tensor | None = None,
    masked_channels: torch.Tensor | None = None,
) -> MaskedGraphView:
    """Declare which pair edges and layer/head channels are visible."""
    device = graph.edge_index.device
    visible_edges = torch.ones(graph.edge_index.shape[1], dtype=torch.bool, device=device)
    visible_channels = torch.ones(graph.num_channels, dtype=torch.bool, device=device)
    if masked_edges is not None:
        visible_edges[torch.as_tensor(masked_edges, device=device).long()] = False
    if masked_channels is not None:
        visible_channels[torch.as_tensor(masked_channels, device=device).long()] = False
    return MaskedGraphView(visible_edges, visible_channels)


class _MessageLayer(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU())
        self.self_update = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor, edge_index: torch.Tensor, edge_embedding: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return self.norm(hidden + self.dropout(F.gelu(self.self_update(hidden))))
        source, target = edge_index
        message = self.message(torch.cat((hidden[source], edge_embedding), dim=1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, message)
        degree = hidden.new_zeros(len(hidden))
        degree.index_add_(0, target, hidden.new_ones(len(target)))
        update = F.gelu(self.self_update(hidden) + aggregate / degree.clamp_min(1).unsqueeze(1))
        return self.norm(hidden + self.dropout(update))


class RelationChannelEncoder(nn.Module):
    """Fuse sparse layer/head edge values into directed response-node vectors."""

    def __init__(
        self, *, num_channels: int, embedding_dim: int,
        message_passing_steps: int, dropout: float,
    ) -> None:
        super().__init__()
        if num_channels < 1 or embedding_dim < 1 or message_passing_steps < 0:
            raise ValueError("encoder dimensions must be positive and steps non-negative")
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.channel_embedding = nn.Embedding(num_channels, embedding_dim)
        self.relation_embedding = nn.Embedding(2, embedding_dim)
        self.context_encoder = nn.Linear(3, embedding_dim)
        self.value_encoder = nn.Linear(1, embedding_dim)
        self.layers = nn.ModuleList(
            _MessageLayer(embedding_dim, dropout) for _ in range(message_passing_steps)
        )

    def _node_embedding(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> torch.Tensor:
        if graph.num_channels != self.num_channels:
            raise ValueError("graph and encoder channel counts differ")
        keep = view.visible_channels.to(graph.node_attr.dtype)
        node = (graph.node_attr.float() * keep) @ self.channel_embedding.weight
        node /= keep.sum().clamp_min(1)
        return node + self.context_encoder(graph.node_context.float())

    def _edge_embedding(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_count = graph.edge_index.shape[1]
        trace_edge = torch.repeat_interleave(
            torch.arange(edge_count, device=graph.edge_index.device), graph.edge_ptr.diff()
        )
        visible_trace = view.visible_edges[trace_edge] & view.visible_channels[
            graph.edge_channel.long()
        ]
        trace_edge = trace_edge[visible_trace]
        channel = graph.edge_channel[visible_trace].long()
        value = graph.edge_value[visible_trace].float()

        edge = graph.edge_value.new_zeros((edge_count, self.embedding_dim), dtype=torch.float32)
        if len(value):
            trace = self.channel_embedding(channel) * value.unsqueeze(1)
            trace += self.value_encoder(value.unsqueeze(1))
            edge.index_add_(0, trace_edge, trace)
            edge /= float(graph.num_channels)

        edge_ids = torch.nonzero(view.visible_edges, as_tuple=False).flatten()
        relation = graph.edge_type[edge_ids].long()
        return graph.edge_index[:, edge_ids], edge[edge_ids] + self.relation_embedding(relation)

    def encode(self, graph: AttentionGraph, view: MaskedGraphView) -> torch.Tensor:
        hidden = self._node_embedding(graph, view)
        if not self.layers:
            return hidden
        edge_index, edge_embedding = self._edge_embedding(graph, view)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_embedding)
        return hidden

    def forward(self, graph: AttentionGraph, view: MaskedGraphView) -> torch.Tensor:
        return self.encode(graph, view)


class RelationChannelAutoencoder(nn.Module):
    """Encode a masked graph and decode its held-out support and channel values."""

    def __init__(
        self, *, num_channels: int, embedding_dim: int,
        message_passing_steps: int, dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = RelationChannelEncoder(
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
        )
        self.support_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1)
        )
        self.weight_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1)
        )
        self.distribution_decoder = nn.Linear(embedding_dim * 5, 1)
        self.other_decoder = nn.Linear(embedding_dim * 2, 1)

    def encode(self, graph: AttentionGraph, view: MaskedGraphView) -> torch.Tensor:
        return self.encoder(graph, view)

    def support_logits(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        source_hidden = hidden[source]
        target_hidden = hidden[target]
        relation = self.encoder.relation_embedding(edge_type.long())
        features = torch.cat(
            (source_hidden, target_hidden, source_hidden * target_hidden, relation), dim=1
        )
        return self.support_decoder(features).squeeze(1)

    def weight_prediction(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        channel: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight_decoder(
            self._trace_features(hidden, edge_index, edge_type, channel)
        ).squeeze(1).sigmoid()

    def _trace_features(
        self, hidden: torch.Tensor, edge_index: torch.Tensor,
        edge_type: torch.Tensor, channel: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        return torch.cat(
            (hidden[source], hidden[target], hidden[source] * hidden[target],
             self.encoder.relation_embedding(edge_type.long()),
             self.encoder.channel_embedding(channel.long())), dim=1
        )

    def distribution_logits(
        self, hidden: torch.Tensor, edge_index: torch.Tensor,
        edge_type: torch.Tensor, channel: torch.Tensor,
    ) -> torch.Tensor:
        features = self._trace_features(hidden, edge_index, edge_type, channel)
        return self.distribution_decoder(features).squeeze(1)

    def forward(self, graph: AttentionGraph, view: MaskedGraphView) -> torch.Tensor:
        return self.encode(graph, view)


def _support_negatives(
    graph: AttentionGraph, positive_edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the first absent causal source for each masked target/relation."""
    device = graph.edge_index.device
    if positive_edges.numel() == 0:
        return graph.edge_index[:, :0], graph.edge_type[:0]

    target = graph.edge_index[1, positive_edges]
    relation = graph.edge_type[positive_edges].long()
    domain_start = torch.where(
        relation == 0, torch.zeros_like(target),
        torch.full_like(target, graph.response_idx)
    )
    domain_size = torch.where(
        relation == 0,
        torch.full_like(target, graph.response_idx),
        target - graph.response_idx,
    )

    pair_key = torch.unique(
        graph.edge_index[1] * graph.num_nodes + graph.edge_index[0], sorted=True
    )
    group_start = target * graph.num_nodes + domain_start
    group_end = group_start + domain_size
    occupied = torch.searchsorted(pair_key, group_end) - torch.searchsorted(
        pair_key, group_start
    )
    sampleable = domain_size > occupied
    if not torch.any(sampleable):
        return graph.edge_index[:, :0], graph.edge_type[:0]

    target = target[sampleable]
    relation = relation[sampleable]
    domain_start = domain_start[sampleable]
    domain_size = domain_size[sampleable]
    group_start = group_start[sampleable]
    group_begin = torch.searchsorted(pair_key, group_start)

    lower = torch.zeros_like(domain_size)
    upper = domain_size - 1
    steps = int(domain_size.max().item()).bit_length()
    for _ in range(steps):
        middle = torch.div(lower + upper, 2, rounding_mode="floor")
        probe_key = group_start + middle
        occupied_through = torch.searchsorted(pair_key, probe_key, right=True) - group_begin
        has_missing = middle + 1 > occupied_through
        upper = torch.where(has_missing, middle, upper)
        lower = torch.where(has_missing, lower, middle + 1)

    source = domain_start + lower
    return torch.stack((source, target)), relation.to(device=device)


def _distribution_loss(
    model: RelationChannelAutoencoder, hidden: torch.Tensor, graph: AttentionGraph,
    view: MaskedGraphView, trace_edge: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct active attention rows with an explicit censored OTHER bucket."""
    channel = graph.edge_channel.long()
    masked = ~view.visible_edges[trace_edge] | ~view.visible_channels[channel]
    if not torch.any(masked):
        return hidden.sum() * 0.0

    target = graph.edge_index[1, trace_edge]
    group_key = target * graph.num_channels + channel
    active_key = torch.unique(group_key[masked], sorted=True)
    selected = masked
    group = torch.searchsorted(active_key, group_key[selected])

    edge = trace_edge[selected]
    channel = channel[selected]
    logits = model.distribution_logits(
        hidden, graph.edge_index[:, edge], graph.edge_type[edge], channel
    )
    weight = graph.edge_value[selected].float()
    active_target = active_key // graph.num_channels
    active_channel = active_key.remainder(graph.num_channels)
    diagonal = graph.node_attr[active_target, active_channel].float()
    active_trace = torch.isin(group_key, active_key)
    full_mass = hidden.new_zeros(len(active_key))
    full_mass.index_add_(
        0,
        torch.searchsorted(active_key, group_key[active_trace]),
        graph.edge_value[active_trace].float(),
    )
    other_mass = (1.0 - diagonal - full_mass).clamp_min(graph.attention_floor)
    hidden_mass = hidden.new_zeros(len(active_key)).index_add_(0, group, weight)
    normalizer = hidden_mass + other_mass
    target_probability = weight / normalizer[group]
    other_probability = other_mass / normalizer

    other_features = torch.cat(
        (hidden[active_target], model.encoder.channel_embedding(active_channel)), dim=1
    )
    other_logits = model.other_decoder(other_features).squeeze(1)
    maximum = hidden.new_full((len(active_key),), -torch.inf)
    maximum.scatter_reduce_(0, group, logits, reduce="amax", include_self=True)
    maximum = torch.maximum(maximum, other_logits)
    partition = torch.exp(other_logits - maximum).index_add(
        0, group, torch.exp(logits - maximum[group])
    )
    log_partition = maximum + partition.log()

    cross_entropy = hidden.new_zeros(len(active_key))
    cross_entropy.index_add_(
        0, group, -target_probability * (logits - log_partition[group])
    )
    cross_entropy -= other_probability * (other_logits - log_partition)
    return cross_entropy.mean()


def reconstruction_loss(
    model: RelationChannelAutoencoder, graph: AttentionGraph, view: MaskedGraphView
) -> dict[str, torch.Tensor]:
    """Reconstruct only graph content hidden by ``view``."""
    hidden = model.encode(graph, view)
    masked_edges = torch.nonzero(~view.visible_edges, as_tuple=False).flatten()
    zero = hidden.sum() * 0.0

    if masked_edges.numel():
        positive_logits = model.support_logits(
            hidden, graph.edge_index[:, masked_edges], graph.edge_type[masked_edges]
        )
        negative_index, negative_type = _support_negatives(graph, masked_edges)
        support_logits = positive_logits
        support_target = torch.ones_like(positive_logits)
        if negative_index.shape[1]:
            negative_logits = model.support_logits(hidden, negative_index, negative_type)
            support_logits = torch.cat((positive_logits, negative_logits))
            support_target = torch.cat((support_target, torch.zeros_like(negative_logits)))
        support_loss = F.binary_cross_entropy_with_logits(support_logits, support_target)
    else:
        support_loss = zero

    edge_count = graph.edge_index.shape[1]
    trace_edge = torch.repeat_interleave(
        torch.arange(edge_count, device=graph.edge_index.device), graph.edge_ptr.diff()
    )
    masked_trace = ~view.visible_edges[trace_edge] | ~view.visible_channels[
        graph.edge_channel.long()
    ]
    held_out_edge = trace_edge[masked_trace]
    if held_out_edge.numel():
        prediction = model.weight_prediction(
            hidden,
            graph.edge_index[:, held_out_edge],
            graph.edge_type[held_out_edge],
            graph.edge_channel[masked_trace],
        )
        weight_loss = F.smooth_l1_loss(prediction, graph.edge_value[masked_trace].float())
    else:
        weight_loss = zero

    distribution_loss = _distribution_loss(model, hidden, graph, view, trace_edge)
    total = support_loss + weight_loss + distribution_loss
    return {"support": support_loss, "weight": weight_loss,
            "distribution": distribution_loss, "total": total}
