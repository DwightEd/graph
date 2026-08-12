"""Relation/channel-aware masked graph autoencoder.

The encoder learns how to fuse sparse layer/head attention traces into edge
representations and then performs CHARM-style message passing. Hallucination
labels never enter this module; parameters are learned only by reconstructing
masked graph content.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .graph import AttentionGraph, RP


@dataclass(frozen=True)
class MaskedGraphView:
    visible_edge_mask: torch.Tensor
    node_mask: torch.Tensor
    channel_keep_mask: torch.Tensor

    @property
    def masked_edge_ids(self):
        return torch.nonzero(~self.visible_edge_mask, as_tuple=False).flatten()


@dataclass(frozen=True)
class ReconstructionLosses:
    support: torch.Tensor
    weight: torch.Tensor
    distribution: torch.Tensor
    node: torch.Tensor
    total: torch.Tensor


def full_view(graph: AttentionGraph):
    device = graph.node_attr.device
    return MaskedGraphView(
        visible_edge_mask=torch.ones(graph.num_edges, dtype=torch.bool, device=device),
        node_mask=torch.zeros(graph.num_nodes, dtype=torch.bool, device=device),
        channel_keep_mask=torch.ones(graph.num_channels, dtype=torch.bool, device=device),
    )


def target_masked_view(
    graph: AttentionGraph,
    targets: torch.Tensor,
    *,
    channel_drop_rate: float = 0.0,
    generator: torch.Generator | None = None,
):
    """Hide the complete incoming attention state of selected response tokens."""
    device = graph.node_attr.device
    targets = torch.as_tensor(targets, dtype=torch.long, device=device).flatten()
    if targets.numel() == 0:
        raise ValueError("targets must contain at least one response node")
    if bool(((targets < graph.response_idx) | (targets >= graph.num_nodes)).any()):
        raise ValueError("masked targets must be response nodes")
    node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    node_mask[targets] = True
    visible_edges = ~torch.isin(graph.edge_index[1], targets)
    channel_keep = torch.ones(graph.num_channels, dtype=torch.bool, device=device)
    if channel_drop_rate:
        if not 0.0 <= channel_drop_rate < 1.0:
            raise ValueError("channel_drop_rate must be in [0,1)")
        count = min(
            max(1, round(graph.num_channels * channel_drop_rate)), graph.num_channels - 1
        )
        if count > 0:
            random_device = torch.device(generator.device) if generator is not None else device
            order = torch.randperm(
                graph.num_channels, generator=generator, device=random_device
            ).to(device)
            channel_keep[order[:count]] = False
    return MaskedGraphView(visible_edges, node_mask, channel_keep)


def random_target_view(
    graph: AttentionGraph,
    *,
    target_mask_rate: float,
    channel_drop_rate: float,
    generator: torch.Generator,
):
    """Training view: mask complete response-token graph states, not random labels."""
    if not 0.0 < target_mask_rate <= 1.0:
        raise ValueError("target_mask_rate must be in (0,1]")
    response_nodes = torch.nonzero(graph.response_mask, as_tuple=False).flatten()
    count = min(
        max(1, round(len(response_nodes) * target_mask_rate)), len(response_nodes)
    )
    random_device = torch.device(generator.device)
    order = torch.randperm(len(response_nodes), generator=generator, device=random_device).to(
        response_nodes.device
    )
    return target_masked_view(
        graph,
        response_nodes[order[:count]],
        channel_drop_rate=channel_drop_rate,
        generator=generator,
    )


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values):
        return self.net(values)


class _MessageLayer(nn.Module):
    """CHARM-style edge-aware message and target update."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.message_mlp = _MLP(dim * 2, dim * 2, dim)
        self.update_mlp = _MLP(dim * 2, dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_index, edge_embedding):
        if edge_index.shape[1] == 0:
            return hidden
        source, target = edge_index
        message = self.message_mlp(torch.cat((hidden[source], edge_embedding), dim=1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, message)
        degree = hidden.new_zeros(hidden.shape[0])
        degree.index_add_(0, target, hidden.new_ones(len(target)))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
        update = self.update_mlp(torch.cat((hidden, aggregate), dim=1))
        return self.norm(hidden + self.dropout(update))


class AttentionGraphEncoder(nn.Module):
    """Learn layer/head fusion, relation semantics, and neighborhood aggregation."""

    def __init__(
        self,
        *,
        num_channels: int,
        embedding_dim: int = 64,
        message_steps: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(num_channels, embedding_dim) < 1 or message_steps < 0:
            raise ValueError("invalid encoder dimensions")
        self.num_channels = int(num_channels)
        self.embedding_dim = int(embedding_dim)
        self.node_channel_basis = nn.Embedding(num_channels, embedding_dim)
        self.edge_value_basis = nn.Embedding(num_channels, embedding_dim)
        self.edge_presence_basis = nn.Embedding(num_channels, embedding_dim)
        self.relation_embedding = nn.Embedding(2, embedding_dim)
        self.context_encoder = nn.Linear(4, embedding_dim)
        self.edge_magnitude_encoder = nn.Linear(1, embedding_dim)
        self.node_mask_token = nn.Parameter(torch.zeros(embedding_dim))
        self.layers = nn.ModuleList(
            _MessageLayer(embedding_dim, dropout) for _ in range(message_steps)
        )

    def _nodes(self, graph: AttentionGraph, view: MaskedGraphView):
        if graph.num_channels != self.num_channels:
            raise ValueError("graph and encoder channel dimensions differ")
        keep = view.channel_keep_mask.to(graph.node_attr.dtype)
        node_signal = (graph.node_attr * keep) @ self.node_channel_basis.weight
        node_signal /= keep.sum().clamp_min(1.0)
        context = self.context_encoder(graph.node_context.float())
        node = node_signal + context
        if bool(view.node_mask.any()):
            node = torch.where(
                view.node_mask.unsqueeze(1), context + self.node_mask_token, node
            )
        return node

    def _edges(self, graph: AttentionGraph, view: MaskedGraphView):
        visible_ids = torch.nonzero(view.visible_edge_mask, as_tuple=False).flatten()
        if visible_ids.numel() == 0:
            return graph.edge_index[:, :0], graph.node_attr.new_empty((0, self.embedding_dim))
        trace_visible = (
            view.visible_edge_mask[graph.trace_edge_id]
            & view.channel_keep_mask[graph.trace_channel]
        )
        trace_edge = graph.trace_edge_id[trace_visible]
        trace_channel = graph.trace_channel[trace_visible]
        trace_value = graph.trace_value[trace_visible].float()
        edge_count = torch.bincount(trace_edge, minlength=graph.num_edges).float()
        edge_mass = torch.bincount(
            trace_edge, weights=trace_value, minlength=graph.num_edges
        )
        edge_all = graph.node_attr.new_zeros((graph.num_edges, self.embedding_dim))
        if trace_value.numel():
            indices = torch.stack((trace_edge, trace_channel))
            weighted = torch.sparse_coo_tensor(
                indices,
                trace_value,
                (graph.num_edges, graph.num_channels),
                device=trace_value.device,
            ).coalesce()
            presence = torch.sparse_coo_tensor(
                indices,
                torch.ones_like(trace_value),
                (graph.num_edges, graph.num_channels),
                device=trace_value.device,
            ).coalesce()
            denom = edge_count.clamp_min(1).unsqueeze(1)
            edge_all = (
                torch.sparse.mm(weighted, self.edge_value_basis.weight)
                + torch.sparse.mm(presence, self.edge_presence_basis.weight)
            ) / denom
            edge_all += self.edge_magnitude_encoder(
                (edge_mass / edge_count.clamp_min(1)).unsqueeze(1)
            )
        relation = self.relation_embedding(graph.edge_type[visible_ids])
        return graph.edge_index[:, visible_ids], edge_all[visible_ids] + relation

    def encode_stages(self, graph: AttentionGraph, view: MaskedGraphView):
        """Return node states before and after the same message-passing stack."""
        hidden = self._nodes(graph, view)
        before = hidden
        if not self.layers:
            return before, before
        edge_index, edge_embedding = self._edges(graph, view)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_embedding)
        return before, hidden

    def forward(self, graph: AttentionGraph, view: MaskedGraphView):
        return self.encode_stages(graph, view)[1]


class MaskedAttentionAutoencoder(nn.Module):
    """Self-supervised graph encoder with four reconstruction objectives."""

    def __init__(self, *, num_channels, embedding_dim=64, message_steps=2, dropout=0.1):
        super().__init__()
        self.encoder = AttentionGraphEncoder(
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            message_steps=message_steps,
            dropout=dropout,
        )
        dim = embedding_dim
        pair_dim = dim * 4
        self.support_decoder = _MLP(pair_dim, dim * 2, 1)
        self.weight_decoder = _MLP(pair_dim + dim, dim * 2, 1)
        self.distribution_decoder = _MLP(pair_dim + dim, dim * 2, 1)
        self.other_decoder = _MLP(dim * 2, dim * 2, 1)
        self.node_decoder = nn.Linear(dim, num_channels)

    @property
    def embedding_dim(self):
        return self.encoder.embedding_dim

    @property
    def num_channels(self):
        return self.encoder.num_channels

    def encode(self, graph, view=None):
        return self.encoder(graph, full_view(graph) if view is None else view)

    def encode_stages(self, graph, view=None):
        return self.encoder.encode_stages(graph, full_view(graph) if view is None else view)

    def _pair_features(self, hidden, edge_index, edge_type):
        source, target = edge_index
        source_h, target_h = hidden[source], hidden[target]
        return torch.cat((
            source_h,
            target_h,
            source_h * target_h,
            self.encoder.relation_embedding(edge_type.long()),
        ), dim=1)

    def support_logits(self, hidden, edge_index, edge_type):
        return self.support_decoder(self._pair_features(hidden, edge_index, edge_type)).squeeze(1)

    def weight_prediction(self, hidden, edge_index, edge_type, channel):
        pair = self._pair_features(hidden, edge_index, edge_type)
        channel_h = self.encoder.edge_value_basis(channel.long())
        return self.weight_decoder(torch.cat((pair, channel_h), dim=1)).squeeze(1).sigmoid()

    def distribution_logits(self, hidden, edge_index, edge_type, channel):
        pair = self._pair_features(hidden, edge_index, edge_type)
        channel_h = self.encoder.edge_value_basis(channel.long())
        return self.distribution_decoder(torch.cat((pair, channel_h), dim=1)).squeeze(1)


def _sample_cap(indices, maximum, generator=None):
    if maximum is None or indices.numel() <= maximum:
        return indices
    if maximum < 1:
        raise ValueError("sampling cap must be positive")
    random_device = torch.device(generator.device) if generator is not None else indices.device
    order = torch.randperm(len(indices), generator=generator, device=random_device).to(indices.device)
    return indices[order[:maximum]]


def _support_sampleable_edges(graph: AttentionGraph, edge_ids: torch.Tensor):
    if edge_ids.numel() == 0:
        return edge_ids
    relation = (graph.edge_index[0] >= graph.response_idx).long()
    group = graph.edge_index[1] * 2 + relation
    _, inverse, counts = torch.unique(group, sorted=True, return_inverse=True, return_counts=True)
    target = graph.edge_index[1, edge_ids]
    selected_relation = relation[edge_ids]
    domain = torch.where(
        selected_relation == RP,
        torch.full_like(target, graph.response_idx),
        target - graph.response_idx,
    )
    return edge_ids[counts[inverse[edge_ids]] < domain]


def sample_support_negatives(graph, positive_edge_ids, generator=None):
    """Uniformly sample one absent causal source with the same target/relation."""
    device = graph.edge_index.device
    positive_edge_ids = _support_sampleable_edges(graph, positive_edge_ids)
    if positive_edge_ids.numel() == 0:
        return graph.edge_index[:, :0], graph.edge_type[:0], positive_edge_ids
    targets = graph.edge_index[1, positive_edge_ids]
    relations = (graph.edge_index[0, positive_edge_ids] >= graph.response_idx).long()
    domain_start = torch.where(
        relations == RP, torch.zeros_like(targets), torch.full_like(targets, graph.response_idx)
    )
    domain_size = torch.where(
        relations == RP, torch.full_like(targets, graph.response_idx), targets - graph.response_idx
    )
    all_relation = (graph.edge_index[0] >= graph.response_idx).long()
    group = graph.edge_index[1] * 2 + all_relation
    _, inverse, counts = torch.unique(group, sorted=True, return_inverse=True, return_counts=True)
    free_count = domain_size - counts[inverse[positive_edge_ids]]

    random_device = torch.device(generator.device) if generator is not None else device
    random = torch.rand(len(positive_edge_ids), generator=generator, device=random_device).to(device)
    missing_rank = torch.floor(random * free_count.float()).long()
    pair_key = torch.sort(graph.edge_index[1] * graph.num_nodes + graph.edge_index[0]).values
    group_start = targets * graph.num_nodes + domain_start
    group_begin = torch.searchsorted(pair_key, group_start)
    lower = torch.zeros_like(domain_size)
    upper = domain_size - 1
    for _ in range(max(int(domain_size.max()).bit_length(), 1)):
        offset = (lower + upper) // 2
        probe = targets * graph.num_nodes + domain_start + offset
        occupied_through = torch.searchsorted(pair_key, probe, right=True) - group_begin
        missing_through = offset + 1 - occupied_through
        enough = missing_through >= missing_rank + 1
        upper = torch.where(enough, offset, upper)
        lower = torch.where(enough, lower, offset + 1)
    source = domain_start + lower
    return torch.stack((source, targets)), relations, positive_edge_ids


def _distribution_group_energy(model, hidden, graph, view, max_groups=None, generator=None):
    trace_masked = ~view.visible_edge_mask[graph.trace_edge_id]
    if not bool(trace_masked.any()):
        return hidden.new_empty(0), torch.empty(0, dtype=torch.long, device=hidden.device)
    trace_target = graph.edge_index[1, graph.trace_edge_id]
    group_key = trace_target * graph.num_channels + graph.trace_channel
    all_keys, inverse = torch.unique(group_key, sorted=True, return_inverse=True)
    active = torch.zeros(len(all_keys), dtype=torch.bool, device=hidden.device)
    active[inverse[trace_masked]] = True
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    active_ids = _sample_cap(active_ids, max_groups, generator)
    selected_group = torch.zeros_like(active)
    selected_group[active_ids] = True
    selected_trace = selected_group[inverse]
    if not bool(selected_trace.any()):
        return hidden.new_empty(0), torch.empty(0, dtype=torch.long, device=hidden.device)
    remap = torch.full((len(all_keys),), -1, dtype=torch.long, device=hidden.device)
    remap[active_ids] = torch.arange(len(active_ids), device=hidden.device)
    local_group = remap[inverse[selected_trace]]
    edge = graph.trace_edge_id[selected_trace]
    channel = graph.trace_channel[selected_trace]
    logits = model.distribution_logits(
        hidden, graph.edge_index[:, edge], graph.edge_type[edge], channel
    )
    weight = graph.trace_value[selected_trace].float()
    active_key = all_keys[active_ids]
    active_target = active_key // graph.num_channels
    active_channel = active_key.remainder(graph.num_channels)
    retained_mass = hidden.new_zeros(len(active_ids)).index_add_(0, local_group, weight)
    history_mass = (1.0 - graph.node_attr[active_target, active_channel]).clamp_min(1e-8)
    other_mass = (history_mass - retained_mass).clamp_min(graph.attention_floor)
    normalizer = retained_mass + other_mass
    target_probability = weight / normalizer[local_group]
    other_probability = other_mass / normalizer
    other_features = torch.cat((
        hidden[active_target], model.encoder.edge_value_basis(active_channel)
    ), dim=1)
    other_logits = model.other_decoder(other_features).squeeze(1)
    maximum = hidden.new_full((len(active_ids),), -torch.inf)
    maximum.scatter_reduce_(0, local_group, logits, reduce="amax", include_self=True)
    maximum = torch.maximum(maximum, other_logits)
    partition = torch.exp(other_logits - maximum).index_add_(
        0, local_group, torch.exp(logits - maximum[local_group])
    )
    log_partition = maximum + partition.log()
    ce = hidden.new_zeros(len(active_ids))
    ce.index_add_(0, local_group, -target_probability * (logits - log_partition[local_group]))
    ce -= other_probability * (other_logits - log_partition)
    return ce, active_target


def _mean_by_target(values, targets, node_count):
    output = values.new_zeros(node_count)
    count = values.new_zeros(node_count)
    if values.numel():
        output.index_add_(0, targets, values)
        count.index_add_(0, targets, torch.ones_like(values))
    return output / count.clamp_min(1)


def reconstruction_energy_by_node(
    model,
    graph,
    view,
    *,
    hidden=None,
    max_support_edges=None,
    max_weight_traces=None,
    max_distribution_groups=None,
    generator=None,
):
    """Assign self-supervised reconstruction residuals to response targets."""
    hidden = model.encode(graph, view) if hidden is None else hidden
    zero = hidden.new_zeros(graph.num_nodes)
    masked_edges = view.masked_edge_ids

    support = zero.clone()
    support_rp = zero.clone()
    support_rr = zero.clone()
    sampleable = _support_sampleable_edges(graph, masked_edges)
    sampleable = _sample_cap(sampleable, max_support_edges, generator)
    negative_index, negative_type, sampleable = sample_support_negatives(
        graph, sampleable, generator
    )
    if sampleable.numel():
        positive_index = graph.edge_index[:, sampleable]
        positive_type = graph.edge_type[sampleable]
        positive = F.binary_cross_entropy_with_logits(
            model.support_logits(hidden, positive_index, positive_type),
            torch.ones(len(sampleable), device=hidden.device), reduction="none"
        )
        negative = F.binary_cross_entropy_with_logits(
            model.support_logits(hidden, negative_index, negative_type),
            torch.zeros(len(sampleable), device=hidden.device), reduction="none"
        )
        entry = 0.5 * (positive + negative)
        target = positive_index[1]
        support = _mean_by_target(entry, target, graph.num_nodes)
        rp = positive_type == RP
        support_rp = _mean_by_target(entry[rp], target[rp], graph.num_nodes)
        support_rr = _mean_by_target(entry[~rp], target[~rp], graph.num_nodes)

    trace_masked = ~view.visible_edge_mask[graph.trace_edge_id]
    trace_ids = _sample_cap(
        torch.nonzero(trace_masked, as_tuple=False).flatten(), max_weight_traces, generator
    )
    weight = zero.clone()
    weight_rp = zero.clone()
    weight_rr = zero.clone()
    if trace_ids.numel():
        edge = graph.trace_edge_id[trace_ids]
        prediction = model.weight_prediction(
            hidden,
            graph.edge_index[:, edge],
            graph.edge_type[edge],
            graph.trace_channel[trace_ids],
        )
        residual = F.smooth_l1_loss(prediction, graph.trace_value[trace_ids].float(), reduction="none")
        target = graph.edge_index[1, edge]
        weight = _mean_by_target(residual, target, graph.num_nodes)
        rp = graph.edge_type[edge] == RP
        weight_rp = _mean_by_target(residual[rp], target[rp], graph.num_nodes)
        weight_rr = _mean_by_target(residual[~rp], target[~rp], graph.num_nodes)

    group_energy, group_target = _distribution_group_energy(
        model, hidden, graph, view,
        max_groups=max_distribution_groups, generator=generator
    )
    distribution = _mean_by_target(group_energy, group_target, graph.num_nodes)

    node = zero.clone()
    if bool(view.node_mask.any()):
        prediction = model.node_decoder(hidden[view.node_mask]).sigmoid()
        target = graph.node_attr[view.node_mask]
        per_channel = F.smooth_l1_loss(prediction, target, reduction="none")
        node[view.node_mask] = per_channel.mean(dim=1)

    total = support + weight + distribution + node
    return {
        "support": support,
        "support_rp": support_rp,
        "support_rr": support_rr,
        "weight": weight,
        "weight_rp": weight_rp,
        "weight_rr": weight_rr,
        "distribution": distribution,
        "node": node,
        "total": total,
    }


def reconstruction_losses(
    model,
    graph,
    view,
    *,
    support_weight=1.0,
    attention_weight=1.0,
    distribution_weight=1.0,
    node_weight=0.25,
    max_support_edges=8192,
    max_weight_traces=65536,
    max_distribution_groups=512,
    generator=None,
):
    energies = reconstruction_energy_by_node(
        model, graph, view,
        max_support_edges=max_support_edges,
        max_weight_traces=max_weight_traces,
        max_distribution_groups=max_distribution_groups,
        generator=generator,
    )
    target_nodes = torch.nonzero(view.node_mask, as_tuple=False).flatten()
    if target_nodes.numel() == 0:
        raise ValueError("training view must mask at least one response target")
    support = energies["support"][target_nodes].mean()
    weight = energies["weight"][target_nodes].mean()
    distribution = energies["distribution"][target_nodes].mean()
    node = energies["node"][target_nodes].mean()
    total = (
        support_weight * support
        + attention_weight * weight
        + distribution_weight * distribution
        + node_weight * node
    )
    return ReconstructionLosses(support, weight, distribution, node, total)
