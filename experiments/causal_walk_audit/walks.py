"""De Bruijn-style causal walk states and nested Markov-order features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lineage import DIRECT, MULTI_HOP, ONE_HOP, LineageTrace


@dataclass(frozen=True)
class LayerEventGraph:
    source: torch.Tensor          # [event]
    target: torch.Tensor          # [event]
    layer: torch.Tensor           # [event]
    head_value: torch.Tensor      # [event, head]
    head_observed: torch.Tensor   # [event, head]
    predecessor_ptr: torch.Tensor # [event + 1]
    predecessor: torch.Tensor     # [walk relation]
    response_idx: int
    num_response_tokens: int
    num_layers: int
    num_heads: int

    @property
    def num_events(self) -> int:
        return int(self.source.numel())


@dataclass(frozen=True)
class NestedFeatures:
    order1: torch.Tensor
    order2: torch.Tensor
    order3: torch.Tensor
    target: torch.Tensor
    token_index: torch.Tensor
    layer_index: torch.Tensor


def build_layer_event_graph(routing) -> LayerEventGraph:
    edges = routing.edges
    target = edges.response_idx + edges.query
    key = (
        edges.layer.to(torch.int64) * edges.num_tokens * edges.num_tokens
        + target.to(torch.int64) * edges.num_tokens
        + edges.source.to(torch.int64)
    )
    unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
    event_layer = torch.div(
        unique,
        edges.num_tokens * edges.num_tokens,
        rounding_mode="floor",
    )
    remainder = unique % (edges.num_tokens * edges.num_tokens)
    event_target = torch.div(remainder, edges.num_tokens, rounding_mode="floor")
    event_source = remainder % edges.num_tokens

    head_value = routing.edge_weight.new_zeros((len(unique), edges.num_heads))
    head_observed = torch.zeros_like(head_value, dtype=torch.bool)
    head_value.index_put_((inverse, edges.head), routing.edge_weight, accumulate=True)
    head_observed.index_put_(
        (inverse, edges.head),
        torch.ones_like(edges.head, dtype=torch.bool),
        accumulate=False,
    )

    lookup: dict[tuple[int, int], list[int]] = {}
    for index, (layer, target_token) in enumerate(
        zip(event_layer.cpu().tolist(), event_target.cpu().tolist())
    ):
        lookup.setdefault((int(layer), int(target_token)), []).append(index)

    predecessor_rows: list[int] = []
    pointer = [0]
    for layer, source_token in zip(event_layer.cpu().tolist(), event_source.cpu().tolist()):
        if layer > 0 and source_token >= edges.response_idx:
            predecessor_rows.extend(lookup.get((int(layer) - 1, int(source_token)), ()))
        pointer.append(len(predecessor_rows))

    return LayerEventGraph(
        source=event_source.long(),
        target=event_target.long(),
        layer=event_layer.long(),
        head_value=head_value,
        head_observed=head_observed,
        predecessor_ptr=torch.tensor(pointer, dtype=torch.long, device=edges.device),
        predecessor=torch.tensor(
            predecessor_rows, dtype=torch.long, device=edges.device
        ),
        response_idx=edges.response_idx,
        num_response_tokens=edges.num_response_tokens,
        num_layers=edges.num_layers,
        num_heads=edges.num_heads,
    )


def _predecessor_context(graph: LayerEventGraph, steps: int) -> torch.Tensor:
    context = graph.head_value
    for _ in range(steps):
        updated = context.new_zeros(context.shape)
        for event in range(graph.num_events):
            start = int(graph.predecessor_ptr[event])
            stop = int(graph.predecessor_ptr[event + 1])
            if start == stop:
                continue
            predecessor = graph.predecessor[start:stop]
            weight = graph.head_value[predecessor].sum(dim=-1).clamp_min(1e-8)
            updated[event] = (
                context[predecessor] * weight[:, None]
            ).sum(dim=0) / weight.sum()
        context = updated
    return context


def _aggregate_event_context(graph: LayerEventGraph, context: torch.Tensor) -> torch.Tensor:
    rows = graph.num_response_tokens * graph.num_layers
    mean = context.new_zeros((rows, graph.num_heads))
    square = context.new_zeros((rows, graph.num_heads))
    normalizer = context.new_zeros(rows)
    token = graph.target - graph.response_idx
    row = token * graph.num_layers + graph.layer
    weight = graph.head_value.sum(dim=-1).clamp_min(1e-8)
    mean.index_add_(0, row, context * weight[:, None])
    square.index_add_(0, row, context.square() * weight[:, None])
    normalizer.index_add_(0, row, weight)
    mean = mean / normalizer.clamp_min(1e-8)[:, None]
    variance = square / normalizer.clamp_min(1e-8)[:, None] - mean.square()
    return torch.cat((mean, variance.clamp_min(0).sqrt()), dim=-1).reshape(
        graph.num_response_tokens, graph.num_layers, -1
    )


def causal_walk_contexts(graph: LayerEventGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Return order-2 and order-3 predecessor contexts for every token/layer."""

    order2 = _aggregate_event_context(graph, _predecessor_context(graph, 1))
    order3 = _aggregate_event_context(graph, _predecessor_context(graph, 2))
    return order2, order3


def _pad_anchor(value: torch.Tensor, max_anchors: int) -> torch.Tensor:
    pad = max_anchors - value.shape[-1]
    if pad <= 0:
        return value[..., :max_anchors]
    return torch.nn.functional.pad(value, (0, pad))


def build_nested_features(
    routing,
    lineage: LineageTrace,
    event_graph: LayerEventGraph,
    *,
    max_anchors: int,
) -> NestedFeatures:
    """Build nested order-1/2/3 features and the next-layer routing target."""

    tokens = routing.edges.num_response_tokens
    layers = routing.edges.num_layers
    heads = routing.edges.num_heads
    transitions = layers - 1
    position = torch.linspace(0.0, 1.0, tokens, device=routing.edges.device)
    depth = torch.linspace(0.0, 1.0, transitions, device=routing.edges.device)

    role = routing.role_probability[:, :-1].reshape(tokens, transitions, -1)
    base = torch.cat(
        (
            role,
            position[:, None, None].expand(tokens, transitions, 1),
            depth[None, :, None].expand(tokens, transitions, 1),
        ),
        dim=-1,
    )

    direct_anchor = _pad_anchor(
        lineage.direct_anchor()[:, :-1].reshape(tokens, transitions, heads, -1),
        max_anchors,
    ).reshape(tokens, transitions, -1)
    one_anchor = _pad_anchor(
        lineage.state[:, :-1, :, : lineage.anchor_count, ONE_HOP],
        max_anchors,
    ).reshape(tokens, transitions, -1)
    response_direct = lineage.state[
        :, :-1, :, lineage.response_base_index, DIRECT
    ]
    response_one = lineage.response_base_one_hop()[:, :-1]

    order2_walk, order3_walk = causal_walk_contexts(event_graph)
    one = torch.cat(
        (
            direct_anchor,
            one_anchor,
            response_direct.reshape(tokens, transitions, -1),
            response_one.reshape(tokens, transitions, -1),
            order2_walk[:, :-1],
        ),
        dim=-1,
    )

    multi_anchor = _pad_anchor(
        lineage.state[:, :-1, :, : lineage.anchor_count, MULTI_HOP],
        max_anchors,
    ).reshape(tokens, transitions, -1)
    response_multi = lineage.response_base_multihop()[:, :-1]
    multi = torch.cat(
        (
            multi_anchor,
            response_multi.reshape(tokens, transitions, -1),
            order3_walk[:, :-1],
        ),
        dim=-1,
    )

    next_role = routing.role_probability[:, 1:].reshape(tokens, transitions, -1)
    next_anchor = lineage.state[:, 1:, :, : lineage.anchor_count].mean(dim=2)
    next_anchor = _pad_anchor(
        next_anchor.transpose(-1, -2), max_anchors
    ).transpose(-1, -2).reshape(tokens, transitions, -1)
    next_response = lineage.state[
        :, 1:, :, lineage.response_base_index
    ].mean(dim=2)
    target = torch.cat((next_role, next_anchor, next_response), dim=-1)

    token_index = torch.arange(tokens, device=routing.edges.device)[:, None].expand(
        tokens, transitions
    )
    layer_index = torch.arange(transitions, device=routing.edges.device)[None].expand(
        tokens, transitions
    )
    return NestedFeatures(
        order1=base.reshape(-1, base.shape[-1]),
        order2=torch.cat((base, one), dim=-1).reshape(-1, base.shape[-1] + one.shape[-1]),
        order3=torch.cat((base, one, multi), dim=-1).reshape(
            -1, base.shape[-1] + one.shape[-1] + multi.shape[-1]
        ),
        target=target.reshape(-1, target.shape[-1]),
        token_index=token_index.reshape(-1),
        layer_index=layer_index.reshape(-1),
    )
