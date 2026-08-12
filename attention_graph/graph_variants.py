"""Auditable counterfactuals of a causal RP/RR attention graph."""

from __future__ import annotations

import torch

from .graph import AttentionGraph, RP


VARIANTS = (
    "full", "no_edges", "marginals", "source_rewire", "binary",
    "collapse_relations", "mean_heads", "shuffle_layers",
)


def _replace(graph: AttentionGraph, **values) -> AttentionGraph:
    return AttentionGraph(**{**graph.__dict__, **values})


def _empty_edges(graph: AttentionGraph) -> AttentionGraph:
    device = graph.node_attr.device
    return _replace(
        graph,
        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
        edge_type=torch.empty(0, dtype=torch.long, device=device),
        edge_score=torch.empty(0, dtype=torch.float32, device=device),
        trace_edge_id=torch.empty(0, dtype=torch.long, device=device),
        trace_channel=torch.empty(0, dtype=torch.long, device=device),
        trace_value=torch.empty(0, dtype=torch.float32, device=device),
    )


def _edge_scores(edge_count, trace_edge_id, trace_value, channels):
    scores = torch.zeros(edge_count, dtype=torch.float32, device=trace_value.device)
    scores.index_add_(0, trace_edge_id, trace_value.float())
    return scores / float(channels)


def _sort_traces(edge_id, channel, value, channels):
    order = torch.argsort(edge_id * channels + channel, stable=True)
    return edge_id[order], channel[order], value[order]


def _marginals(graph: AttentionGraph) -> AttentionGraph:
    """Aggregate source identities into deterministic RP/RR canonical endpoints."""
    if graph.num_edges == 0:
        return graph
    trace_edge = graph.trace_edge_id
    target = graph.edge_index[1, trace_edge]
    relation = (graph.edge_index[0, trace_edge] >= graph.response_idx).long()
    trace_key = (target * 2 + relation) * graph.num_channels + graph.trace_channel
    unique_trace, inverse = torch.unique(trace_key, sorted=True, return_inverse=True)
    trace_value = torch.zeros(len(unique_trace), dtype=torch.float32, device=graph.node_attr.device)
    trace_value.index_add_(0, inverse, graph.trace_value.float())
    trace_channel = unique_trace.remainder(graph.num_channels)
    pair_key = torch.div(unique_trace, graph.num_channels, rounding_mode="floor")
    pair_key, trace_edge_id = torch.unique(pair_key, sorted=True, return_inverse=True)
    pair_target = torch.div(pair_key, 2, rounding_mode="floor")
    edge_type = pair_key.remainder(2).long()
    edge_source = torch.where(edge_type == RP, torch.zeros_like(pair_target), pair_target - 1)
    trace_edge_id, trace_channel, trace_value = _sort_traces(
        trace_edge_id, trace_channel, trace_value, graph.num_channels
    )
    return _replace(
        graph,
        edge_index=torch.stack((edge_source, pair_target)), edge_type=edge_type,
        edge_score=_edge_scores(len(pair_key), trace_edge_id, trace_value, graph.num_channels),
        trace_edge_id=trace_edge_id, trace_channel=trace_channel, trace_value=trace_value,
    )


def _rewire_sources(graph: AttentionGraph, seed: int) -> AttentionGraph:
    """Swap source endpoints only when two bundles admit a legal non-duplicate swap."""
    if graph.num_edges < 2:
        return graph
    source, target = graph.edge_index
    device = source.device
    source_cpu, target_cpu = source.cpu().tolist(), target.cpu().tolist()
    relation_cpu = graph.edge_type.cpu().tolist()
    rewired_cpu = source_cpu.copy()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Each swap preserves global source degree, target/relation degrees, exact
    # trace payload, causality, and simple pair support.  A source can only move
    # to a target where it remains in the same RP/RR partition.
    occupied = set(zip(source_cpu, target_cpu))
    used = [False] * graph.num_edges
    for relation in (0, 1):
        candidates = [index for index, value in enumerate(relation_cpu) if value == relation]
        for _ in range(4):
            order = torch.randperm(len(candidates), generator=generator).tolist()
            for offset in range(0, len(order) - 1, 2):
                first, second = candidates[order[offset]], candidates[order[offset + 1]]
                if used[first] or used[second]:
                    continue
                a_source, a_target = source_cpu[first], target_cpu[first]
                b_source, b_target = source_cpu[second], target_cpu[second]
                if a_target == b_target or a_source == b_source:
                    continue
                valid_a = (b_source < graph.response_idx) == (relation == RP) and b_source < a_target
                valid_b = (a_source < graph.response_idx) == (relation == RP) and a_source < b_target
                if not (valid_a and valid_b):
                    continue
                if (b_source, a_target) in occupied or (a_source, b_target) in occupied:
                    continue
                rewired_cpu[first], rewired_cpu[second] = b_source, a_source
                used[first], used[second] = True, True
                occupied.remove((a_source, a_target))
                occupied.remove((b_source, b_target))
                occupied.add((b_source, a_target))
                occupied.add((a_source, b_target))
    rewired = torch.tensor(rewired_cpu, dtype=source.dtype, device=device)
    return _replace(graph, edge_index=torch.stack((rewired, target)))


def rewire_moved_fractions(original: AttentionGraph, transformed: AttentionGraph) -> dict[str, float]:
    moved = original.edge_index[0] != transformed.edge_index[0]
    result = {"overall": float(moved.float().mean()) if original.num_edges else 0.0}
    for name, relation in (("rp", RP), ("rr", 1)):
        selected = original.edge_type == relation
        result[name] = float(moved[selected].float().mean()) if bool(selected.any()) else 0.0
    return result


def _mean_heads(graph: AttentionGraph) -> AttentionGraph:
    if graph.num_edges == 0:
        return graph
    layer = torch.div(graph.trace_channel, graph.num_heads, rounding_mode="floor")
    key = graph.trace_edge_id * graph.num_layers + layer
    unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
    values = torch.zeros(len(unique), dtype=torch.float32, device=graph.node_attr.device)
    values.index_add_(0, inverse, graph.trace_value.float())
    edge_id = torch.div(unique, graph.num_layers, rounding_mode="floor")
    channel = unique.remainder(graph.num_layers) * graph.num_heads
    edge_id, channel, values = _sort_traces(edge_id, channel, values, graph.num_channels)
    diagonal = graph.node_attr.reshape(graph.num_nodes, graph.num_layers, graph.num_heads).mean(2, keepdim=True)
    diagonal = diagonal.repeat(1, 1, graph.num_heads).reshape(graph.num_nodes, graph.num_channels)
    return _replace(
        graph, node_attr=diagonal, trace_edge_id=edge_id, trace_channel=channel,
        trace_value=values,
        edge_score=_edge_scores(graph.num_edges, edge_id, values, graph.num_channels),
    )


def _shuffle_layers(graph: AttentionGraph, seed: int) -> AttentionGraph:
    generator = torch.Generator(device=graph.node_attr.device).manual_seed(seed)
    order = torch.randperm(graph.num_layers, generator=generator, device=graph.node_attr.device)
    layer = torch.div(graph.trace_channel, graph.num_heads, rounding_mode="floor")
    head = graph.trace_channel.remainder(graph.num_heads)
    # old layer `layer` moves to its new index `order[layer]`; node rows use
    # exactly that old-to-new mapping, not inverse indexing.
    channel = order[layer] * graph.num_heads + head
    edge_id, channel, value = _sort_traces(graph.trace_edge_id, channel, graph.trace_value, graph.num_channels)
    diagonal = torch.empty_like(graph.node_attr.reshape(graph.num_nodes, graph.num_layers, graph.num_heads))
    diagonal[:, order] = graph.node_attr.reshape(graph.num_nodes, graph.num_layers, graph.num_heads)
    return _replace(
        graph, node_attr=diagonal.reshape(graph.num_nodes, graph.num_channels),
        trace_edge_id=edge_id, trace_channel=channel, trace_value=value,
    )


def transform_graph(graph: AttentionGraph, variant: str, *, seed: int) -> AttentionGraph:
    if variant == "full":
        return graph
    if variant == "no_edges":
        return _empty_edges(graph)
    if variant == "marginals":
        return _marginals(graph)
    if variant == "source_rewire":
        return _rewire_sources(graph, seed)
    if variant == "binary":
        values = torch.ones_like(graph.trace_value)
        return _replace(graph, node_attr=(graph.node_attr != 0).float(), trace_value=values,
                        edge_score=_edge_scores(graph.num_edges, graph.trace_edge_id, values, graph.num_channels))
    if variant == "collapse_relations":
        return _replace(graph, edge_type=torch.zeros_like(graph.edge_type))
    if variant == "mean_heads":
        return _mean_heads(graph)
    if variant == "shuffle_layers":
        return _shuffle_layers(graph, seed)
    raise ValueError(f"unknown graph validation variant: {variant}")
