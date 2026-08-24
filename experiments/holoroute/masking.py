"""Structure-aware masking for HoloRoute self-supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.attention_holonomy_audit.graph import AttentionEventGraph


@dataclass(frozen=True)
class MaskedGraphInputs:
    head_value: torch.Tensor
    head_observed: torch.Tensor
    event_mask: torch.Tensor
    relay_keep: torch.Tensor
    dropped_relay_target: torch.Tensor


def structurally_supported_events(graph: AttentionEventGraph) -> torch.Tensor:
    supported = torch.zeros(graph.num_events, dtype=torch.bool, device=graph.device)
    for edge_index in (graph.depth_edge_index, graph.relay_edge_index):
        if edge_index.shape[1]:
            supported[edge_index[1]] = True
    groups = len(graph.query_ptr) - 1
    for group in range(groups):
        start = int(graph.query_ptr[group].item())
        stop = int(graph.query_ptr[group + 1].item())
        index = graph.query_event_index[start:stop]
        if len(index) > 1:
            supported[index] = True
    return supported


def sample_boolean_mask(
    eligible: torch.Tensor,
    fraction: float,
    minimum: int,
    generator: torch.Generator,
) -> torch.Tensor:
    index = torch.nonzero(eligible, as_tuple=False).flatten()
    output = torch.zeros_like(eligible)
    if len(index) == 0:
        return output
    count = max(int(round(len(index) * float(fraction))), int(minimum))
    count = min(count, len(index))
    selected = index[torch.randperm(len(index), generator=generator, device=index.device)[:count]]
    output[selected] = True
    return output


def _relay_keep_mask(
    graph: AttentionEventGraph,
    fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = graph.relay_edge_index.shape[1]
    keep = torch.ones(edges, dtype=torch.bool, device=graph.device)
    affected = torch.zeros(graph.num_events, dtype=torch.bool, device=graph.device)
    if edges == 0:
        return keep, affected

    target = graph.relay_edge_index[1]
    degree = torch.zeros(graph.num_events, dtype=torch.long, device=graph.device)
    degree.index_add_(0, target, torch.ones_like(target))
    eligible = degree[target] > 1
    edge_mask = sample_boolean_mask(
        eligible,
        fraction,
        minimum=1,
        generator=generator,
    )
    keep[edge_mask] = False
    if bool(edge_mask.any()):
        affected[target[edge_mask]] = True
    return keep, affected


def mask_graph_inputs(
    graph: AttentionEventGraph,
    *,
    event_fraction: float,
    relay_fraction: float,
    minimum_events: int,
    generator: torch.Generator,
) -> MaskedGraphInputs:
    eligible = structurally_supported_events(graph)
    if not bool(eligible.any()):
        eligible = torch.ones(graph.num_events, dtype=torch.bool, device=graph.device)
    event_mask = sample_boolean_mask(
        eligible,
        event_fraction,
        minimum_events,
        generator,
    )
    value = graph.event_head_value.clone()
    observed = graph.event_head_observed.clone()
    value[event_mask] = 0.0
    observed[event_mask] = False
    relay_keep, dropped_target = _relay_keep_mask(
        graph,
        relay_fraction,
        generator,
    )
    return MaskedGraphInputs(
        head_value=value,
        head_observed=observed,
        event_mask=event_mask,
        relay_keep=relay_keep,
        dropped_relay_target=dropped_target,
    )
