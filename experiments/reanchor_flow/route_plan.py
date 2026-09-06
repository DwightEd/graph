"""Budgeted route selection before any exact causal intervention.

The planner only reads a frozen, head-resolved native flow.  It ranks source
units and internal carrier nodes with the already captured target action, then
returns a connected root-to-query corridor.  Exact cut/patch reruns belong to
the caller and are never used for selection here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .flow import PairedFlow
from .native_world import NativeWorld
from .throughput import FlowKernel, FlowThroughput


@dataclass(frozen=True)
class RouteBudget:
    """Fixed candidate budget shared by screening and later confirmation."""

    edges_per_head: int = 2
    max_rows: int = 256
    root_candidates: int = 4
    hub_candidates: int = 8
    corridor_edges: int = 64
    confirm: bool = False

    def __post_init__(self) -> None:
        if self.edges_per_head < 1:
            raise ValueError("edges_per_head must be positive")
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")
        if self.root_candidates < 1:
            raise ValueError("root_candidates must be positive")
        if self.hub_candidates < 0:
            raise ValueError("hub_candidates must be non-negative")
        if self.corridor_edges < 1:
            raise ValueError("corridor_edges must be positive")


@dataclass(frozen=True)
class RootRouteScore:
    """One source unit ranked by connected, target-specific route action."""

    unit_id: int
    route_mass: float
    signed_action: float
    absolute_action: float
    functional_agreement: float
    score: float


@dataclass(frozen=True)
class HubCandidate:
    """One internal token-layer node with functional flow on both sides."""

    layer: int
    position: int
    route_mass: float
    signed_action: float
    score: float


@dataclass(frozen=True)
class AuditPlan:
    """Frozen candidates that an optional intervention runner may confirm."""

    budget: RouteBudget
    roots: tuple[RootRouteScore, ...]
    selected_root_unit_id: int
    root_selection_fallback: bool
    throughput: FlowThroughput
    backbone_edge_index: Tensor
    backbone_position: Tensor
    corridor_edge_index: Tensor
    hubs: tuple[HubCandidate, ...]


def _root_score(
    flow: PairedFlow,
    unit_id: int,
    throughput: FlowThroughput,
) -> RootRouteScore:
    """Average signed and absolute path action over decoder depth."""

    layers = flow.clean_cache.layer_count
    heads = flow.row_total.shape[1]
    edge_layer = flow.edges.layer.long()
    edge_head = flow.edges.head.long()
    action = torch.nan_to_num(flow.edges.clean_target_score.float())
    routed_action = throughput.edge.float() * action
    head_action = torch.zeros(layers * heads)
    head_action.index_add_(0, edge_layer * heads + edge_head, routed_action)
    head_action = head_action.view(layers, heads)
    signed_by_layer = head_action.sum(dim=1)
    absolute_by_layer = head_action.abs().sum(dim=1)
    signed = float(signed_by_layer.mean())
    absolute = float(absolute_by_layer.mean())
    agreement = abs(signed) / absolute if absolute > 0 else 0.0
    route_mass = float(throughput.root_mass)
    return RootRouteScore(
        unit_id=unit_id,
        route_mass=route_mass,
        signed_action=signed,
        absolute_action=absolute,
        functional_agreement=agreement,
        score=route_mass * absolute * agreement,
    )


def _residual_throughput(throughput: FlowThroughput) -> Tensor:
    """Selected-root path mass carried by the implicit residual edge."""

    visit = throughput.reverse_visit.float()
    node = throughput.node.float()
    root_reach = torch.where(visit > 0, node / visit, torch.zeros_like(node))
    return visit[1:] * throughput.residual_probability.float() * root_reach[:-1]


def _widest_backbone(
    flow: PairedFlow,
    world: NativeWorld,
    root_unit_id: int,
    throughput: FlowThroughput,
) -> tuple[Tensor, Tensor]:
    """Return message edges and positions on one widest connected path."""

    node = throughput.node.float().cpu()
    layers = node.shape[0] - 1
    tokens = node.shape[1]
    query = int(flow.target.query_position)
    token_unit = world.units.token_unit_id.long().cpu()
    root_position = torch.nonzero(
        (token_unit == root_unit_id) & (torch.arange(tokens) < world.response_start),
        as_tuple=False,
    ).flatten()

    best = torch.full((tokens,), -torch.inf)
    best[root_position] = torch.inf
    predecessor_position = torch.full((layers, tokens), -1, dtype=torch.long)
    predecessor_edge = torch.full((layers, tokens), -2, dtype=torch.long)
    residual = _residual_throughput(throughput).cpu()
    edge_flow = throughput.edge.float().cpu()
    edges = flow.edges

    for layer in range(layers):
        following = torch.full_like(best, -torch.inf)
        residual_position = torch.nonzero(residual[layer] > 0, as_tuple=False).flatten()
        for position in residual_position.tolist():
            candidate = min(float(best[position]), float(residual[layer, position]))
            if candidate > float(following[position]):
                following[position] = candidate
                predecessor_position[layer, position] = position
                predecessor_edge[layer, position] = -1

        layer_edge = torch.nonzero(
            (edges.layer.long() == layer) & (edge_flow > 0),
            as_tuple=False,
        ).flatten()
        for edge_index in layer_edge.tolist():
            source = int(edges.source[edge_index])
            target = int(edges.target[edge_index])
            candidate = min(float(best[source]), float(edge_flow[edge_index]))
            if candidate > float(following[target]):
                following[target] = candidate
                predecessor_position[layer, target] = source
                predecessor_edge[layer, target] = edge_index
        best = following

    if not torch.isfinite(best[query]):
        if throughput.root_mass > 0:
            raise ValueError("positive selected-root flow has no connected backbone")
        return torch.empty(0, dtype=torch.long), torch.tensor([query])

    position = torch.empty(layers + 1, dtype=torch.long)
    position[layers] = query
    path_edges: list[int] = []
    for layer in range(layers - 1, -1, -1):
        target = int(position[layer + 1])
        source = int(predecessor_position[layer, target])
        edge_index = int(predecessor_edge[layer, target])
        position[layer] = source
        if edge_index >= 0:
            path_edges.append(edge_index)
    path_edges.reverse()
    return torch.tensor(path_edges, dtype=torch.long), position


def _corridor_edges(
    flow: PairedFlow,
    throughput: FlowThroughput,
    backbone: Tensor,
    budget: RouteBudget,
) -> Tensor:
    """Keep the backbone, then fill the budget with strongest routed edges."""

    chosen = backbone.tolist()
    if len(chosen) > budget.corridor_edges:
        raise ValueError(
            "corridor_edges is smaller than the connected root-to-query backbone"
        )
    group_count: dict[tuple[int, int, int], int] = {}
    for edge_index in chosen:
        key = (
            int(flow.edges.layer[edge_index]),
            int(flow.edges.head[edge_index]),
            int(flow.edges.target[edge_index]),
        )
        group_count[key] = group_count.get(key, 0) + 1

    edge_flow = throughput.edge.float()
    edge_utility = (
        edge_flow * torch.nan_to_num(flow.edges.clean_target_score.float()).abs()
    )
    eligible = edge_flow > 0
    if chosen:
        eligible[torch.tensor(chosen, dtype=torch.long)] = False
    ranked_utility = torch.where(
        eligible,
        edge_utility,
        torch.full_like(edge_utility, -torch.inf),
    )
    candidates = torch.argsort(ranked_utility, descending=True, stable=True)
    target_count = budget.corridor_edges
    for edge_index in candidates:
        if not bool(eligible[edge_index]):
            break
        edge_index = int(edge_index)
        if len(chosen) >= target_count:
            break
        key = (
            int(flow.edges.layer[edge_index]),
            int(flow.edges.head[edge_index]),
            int(flow.edges.target[edge_index]),
        )
        if group_count.get(key, 0) >= budget.edges_per_head:
            continue
        chosen.append(edge_index)
        group_count[key] = group_count.get(key, 0) + 1
    return torch.tensor(chosen, dtype=torch.long)


def _hub_candidates(
    flow: PairedFlow,
    throughput: FlowThroughput,
    budget: RouteBudget,
) -> tuple[HubCandidate, ...]:
    """Rank two-sided functional bottlenecks and suppress residual duplicates."""

    if budget.hub_candidates == 0:
        return ()
    layers = flow.clean_cache.layer_count
    tokens = throughput.node.shape[1]
    edge_layer = flow.edges.layer.long()
    source = flow.edges.source.long()
    target = flow.edges.target.long()
    action = torch.nan_to_num(flow.edges.clean_target_score.float())
    routed = throughput.edge.float() * action

    incoming_signed = torch.zeros(layers + 1, tokens)
    incoming_absolute = torch.zeros_like(incoming_signed)
    outgoing_signed = torch.zeros_like(incoming_signed)
    outgoing_absolute = torch.zeros_like(incoming_signed)
    incoming_signed.index_put_((edge_layer + 1, target), routed, accumulate=True)
    incoming_absolute.index_put_(
        (edge_layer + 1, target), routed.abs(), accumulate=True
    )
    outgoing_signed.index_put_((edge_layer, source), routed, accumulate=True)
    outgoing_absolute.index_put_((edge_layer, source), routed.abs(), accumulate=True)

    residual = throughput.residual_probability.float()
    left_signed = incoming_signed.clone()
    left_absolute = incoming_absolute.clone()
    for layer in range(layers):
        left_signed[layer + 1] += left_signed[layer] * residual[layer]
        left_absolute[layer + 1] += left_absolute[layer] * residual[layer]
    right_signed = outgoing_signed.clone()
    right_absolute = outgoing_absolute.clone()
    for layer in range(layers - 1, -1, -1):
        right_signed[layer] += right_signed[layer + 1] * residual[layer]
        right_absolute[layer] += right_absolute[layer + 1] * residual[layer]

    represented = set(flow.row_position.long().tolist())
    query_position = int(flow.target.query_position)
    candidates: list[HubCandidate] = []
    for layer in range(1, layers):
        for position in range(tokens):
            if position not in represented or position >= query_position:
                continue
            route_mass = float(throughput.node[layer, position])
            two_sided = min(
                float(left_absolute[layer, position]),
                float(right_absolute[layer, position]),
            )
            left_budget = float(left_absolute[layer, position])
            right_budget = float(right_absolute[layer, position])
            left_agreement = (
                abs(float(left_signed[layer, position])) / left_budget
                if left_budget > 0
                else 0.0
            )
            right_agreement = (
                abs(float(right_signed[layer, position])) / right_budget
                if right_budget > 0
                else 0.0
            )
            score = route_mass * two_sided * min(left_agreement, right_agreement)
            if score <= 0:
                continue
            signed = 0.5 * (
                float(left_signed[layer, position])
                + float(right_signed[layer, position])
            )
            candidates.append(HubCandidate(layer, position, route_mass, signed, score))
    candidates.sort(
        key=lambda item: (-item.score, -item.route_mass, item.layer, item.position)
    )

    selected: list[HubCandidate] = []
    for candidate in candidates:
        if any(
            candidate.position == previous.position
            and abs(candidate.layer - previous.layer) <= 1
            for previous in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == budget.hub_candidates:
            break
    return tuple(selected)


def plan_audit(
    flow: PairedFlow,
    world: NativeWorld,
    budget: RouteBudget,
) -> AuditPlan:
    """Plan roots, a connected corridor, and hubs without a model rerun."""

    kernel = FlowKernel.from_flow(
        flow,
        world.units.token_unit_id,
        world.units.count,
    )
    scored: list[RootRouteScore] = []
    for unit_id in world.evidence_unit_id:
        throughput = kernel.condition(
            flow,
            world.units.token_unit_id,
            (unit_id,),
        )
        scored.append(_root_score(flow, unit_id, throughput))
        del throughput

    ranked = sorted(
        scored,
        key=lambda item: (-item.score, -item.route_mass, item.unit_id),
    )
    root_selection_fallback = not any(item.score > 0 for item in ranked)
    if not root_selection_fallback:
        selected_score = ranked[0]
    else:
        selected_score = max(
            ranked,
            key=lambda item: (item.route_mass, -item.unit_id),
        )
    selected_throughput = kernel.condition(
        flow,
        world.units.token_unit_id,
        (selected_score.unit_id,),
    )
    roots = tuple(ranked[: budget.root_candidates])
    backbone, backbone_position = _widest_backbone(
        flow,
        world,
        selected_score.unit_id,
        selected_throughput,
    )
    corridor = _corridor_edges(
        flow,
        selected_throughput,
        backbone,
        budget,
    )
    hubs = _hub_candidates(
        flow,
        selected_throughput,
        budget,
    )
    return AuditPlan(
        budget=budget,
        roots=roots,
        selected_root_unit_id=selected_score.unit_id,
        root_selection_fallback=root_selection_fallback,
        throughput=selected_throughput,
        backbone_edge_index=backbone,
        backbone_position=backbone_position,
        corridor_edge_index=corridor,
        hubs=hubs,
    )
