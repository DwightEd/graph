"""Conserved source-to-target flow on the layer-unrolled ETCC graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .flow import PairedFlow


@dataclass(frozen=True)
class FlowThroughput:
    """Reverse path mass and root-conditioned node/edge participation.

    ``unit_mass[u]`` is the retained target-to-source path mass ending in
    source unit ``u``.  ``node`` and ``edge`` are conditioned on the selected
    root, so a non-empty retained root flow sums to one at every unrolled
    depth.  The sparse coverage complement is deliberately not redistributed.
    """

    edge_probability: Tensor
    residual_probability: Tensor
    reverse_visit: Tensor
    unit_mass: Tensor
    root_unit_id: tuple[int, ...]
    root_mass: float
    node: Tensor
    edge: Tensor


def transition_probabilities(
    flow: PairedFlow,
    tokens: int,
) -> tuple[Tensor, Tensor]:
    """Turn either backend into one explicit residual-aware transition law.

    Paired ETCC keeps its registered half-residual law. A native flow may
    instead provide an explicit non-negative ``residual_weight`` for every
    represented layer/row. In both cases coverage-pruned mass goes to an
    unobserved sink rather than being silently renormalized onto retained
    edges.
    """

    edges = flow.edges
    layers, _, position_count = flow.row_total.shape
    lookup = torch.full((tokens,), -1, dtype=torch.long)
    lookup[flow.row_position.long()] = torch.arange(position_count)
    target_slot = lookup.index_select(0, edges.target.long())
    if bool((target_slot < 0).any()):
        raise ValueError("an edge target is absent from the represented rows")

    row_total = flow.row_total.sum(dim=1)
    selected_total = row_total[edges.layer.long(), target_slot]
    magnitude = edges.score.float().abs()
    residual = torch.ones(layers, tokens)
    residual_weight = getattr(flow, "residual_weight", None)
    if residual_weight is None:
        probability = torch.where(
            selected_total > 0,
            0.5 * magnitude / selected_total,
            torch.zeros_like(magnitude),
        )
        represented_total = row_total > 0
        for slot, position in enumerate(flow.row_position.tolist()):
            residual[:, position] = torch.where(
                represented_total[:, slot],
                torch.full((layers,), 0.5),
                torch.ones(layers),
            )
        return probability, residual

    residual_weight = residual_weight.float()
    if residual_weight.shape != row_total.shape:
        raise ValueError("native residual weights do not match represented rows")
    if not bool(torch.isfinite(residual_weight).all()) or bool(
        (residual_weight < 0).any()
    ):
        raise ValueError("native residual weights must be finite and non-negative")
    denominator = selected_total + residual_weight[edges.layer.long(), target_slot]
    probability = torch.where(
        denominator > 0,
        magnitude / denominator,
        torch.zeros_like(magnitude),
    )
    represented_denominator = row_total + residual_weight
    for slot, position in enumerate(flow.row_position.tolist()):
        denominator_row = represented_denominator[:, slot]
        residual[:, position] = torch.where(
            denominator_row > 0,
            residual_weight[:, slot] / denominator_row,
            torch.ones(layers),
        )
    return probability, residual


def reverse_visitation(
    flow: PairedFlow,
    edge_probability: Tensor,
    residual_probability: Tensor,
    tokens: int,
) -> Tensor:
    """Propagate one unit of target mass backwards through the sparse DAG."""

    layers = flow.clean_cache.layer_count
    visit = torch.zeros(layers + 1, tokens)
    visit[layers, flow.target.query_position] = 1.0
    edges = flow.edges
    for layer in range(layers - 1, -1, -1):
        visit[layer] += visit[layer + 1] * residual_probability[layer]
        selected = torch.nonzero(edges.layer == layer, as_tuple=False).flatten()
        if not len(selected):
            continue
        source = edges.source.index_select(0, selected).long()
        target = edges.target.index_select(0, selected).long()
        transported = visit[layer + 1].index_select(0, target)
        transported *= edge_probability.index_select(0, selected)
        visit[layer].index_add_(0, source, transported)
    return visit


def source_unit_mass(
    visit: Tensor,
    token_unit_id: Tensor,
    unit_count: int,
) -> Tensor:
    mass = torch.zeros(unit_count)
    mass.index_add_(0, token_unit_id.long(), visit[0])
    return mass


def root_reachability(
    flow: PairedFlow,
    edge_probability: Tensor,
    residual_probability: Tensor,
    root_position: Tensor,
    tokens: int,
) -> Tensor:
    """Probability that a reverse route from each node terminates at the root."""

    layers = flow.clean_cache.layer_count
    reach = torch.zeros(layers + 1, tokens)
    reach[0, root_position.long()] = 1.0
    edges = flow.edges
    for layer in range(layers):
        reach[layer + 1] = reach[layer] * residual_probability[layer]
        selected = torch.nonzero(edges.layer == layer, as_tuple=False).flatten()
        if not len(selected):
            continue
        source = edges.source.index_select(0, selected).long()
        target = edges.target.index_select(0, selected).long()
        transported = reach[layer].index_select(0, source)
        transported *= edge_probability.index_select(0, selected)
        reach[layer + 1].index_add_(0, target, transported)
    return reach


def compute_throughput(
    flow: PairedFlow,
    token_unit_id: Tensor,
    unit_count: int,
    root_unit_id: tuple[int, ...],
) -> FlowThroughput:
    """Compute ``C(u→t)`` and ``T(v|u,t)`` for one selected root group."""

    tokens = len(token_unit_id)
    probability, residual = transition_probabilities(flow, tokens)
    visit = reverse_visitation(flow, probability, residual, tokens)
    unit_mass = source_unit_mass(visit, token_unit_id, unit_count)
    root_position = torch.zeros(tokens, dtype=torch.bool)
    for unit_id in root_unit_id:
        root_position |= token_unit_id == unit_id
    position = torch.nonzero(root_position, as_tuple=False).flatten()
    reach = root_reachability(flow, probability, residual, position, tokens)
    root_mass = float(reach[-1, flow.target.query_position])

    if root_mass <= 0:
        node = torch.zeros_like(visit)
        edge = torch.zeros(flow.edges.count)
    else:
        node = visit * reach / root_mass
        layers = flow.edges.layer.long()
        source = flow.edges.source.long()
        target = flow.edges.target.long()
        edge = visit[layers + 1, target] * probability
        edge *= reach[layers, source] / root_mass
    return FlowThroughput(
        probability,
        residual,
        visit,
        unit_mass,
        root_unit_id,
        root_mass,
        node,
        edge,
    )
