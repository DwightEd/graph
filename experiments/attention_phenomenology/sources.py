"""Exact token-source statistics without dense attention tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .routing import RoutingState


@dataclass(frozen=True)
class ExactSourceStatistics:
    """Per-head fields with shape ``[token, layer, head]``."""

    mass: torch.Tensor
    valid: torch.Tensor
    effective_sources: torch.Tensor
    top1_share: torch.Tensor
    top_source: torch.Tensor
    velocity: torch.Tensor
    transition_valid: torch.Tensor


def _edge_group(routing: RoutingState) -> torch.Tensor:
    edges = routing.edges
    return (
        (edges.query * edges.num_layers + edges.layer) * edges.num_heads
        + edges.head
    )


def _adjacent_velocity(
    routing: RoutingState,
    *,
    selected: torch.Tensor,
    group: torch.Tensor,
    source: torch.Tensor,
    probability: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hellinger distance between adjacent exact-source distributions."""

    edges = routing.edges
    transition_valid = torch.zeros_like(valid)
    transition_valid[1:] = valid[1:] & valid[:-1]
    affinity = probability.new_zeros(valid.numel())

    query = edges.query[selected]
    current = query > 0
    if current.any():
        source_domain = max(edges.num_tokens, 1)
        groups_per_token = edges.num_layers * edges.num_heads
        edge_key = group * source_domain + source
        sorted_key, order = edge_key.sort()
        previous_key = (
            (group[current] - groups_per_token) * source_domain + source[current]
        )
        position = torch.searchsorted(sorted_key, previous_key)
        in_bounds = position < len(sorted_key)
        safe_position = position.clamp_max(len(sorted_key) - 1)
        matched = in_bounds & (sorted_key[safe_position] == previous_key)

        if matched.any():
            previous_edge = order[safe_position[matched]]
            contribution = (
                probability[current][matched] * probability[previous_edge]
            ).sqrt()
            affinity.index_add_(0, group[current][matched], contribution)

    velocity = (1.0 - affinity.clamp(0.0, 1.0)).sqrt().reshape(valid.shape)
    velocity = torch.where(
        transition_valid,
        velocity,
        torch.zeros_like(velocity),
    )
    return velocity, transition_valid


def summarize_exact_sources(
    routing: RoutingState,
    *,
    role: Literal["prompt", "response"],
    epsilon: float,
) -> ExactSourceStatistics:
    """Summarize one exact-source role for every token/layer/head row."""

    edges = routing.edges
    shape = (edges.num_response_tokens, edges.num_layers, edges.num_heads)
    group_count = edges.num_response_tokens * edges.num_layers * edges.num_heads
    is_prompt = edges.source < edges.response_idx
    selected = is_prompt if role == "prompt" else ~is_prompt
    mass = routing.prompt_mass if role == "prompt" else routing.response_mass
    valid = mass > epsilon

    zeros = mass.new_zeros(shape)
    no_source = torch.full(shape, -1, dtype=torch.long, device=edges.device)
    if not selected.any():
        return ExactSourceStatistics(
            mass=mass,
            valid=valid,
            effective_sources=zeros,
            top1_share=zeros,
            top_source=no_source,
            velocity=zeros,
            transition_valid=torch.zeros_like(valid),
        )

    group = _edge_group(routing)[selected]
    source = edges.source[selected]
    if role == "response":
        source = source - edges.response_idx
    weight = routing.edge_weight[selected]
    total = mass.reshape(-1)
    probability = weight / total[group].clamp_min(epsilon)

    entropy = weight.new_zeros(group_count)
    entropy.index_add_(
        0,
        group,
        -probability * probability.clamp_min(epsilon).log(),
    )
    effective_sources = entropy.exp()
    effective_sources[~valid.reshape(-1)] = 0.0

    maximum_weight = torch.full(
        (group_count,),
        -torch.inf,
        dtype=weight.dtype,
        device=weight.device,
    )
    maximum_weight.scatter_reduce_(
        0,
        group,
        weight,
        reduce="amax",
        include_self=True,
    )
    top1_share = maximum_weight / total.clamp_min(epsilon)
    top1_share[~valid.reshape(-1)] = 0.0

    top_source = torch.full(
        (group_count,),
        edges.num_tokens,
        dtype=torch.long,
        device=edges.device,
    )
    is_maximum = weight == maximum_weight[group]
    top_source.scatter_reduce_(
        0,
        group[is_maximum],
        source[is_maximum],
        reduce="amin",
        include_self=True,
    )
    top_source[~valid.reshape(-1)] = -1

    velocity, transition_valid = _adjacent_velocity(
        routing,
        selected=selected,
        group=group,
        source=source,
        probability=probability,
        valid=valid,
    )
    return ExactSourceStatistics(
        mass=mass,
        valid=valid,
        effective_sources=effective_sources.reshape(shape),
        top1_share=top1_share.reshape(shape),
        top_source=top_source.reshape(shape),
        velocity=velocity,
        transition_valid=transition_valid,
    )


def response_lag_statistics(
    routing: RoutingState,
    response_sources: ExactSourceStatistics,
    *,
    recent_tokens: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return recent-source share and mean lag within observed RR mass."""

    edges = routing.edges
    selected = edges.source >= edges.response_idx
    recent_share = torch.zeros_like(routing.response_mass)
    mean_lag = torch.zeros_like(recent_share)
    if not selected.any():
        return recent_share, mean_lag

    group = _edge_group(routing)[selected]
    response_source = edges.source[selected] - edges.response_idx
    lag = edges.query[selected] - response_source
    total = response_sources.mass.reshape(-1)
    probability = routing.edge_weight[selected] / total[group].clamp_min(epsilon)
    recent_share.reshape(-1).index_add_(
        0,
        group,
        probability * ((lag > 0) & (lag <= recent_tokens)),
    )
    mean_lag.reshape(-1).index_add_(0, group, probability * lag.float())
    return recent_share, mean_lag
