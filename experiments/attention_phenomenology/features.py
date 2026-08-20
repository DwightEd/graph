"""Compose routing, exact sources, and provenance into mechanism features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig
from .hypotheses import FEATURE_NAMES
from .provenance import PromptProvenance, layered_prompt_provenance
from .routing import RoutingEdges, RoutingState, build_routing_state
from .sources import (
    ExactSourceStatistics,
    response_lag_statistics,
    summarize_exact_sources,
)


@dataclass(frozen=True)
class SamplePhenomenology:
    """All label-free mechanism fields for one response."""

    layer_features: torch.Tensor
    routing: RoutingState
    prompt_sources: ExactSourceStatistics
    response_sources: ExactSourceStatistics
    provenance: PromptProvenance


def _valid_head_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=-1)
    total = torch.where(valid, values, torch.zeros_like(values)).sum(dim=-1)
    return torch.where(
        count > 0,
        total / count.clamp_min(1),
        torch.zeros_like(total),
    )


def _head_anchor_agreement(statistics: ExactSourceStatistics) -> torch.Tensor:
    """Fraction of valid head pairs with the same strongest exact source."""

    source = statistics.top_source
    valid = statistics.valid
    heads = source.shape[-1]
    unique_pairs = torch.triu(
        torch.ones((heads, heads), dtype=torch.bool, device=source.device),
        diagonal=1,
    )
    valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2) & unique_pairs
    same_source = source.unsqueeze(-1) == source.unsqueeze(-2)
    pair_count = valid_pair.sum(dim=(-2, -1))
    matching = (valid_pair & same_source).sum(dim=(-2, -1))
    return torch.where(
        pair_count > 0,
        matching.float() / pair_count.clamp_min(1),
        torch.zeros_like(pair_count, dtype=torch.float32),
    )


def _prompt_response_head_disagreement(
    routing: RoutingState,
    epsilon: float,
) -> torch.Tensor:
    """Mean pairwise Hellinger distance over each head's prompt/RR balance."""

    route_mass = torch.stack((routing.prompt_mass, routing.response_mass), dim=-1)
    off_diagonal_mass = route_mass.sum(dim=-1)
    valid = off_diagonal_mass > epsilon
    probability = route_mass / off_diagonal_mass.unsqueeze(-1).clamp_min(epsilon)
    root_probability = probability.sqrt()
    affinity = (
        root_probability.unsqueeze(-2) * root_probability.unsqueeze(-3)
    ).sum(dim=-1)
    distance = (1.0 - affinity.clamp(0.0, 1.0)).sqrt()

    heads = routing.edges.num_heads
    unique_pairs = torch.triu(
        torch.ones((heads, heads), dtype=torch.bool, device=distance.device),
        diagonal=1,
    )
    valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2) & unique_pairs
    pair_count = valid_pair.sum(dim=(-2, -1))
    total_distance = torch.where(
        valid_pair,
        distance,
        torch.zeros_like(distance),
    ).sum(dim=(-2, -1))
    return torch.where(
        pair_count > 0,
        total_distance / pair_count.clamp_min(1),
        torch.zeros_like(total_distance),
    )


def _feature_fields(
    routing: RoutingState,
    prompt: ExactSourceStatistics,
    response: ExactSourceStatistics,
    provenance: PromptProvenance,
    config: PhenomenologyConfig,
) -> dict[str, torch.Tensor]:
    recent_share, mean_lag = response_lag_statistics(
        routing,
        response,
        recent_tokens=config.recent_response_tokens,
        epsilon=config.epsilon,
    )
    off_diagonal_mass = routing.prompt_mass + routing.response_mass
    response_takeover = routing.response_mass / off_diagonal_mass.clamp_min(
        config.epsilon
    )
    response_takeover = torch.where(
        off_diagonal_mass > config.epsilon,
        response_takeover,
        torch.zeros_like(response_takeover),
    )
    return {
        "prompt_mass_mean": routing.prompt_mass.mean(dim=-1),
        "prompt_effective_sources_mean": _valid_head_mean(
            prompt.effective_sources, prompt.valid
        ),
        "prompt_top1_share_mean": _valid_head_mean(prompt.top1_share, prompt.valid),
        "prompt_source_velocity_mean": _valid_head_mean(
            prompt.velocity, prompt.transition_valid
        ),
        "prompt_response_head_disagreement": _prompt_response_head_disagreement(
            routing, config.epsilon
        ),
        "prompt_mass_head_std": routing.prompt_mass.std(dim=-1, unbiased=False),
        "prompt_provenance_head_std": provenance.head_lower.std(
            dim=-1, unbiased=False
        ),
        "prompt_anchor_head_agreement": _head_anchor_agreement(prompt),
        "prompt_provenance_lower_mean": provenance.head_lower.mean(dim=-1),
        "prompt_provenance_uncertainty": (
            provenance.head_upper - provenance.head_lower
        ).mean(dim=-1),
        "unsupported_response_mass_mean": provenance.unsupported_response_lower.mean(
            dim=-1
        ),
        "response_takeover_mean": _valid_head_mean(
            response_takeover,
            off_diagonal_mass > config.epsilon,
        ),
        "response_effective_sources_mean": _valid_head_mean(
            response.effective_sources, response.valid
        ),
        "response_top1_share_mean": _valid_head_mean(
            response.top1_share, response.valid
        ),
        "recent_response_share_mean": _valid_head_mean(recent_share, response.valid),
        "response_mean_lag_mean": _valid_head_mean(mean_lag, response.valid),
        "response_source_velocity_mean": _valid_head_mean(
            response.velocity, response.transition_valid
        ),
        "response_anchor_head_agreement": _head_anchor_agreement(response),
        "self_mass_mean": routing.self_mass.mean(dim=-1),
        "unresolved_mass_mean": routing.unresolved_mass.mean(dim=-1),
        "known_mass_mean": routing.known_mass.mean(dim=-1),
    }


def analyze_routing(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
) -> SamplePhenomenology:
    """Return all causal, label-free mechanism fields for one response."""

    config = PhenomenologyConfig() if config is None else config
    routing = build_routing_state(edges)
    prompt_sources = summarize_exact_sources(
        routing,
        role="prompt",
        epsilon=config.epsilon,
    )
    response_sources = summarize_exact_sources(
        routing,
        role="response",
        epsilon=config.epsilon,
    )
    provenance = layered_prompt_provenance(routing)
    fields = _feature_fields(
        routing,
        prompt_sources,
        response_sources,
        provenance,
        config,
    )
    layer_features = torch.stack([fields[name] for name in FEATURE_NAMES], dim=-1)
    return SamplePhenomenology(
        layer_features=layer_features,
        routing=routing,
        prompt_sources=prompt_sources,
        response_sources=response_sources,
        provenance=provenance,
    )
