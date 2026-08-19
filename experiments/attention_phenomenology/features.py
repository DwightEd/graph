"""Mechanism fields for attention analogues of detection, fracture, and breach."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import FEATURE_NAMES, PhenomenologyConfig
from .geometry import HeadSetGeometry, analyze_head_sets
from .provenance import PromptProvenance, layered_prompt_provenance
from .routing import RoutingEdges, RoutingTensor, build_routing_tensor


@dataclass(frozen=True)
class SamplePhenomenology:
    """Full layer-resolved mechanism analysis for one response."""

    layer_features: torch.Tensor  # [response, layer, feature]
    geometry: HeadSetGeometry
    provenance: PromptProvenance
    routing: RoutingTensor


@dataclass(frozen=True)
class ResponseSourceDynamics:
    effective_number: torch.Tensor
    top1_share: torch.Tensor
    recent_mass_fraction: torch.Tensor
    mean_lag: torch.Tensor
    anchor_turnover: torch.Tensor


def response_source_dynamics(
    source_mass: torch.Tensor,
    *,
    config: PhenomenologyConfig,
) -> ResponseSourceDynamics:
    """Exact response-anchor concentration and persistence at every layer."""

    r, l, _ = source_mass.shape
    total = source_mass.sum(dim=2)
    probability = source_mass / total[:, :, None].clamp_min(config.epsilon)
    positive = probability > 0
    entropy = -torch.where(
        positive,
        probability * probability.clamp_min(config.epsilon).log(),
        torch.zeros_like(probability),
    ).sum(dim=2)
    effective = torch.where(
        total > config.epsilon, entropy.exp(), torch.zeros_like(entropy)
    )
    top1 = torch.where(
        total > config.epsilon,
        probability.max(dim=2).values,
        torch.zeros_like(total),
    )

    query = torch.arange(r, device=source_mass.device)[:, None]
    source = torch.arange(r, device=source_mass.device)[None, :]
    lag = query - source
    recent_mask = (lag > 0) & (lag <= config.recent_lag_max)
    recent = (probability * recent_mask[:, None, :]).sum(dim=2)
    mean_lag = (probability * lag.clamp_min(0)[:, None, :]).sum(dim=2)

    keep = min(config.anchor_count, r)
    turnover = torch.zeros((r, l), dtype=torch.float32, device=source_mass.device)
    if keep > 0 and r > 1:
        values, indices = torch.topk(source_mass, k=keep, dim=2)
        valid = values > config.epsilon
        current_indices = indices[1:]
        previous_indices = indices[:-1]
        current_valid = valid[1:]
        previous_valid = valid[:-1]
        match = current_indices[:, :, :, None] == previous_indices[:, :, None, :]
        match &= current_valid[:, :, :, None] & previous_valid[:, :, None, :]
        intersection = match.any(dim=3).sum(dim=2).float()
        current_count = current_valid.sum(dim=2).float()
        previous_count = previous_valid.sum(dim=2).float()
        union = current_count + previous_count - intersection
        turnover[1:] = torch.where(
            union > 0,
            1.0 - intersection / union,
            torch.zeros_like(union),
        )

    return ResponseSourceDynamics(
        effective_number=effective,
        top1_share=top1,
        recent_mass_fraction=recent,
        mean_lag=mean_lag,
        anchor_turnover=turnover,
    )


def _feature_tensor(
    routing: RoutingTensor,
    geometry: HeadSetGeometry,
    provenance: PromptProvenance,
    source: ResponseSourceDynamics,
) -> torch.Tensor:
    direct_prompt_mean = routing.prompt_mass.mean(dim=2)
    direct_prompt_std = routing.prompt_mass.std(dim=2, unbiased=False)
    grounding_lower_mean = provenance.head_lower.mean(dim=2)
    grounding_lower_std = provenance.head_lower.std(dim=2, unbiased=False)
    grounding_upper_mean = provenance.head_upper.mean(dim=2)
    grounding_interval = (provenance.head_upper - provenance.head_lower).mean(dim=2)
    unsupported_lower = provenance.unsupported_lower.mean(dim=2)
    unsupported_upper = provenance.unsupported_upper.mean(dim=2)
    self_mean = routing.self_mass.mean(dim=2)
    unresolved_mean = routing.unresolved_mass.mean(dim=2)
    known_mean = routing.known_mass.mean(dim=2)

    fields = (
        geometry.mean_death,
        geometry.max_death,
        geometry.persistence_entropy,
        geometry.largest_gap,
        geometry.effective_rank,
        geometry.local_intrinsic_dimension,
        direct_prompt_mean,
        direct_prompt_std,
        grounding_lower_mean,
        grounding_lower_std,
        grounding_upper_mean,
        grounding_interval,
        unsupported_lower,
        unsupported_upper,
        source.effective_number,
        source.top1_share,
        source.recent_mass_fraction,
        source.mean_lag,
        source.anchor_turnover,
        geometry.layer_transition,
        geometry.temporal_transition,
        self_mean,
        unresolved_mean,
        known_mean,
    )
    if len(fields) != len(FEATURE_NAMES):
        raise RuntimeError("feature registry and extracted fields differ")
    return torch.stack(fields, dim=2)


def analyze_routing(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
) -> SamplePhenomenology:
    """Extract all label-free mechanism fields from one attention sample."""

    config = PhenomenologyConfig() if config is None else config
    routing = build_routing_tensor(edges, config=config)
    geometry = analyze_head_sets(routing.role_probability, config=config)
    provenance = layered_prompt_provenance(routing)
    source = response_source_dynamics(routing.source_mass, config=config)
    features = _feature_tensor(routing, geometry, provenance, source)
    return SamplePhenomenology(
        layer_features=features,
        geometry=geometry,
        provenance=provenance,
        routing=routing,
    )
