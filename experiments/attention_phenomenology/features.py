"""Compose routing, geometry, provenance, and dynamics into mechanism fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig
from .dynamics import ResponseSourceDynamics, response_source_dynamics
from .geometry import HeadSetGeometry, analyze_head_sets
from .hypotheses import FEATURE_NAMES
from .provenance import PromptProvenance, layered_prompt_provenance
from .routing import RoutingEdges, RoutingTensor, build_routing_tensor


@dataclass(frozen=True)
class SamplePhenomenology:
    """Layer-resolved attention mechanism analysis for one response."""

    layer_features: torch.Tensor
    known_geometry: HeadSetGeometry
    full_geometry: HeadSetGeometry
    provenance: PromptProvenance
    source_dynamics: ResponseSourceDynamics
    routing: RoutingTensor


def _feature_fields(
    routing: RoutingTensor,
    known_geometry: HeadSetGeometry,
    full_geometry: HeadSetGeometry,
    provenance: PromptProvenance,
    source: ResponseSourceDynamics,
) -> dict[str, torch.Tensor]:
    return {
        "known_ph_mean_death": known_geometry.mean_death,
        "known_ph_max_death": known_geometry.max_death,
        "known_ph_persistence_entropy": known_geometry.persistence_entropy,
        "known_ph_largest_gap": known_geometry.largest_gap,
        "known_head_route_effective_rank": known_geometry.effective_rank,
        "known_head_local_intrinsic_dimension": known_geometry.local_intrinsic_dimension,
        "full_ph_mean_death": full_geometry.mean_death,
        "full_ph_max_death": full_geometry.max_death,
        "direct_prompt_mean": routing.prompt_mass.mean(dim=2),
        "direct_prompt_std": routing.prompt_mass.std(dim=2, unbiased=False),
        "grounding_lower_mean": provenance.head_lower.mean(dim=2),
        "grounding_lower_std": provenance.head_lower.std(dim=2, unbiased=False),
        "grounding_upper_mean": provenance.head_upper.mean(dim=2),
        "grounding_interval_width": (
            provenance.head_upper - provenance.head_lower
        ).mean(dim=2),
        "unsupported_rr_lower": provenance.unsupported_rr_lower.mean(dim=2),
        "unsupported_rr_upper": provenance.unsupported_rr_upper.mean(dim=2),
        "response_source_effective_number": source.effective_number,
        "response_source_top1_share": source.top1_share,
        "recent_response_mass_fraction": source.recent_mass_fraction,
        "response_mean_lag": source.mean_lag,
        "response_anchor_turnover": source.anchor_turnover,
        "source_distribution_velocity": source.distribution_velocity,
        "layer_headset_transition": known_geometry.layer_transition,
        "temporal_headset_transition": known_geometry.temporal_transition,
        "self_mass_mean": routing.self_mass.mean(dim=2),
        "unresolved_mass_mean": routing.unresolved_mass.mean(dim=2),
        "known_mass_mean": routing.known_mass.mean(dim=2),
    }


def analyze_routing(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
) -> SamplePhenomenology:
    """Extract all label-free attention mechanism fields for one sample."""

    config = PhenomenologyConfig() if config is None else config
    routing = build_routing_tensor(edges, config=config)
    known_geometry = analyze_head_sets(
        routing.known_role_probability, config=config
    )
    full_geometry = analyze_head_sets(routing.role_probability, config=config)
    provenance = layered_prompt_provenance(routing)
    source = response_source_dynamics(routing.source_mass, config=config)
    fields = _feature_fields(
        routing, known_geometry, full_geometry, provenance, source
    )
    layer_features = torch.stack([fields[name] for name in FEATURE_NAMES], dim=2)
    return SamplePhenomenology(
        layer_features=layer_features,
        known_geometry=known_geometry,
        full_geometry=full_geometry,
        provenance=provenance,
        source_dynamics=source,
        routing=routing,
    )
