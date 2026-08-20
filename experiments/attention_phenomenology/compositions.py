"""Simplex-valued attention compositions for distributional validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompositionView:
    name: str
    values: torch.Tensor  # [token, layer, head, component]
    component_names: tuple[str, ...]


ROLE_COMPONENTS = (
    "prompt",
    "response_history",
    "self",
    "unresolved",
)

PROVENANCE_COMPONENTS = (
    "direct_prompt",
    "grounded_response_lower",
    "unsupported_response_lower",
    "uncertain_response",
    "self",
    "unresolved",
)


def _close(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    values = values.clamp_min(0.0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def role_composition(analysis, *, epsilon: float = 1e-8) -> CompositionView:
    """Return the direct four-role routing composition."""

    return CompositionView(
        name="role",
        values=_close(analysis.routing.role_probability, epsilon),
        component_names=ROLE_COMPONENTS,
    )


def provenance_composition(analysis, *, epsilon: float = 1e-8) -> CompositionView:
    """Split response mass using exact-source prompt-provenance bounds.

    The response branch is decomposed into mass known to be prompt grounded,
    mass known to be unsupported, and the remaining provenance-ambiguous mass.
    """

    routing = analysis.routing
    provenance = analysis.provenance
    previous_lower = provenance.aggregate_lower[:, :-1].unsqueeze(-1)
    grounded = (
        provenance.head_lower
        - routing.prompt_mass
        - routing.self_mass * previous_lower
    ).clamp_min(0.0)
    unsupported = provenance.unsupported_response_lower.clamp_min(0.0)

    explained = grounded + unsupported
    rescale = torch.where(
        explained > routing.response_mass,
        routing.response_mass / explained.clamp_min(epsilon),
        torch.ones_like(explained),
    )
    grounded = grounded * rescale
    unsupported = unsupported * rescale
    uncertain = (routing.response_mass - grounded - unsupported).clamp_min(0.0)

    values = torch.stack(
        (
            routing.prompt_mass,
            grounded,
            unsupported,
            uncertain,
            routing.self_mass,
            routing.unresolved_mass,
        ),
        dim=-1,
    )
    return CompositionView(
        name="provenance",
        values=_close(values, epsilon),
        component_names=PROVENANCE_COMPONENTS,
    )


def composition_views(analysis, *, epsilon: float = 1e-8) -> dict[str, CompositionView]:
    views = (
        role_composition(analysis, epsilon=epsilon),
        provenance_composition(analysis, epsilon=epsilon),
    )
    return {view.name: view for view in views}
