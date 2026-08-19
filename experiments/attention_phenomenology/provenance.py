"""Ordered-layer prompt provenance with explicit censoring bounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .routing import RoutingTensor


@dataclass(frozen=True)
class PromptProvenance:
    head_lower: torch.Tensor
    head_upper: torch.Tensor
    unsupported_rr_lower: torch.Tensor
    unsupported_rr_upper: torch.Tensor
    aggregate_lower: torch.Tensor
    aggregate_upper: torch.Tensor


def layered_prompt_provenance(routing: RoutingTensor) -> PromptProvenance:
    """Propagate prompt ancestry through ordered attention layers."""

    response_count = routing.edges.num_response_tokens
    layers = routing.edges.num_layers
    heads = routing.edges.num_heads
    shape = (response_count, layers, heads)
    head_lower = torch.zeros(shape, device=routing.edges.device)
    head_upper = torch.zeros_like(head_lower)
    unsupported_lower = torch.zeros_like(head_lower)
    unsupported_upper = torch.zeros_like(head_lower)
    aggregate_lower = torch.zeros(
        (response_count, layers + 1), device=routing.edges.device
    )
    aggregate_upper = torch.zeros_like(aggregate_lower)

    previous_lower = aggregate_lower[:, 0]
    previous_upper = aggregate_upper[:, 0]

    for layer in range(layers):
        lower = routing.prompt_mass[:, layer].clone()
        upper = routing.prompt_mass[:, layer].clone()
        lower += routing.self_mass[:, layer] * previous_lower[:, None]
        upper += routing.self_mass[:, layer] * previous_upper[:, None]
        upper += routing.unresolved_mass[:, layer]

        unsupported_rr_lower = torch.zeros_like(lower)
        unsupported_rr_upper = routing.unresolved_mass[:, layer].clone()
        selected = routing.rr_layer == layer
        if selected.any():
            query = routing.rr_query[selected]
            head = routing.rr_head[selected]
            source = routing.rr_source[selected]
            weight = routing.rr_weight[selected]
            lower.index_put_(
                (query, head), weight * previous_lower[source], accumulate=True
            )
            upper.index_put_(
                (query, head), weight * previous_upper[source], accumulate=True
            )
            unsupported_rr_lower.index_put_(
                (query, head), weight * (1.0 - previous_upper[source]), accumulate=True
            )
            unsupported_rr_upper.index_put_(
                (query, head), weight * (1.0 - previous_lower[source]), accumulate=True
            )

        head_lower[:, layer] = lower.clamp(0.0, 1.0)
        head_upper[:, layer] = upper.clamp(0.0, 1.0)
        unsupported_lower[:, layer] = unsupported_rr_lower.clamp(0.0, 1.0)
        unsupported_upper[:, layer] = unsupported_rr_upper.clamp(0.0, 1.0)

        previous_lower = head_lower[:, layer].mean(dim=1)
        previous_upper = head_upper[:, layer].mean(dim=1)
        aggregate_lower[:, layer + 1] = previous_lower
        aggregate_upper[:, layer + 1] = previous_upper

    return PromptProvenance(
        head_lower=head_lower,
        head_upper=head_upper,
        unsupported_rr_lower=unsupported_lower,
        unsupported_rr_upper=unsupported_upper,
        aggregate_lower=aggregate_lower,
        aggregate_upper=aggregate_upper,
    )
