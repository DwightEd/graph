"""Layer-composed prompt provenance with explicit censoring bounds.

This is an attention-only routing proxy. Prompt states are fixed anchors. Each
layer reads provenance from the preceding layer, and heads are aggregated by an
unweighted mean because value/output projections are unavailable in the cache.
The lower bound assigns unresolved mass provenance zero; the upper bound assigns
it provenance one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .routing import RoutingTensor


@dataclass(frozen=True)
class PromptProvenance:
    head_lower: torch.Tensor  # [response, layer, head]
    head_upper: torch.Tensor
    unsupported_lower: torch.Tensor
    unsupported_upper: torch.Tensor
    aggregate_lower: torch.Tensor  # [response, layer + 1]
    aggregate_upper: torch.Tensor


def layered_prompt_provenance(routing: RoutingTensor) -> PromptProvenance:
    """Compose prompt ancestry through ordered Transformer attention layers."""

    r = routing.edges.num_response_tokens
    l = routing.edges.num_layers
    h = routing.edges.num_heads
    device = routing.edges.device

    head_lower = torch.zeros((r, l, h), dtype=torch.float32, device=device)
    head_upper = torch.zeros_like(head_lower)
    unsupported_lower = torch.zeros_like(head_lower)
    unsupported_upper = torch.zeros_like(head_lower)
    aggregate_lower = torch.zeros((r, l + 1), dtype=torch.float32, device=device)
    aggregate_upper = torch.zeros_like(aggregate_lower)

    previous_lower = aggregate_lower[:, 0]
    previous_upper = aggregate_upper[:, 0]

    for layer in range(l):
        lower = routing.prompt_mass[:, layer].clone()
        upper = routing.prompt_mass[:, layer].clone()
        lower += routing.self_mass[:, layer] * previous_lower[:, None]
        upper += routing.self_mass[:, layer] * previous_upper[:, None]
        upper += routing.unresolved_mass[:, layer]

        unsupported_known_lower = routing.self_mass[:, layer] * (
            1.0 - previous_upper[:, None]
        )
        unsupported_known_upper = routing.self_mass[:, layer] * (
            1.0 - previous_lower[:, None]
        )
        unsupported_known_upper += routing.unresolved_mass[:, layer]

        selected = routing.rr_layer == layer
        if bool(selected.any()):
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
            unsupported_known_lower.index_put_(
                (query, head), weight * (1.0 - previous_upper[source]), accumulate=True
            )
            unsupported_known_upper.index_put_(
                (query, head), weight * (1.0 - previous_lower[source]), accumulate=True
            )

        head_lower[:, layer] = lower.clamp(0.0, 1.0)
        head_upper[:, layer] = upper.clamp(0.0, 1.0)
        unsupported_lower[:, layer] = unsupported_known_lower.clamp(0.0, 1.0)
        unsupported_upper[:, layer] = unsupported_known_upper.clamp(0.0, 1.0)

        previous_lower = head_lower[:, layer].mean(dim=1)
        previous_upper = head_upper[:, layer].mean(dim=1)
        aggregate_lower[:, layer + 1] = previous_lower
        aggregate_upper[:, layer + 1] = previous_upper

    return PromptProvenance(
        head_lower=head_lower,
        head_upper=head_upper,
        unsupported_lower=unsupported_lower,
        unsupported_upper=unsupported_upper,
        aggregate_lower=aggregate_lower,
        aggregate_upper=aggregate_upper,
    )
