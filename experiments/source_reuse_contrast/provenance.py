"""Prompt-origin lower bounds and graph-derived self-supervised targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .baselines import received_support_topk
from .data import SourceReuseGraph


@dataclass(frozen=True)
class GroundingTargets:
    """Frozen targets derived from the unmodified attention graph."""

    provenance: torch.Tensor
    edge_origin: torch.Tensor
    grounding_field: torch.Tensor
    received_support: torch.Tensor

    def token(self, index: int) -> "TokenGroundingTargets":
        return TokenGroundingTargets(
            provenance=self.provenance[index, 1:],
            grounding_field=self.grounding_field[index],
            received_support=self.received_support[index],
        )


@dataclass(frozen=True)
class TokenGroundingTargets:
    provenance: torch.Tensor
    grounding_field: torch.Tensor
    received_support: torch.Tensor


def compute_grounding_targets(
    graph: SourceReuseGraph,
    *,
    received_topk: int = 5,
) -> GroundingTargets:
    """Propagate a conservative prompt-origin lower bound through layers.

    Retained prompt edges are fully prompt-origin. Retained response edges inherit
    the source token's prompt provenance from the previous transformer depth.
    Unresolved attention mass is excluded, so the result is a lower bound.
    """

    tokens = graph.num_response_tokens
    layers = graph.num_layers
    heads = graph.num_heads
    provenance = graph.weight.new_zeros((tokens, layers + 1))
    grounding = graph.weight.new_zeros((tokens, layers, heads, 3))
    edge_origin = graph.weight.new_zeros(graph.num_edges)

    for layer in range(layers):
        previous = provenance[:, layer]
        head_lower = graph.diagonal[:, layer] * previous[:, None]
        selected = graph.layer == layer
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if indices.numel():
            query = graph.query[indices]
            source = graph.source[indices]
            head = graph.head[indices]
            weight = graph.weight[indices]
            prompt = source < graph.response_idx

            current_origin = weight.new_ones(weight.shape)
            if bool((~prompt).any()):
                response_source = source[~prompt] - graph.response_idx
                current_origin[~prompt] = previous[response_source]
            edge_origin[indices] = current_origin

            contribution = weight * current_origin
            head_lower.index_put_((query, head), contribution, accumulate=True)

            if bool(prompt.any()):
                grounding[:, layer, :, 0].index_put_(
                    (query[prompt], head[prompt]),
                    weight[prompt],
                    accumulate=True,
                )
            response = ~prompt
            if bool(response.any()):
                grounded = weight[response] * current_origin[response]
                unsupported = weight[response] * (1.0 - current_origin[response])
                grounding[:, layer, :, 1].index_put_(
                    (query[response], head[response]),
                    grounded,
                    accumulate=True,
                )
                grounding[:, layer, :, 2].index_put_(
                    (query[response], head[response]),
                    unsupported,
                    accumulate=True,
                )

        provenance[:, layer + 1] = head_lower.mean(dim=1).clamp(0.0, 1.0)

    return GroundingTargets(
        provenance=provenance.detach(),
        edge_origin=edge_origin.detach(),
        grounding_field=grounding.detach(),
        received_support=received_support_topk(
            graph,
            topk=received_topk,
        ).detach(),
    )
