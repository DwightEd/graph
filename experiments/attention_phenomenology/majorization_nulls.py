"""Registered null controls for causal prompt-route mechanisms."""

from __future__ import annotations

from dataclasses import replace

import torch

from .routing import RoutingEdges


def uniform_prompt_excess(edges: RoutingEdges) -> RoutingEdges:
    """Keep exact prompt support but give every retained edge equal excess."""

    selected = (edges.source < edges.response_idx) & (
        edges.weight > edges.attention_floor
    )
    weight = edges.weight.clone()
    weight[selected] = edges.attention_floor + 1.0
    return replace(edges, weight=weight)


def shuffle_prompt_source_identity(
    edges: RoutingEdges,
    *,
    seed: int,
) -> RoutingEdges:
    """Break cross-token source identity while preserving each row's weights.

    One bijection of prompt source IDs is drawn per response token and shared
    by its layer/head channels. Thus every row keeps exactly the same sorted
    weight vector, while recurrence of a named source across time is removed.
    """

    generator = torch.Generator(device=edges.source.device).manual_seed(seed)
    source = edges.source.clone()
    permutation = torch.stack(
        [
            torch.randperm(
                edges.response_idx,
                generator=generator,
                device=edges.source.device,
            )
            for _ in range(edges.num_response_tokens)
        ]
    )
    selected = edges.source < edges.response_idx
    source[selected] = permutation[
        edges.query[selected], edges.source[selected]
    ]
    return replace(edges, source=source)


def shuffle_prompt_time(
    edges: RoutingEdges,
    *,
    seed: int,
) -> RoutingEdges:
    """Permute complete prompt-route rows to destroy chronological dynamics."""

    generator = torch.Generator(device=edges.query.device).manual_seed(seed)
    permutation = torch.randperm(
        edges.num_response_tokens,
        generator=generator,
        device=edges.query.device,
    )
    query = edges.query.clone()
    selected = edges.source < edges.response_idx
    query[selected] = permutation[query[selected]]
    return replace(edges, query=query)
