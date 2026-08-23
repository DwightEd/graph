from __future__ import annotations

import torch

from experiments.attention_phenomenology.routing import (
    RoutingEdges,
    build_routing_state,
)


def routing_state(
    *,
    layers,
    heads,
    queries,
    sources,
    weights,
    diagonal,
    response_idx=1,
    response_tokens=2,
    num_layers=2,
    num_heads=1,
):
    edges = RoutingEdges(
        num_layers=num_layers,
        num_heads=num_heads,
        num_response_tokens=response_tokens,
        num_tokens=response_idx + response_tokens,
        response_idx=response_idx,
        attention_floor=0.01,
        layer=torch.tensor(layers, dtype=torch.long),
        head=torch.tensor(heads, dtype=torch.long),
        query=torch.tensor(queries, dtype=torch.long),
        source=torch.tensor(sources, dtype=torch.long),
        weight=torch.tensor(weights, dtype=torch.float32),
        diagonal=torch.as_tensor(diagonal, dtype=torch.float32).clone(),
    )
    return build_routing_state(edges)
