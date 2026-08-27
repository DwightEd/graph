"""Layer-ordered endpoint layouts for the sparse attention-flow proxy."""

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenGraph


@dataclass(frozen=True)
class EndpointLayout:
    """Final input-endpoint distribution for every response token.

    Columns ``[:graph.token_count]`` are exact token endpoints.  The final
    column is an absorbing bucket for attention mass censored by the sparse
    cache.  This is an attention-only routing layout; it is not an OV-aware
    functional contribution matrix.
    """

    distribution: torch.Tensor


@torch.no_grad()
def ordered_endpoint_layout(
    graph: TokenGraph,
    *,
    residual_weight: float = 1.0,
    layer_order: tuple[int, ...] | None = None,
) -> EndpointLayout:
    """Compose response-row transitions in actual Transformer layer order.

    Prompt rows are unavailable in the cache and therefore remain identity
    endpoints.  Heads stay separate in the sparse edge table and are merged by
    a fixed uniform mean only when forming one layer transition.  The explicit
    residual term is a registered proxy rather than a measured residual-stream
    attribution.
    """

    if residual_weight < 0:
        raise ValueError("residual_weight must be non-negative")
    graph = graph.canonicalize()
    order = (
        tuple(range(graph.layer_count))
        if layer_order is None
        else tuple(map(int, layer_order))
    )
    if sorted(order) != list(range(graph.layer_count)):
        raise ValueError("layer_order must be a permutation of all layers")

    device = graph.device
    dtype = graph.diagonal.dtype
    unresolved_endpoint = graph.token_count
    endpoint_count = graph.token_count + 1
    response = torch.zeros(
        (graph.response_count, endpoint_count),
        device=device,
        dtype=dtype,
    )
    response_index = torch.arange(graph.response_count, device=device)
    response[response_index, graph.response_start + response_index] = 1.0

    for layer in order:
        edges = graph.layer_edges(layer, device)
        attention = torch.zeros_like(response)
        if edges.count:
            target = edges.target - graph.response_start
            prompt = edges.source < graph.response_start
            if bool(prompt.any()):
                attention.index_put_(
                    (target[prompt], edges.source[prompt]),
                    edges.weight[prompt] / graph.head_count,
                    accumulate=True,
                )
            history = ~prompt
            if bool(history.any()):
                transition = torch.sparse_coo_tensor(
                    torch.stack(
                        (
                            target[history],
                            edges.source[history] - graph.response_start,
                        )
                    ),
                    edges.weight[history] / graph.head_count,
                    (graph.response_count, graph.response_count),
                    device=device,
                    dtype=dtype,
                    check_invariants=False,
                ).coalesce()
                attention = attention + torch.sparse.mm(
                    transition,
                    response,
                )

        diagonal = graph.diagonal[:, layer].to(device=device, dtype=dtype).mean(1)
        unresolved = graph.unresolved[:, layer].to(
            device=device,
            dtype=dtype,
        ).mean(1)
        attention = attention + diagonal[:, None] * response
        attention[:, unresolved_endpoint] += unresolved
        response = (
            residual_weight * response + attention
        ) / (residual_weight + 1.0)

    return EndpointLayout(distribution=response)
