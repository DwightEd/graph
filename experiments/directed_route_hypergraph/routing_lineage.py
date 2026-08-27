"""Ordered D/I/E/U lineage for the observed sparse attention rows.

This module is a deterministic attention-routing diagnostic.  It does not use
hallucination labels, learn parameters, or claim to recover value-aware
functional contribution.  In particular, it does not insert an unobserved
Transformer residual transition.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenGraph

DIRECT = 0
INDIRECT = 1
ENDOGENOUS = 2
UNRESOLVED = 3
ROUTING_COMPONENTS = 4


@dataclass(frozen=True)
class RoutingLineage:
    """Layer trace and the predecessor-query-aligned response-token view.

    ``query_trace`` has shape ``[R, K, 4]`` for the ``K`` selected layers and
    follows ``layer_order`` on its layer axis.  Components are direct
    prompt-rooted (D), prompt-rooted after an earlier response carrier (I),
    response-embedding-rooted (E), and unresolved/censored (U).

    A cached response query at index ``i`` predicts response token ``i + 1``.
    Therefore ``token_lineage`` has ``R - 1`` rows: the first generated token
    is omitted because its last-prompt predictor is absent, and the final
    cached query is omitted because it predicts beyond the saved response.
    """

    layer_order: tuple[int, ...]
    query_trace: torch.Tensor
    predictor_response_index: torch.Tensor
    token_response_index: torch.Tensor
    token_id: torch.Tensor
    token_lineage: torch.Tensor

    @property
    def query_lineage(self) -> torch.Tensor:
        """Final-layer lineage for every cached response query."""

        return self.query_trace[:, -1]


def validated_layer_order(
    graph: TokenGraph,
    layer_order: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return a non-empty ordered selection of cached Transformer layers.

    Full permutations implement ordered, reverse, and random-order controls.
    A one-layer selection is needed for the last-layer control, which applies
    that cached layer directly to the initial response roots.
    """

    order = (
        tuple(range(graph.layer_count))
        if layer_order is None
        else tuple(map(int, layer_order))
    )
    if not order:
        raise ValueError("routing lineage requires at least one layer")
    if len(set(order)) != len(order) or any(
        layer < 0 or layer >= graph.layer_count for layer in order
    ):
        raise ValueError(
            "layer_order must be a partial permutation of cached layers"
        )
    return order


def initial_routing_lineage(graph: TokenGraph) -> torch.Tensor:
    """Root each response position at its own response-token embedding."""

    lineage = graph.diagonal.new_zeros(
        (graph.response_count, ROUTING_COMPONENTS)
    )
    lineage[:, ENDOGENOUS] = 1.0
    return lineage


def response_source_lineage(
    graph: TokenGraph,
    previous: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Classify retained sources before one attention-row transition.

    A prompt endpoint enters the current response query directly.  Any prompt
    ancestry read through an earlier response position has crossed a response
    carrier, so both its D and I mass become I at the new target.
    """

    lineage = previous.new_zeros((len(source), ROUTING_COMPONENTS))
    prompt = source < graph.response_start
    lineage[prompt, DIRECT] = 1.0

    response = ~prompt
    if bool(response.any()):
        inherited = previous[source[response] - graph.response_start]
        lineage[response, INDIRECT] = (
            inherited[:, DIRECT] + inherited[:, INDIRECT]
        )
        lineage[response, ENDOGENOUS] = inherited[:, ENDOGENOUS]
        lineage[response, UNRESOLVED] = inherited[:, UNRESOLVED]
    return lineage


def routing_lineage_step(
    graph: TokenGraph,
    previous: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Apply one observed attention layer and uniformly merge its heads.

    Retained off-diagonal edges, the exact cached attention diagonal, and the
    native unresolved bucket are the complete row.  Heads remain distinct
    while rows are accumulated and are averaged only to form the shared state
    consumed by the next layer.
    """

    device = previous.device
    edges = graph.layer_edges(layer, device)
    head_lineage = previous.new_zeros(
        (graph.response_count, graph.head_count, ROUTING_COMPONENTS)
    )

    if edges.count:
        source_lineage = response_source_lineage(
            graph,
            previous,
            edges.source,
        )
        row = (
            (edges.target - graph.response_start) * graph.head_count
            + edges.head
        )
        head_lineage.view(-1, ROUTING_COMPONENTS).index_add_(
            0,
            row,
            source_lineage * edges.weight[:, None],
        )

    diagonal = graph.diagonal[:, layer].to(
        device=device,
        dtype=previous.dtype,
    )
    unresolved = graph.unresolved[:, layer].to(
        device=device,
        dtype=previous.dtype,
    )
    head_lineage = head_lineage + diagonal[..., None] * previous[:, None]
    head_lineage[..., UNRESOLVED] += unresolved
    return head_lineage.mean(dim=1)


@torch.no_grad()
def ordered_routing_lineage(
    graph: TokenGraph,
    *,
    layer_order: Sequence[int] | None = None,
) -> RoutingLineage:
    """Compose D/I/E/U attention-routing lineage in a supplied layer order."""

    graph = graph.canonicalize()
    order = validated_layer_order(graph, layer_order)
    previous = initial_routing_lineage(graph)
    history = []
    for layer in order:
        previous = routing_lineage_step(graph, previous, layer)
        history.append(previous)

    query_trace = torch.stack(history, dim=1)
    aligned_count = max(graph.response_count - 1, 0)
    predictor = torch.arange(aligned_count, device=query_trace.device)
    token_index = predictor + 1
    token_id = graph.response_token_ids[1:].to(query_trace.device)
    return RoutingLineage(
        layer_order=order,
        query_trace=query_trace,
        predictor_response_index=predictor,
        token_response_index=token_index,
        token_id=token_id,
        token_lineage=query_trace[:-1, -1],
    )
