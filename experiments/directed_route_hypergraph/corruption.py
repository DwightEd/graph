"""Mass-conserving corruptions for denoising route reconstruction."""

from dataclasses import replace

import torch

from experiments.grounded_route.graph import TokenEdges, TokenGraph


def row_ids(graph: TokenGraph) -> torch.Tensor:
    """Return the canonical ``(layer, head, response target)`` edge row."""

    return (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )


def corrupt_graph(
    graph: TokenGraph,
    *,
    incidence_dropout: float,
    head_dropout: float,
    generator: torch.Generator,
) -> TokenGraph:
    """Hide retained incidences and move exactly their mass to unresolved.

    Head dropout masks a complete ``(layer, head)`` channel.  It is combined
    with independent incidence dropout.  Neither operation fabricates a new
    endpoint or changes the diagonal mass.
    """

    if not 0.0 <= incidence_dropout < 1.0:
        raise ValueError("incidence_dropout must be in [0, 1)")
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError("head_dropout must be in [0, 1)")
    if graph.edge_count == 0 or (
        incidence_dropout == 0.0 and head_dropout == 0.0
    ):
        return graph

    edge_keep = torch.ones(graph.edge_count, dtype=torch.bool)
    if incidence_dropout:
        edge_keep &= torch.rand(
            graph.edge_count,
            generator=generator,
        ) >= incidence_dropout
    if head_dropout:
        channel_keep = torch.rand(
            graph.layer_count * graph.head_count,
            generator=generator,
        ) >= head_dropout
        channel = graph.edges.layer * graph.head_count + graph.edges.head
        edge_keep &= channel_keep[channel]

    removed = ~edge_keep
    if not bool(removed.any()):
        return graph

    total_rows = graph.response_count * graph.layer_count * graph.head_count
    removed_mass = torch.zeros(total_rows, dtype=graph.edges.weight.dtype)
    removed_mass.index_add_(
        0,
        row_ids(graph)[removed],
        graph.edges.weight[removed],
    )
    # row_ids are layer/head/response-major; dense storage is response/layer/head.
    removed_mass = removed_mass.view(
        graph.layer_count,
        graph.head_count,
        graph.response_count,
    ).permute(2, 0, 1)

    corrupted = replace(
        graph,
        edges=TokenEdges(
            source=graph.edges.source[edge_keep],
            target=graph.edges.target[edge_keep],
            layer=graph.edges.layer[edge_keep],
            head=graph.edges.head[edge_keep],
            weight=graph.edges.weight[edge_keep],
        ),
        unresolved=graph.unresolved
        + removed_mass.to(device=graph.device, dtype=graph.unresolved.dtype),
    ).check()
    return corrupted.canonicalize()
