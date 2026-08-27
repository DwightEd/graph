"""Retained-edge masking with native and artificial missing mass separated."""

from dataclasses import dataclass, replace

import torch

from experiments.grounded_route.graph import (
    TokenEdges,
    TokenGraph,
    endpoint_storage_keys,
)


@dataclass(frozen=True)
class CorruptionResult:
    graph: TokenGraph
    masked_mass: torch.Tensor
    masked_edge: torch.Tensor


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
    forced_edge: torch.Tensor | None = None,
) -> CorruptionResult:
    """Hide retained incidences while keeping native censoring unchanged.

    Head dropout masks a complete ``(layer, head)`` channel.  It is combined
    with independent incidence dropout and endpoint-recovery holdouts.  Removed
    mass is returned as a separate student-only channel rather than being
    merged with sparse-cache ``unresolved`` mass.
    """

    if not 0.0 <= incidence_dropout < 1.0:
        raise ValueError("incidence_dropout must be in [0, 1)")
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError("head_dropout must be in [0, 1)")
    if forced_edge is None:
        forced_edge = torch.empty(0, dtype=torch.long)
    else:
        forced_edge = torch.unique(forced_edge.detach().to("cpu", torch.long))
    if len(forced_edge) and bool(
        ((forced_edge < 0) | (forced_edge >= graph.edge_count)).any()
    ):
        raise ValueError("forced edge index is outside the graph")
    if graph.edge_count == 0:
        return CorruptionResult(
            graph=graph,
            masked_mass=torch.zeros_like(graph.unresolved),
            masked_edge=forced_edge,
        )

    edge_keep = torch.ones(graph.edge_count, dtype=torch.bool)
    if len(forced_edge):
        endpoint_key = endpoint_storage_keys(graph)
        forced_key = torch.unique(endpoint_key[forced_edge], sorted=True)
        location = torch.searchsorted(forced_key, endpoint_key)
        lookup = location.clamp_max(len(forced_key) - 1)
        edge_keep &= ~(
            (location < len(forced_key))
            & (forced_key[lookup] == endpoint_key)
        )
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
        return CorruptionResult(
            graph=graph,
            masked_mass=torch.zeros_like(graph.unresolved),
            masked_edge=torch.empty(0, dtype=torch.long),
        )

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

    masked_mass = removed_mass.to(
        device=graph.device,
        dtype=graph.unresolved.dtype,
    )
    corrupted = replace(
        graph,
        edges=TokenEdges(
            source=graph.edges.source[edge_keep],
            target=graph.edges.target[edge_keep],
            layer=graph.edges.layer[edge_keep],
            head=graph.edges.head[edge_keep],
            weight=graph.edges.weight[edge_keep],
        ),
    ).check()
    return CorruptionResult(
        graph=corrupted.canonicalize(),
        masked_mass=masked_mass,
        masked_edge=torch.nonzero(removed, as_tuple=False).flatten(),
    )
