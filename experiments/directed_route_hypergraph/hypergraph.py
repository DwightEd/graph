"""Expose one Transformer layer as a directed node-row-node hypergraph."""

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenEdges, TokenGraph


@dataclass(frozen=True)
class DirectedLayerHypergraph:
    """Incidence tensors for ``source node -> attention row -> target node``."""

    layer: int
    source: torch.Tensor
    hyperedge: torch.Tensor
    weight: torch.Tensor
    role: torch.Tensor
    target: torch.Tensor
    head: torch.Tensor
    diagonal: torch.Tensor
    unresolved: torch.Tensor
    masked: torch.Tensor

    @property
    def incidence_count(self) -> int:
        return int(self.source.numel())

    @property
    def hyperedge_count(self) -> int:
        return int(self.target.numel())

    @property
    def source_to_hyperedge(self) -> torch.Tensor:
        return torch.stack((self.source, self.hyperedge))

    @property
    def hyperedge_to_target(self) -> torch.Tensor:
        hyperedge = torch.arange(self.hyperedge_count, device=self.target.device)
        return torch.stack((hyperedge, self.target))


def layer_hypergraph(
    graph: TokenGraph,
    layer: int,
    device: str | torch.device,
    edges: TokenEdges | None = None,
    masked_mass: torch.Tensor | None = None,
) -> DirectedLayerHypergraph:
    """Build the explicit row-hyperedge view without copying the whole graph."""

    if edges is None:
        edges = graph.layer_edges(layer, device)
    response = torch.arange(graph.response_count, device=device)
    head = torch.arange(graph.head_count, device=device)
    target = (graph.response_start + response[:, None]).expand(-1, graph.head_count)
    hyperedge_head = head[None].expand(graph.response_count, -1)
    hyperedge = (
        (edges.target - graph.response_start) * graph.head_count + edges.head
    )
    if masked_mass is None:
        masked = torch.zeros_like(graph.unresolved[:, layer]).to(device)
    else:
        if masked_mass.shape != graph.unresolved.shape:
            raise ValueError("masked mass must be [R,L,H]")
        masked = masked_mass[:, layer].to(device)
    return DirectedLayerHypergraph(
        layer=int(layer),
        source=edges.source,
        hyperedge=hyperedge,
        weight=edges.weight,
        role=(edges.source >= graph.response_start).long(),
        target=target.reshape(-1),
        head=hyperedge_head.reshape(-1),
        diagonal=graph.diagonal[:, layer].to(device).reshape(-1),
        unresolved=graph.unresolved[:, layer].to(device).reshape(-1),
        masked=masked.reshape(-1),
    )
