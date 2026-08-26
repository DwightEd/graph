"""Lift one saved token graph to the order-2 graph expected by DBGNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.grounded_route.artifacts import EncodedTokenGraph


@dataclass(frozen=True)
class DBGNNGraph:
    """Tensor-only input contract used by the original ``HO_GCN`` model."""

    x_ho: torch.Tensor
    x_fo: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_index_fo: torch.Tensor
    edge_weight_fo: torch.Tensor
    edge_index_hon_to_fon: torch.Tensor
    ho_endpoints: torch.Tensor
    response_start: int

    @property
    def num_nodes(self) -> int:
        return int(self.x_fo.shape[0])

    @property
    def num_ho_nodes(self) -> int:
        return int(self.x_ho.shape[0])

    @property
    def edge_weight_ho(self) -> torch.Tensor:
        return self.edge_weight

    def to(self, device: str | torch.device) -> "DBGNNGraph":
        return DBGNNGraph(
            x_ho=self.x_ho.to(device),
            x_fo=self.x_fo.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            edge_index_fo=self.edge_index_fo.to(device),
            edge_weight_fo=self.edge_weight_fo.to(device),
            edge_index_hon_to_fon=self.edge_index_hon_to_fon.to(device),
            ho_endpoints=self.ho_endpoints.to(device),
            response_start=self.response_start,
        )


def build_dbgnn_graph(
    graph: EncodedTokenGraph,
    *,
    delta_layers: int = 1,
    higher_order_mode: str = "causal",
) -> DBGNNGraph:
    """Aggregate typed attention edges and form an order-2 causal graph.

    Transformer layer is the event timestamp.  Head-specific edges first form
    one event per ``(layer, source, target)``; causal compositions then join an
    event at layer ``l`` to a relay event in the next ``delta_layers`` layers.
    A first-order edge stores mean retained attention mass per layer-head
    channel.  Censored mass remains in the source artifact's unresolved tensor.
    """

    if delta_layers < 1:
        raise ValueError("delta_layers must be positive")
    if higher_order_mode not in {"causal", "no_transition"}:
        raise ValueError("higher_order_mode must be 'causal' or 'no_transition'")

    event_index, event_weight = _layer_events(graph)
    edge_index_fo, edge_weight_fo = _first_order_edges(
        graph,
        event_index,
        event_weight,
    )
    x_fo = _first_order_features(graph)
    x_ho = _higher_order_features(x_fo, edge_index_fo, edge_weight_fo)
    if higher_order_mode == "no_transition":
        edge_index = edge_index_fo.new_empty((2, 0))
        edge_weight = edge_weight_fo.new_empty(0)
    else:
        edge_index, edge_weight = _causal_transitions(
            edge_index_fo,
            event_index,
            event_weight,
            layer_count=graph.layer_count,
            delta_layers=delta_layers,
        )
    ho_nodes = edge_index_fo.shape[1]
    edge_index_hon_to_fon = torch.stack(
        (
            torch.arange(ho_nodes, device=edge_index_fo.device),
            edge_index_fo[1],
        )
    )
    return DBGNNGraph(
        x_ho=x_ho,
        x_fo=x_fo,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_index_fo=edge_index_fo,
        edge_weight_fo=edge_weight_fo,
        edge_index_hon_to_fon=edge_index_hon_to_fon,
        ho_endpoints=edge_index_fo,
        response_start=graph.response_start,
    )


def _layer_events(graph: EncodedTokenGraph) -> tuple[torch.Tensor, torch.Tensor]:
    if not graph.edge_weight.numel():
        return graph.edge_index.new_empty((3, 0)), graph.edge_weight.clone()

    node_count = int(graph.token_ids.numel())
    key = (
        graph.edge_layer * node_count * node_count
        + graph.edge_index[0] * node_count
        + graph.edge_index[1]
    )
    event_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
    weight = torch.zeros(
        len(event_key),
        dtype=graph.edge_weight.dtype,
        device=graph.edge_weight.device,
    )
    weight.index_add_(0, inverse, graph.edge_weight)
    weight /= graph.head_count
    layer = event_key // (node_count * node_count)
    endpoint = event_key.remainder(node_count * node_count)
    source = endpoint // node_count
    target = endpoint.remainder(node_count)
    return torch.stack((layer, source, target)), weight


def _first_order_edges(
    graph: EncodedTokenGraph,
    event_index: torch.Tensor,
    event_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not event_weight.numel():
        return graph.edge_index.clone(), event_weight

    node_count = int(graph.token_ids.numel())
    pair_key = event_index[1] * node_count + event_index[2]
    endpoint_key, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)
    weight = torch.zeros(
        len(endpoint_key),
        dtype=event_weight.dtype,
        device=event_weight.device,
    )
    weight.index_add_(0, inverse, event_weight)
    weight /= graph.layer_count
    source = endpoint_key // node_count
    target = endpoint_key.remainder(node_count)
    return torch.stack((source, target)), weight


def _first_order_features(graph: EncodedTokenGraph) -> torch.Tensor:
    node_count = int(graph.token_ids.numel())
    position = torch.arange(
        node_count,
        dtype=graph.edge_weight.dtype,
        device=graph.edge_weight.device,
    )
    is_response = position >= graph.response_start
    absolute_position = position / (position + 1.0)
    response_ordinal = (position - graph.response_start + 1).clamp_min(0.0)
    response_position = torch.where(
        is_response,
        response_ordinal / (response_ordinal + 1.0),
        0.0,
    )
    return torch.stack(
        (
            (~is_response).to(position.dtype),
            is_response.to(position.dtype),
            absolute_position,
            response_position,
        ),
        dim=-1,
    )


def _higher_order_features(
    x_fo: torch.Tensor,
    edge_index_fo: torch.Tensor,
    edge_weight_fo: torch.Tensor,
) -> torch.Tensor:
    source, target = edge_index_fo
    lag = (target - source).to(x_fo.dtype)
    lag = lag / (lag + 1.0)
    return torch.cat(
        (
            x_fo[source],
            x_fo[target],
            edge_weight_fo[:, None],
            lag[:, None],
        ),
        dim=-1,
    )


def _causal_transitions(
    edge_index_fo: torch.Tensor,
    event_index: torch.Tensor,
    event_weight: torch.Tensor,
    *,
    layer_count: int,
    delta_layers: int,
    block_size: int = 65_536,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not event_weight.numel():
        empty_index = edge_index_fo.new_empty((2, 0))
        return empty_index, event_weight.new_empty(0)

    node_count = int(edge_index_fo.max().item()) + 1
    fo_key = edge_index_fo[0] * node_count + edge_index_fo[1]
    event_key = event_index[1] * node_count + event_index[2]
    event_to_ho = torch.searchsorted(fo_key, event_key)
    ho_count = edge_index_fo.shape[1]
    event_mass = event_weight.new_zeros((ho_count, layer_count))
    event_mass.index_put_(
        (event_to_ho, event_index[0]),
        event_weight,
        accumulate=True,
    )

    endpoint_rows = edge_index_fo.T.detach().cpu().tolist()
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    for index, (source, target) in enumerate(endpoint_rows):
        incoming.setdefault(target, []).append(index)
        outgoing.setdefault(source, []).append(index)

    left_blocks: list[torch.Tensor] = []
    right_blocks: list[torch.Tensor] = []
    weight_blocks: list[torch.Tensor] = []
    for middle in incoming.keys() & outgoing.keys():
        left = torch.tensor(incoming[middle], device=event_weight.device)
        right = torch.tensor(outgoing[middle], device=event_weight.device)
        count = len(left) * len(right)
        for start in range(0, count, block_size):
            linear = torch.arange(
                start,
                min(start + block_size, count),
                device=event_weight.device,
            )
            left_index = left[torch.div(linear, len(right), rounding_mode="floor")]
            right_index = right[linear.remainder(len(right))]
            weight = event_weight.new_zeros(len(linear))
            for wait in range(1, min(delta_layers, layer_count - 1) + 1):
                weight += (
                    event_mass[left_index, :-wait]
                    * event_mass[right_index, wait:]
                ).sum(dim=1)
            keep = weight > 0
            if bool(keep.any()):
                left_blocks.append(left_index[keep])
                right_blocks.append(right_index[keep])
                weight_blocks.append(weight[keep])

    if not weight_blocks:
        empty_index = edge_index_fo.new_empty((2, 0))
        return empty_index, event_weight.new_empty(0)
    left = torch.cat(left_blocks)
    right = torch.cat(right_blocks)
    weight = torch.cat(weight_blocks)
    order = torch.argsort(left * ho_count + right, stable=True)
    return torch.stack((left[order], right[order])).long(), weight[order]
