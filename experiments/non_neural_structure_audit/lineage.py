"""Finite attention-routing lineage on the layer-unrolled causal graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.attention_phenomenology.routing import RoutingState


LINEAGE_NAMES = (
    "prompt_direct",
    "prompt_relay",
    "response_base",
    "response_relay_one_hop",
    "response_relay_multihop",
    "unresolved",
)
LINEAGE_INDEX = {name: index for index, name in enumerate(LINEAGE_NAMES)}


@dataclass(frozen=True)
class LineageTrace:
    state: torch.Tensor
    layer_order: tuple[int, ...]


def _through_response(source_state: torch.Tensor) -> torch.Tensor:
    routed = torch.zeros_like(source_state)
    routed[:, LINEAGE_INDEX["prompt_relay"]] = (
        source_state[:, LINEAGE_INDEX["prompt_direct"]]
        + source_state[:, LINEAGE_INDEX["prompt_relay"]]
    )
    routed[:, LINEAGE_INDEX["response_relay_one_hop"]] = source_state[
        :, LINEAGE_INDEX["response_base"]
    ]
    routed[:, LINEAGE_INDEX["response_relay_multihop"]] = (
        source_state[:, LINEAGE_INDEX["response_relay_one_hop"]]
        + source_state[:, LINEAGE_INDEX["response_relay_multihop"]]
    )
    routed[:, LINEAGE_INDEX["unresolved"]] = source_state[
        :, LINEAGE_INDEX["unresolved"]
    ]
    return routed


class LineageOperator:
    """Reuse one grouped edge plan across real, rewired, and shuffled traces."""

    def __init__(self, routing: RoutingState):
        self.routing = routing
        edges = routing.edges
        sorted_index = torch.argsort(edges.layer, stable=True)
        count = torch.bincount(edges.layer, minlength=edges.num_layers)
        pointer = torch.cat((count.new_zeros(1), count.cumsum(dim=0)))
        self.layer_edges = tuple(
            sorted_index[int(pointer[layer]) : int(pointer[layer + 1])]
            for layer in range(edges.num_layers)
        )

    def run(
        self,
        *,
        source: torch.Tensor | None = None,
        layer_order: tuple[int, ...] | None = None,
    ) -> LineageTrace:
        """Propagate a conserved attention-routing proxy through finite layers."""

        routing = self.routing
        edges = routing.edges
        source = edges.source if source is None else source
        if source.shape != edges.source.shape:
            raise ValueError("source override must match retained edge rows")
        order = tuple(range(edges.num_layers)) if layer_order is None else layer_order
        if sorted(order) != list(range(edges.num_layers)):
            raise ValueError("layer_order must be a permutation of all layers")

        previous = routing.prompt_mass.new_zeros(
            (edges.num_response_tokens, len(LINEAGE_NAMES))
        )
        previous[:, LINEAGE_INDEX["response_base"]] = 1.0
        states = []

        for layer in order:
            head_state = previous.new_zeros(
                (edges.num_response_tokens, edges.num_heads, len(LINEAGE_NAMES))
            )
            head_state[:, :, LINEAGE_INDEX["prompt_direct"]] = routing.prompt_mass[
                :, layer
            ]
            head_state += routing.self_mass[:, layer, :, None] * previous[:, None, :]
            head_state[:, :, LINEAGE_INDEX["unresolved"]] += routing.unresolved_mass[
                :, layer
            ]

            indices = self.layer_edges[layer]
            response = source[indices] >= edges.response_idx
            selected = indices[response]
            if selected.numel():
                query = edges.query[selected]
                head = edges.head[selected]
                response_source = source[selected] - edges.response_idx
                message = _through_response(previous[response_source])
                message *= routing.edge_weight[selected, None]
                head_state.index_put_((query, head), message, accumulate=True)

            previous = head_state.mean(dim=1)
            states.append(previous)

        return LineageTrace(state=torch.stack(states, dim=1), layer_order=order)


def propagate_lineage(
    routing: RoutingState,
    *,
    layer_order: tuple[int, ...] | None = None,
) -> LineageTrace:
    """Convenience interface for a single lineage trace."""

    return LineageOperator(routing).run(layer_order=layer_order)
