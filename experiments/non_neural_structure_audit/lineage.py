"""Finite attention-routing lineage on the layer-unrolled causal graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.attention_phenomenology.routing import RoutingState


LINEAGE_NAMES = (
    "prompt_direct",
    "prompt_relay",
    "local_self",
    "detached_one_hop",
    "detached_multihop",
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
    routed[:, LINEAGE_INDEX["detached_one_hop"]] = source_state[
        :, LINEAGE_INDEX["local_self"]
    ]
    routed[:, LINEAGE_INDEX["detached_multihop"]] = (
        source_state[:, LINEAGE_INDEX["detached_one_hop"]]
        + source_state[:, LINEAGE_INDEX["detached_multihop"]]
    )
    routed[:, LINEAGE_INDEX["unresolved"]] = source_state[
        :, LINEAGE_INDEX["unresolved"]
    ]
    return routed


def propagate_lineage(
    routing: RoutingState,
    *,
    layer_order: tuple[int, ...] | None = None,
) -> LineageTrace:
    """Propagate conserved prompt/local ancestry through a finite layer DAG.

    This is an attention-routing proxy. ``self_mass`` is the attention
    diagonal; it is not a reconstructed Transformer residual connection.
    """

    edges = routing.edges
    order = tuple(range(edges.num_layers)) if layer_order is None else layer_order
    if sorted(order) != list(range(edges.num_layers)):
        raise ValueError("layer_order must be a permutation of all layers")

    previous = routing.prompt_mass.new_zeros(
        (edges.num_response_tokens, len(LINEAGE_NAMES))
    )
    previous[:, LINEAGE_INDEX["local_self"]] = 1.0
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

        selected = (edges.layer == layer) & (edges.source >= edges.response_idx)
        if bool(selected.any()):
            query = edges.query[selected]
            head = edges.head[selected]
            source = edges.source[selected] - edges.response_idx
            message = _through_response(previous[source])
            message *= routing.edge_weight[selected, None]
            head_state.index_put_((query, head), message, accumulate=True)

        previous = head_state.mean(dim=1)
        states.append(previous)

    return LineageTrace(state=torch.stack(states, dim=1), layer_order=order)
