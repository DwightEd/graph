"""Typed, layer-unfolded attention-lineage automaton."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import PROMPT, RESPONSE_FAR, RESPONSE_NEAR, RoutingGraph

P0 = 0
P_PLUS = 1
R0 = 2
R_NEAR = 3
R_FAR = 4
R_MIXED = 5
U = 6

STATE_NAMES = (
    "prompt_direct",
    "prompt_relay",
    "response_base",
    "response_near_closed",
    "response_far_closed",
    "response_mixed_closed",
    "unresolved",
)
CLOSED_STATES = (R_NEAR, R_FAR, R_MIXED)


@dataclass(frozen=True)
class AutomatonTrace:
    route: torch.Tensor  # [token, layer, head, state]

    @property
    def flat(self) -> torch.Tensor:
        tokens, layers, heads, states = self.route.shape
        return self.route.reshape(tokens, layers * heads, states)

    @property
    def prompt_lineage(self) -> torch.Tensor:
        return self.route[..., P0] + self.route[..., P_PLUS]

    @property
    def response_closed(self) -> torch.Tensor:
        return self.route[..., list(CLOSED_STATES)].sum(dim=-1)


def _response_transport(
    state: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    """Transport source lineage through a typed response edge."""

    result = torch.zeros_like(state)
    result[..., P_PLUS] = state[..., P0] + state[..., P_PLUS]
    result[..., U] = state[..., U]

    near = relation == RESPONSE_NEAR
    if bool(near.any()):
        current = state[near]
        result[near, R_NEAR] = current[:, R0] + current[:, R_NEAR]
        result[near, R_MIXED] = current[:, R_FAR] + current[:, R_MIXED]

    far = relation == RESPONSE_FAR
    if bool(far.any()):
        current = state[far]
        result[far, R_FAR] = current[:, R0] + current[:, R_FAR]
        result[far, R_MIXED] = current[:, R_NEAR] + current[:, R_MIXED]
    return result


@torch.no_grad()
def run_typed_automaton(graph: RoutingGraph) -> AutomatonTrace:
    """Propagate prompt and response lineage through exact causal endpoints.

    Cross-layer head transport is the permutation-invariant mean of the
    preceding layer. This is an attention-only proxy; it is not a reconstruction
    of W_V/W_O or residual-stream contribution.
    """

    graph.validate()
    tokens, layers, heads = (
        graph.num_response_tokens,
        graph.num_layers,
        graph.num_heads,
    )
    route = graph.weight.new_zeros((tokens, layers, heads, len(STATE_NAMES)))
    previous = graph.weight.new_zeros((tokens, len(STATE_NAMES)))
    previous[:, R0] = 1.0

    response_edge = graph.relation != PROMPT
    edge_layer = graph.layer[response_edge]
    edge_head = graph.head[response_edge]
    edge_target = graph.query[response_edge]
    edge_source = graph.source[response_edge] - graph.response_idx
    edge_relation = graph.relation[response_edge]
    edge_weight = graph.weight[response_edge]

    for layer in range(layers):
        current = graph.weight.new_zeros((tokens, heads, len(STATE_NAMES)))
        current[..., P0] = graph.prompt_mass[:, layer]
        current += graph.self_mass[:, layer, :, None] * previous[:, None, :]
        current[..., U] += graph.unresolved_mass[:, layer]

        selected = edge_layer == layer
        if bool(selected.any()):
            transported = _response_transport(
                previous[edge_source[selected]],
                edge_relation[selected],
            )
            message = transported * edge_weight[selected, None]
            flat_target = edge_target[selected] * heads + edge_head[selected]
            aggregate = graph.weight.new_zeros((tokens * heads, len(STATE_NAMES)))
            aggregate.index_add_(0, flat_target, message)
            current += aggregate.reshape(tokens, heads, len(STATE_NAMES))

        total = current.sum(dim=-1, keepdim=True)
        if bool((total <= 0).any()):
            raise ValueError("typed automaton encountered an empty routing row")
        current = current / total
        route[:, layer] = current
        previous = current.mean(dim=1)

    return AutomatonTrace(route=route)
