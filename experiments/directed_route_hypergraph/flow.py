"""Deterministic prompt/response/unresolved flow through attention layers."""

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenEdges, TokenGraph
from experiments.grounded_route.lineage import (
    LINEAGE_STATES,
    PROMPT_ORIGIN,
    RESPONSE_CLOSED,
    UNRESOLVED,
)

MASKED = LINEAGE_STATES
STUDENT_LINEAGE_STATES = LINEAGE_STATES + 1


@dataclass(frozen=True)
class FlowStep:
    """One layer's endpoint provenance and post-attention token state."""

    provenance: torch.Tensor
    head_flow: torch.Tensor
    token_state: torch.Tensor


@dataclass(frozen=True)
class FlowOutput:
    """Ordered head-level flow and post-layer response-token trajectory."""

    head_trace: torch.Tensor
    token_trace: torch.Tensor


def initial_flow(graph: TokenGraph) -> torch.Tensor:
    """Assign prompt and response tokens to distinct path roots."""

    state = graph.diagonal.new_zeros((graph.token_count, LINEAGE_STATES))
    state[: graph.response_start, PROMPT_ORIGIN] = 1.0
    state[graph.response_start :, RESPONSE_CLOSED] = 1.0
    return state


def initial_student_flow(graph: TokenGraph) -> torch.Tensor:
    """Add a private MASK state without changing the public P/R/U contract."""

    state = initial_flow(graph)
    masked = state.new_zeros((graph.token_count, 1))
    return torch.cat((state, masked), dim=-1)


def flow_step(
    graph: TokenGraph,
    token_state: torch.Tensor,
    layer: int,
    *,
    residual_weight: float = 1.0,
    edges: TokenEdges | None = None,
    masked_mass: torch.Tensor | None = None,
) -> FlowStep:
    """Apply one typed attention layer without learning a head transition.

    Head-specific rows remain separate in ``head_flow``.  Their uniform mean is
    combined with an explicit residual proxy to form the shared token state
    consumed by the next Transformer layer.  Prompt states remain fixed because
    the cache does not contain prompt-query rows.
    """

    if residual_weight < 0:
        raise ValueError("residual_weight must be non-negative")

    device = token_state.device
    state_count = int(token_state.shape[-1])
    expected_states = (
        STUDENT_LINEAGE_STATES if masked_mass is not None else LINEAGE_STATES
    )
    if state_count != expected_states:
        raise ValueError("token flow has incompatible lineage dimensions")
    if masked_mass is not None and masked_mass.shape != graph.unresolved.shape:
        raise ValueError("masked mass must be [R,L,H]")
    if edges is None:
        edges = graph.layer_edges(layer, device)
    provenance = token_state[edges.source]
    head_flow = token_state.new_zeros(
        (graph.response_count, graph.head_count, state_count)
    )

    if edges.count:
        group = (
            (edges.target - graph.response_start) * graph.head_count
            + edges.head
        )
        head_flow.view(-1, state_count).index_add_(
            0,
            group,
            provenance * edges.weight[:, None],
        )

    response_state = token_state[graph.response_start :]
    diagonal = graph.diagonal[:, layer].to(
        device=device,
        dtype=token_state.dtype,
    )
    unresolved = graph.unresolved[:, layer].to(
        device=device,
        dtype=token_state.dtype,
    )
    head_flow = head_flow + diagonal[..., None] * response_state[:, None]
    head_flow[..., UNRESOLVED] = (
        head_flow[..., UNRESOLVED] + unresolved
    )
    if masked_mass is not None:
        masked = masked_mass[:, layer].to(
            device=device,
            dtype=token_state.dtype,
        )
        head_flow[..., MASKED] = head_flow[..., MASKED] + masked

    attention_state = head_flow.mean(dim=1)
    next_response = (
        residual_weight * response_state + attention_state
    ) / (residual_weight + 1.0)
    next_state = torch.cat(
        (token_state[: graph.response_start], next_response),
        dim=0,
    )
    return FlowStep(
        provenance=provenance,
        head_flow=head_flow,
        token_state=next_state,
    )


def ordered_flow(
    graph: TokenGraph,
    *,
    residual_weight: float = 1.0,
    layer_order: tuple[int, ...] | None = None,
) -> FlowOutput:
    """Compose all layers and retain exact head and token flow trajectories.

    ``layer_order`` exists for matched order controls.  Trace axis one follows
    the processing order, and the default is the Transformer's forward order.
    """

    graph = graph.canonicalize()
    order = (
        tuple(range(graph.layer_count))
        if layer_order is None
        else tuple(map(int, layer_order))
    )
    if sorted(order) != list(range(graph.layer_count)):
        raise ValueError("layer_order must be a permutation of all layers")

    state = initial_flow(graph)
    head_history = []
    token_history = []
    for layer in order:
        step = flow_step(
            graph,
            state,
            layer,
            residual_weight=residual_weight,
        )
        state = step.token_state
        head_history.append(step.head_flow)
        token_history.append(state[graph.response_start :])

    return FlowOutput(
        head_trace=torch.stack(head_history, dim=1),
        token_trace=torch.stack(token_history, dim=1),
    )
