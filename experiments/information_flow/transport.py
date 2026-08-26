"""Compose sparse layer-wise attention into token information-flow sketches."""

from dataclasses import dataclass

import torch

from .basis import source_basis
from .config import FlowConfig


@dataclass(frozen=True)
class FlowViews:
    full_trace: torch.Tensor
    full_final: torch.Tensor
    reverse_trace: torch.Tensor
    reverse_final: torch.Tensor
    last_layer: torch.Tensor
    layer_mean: torch.Tensor
    identity_trace: torch.Tensor
    identity_final: torch.Tensor
    trajectory: torch.Tensor

    def embeddings(self) -> dict[str, torch.Tensor]:
        return {
            "full_trace": self.full_trace,
            "full_final": self.full_final,
            "reverse_trace": self.reverse_trace,
            "reverse_final": self.reverse_final,
            "last_layer": self.last_layer,
            "layer_mean": self.layer_mean,
            "identity_trace": self.identity_trace,
            "identity_final": self.identity_final,
        }


def attention_output(
    graph,
    state: torch.Tensor,
    layer: int,
    unresolved: str,
) -> torch.Tensor:
    """Apply one attention-only layer to response nodes while prompt nodes stay fixed."""

    edges = graph.layer_edges(layer, state.device)
    response = state[graph.response_start :]
    response_count, hidden = response.shape
    head_count = graph.head_count
    cells = state.new_zeros((response_count * head_count, hidden))

    if edges.count:
        target = edges.target - graph.response_start
        group = target * head_count + edges.head
        cells.index_add_(
            0,
            group,
            state[edges.source] * edges.weight[:, None],
        )

    cells = cells.view(response_count, head_count, hidden)
    diagonal = graph.diagonal[:, layer].to(state.device)
    cells = cells + diagonal[..., None] * response[:, None]

    if unresolved == "self":
        missing = graph.unresolved[:, layer].to(state.device)
        cells = cells + missing[..., None] * response[:, None]
    else:
        retained = diagonal.new_zeros((response_count, head_count))
        if edges.count:
            target = edges.target - graph.response_start
            retained.index_put_(
                (target, edges.head),
                edges.weight,
                accumulate=True,
            )
        mass = retained + diagonal
        cells = cells / mass.clamp_min(1e-8)[..., None]
        cells = torch.where(
            mass[..., None] > 0,
            cells,
            response[:, None],
        )

    return cells.mean(dim=1)


def flow_step(
    graph,
    state: torch.Tensor,
    layer: int,
    residual_weight: float,
    unresolved: str,
) -> torch.Tensor:
    response = state[graph.response_start :]
    attended = attention_output(graph, state, layer, unresolved)
    updated = (residual_weight * response + attended) / (residual_weight + 1.0)
    return torch.cat((state[: graph.response_start], updated), dim=0)


def flow_sequence(
    graph,
    initial: torch.Tensor,
    order,
    residual_weight: float,
    unresolved: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial
    trajectory = []
    for layer in order:
        state = flow_step(
            graph,
            state,
            int(layer),
            residual_weight,
            unresolved,
        )
        trajectory.append(state[graph.response_start :])
    return state, torch.stack(trajectory)


def layer_mean_view(
    graph,
    initial: torch.Tensor,
    residual_weight: float,
    unresolved: str,
) -> torch.Tensor:
    """Average one-step outputs of the individual layer operators."""

    states = [
        flow_step(
            graph,
            initial,
            layer,
            residual_weight,
            unresolved,
        )[graph.response_start :]
        for layer in range(graph.layer_count)
    ]
    return torch.stack(states).mean(dim=0)


def encode_views(graph, config: FlowConfig | None = None) -> FlowViews:
    config = FlowConfig() if config is None else config
    initial = source_basis(
        graph.token_count,
        graph.response_start,
        config.sketch_dim,
        graph.device,
    )

    full_state, full_trajectory = flow_sequence(
        graph,
        initial,
        range(graph.layer_count),
        config.residual_weight,
        config.unresolved,
    )
    reverse_state, reverse_trajectory = flow_sequence(
        graph,
        initial,
        reversed(range(graph.layer_count)),
        config.residual_weight,
        config.unresolved,
    )
    last_layer = flow_step(
        graph,
        initial,
        graph.layer_count - 1,
        config.residual_weight,
        config.unresolved,
    )[graph.response_start :]
    layer_mean = layer_mean_view(
        graph,
        initial,
        config.residual_weight,
        config.unresolved,
    )

    identity_final = initial[graph.response_start :]
    identity_trace = identity_final[:, None, :].expand(
        -1,
        graph.layer_count,
        -1,
    ).reshape(graph.response_count, -1)

    return FlowViews(
        full_trace=full_trajectory.transpose(0, 1).reshape(graph.response_count, -1),
        full_final=full_state[graph.response_start :],
        reverse_trace=reverse_trajectory.transpose(0, 1).reshape(
            graph.response_count,
            -1,
        ),
        reverse_final=reverse_state[graph.response_start :],
        last_layer=last_layer,
        layer_mean=layer_mean,
        identity_trace=identity_trace,
        identity_final=identity_final,
        trajectory=full_trajectory.transpose(0, 1),
    )
