"""Turn a multiplex attention graph into one structural feature vector per token.

No neural message passing is used. Edge distributions and one-hop neighbour
provenance are projected into a fixed basis during graph construction. The
resulting node features are the only input seen by the anomaly detector.
"""

from dataclasses import dataclass
import math

import torch

from .config import FeatureConfig
from .graph import AttentionGraph


@dataclass(frozen=True)
class RoutingFeatures:
    node: torch.Tensor
    token_layer: torch.Tensor
    direct_prompt: torch.Tensor
    response_history: torch.Tensor
    inherited_prompt: torch.Tensor


def cosine_basis(position: torch.Tensor, dimension: int) -> torch.Tensor:
    frequency = torch.arange(
        dimension,
        dtype=torch.float32,
        device=position.device,
    )
    return torch.cos(math.pi * position[:, None] * frequency[None])


def head_projection(
    heads: int,
    modes: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    modes = min(heads, modes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((heads, modes), generator=generator, dtype=torch.float32)
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection.to(device)


def direct_edge_features(
    graph: AttentionGraph,
    basis_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (
        graph.response_count,
        graph.layer_count,
        graph.head_count,
        basis_dim,
    )
    prompt = torch.zeros(shape, dtype=torch.float32, device=graph.device)
    history = torch.zeros_like(prompt)

    source = graph.edges.source
    response_target = graph.edges.target - graph.response_start
    layer = graph.edges.layer
    head = graph.edges.head
    weight = graph.edges.weight

    prompt_edge = source < graph.response_start
    if bool(prompt_edge.any()):
        position = (
            source[prompt_edge].float() + 0.5
        ) / max(graph.response_start, 1)
        basis = cosine_basis(position, basis_dim)
        row = (
            (response_target[prompt_edge] * graph.layer_count + layer[prompt_edge])
            * graph.head_count
            + head[prompt_edge]
        )
        prompt.view(-1, basis_dim).index_add_(
            0,
            row,
            weight[prompt_edge, None] * basis,
        )

    history_edge = ~prompt_edge
    if bool(history_edge.any()):
        source_response = source[history_edge] - graph.response_start
        target_response = response_target[history_edge]
        lag = (
            target_response - source_response
        ).float() / (target_response.float() + 1.0)
        basis = cosine_basis(lag.clamp(0.0, 1.0), basis_dim)
        row = (
            (target_response * graph.layer_count + layer[history_edge])
            * graph.head_count
            + head[history_edge]
        )
        history.view(-1, basis_dim).index_add_(
            0,
            row,
            weight[history_edge, None] * basis,
        )

    return prompt, history


def inherited_prompt_features(
    graph: AttentionGraph,
    direct_prompt: torch.Tensor,
) -> torch.Tensor:
    """Aggregate a source token's previous-layer prompt signature into targets."""

    inherited = torch.zeros_like(direct_prompt)

    source = graph.edges.source
    response_target = graph.edges.target - graph.response_start
    layer = graph.edges.layer
    head = graph.edges.head
    weight = graph.edges.weight
    history_edge = (source >= graph.response_start) & (layer > 0)
    if not bool(history_edge.any()):
        return inherited

    source_response = source[history_edge] - graph.response_start
    target_response = response_target[history_edge]
    previous_layer = layer[history_edge] - 1
    message = (
        direct_prompt[source_response, previous_layer, head[history_edge]]
        * weight[history_edge, None]
    )
    row = (
        (target_response * graph.layer_count + layer[history_edge])
        * graph.head_count
        + head[history_edge]
    )
    inherited.view(-1, inherited.shape[-1]).index_add_(0, row, message)
    return inherited


@torch.no_grad()
def build_node_features(
    graph: AttentionGraph,
    config: FeatureConfig | None = None,
) -> RoutingFeatures:
    config = FeatureConfig() if config is None else config
    direct_prompt, response_history = direct_edge_features(
        graph,
        config.source_basis_dim,
    )
    inherited_prompt = inherited_prompt_features(graph, direct_prompt)

    row_state = torch.stack((graph.diagonal, graph.unresolved), dim=-1)
    channel = torch.cat(
        (
            direct_prompt,
            response_history,
            inherited_prompt,
            row_state,
        ),
        dim=-1,
    )
    projection = head_projection(
        graph.head_count,
        config.head_projection_dim,
        config.projection_seed,
        graph.device,
    )
    token_layer = torch.einsum(
        "rlhf,hm->rlmf",
        channel,
        projection,
    ).flatten(2)
    return RoutingFeatures(
        node=token_layer.flatten(1),
        token_layer=token_layer,
        direct_prompt=direct_prompt,
        response_history=response_history,
        inherited_prompt=inherited_prompt,
    )
