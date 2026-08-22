"""Random masks for channel, pair-layer, and node-diagonal recovery."""

from dataclasses import dataclass

import torch

from .config import RecoveryConfig
from .graph_data import MultiplexGraph


@dataclass(frozen=True)
class MaskedGraph:
    edge_value: torch.Tensor
    edge_visible: torch.Tensor
    edge_target: torch.Tensor
    diagonal_value: torch.Tensor
    diagonal_visible: torch.Tensor
    diagonal_target: torch.Tensor


def _keep_one(mask: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()) or not bool(available.any()):
        return mask
    first = torch.nonzero(available, as_tuple=False)[0]
    mask[tuple(first.tolist())] = True
    return mask


def mask_graph(
    graph: MultiplexGraph,
    config: RecoveryConfig,
    *,
    generator: torch.Generator,
) -> MaskedGraph:
    channel_mask = (
        torch.rand(graph.edge_attr.shape, generator=generator, device=graph.device)
        < config.channel_mask_rate
    ) & graph.edge_observed

    pair_layer_available = graph.edge_observed.any(dim=-1)
    pair_layer_mask = (
        torch.rand(pair_layer_available.shape, generator=generator, device=graph.device)
        < config.pair_layer_mask_rate
    ) & pair_layer_available
    edge_target = channel_mask | (pair_layer_mask.unsqueeze(-1) & graph.edge_observed)
    edge_target = _keep_one(edge_target, graph.edge_observed)
    edge_visible = graph.edge_observed & ~edge_target

    diagonal_target = (
        torch.rand(graph.diagonal.shape, generator=generator, device=graph.device)
        < config.diagonal_mask_rate
    ) & graph.diagonal_observed
    diagonal_target = _keep_one(diagonal_target, graph.diagonal_observed)
    diagonal_visible = graph.diagonal_observed & ~diagonal_target

    return MaskedGraph(
        edge_value=graph.edge_attr * edge_visible,
        edge_visible=edge_visible,
        edge_target=edge_target,
        diagonal_value=graph.diagonal * diagonal_visible,
        diagonal_visible=diagonal_visible,
        diagonal_target=diagonal_target,
    )
