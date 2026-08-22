"""Structure-preserving interventions used to audit a trained recovery model."""

from dataclasses import replace
import math

import torch

from .graph_data import MultiplexGraph
from .masking import MaskedGraph


def _masked_mean(value: torch.Tensor, visible: torch.Tensor, dims) -> torch.Tensor:
    total = (value * visible).sum(dim=dims, keepdim=True)
    count = visible.sum(dim=dims, keepdim=True).clamp_min(1)
    return total / count


def collapse_channels(masked: MaskedGraph, mode: str) -> MaskedGraph:
    if mode == "full":
        return masked
    edge_visible = masked.edge_visible
    diagonal_visible = masked.diagonal_visible
    if mode == "layer_mean":
        edge = _masked_mean(masked.edge_value, edge_visible, -1).expand_as(masked.edge_value)
        diagonal = _masked_mean(masked.diagonal_value, diagonal_visible, -1).expand_as(masked.diagonal_value)
        edge_visible = edge_visible.any(dim=-1, keepdim=True).expand_as(edge_visible)
        diagonal_visible = diagonal_visible.any(dim=-1, keepdim=True).expand_as(diagonal_visible)
    else:
        edge = _masked_mean(masked.edge_value, edge_visible, (-2, -1)).expand_as(masked.edge_value)
        diagonal = _masked_mean(masked.diagonal_value, diagonal_visible, (-2, -1)).expand_as(masked.diagonal_value)
        edge_visible = edge_visible.any(dim=(-2, -1), keepdim=True).expand_as(edge_visible)
        diagonal_visible = diagonal_visible.any(dim=(-2, -1), keepdim=True).expand_as(diagonal_visible)
    return replace(
        masked,
        edge_value=edge,
        edge_visible=edge_visible,
        diagonal_value=diagonal,
        diagonal_visible=diagonal_visible,
    )


def shuffle_layers(masked: MaskedGraph, *, generator: torch.Generator) -> MaskedGraph:
    order = torch.randperm(masked.edge_value.shape[1], generator=generator, device=masked.edge_value.device)
    return replace(
        masked,
        edge_value=masked.edge_value[:, order],
        edge_visible=masked.edge_visible[:, order],
        diagonal_value=masked.diagonal_value[:, order],
        diagonal_visible=masked.diagonal_visible[:, order],
    )


def shuffle_heads(masked: MaskedGraph, *, generator: torch.Generator) -> MaskedGraph:
    order = torch.randperm(masked.edge_value.shape[2], generator=generator, device=masked.edge_value.device)
    return replace(
        masked,
        edge_value=masked.edge_value[:, :, order],
        edge_visible=masked.edge_visible[:, :, order],
        diagonal_value=masked.diagonal_value[:, :, order],
        diagonal_visible=masked.diagonal_visible[:, :, order],
    )


def _bin(graph: MultiplexGraph, target: int, source: int, bins: int = 16) -> int:
    if source < graph.response_idx:
        return min(source * bins // max(graph.response_idx, 1), bins - 1)
    return min(int(math.log2(max(target - source, 1))), bins - 1)


def rewire_endpoints(graph: MultiplexGraph, *, generator: torch.Generator) -> MultiplexGraph:
    edge_index = graph.edge_index.clone()
    for token in range(graph.num_response_tokens):
        current = graph.incoming(token)
        target = graph.response_idx + token
        sources = edge_index[0, current]
        occupied = set(int(value) for value in sources.tolist())
        used: set[int] = set()
        for offset, source_tensor in enumerate(sources):
            source = int(source_tensor)
            prompt = source < graph.response_idx
            domain = range(graph.response_idx) if prompt else range(graph.response_idx, target)
            candidates = [
                candidate
                for candidate in domain
                if candidate not in occupied
                and candidate not in used
                and _bin(graph, target, candidate) == _bin(graph, target, source)
            ]
            if candidates:
                index = int(torch.randint(len(candidates), (1,), generator=generator, device=graph.device))
                replacement = candidates[index]
                edge_index[0, current.start + offset] = replacement
                used.add(replacement)
    return replace(graph, edge_index=edge_index)
