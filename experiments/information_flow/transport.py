"""Layer-wise transport sketches built from typed attention graphs."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FlowOutput:
    embedding: torch.Tensor
    trajectory: torch.Tensor


def checkpoint_layers(layer_count: int, count: int) -> tuple[int, ...]:
    steps = {
        max(1, round(layer_count * fraction / count))
        for fraction in range(1, count + 1)
    }
    return tuple(sorted(steps))


def sketch_tables(
    layer_count: int,
    head_count: int,
    dimension: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutations = []
    signs = []
    for _ in range(layer_count * head_count):
        permutations.append(torch.randperm(dimension, generator=generator))
        signs.append(
            torch.randint(0, 2, (dimension,), generator=generator) * 2 - 1
        )
    permutation = torch.stack(permutations).reshape(
        layer_count, head_count, dimension
    )
    sign = torch.stack(signs).reshape(layer_count, head_count, dimension)
    return permutation.to(device), sign.to(device=device, dtype=torch.float32)


def layer_slices(edge_layer: torch.Tensor, layer_count: int) -> tuple[int, ...]:
    counts = torch.bincount(edge_layer.cpu(), minlength=layer_count)
    return (0, *counts.cumsum(0).tolist())


def transform_heads(
    state: torch.Tensor,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "mean":
        return state[:, None].expand(-1, len(permutation), -1)
    return state[:, permutation] * sign[None]


def transport_layer(
    graph,
    state: torch.Tensor,
    layer: int,
    start: int,
    stop: int,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    device = state.device
    heads = graph.head_count
    responses = graph.token_ids.numel() - graph.response_start
    dimension = state.shape[1]
    transformed = transform_heads(state, permutation, sign, mode)

    cells = state.new_zeros((responses * heads, dimension))
    mass = state.new_zeros(responses * heads)
    if stop > start:
        source = graph.edge_index[0, start:stop].to(device)
        target = graph.edge_index[1, start:stop].to(device) - graph.response_start
        head = graph.edge_head[start:stop].to(device)
        weight = graph.edge_weight[start:stop].to(device=device, dtype=state.dtype)
        group = target * heads + head
        message = transformed[source, head]
        cells.index_add_(0, group, message * weight[:, None])
        mass.index_add_(0, group, weight)

    cells = cells.reshape(responses, heads, dimension)
    mass = mass.reshape(responses, heads)
    self_mass = (
        graph.diagonal[:, layer] + graph.unresolved[:, layer]
    ).to(device=device, dtype=state.dtype)
    cells = cells + self_mass[..., None] * transformed[graph.response_start :]
    mass = mass + self_mass
    head_state = cells / mass.clamp_min(1e-8)[..., None]

    if mode == "mean":
        response_state = head_state.mean(dim=1)
        prompt_state = state[: graph.response_start]
    else:
        scale = math.sqrt(heads)
        response_state = head_state.sum(dim=1) / scale
        prompt_state = transformed[: graph.response_start].sum(dim=1) / scale
    return torch.cat((prompt_state, response_state), dim=0)


def flow_embedding(
    graph,
    *,
    mode: str = "sketch",
    checkpoints: int = 4,
    seed: int = 20260827,
    device: str | torch.device = "cpu",
) -> FlowOutput:
    """Compose typed attention transport and keep a few depth snapshots."""

    device = torch.device(device)
    state = graph.node_embedding.to(device=device, dtype=torch.float32)
    base = state[graph.response_start :].clone()
    layers = checkpoint_layers(graph.layer_count, checkpoints)
    offsets = layer_slices(graph.edge_layer, graph.layer_count)
    permutation, sign = sketch_tables(
        graph.layer_count,
        graph.head_count,
        state.shape[1],
        seed,
        device,
    )

    snapshots = []
    for layer in range(graph.layer_count):
        state = transport_layer(
            graph,
            state,
            layer,
            offsets[layer],
            offsets[layer + 1],
            permutation[layer],
            sign[layer],
            mode,
        )
        if layer + 1 in layers:
            snapshots.append(state[graph.response_start :].clone())

    trajectory = torch.stack(snapshots, dim=1)
    embedding = torch.cat((base, *snapshots), dim=1)
    return FlowOutput(embedding=embedding, trajectory=trajectory)
