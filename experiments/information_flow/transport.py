"""Identity-preserving ordered route transport over frozen token graphs."""

import math
from dataclasses import dataclass

import torch

from .basis import source_basis


@dataclass(frozen=True)
class FlowOutput:
    """Keep the frozen GCN base separate from route-only snapshots."""

    embedding: torch.Tensor
    trajectory: torch.Tensor


def checkpoint_layers(layer_count: int, count: int) -> tuple[int, ...]:
    if not 1 <= count <= layer_count:
        raise ValueError("checkpoints must be between 1 and layer_count")
    return tuple(
        round(step * layer_count / count)
        for step in range(1, count + 1)
    )


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


def transform_delta(
    delta: torch.Tensor,
    head: torch.Tensor,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "mean":
        return delta
    if mode != "sketch":
        raise ValueError("mode must be 'sketch' or 'mean'")
    return delta.gather(1, permutation[head]) * sign[head]


def transport_layer(
    graph,
    state: torch.Tensor,
    layer: int,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Transport only off-diagonal increments; self-only rows are identity."""

    device = state.device
    response_count = len(graph.token_ids) - graph.response_start
    dimension = state.shape[1]
    selected = graph.edge_layer == layer
    source = graph.edge_index[0, selected].to(device)
    target = graph.edge_index[1, selected].to(device)
    head = graph.edge_head[selected].to(device)
    weight = graph.edge_weight[selected].to(device=device, dtype=state.dtype)

    head_delta = state.new_zeros(
        (response_count * graph.head_count, dimension)
    )
    if len(source):
        delta = state[source] - state[target]
        delta = transform_delta(delta, head, permutation, sign, mode)
        group = (target - graph.response_start) * graph.head_count + head
        head_delta.index_add_(0, group, weight[:, None] * delta)

    head_delta = head_delta.view(
        response_count,
        graph.head_count,
        dimension,
    )
    if mode == "mean":
        update = head_delta.mean(dim=1)
    else:
        update = head_delta.sum(dim=1) / math.sqrt(graph.head_count)
    response_state = state[graph.response_start :] + update
    return torch.cat((state[: graph.response_start], response_state), dim=0)


def flow_embedding(
    graph,
    *,
    mode: str = "sketch",
    checkpoints: int = 4,
    seed: int = 20260827,
    device: str | torch.device = "cpu",
) -> FlowOutput:
    """Compose ordered route deltas from a graph-independent source basis.

    The frozen GCN embedding is retained only as the base output channel.  It
    is never used as the layer-zero route state because it already aggregates
    all Transformer layers.
    """

    device = torch.device(device)
    base = graph.node_embedding[graph.response_start :].to(
        device=device,
        dtype=torch.float32,
    )
    dimension = int(base.shape[1])
    state = source_basis(
        len(graph.token_ids),
        graph.response_start,
        dimension,
        device=device,
    )
    layers = checkpoint_layers(graph.layer_count, checkpoints)
    permutation, sign = sketch_tables(
        graph.layer_count,
        graph.head_count,
        dimension,
        seed,
        device,
    )

    snapshots = []
    for layer in range(graph.layer_count):
        state = transport_layer(
            graph,
            state,
            layer,
            permutation[layer],
            sign[layer],
            mode,
        )
        if layer + 1 in layers:
            snapshots.append(state[graph.response_start :].clone())

    trajectory = torch.stack(snapshots, dim=1)
    embedding = torch.cat((base, *snapshots), dim=1)
    return FlowOutput(embedding=embedding, trajectory=trajectory)
