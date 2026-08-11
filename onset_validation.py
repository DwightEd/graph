"""Small graph primitives used by onset-aligned validation experiments."""

from __future__ import annotations

import torch

from graphs import TokenGraph


def merge_positive_runs(runs, max_gap: int = 1) -> list[tuple[int, int]]:
    """Merge sorted or overlapping half-open positive-token intervals."""
    if not runs:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in runs):
        if not merged or start > merged[-1][1] + max_gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def causal_rewire(
    graph: TokenGraph, *, seed: int, sweeps: int = 1
) -> tuple[TokenGraph, int]:
    """Randomly swap sources within causal prompt/history edge groups.

    Edge positions retain their targets and sparse channel payloads.  A swap is
    accepted only when both new edges remain causal and absent from the graph.
    """
    source, target = graph.edge_index
    rewired_source = source.clone()
    device = source.device
    occupied = torch.zeros(
        (graph.num_nodes - graph.response_idx, graph.num_nodes),
        dtype=torch.bool,
        device=device,
    )
    target_row = target - graph.response_idx
    occupied[target_row, source] = True
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    accepted = 0

    for is_history in (False, True):
        group = torch.where((source >= graph.response_idx) == is_history)[0]
        if group.numel() < 2:
            continue

        for _ in range(sweeps):
            order = group[
                torch.randperm(group.numel(), generator=generator, device=device)
            ]
            order = order[: 2 * (order.numel() // 2)]
            left, right = order[::2], order[1::2]
            if left.numel() == 0:
                continue

            left_source, right_source = rewired_source[left], rewired_source[right]
            left_target, right_target = target[left], target[right]
            left_row, right_row = target_row[left], target_row[right]
            transition = torch.rand((), generator=generator, device=device) >= 0.5
            valid = transition & (left_source != right_source)
            valid &= right_source < left_target
            valid &= left_source < right_target
            valid &= ~occupied[left_row, right_source]
            valid &= ~occupied[right_row, left_source]
            if not bool(valid.any()):
                continue

            candidate_keys = torch.cat(
                (
                    left_target * graph.num_nodes + right_source,
                    right_target * graph.num_nodes + left_source,
                )
            )
            _, inverse, counts = torch.unique(
                candidate_keys, return_inverse=True, return_counts=True
            )
            unique_candidates = (counts[inverse] == 1).reshape(2, -1).all(dim=0)
            selected = valid & unique_candidates
            if not bool(selected.any()):
                continue

            left, right = left[selected], right[selected]
            left_source, right_source = rewired_source[left], rewired_source[right]
            left_row, right_row = target_row[left], target_row[right]
            occupied[left_row, left_source] = False
            occupied[right_row, right_source] = False
            rewired_source[left] = right_source
            rewired_source[right] = left_source
            occupied[left_row, right_source] = True
            occupied[right_row, left_source] = True
            accepted += int(selected.sum().item())

    return (
        TokenGraph(
            num_nodes=graph.num_nodes,
            response_idx=graph.response_idx,
            edge_index=torch.stack((rewired_source, target.clone())),
            edge_type=graph.edge_type.clone(),
            edge_weight=None if graph.edge_weight is None else graph.edge_weight.clone(),
            edge_ptr=None if graph.edge_ptr is None else graph.edge_ptr.clone(),
            edge_channel=None
            if graph.edge_channel is None
            else graph.edge_channel.clone(),
            edge_value=None if graph.edge_value is None else graph.edge_value.clone(),
        ),
        accepted,
    )
