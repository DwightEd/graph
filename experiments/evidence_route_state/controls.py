"""Matched dense controls applied before lineage construction."""

from __future__ import annotations

import torch


def dense_endpoint_rewire(
    capacity: torch.Tensor,
    support: torch.Tensor,
    query_position: torch.Tensor,
    response_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate physical endpoints while preserving row/head/role accounts."""

    return _rotate_dense(
        capacity, support, query_position, response_start, head_dependent=False
    )


def dense_weight_shuffle(
    capacity: torch.Tensor,
    support: torch.Tensor,
    query_position: torch.Tensor,
    response_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle paired capacity/support within every row/head/source role."""

    return _rotate_dense(
        capacity, support, query_position, response_start, head_dependent=True
    )


def dense_without_messages(
    capacity: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove every edge account for the non-identifiable boundary test."""

    return torch.zeros_like(capacity), torch.zeros_like(support)


def _rotate_dense(
    capacity: torch.Tensor,
    support: torch.Tensor,
    query_position: torch.Tensor,
    response_start: int,
    *,
    head_dependent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries, heads, sources = capacity.shape
    device = capacity.device
    query = query_position.to(device=device, dtype=torch.long)[:, None, None]
    source = torch.arange(sources, device=device)[None, None, :]
    head = torch.arange(heads, device=device)[None, :, None]
    index = source.expand(queries, heads, sources).clone()

    prompt_stop = torch.minimum(
        query,
        torch.tensor(min(response_start, sources), device=device),
    ).clamp_min(0)
    prompt_member = source < prompt_stop
    prompt_shift = (
        1 + head % (prompt_stop - 1).clamp_min(1)
        if head_dependent
        else torch.ones_like(prompt_stop)
    )
    prompt_index = (source + prompt_shift) % prompt_stop.clamp_min(1)
    index = torch.where(prompt_member & (prompt_stop > 1), prompt_index, index)

    history_stop = torch.minimum(query, torch.tensor(sources, device=device))
    history_count = (history_stop - response_start).clamp_min(0)
    history_member = (source >= response_start) & (source < history_stop)
    history_shift = (
        1 + head % (history_count - 1).clamp_min(1)
        if head_dependent
        else torch.ones_like(history_count)
    )
    history_index = response_start + (
        (source - response_start + history_shift) % history_count.clamp_min(1)
    )
    index = torch.where(history_member & (history_count > 1), history_index, index)
    return capacity.gather(2, index), support.gather(2, index)
