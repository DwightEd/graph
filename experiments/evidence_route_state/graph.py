"""Sparse rows of the head-resolved attention-write graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouteEdges:
    """Explicit token edges and an endpoint-free remainder for one query row."""

    layer: int
    query_position: int
    source: torch.Tensor
    head: torch.Tensor
    message: torch.Tensor
    capacity: torch.Tensor
    support: torch.Tensor
    unknown_message: torch.Tensor
    unknown_capacity: torch.Tensor
    unknown_support: torch.Tensor
    unknown_positive_support: torch.Tensor
    residual_support: torch.Tensor

    @property
    def prediction_position(self) -> int:
        return self.query_position + 1

    @property
    def head_count(self) -> int:
        return len(self.unknown_capacity)


@dataclass(frozen=True)
class ResponseGraph:
    """CPU scalar graph for response predictions; AVWO vectors are not saved."""

    edge_start: torch.Tensor
    row_layer: torch.Tensor
    row_query_position: torch.Tensor
    row_prediction_position: torch.Tensor
    edge_head: torch.Tensor
    edge_source: torch.Tensor
    edge_capacity: torch.Tensor
    edge_support: torch.Tensor
    unknown_capacity: torch.Tensor
    unknown_support: torch.Tensor
    unknown_positive_support: torch.Tensor
    unknown_write_norm: torch.Tensor
    reconstructed_head_write_norm: torch.Tensor
    reconstructed_attention_write_norm: torch.Tensor


def joint_cover(
    capacity: torch.Tensor,
    positive_support: torch.Tensor,
    coverage: float,
    max_edges_per_head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source indices and a mask for every query/head row.

    Each head keeps its own shortest joint-cover prefix.  The sparse graph is
    therefore head resolved even when one head carries much more mass than
    another.
    """

    values = capacity.float()
    support = positive_support.float()
    capacity_total = values.sum(2, keepdim=True)
    support_total = support.sum(2, keepdim=True)
    importance = values / capacity_total.clamp_min(1e-12)
    importance += support / support_total.clamp_min(1e-12)

    edge_count = min(max_edges_per_head, capacity.shape[2])
    _, index = importance.topk(edge_count, dim=2, sorted=True)
    selected_capacity = values.gather(2, index).cumsum(2)
    selected_support = support.gather(2, index).cumsum(2)
    capacity_covered = (selected_capacity >= coverage * capacity_total) | (
        capacity_total == 0
    )
    support_covered = (selected_support >= coverage * support_total) | (
        support_total == 0
    )
    covered = capacity_covered & support_covered
    count = torch.where(
        covered.any(2),
        covered.to(torch.int64).argmax(2) + 1,
        torch.full(capacity.shape[:2], edge_count, device=capacity.device),
    )
    has_route = (capacity_total[..., 0] > 0) | (support_total[..., 0] > 0)
    count = torch.where(has_route, count, torch.zeros_like(count))
    valid = torch.arange(edge_count, device=capacity.device) < count[..., None]
    return index, valid


def select_route_indices(
    capacity: torch.Tensor,
    positive_support: torch.Tensor,
    *,
    coverage: float,
    max_edges_per_head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-row oracle for the chunked joint-cover implementation."""

    index, valid = joint_cover(
        capacity[None], positive_support[None], coverage, max_edges_per_head
    )
    head = torch.arange(capacity.shape[0], device=capacity.device)[:, None]
    head = head.expand_as(index[0])[valid[0]]
    source = index[0][valid[0]]
    return head, source


def sparsify_route_chunk(
    capacity: torch.Tensor,
    support: torch.Tensor,
    head_write: torch.Tensor,
    selected_messages: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    *,
    layer: int,
    query_position: Sequence[int] | torch.Tensor,
    coverage: float,
    max_edges_per_head: int,
) -> tuple[RouteEdges, ...]:
    """Sparsify ``[query, head, source]`` statistics with one batched top-k.

    The callback receives local query, head, and source indices for every
    retained edge and materializes those AVWO vectors in one operation.
    """

    queries, heads, _ = capacity.shape
    index, valid = joint_cover(
        capacity, support.clamp_min(0), coverage, max_edges_per_head
    )
    local_query = torch.arange(queries, device=capacity.device)[:, None, None]
    local_query = local_query.expand_as(index)
    local_head = torch.arange(heads, device=capacity.device)[None, :, None]
    local_head = local_head.expand_as(index)
    selected_query = local_query[valid]
    selected_head = local_head[valid]
    selected_source = index[valid]
    # Head-major recovery lets the callback apply each W_O block once without
    # expanding an infeasible [selected edge, head_dim, hidden] tensor.
    head_order = torch.argsort(selected_head, stable=True)
    head_major_message = selected_messages(
        selected_query[head_order],
        selected_head[head_order],
        selected_source[head_order],
    ).float()
    row_order = torch.empty_like(head_order)
    row_order[head_order] = torch.arange(len(head_order), device=head_order.device)
    message = head_major_message[row_order]
    selected_capacity = capacity[selected_query, selected_head, selected_source].float()
    selected_support = support[selected_query, selected_head, selected_source].float()

    hidden = head_write.shape[-1]
    row_head = selected_query * heads + selected_head
    selected_write = torch.zeros(
        queries * heads, hidden, dtype=torch.float32, device=head_write.device
    )
    selected_write.index_add_(0, row_head, message)

    def omitted_total(values: torch.Tensor) -> torch.Tensor:
        kept = torch.zeros(queries * heads, dtype=torch.float32, device=values.device)
        selected = values[selected_query, selected_head, selected_source].float()
        kept.index_add_(0, row_head, selected)
        return values.float().sum(2) - kept.reshape(queries, heads)

    unknown_message = head_write.float() - selected_write.reshape(
        queries, heads, hidden
    )
    unknown_capacity = omitted_total(capacity)
    unknown_support = omitted_total(support)
    unknown_positive = omitted_total(support.clamp_min(0))
    residual_support = 1.0 - support.float().sum((1, 2))
    counts = valid.sum((1, 2)).cpu().tolist()
    positions = torch.as_tensor(query_position).cpu().tolist()

    result = []
    start = 0
    for query, count in enumerate(counts):
        stop = start + count
        result.append(
            RouteEdges(
                layer=layer,
                query_position=positions[query],
                source=selected_source[start:stop],
                head=selected_head[start:stop],
                message=message[start:stop],
                capacity=selected_capacity[start:stop],
                support=selected_support[start:stop],
                unknown_message=unknown_message[query],
                unknown_capacity=unknown_capacity[query],
                unknown_support=unknown_support[query],
                unknown_positive_support=unknown_positive[query],
                residual_support=residual_support[query],
            )
        )
        start = stop
    return tuple(result)


def sparsify_routes(
    capacity: torch.Tensor,
    support: torch.Tensor,
    head_write: torch.Tensor,
    selected_messages: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    layer: int,
    query_position: int,
    coverage: float,
    max_edges_per_head: int,
) -> RouteEdges:
    """Small-row oracle for :func:`sparsify_route_chunk`.

    ``selected_messages`` materializes only chosen AVWO vectors. The dense
    ``[head, source, d_model]`` tensor is therefore never constructed.
    """

    return sparsify_route_chunk(
        capacity[None],
        support[None],
        head_write[None],
        lambda query, head, source: selected_messages(head, source),
        layer=layer,
        query_position=(query_position,),
        coverage=coverage,
        max_edges_per_head=max_edges_per_head,
    )[0]


def attention_write(edges: RouteEdges) -> torch.Tensor:
    """Reconstruct the complete attention write, including the unknown tail."""

    return edges.message.sum(0) + edges.unknown_message.sum(0)


def route_totals(
    edges: RouteEdges,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return conserved capacity, signed support, and positive support per head."""

    capacity = edges.unknown_capacity.clone()
    support = edges.unknown_support.clone()
    positive = edges.unknown_positive_support.clone()
    capacity.index_add_(0, edges.head, edges.capacity)
    support.index_add_(0, edges.head, edges.support)
    positive.index_add_(0, edges.head, edges.support.clamp_min(0))
    return capacity, support, positive


class ResponseGraphBuilder:
    """Persist response endpoints and scalar accounts without message vectors."""

    def __init__(self, response_start: int) -> None:
        self.response_start = response_start
        self.row_layer: list[torch.Tensor] = []
        self.row_query: list[torch.Tensor] = []
        self.edge_count: list[torch.Tensor] = []
        self.edge_head: list[torch.Tensor] = []
        self.edge_source: list[torch.Tensor] = []
        self.edge_capacity: list[torch.Tensor] = []
        self.edge_support: list[torch.Tensor] = []
        self.unknown_capacity: list[torch.Tensor] = []
        self.unknown_support: list[torch.Tensor] = []
        self.unknown_positive_support: list[torch.Tensor] = []
        self.unknown_write_norm: list[torch.Tensor] = []
        self.head_write_norm: list[torch.Tensor] = []
        self.attention_write_norm: list[torch.Tensor] = []
        self.head_count = 0

    def add(self, row: RouteEdges) -> None:
        """Small-row oracle for :meth:`add_many`."""

        self.add_many((row,))

    def add_many(self, rows: Sequence[RouteEdges]) -> None:
        """Move one chunk of response-row scalars to CPU in batched transfers."""

        rows = tuple(
            row for row in rows if row.query_position >= self.response_start - 1
        )
        if not rows:
            return
        self.head_count = rows[0].head_count
        counts = torch.tensor([len(row.head) for row in rows])
        self.row_layer.append(
            torch.tensor([row.layer for row in rows], dtype=torch.int16)
        )
        self.row_query.append(
            torch.tensor([row.query_position for row in rows], dtype=torch.int32)
        )
        self.edge_count.append(counts)

        edge_head = torch.cat([row.head for row in rows])
        edge_source = torch.cat([row.source for row in rows])
        edge_capacity = torch.cat([row.capacity for row in rows]).float()
        edge_support = torch.cat([row.support for row in rows]).float()
        unknown_message = torch.stack([row.unknown_message for row in rows]).float()
        unknown_capacity = torch.stack([row.unknown_capacity for row in rows])
        unknown_support = torch.stack([row.unknown_support for row in rows])
        unknown_positive = torch.stack([row.unknown_positive_support for row in rows])

        messages = torch.cat([row.message for row in rows]).float()
        device = messages.device
        edge_row = torch.repeat_interleave(
            torch.arange(len(rows), device=device), counts.to(device)
        )
        row_head = edge_row * self.head_count + edge_head
        selected_write = torch.zeros(
            len(rows) * self.head_count,
            unknown_message.shape[-1],
            dtype=torch.float32,
            device=device,
        )
        selected_write.index_add_(0, row_head, messages)
        head_write = selected_write.reshape_as(unknown_message) + unknown_message
        edge_index = torch.stack((edge_head, edge_source)).to(torch.int32).cpu()
        edge_account = torch.stack((edge_capacity, edge_support)).detach().cpu()
        row_account = (
            torch.stack(
                (
                    unknown_capacity,
                    unknown_support,
                    unknown_positive,
                    unknown_message.norm(dim=2),
                    head_write.norm(dim=2),
                )
            )
            .detach()
            .cpu()
        )
        attention_norm = head_write.sum(1).norm(dim=1).detach().cpu()

        self.edge_head.append(edge_index[0].to(torch.int16))
        self.edge_source.append(edge_index[1].to(torch.int32))
        self.edge_capacity.append(edge_account[0])
        self.edge_support.append(edge_account[1])
        self.unknown_capacity.append(row_account[0])
        self.unknown_support.append(row_account[1])
        self.unknown_positive_support.append(row_account[2])
        self.unknown_write_norm.append(row_account[3])
        self.head_write_norm.append(row_account[4])
        self.attention_write_norm.append(attention_norm)

    def finish(self) -> ResponseGraph:
        """Assemble the CPU ragged graph for direct artifact storage."""

        row_layer = (
            torch.cat(self.row_layer)
            if self.row_layer
            else torch.empty(0, dtype=torch.int16)
        )
        row_query = (
            torch.cat(self.row_query)
            if self.row_query
            else torch.empty(0, dtype=torch.int32)
        )
        counts = torch.cat(self.edge_count) if self.edge_count else torch.empty(0)
        counts = counts.to(torch.long)
        edge_start = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        empty_edge = torch.empty(0)
        empty_row = torch.empty(0, self.head_count)

        edge_head = (
            torch.cat(self.edge_head) if self.edge_head else empty_edge.to(torch.int16)
        )
        edge_source = (
            torch.cat(self.edge_source)
            if self.edge_source
            else empty_edge.to(torch.int32)
        )
        edge_capacity = (
            torch.cat(self.edge_capacity) if self.edge_capacity else empty_edge
        )
        edge_support = torch.cat(self.edge_support) if self.edge_support else empty_edge
        unknown_capacity = (
            torch.cat(self.unknown_capacity) if self.unknown_capacity else empty_row
        )
        unknown_support = (
            torch.cat(self.unknown_support) if self.unknown_support else empty_row
        )
        unknown_positive = (
            torch.cat(self.unknown_positive_support)
            if self.unknown_positive_support
            else empty_row
        )
        unknown_write_norm = (
            torch.cat(self.unknown_write_norm) if self.unknown_write_norm else empty_row
        )
        head_write_norm = (
            torch.cat(self.head_write_norm) if self.head_write_norm else empty_row
        )
        attention_write_norm = (
            torch.cat(self.attention_write_norm)
            if self.attention_write_norm
            else empty_edge
        )

        return ResponseGraph(
            edge_start=edge_start,
            row_layer=row_layer,
            row_query_position=row_query,
            row_prediction_position=row_query + 1,
            edge_head=edge_head,
            edge_source=edge_source,
            edge_capacity=edge_capacity.float(),
            edge_support=edge_support.float(),
            unknown_capacity=unknown_capacity.float(),
            unknown_support=unknown_support.float(),
            unknown_positive_support=unknown_positive.float(),
            unknown_write_norm=unknown_write_norm.float(),
            reconstructed_head_write_norm=head_write_norm.float(),
            reconstructed_attention_write_norm=attention_write_norm.float(),
        )
