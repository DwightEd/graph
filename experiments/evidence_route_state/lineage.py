"""Streaming boundary-unit ancestry on the attention-write DAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .graph import RouteEdges


@dataclass(frozen=True)
class RouteLineage:
    """Ancestry and response-route accounts, with the head axis intact."""

    query_position: torch.Tensor
    prediction_position: torch.Tensor
    history_valid: torch.Tensor
    ancestry: torch.Tensor
    prompt_evidence: torch.Tensor
    grounded_response_relay: torch.Tensor
    unrooted_response_feedback: torch.Tensor
    predictor_self: torch.Tensor
    unknown: torch.Tensor
    effective_sources: torch.Tensor
    effective_head_rank: torch.Tensor
    anchor_source: torch.Tensor


class LineageTracker:
    """Propagate exact dense routes while retaining only compact state."""

    def __init__(
        self,
        token_root_unit_id: torch.Tensor,
        response_start: int,
        evidence_unit_count: int,
        layer_count: int,
        head_count: int,
        *,
        device: torch.device | str | None = None,
        multi_hop: bool = True,
    ) -> None:
        device = torch.device(device or token_root_unit_id.device)
        roots = token_root_unit_id.to(device=device, dtype=torch.long)
        self.response_start = response_start
        self.evidence_unit_count = evidence_unit_count
        self.layer_count = layer_count
        self.head_count = head_count
        self.token_count = len(roots)
        self.multi_hop = multi_hop

        # Prompt evidence units stay distinct. Response roots share one channel
        # because only their evidence ancestry is needed to split history flow.
        self.response_root = evidence_unit_count + 1
        self.unknown_root = evidence_unit_count + 2
        boundary = torch.where(roots <= evidence_unit_count, roots, self.response_root)
        self.ancestry = torch.zeros(
            layer_count + 1,
            self.token_count,
            evidence_unit_count + 3,
            dtype=torch.float32,
            device=device,
        )
        self.ancestry[0, torch.arange(self.token_count, device=device), boundary] = 1.0
        self.layer_started = torch.zeros(layer_count, dtype=torch.bool)

        query_position = torch.arange(
            response_start - 1, self.token_count - 1, device=device
        )
        self.query_position = query_position
        self.prediction_position = query_position + 1
        self.history_valid = query_position >= response_start + 1
        route_shape = (layer_count, len(query_position), head_count)
        self.prompt_evidence = torch.zeros(
            route_shape, dtype=torch.float32, device=device
        )
        self.grounded_response_relay = torch.zeros_like(self.prompt_evidence)
        self.unrooted_response_feedback = torch.zeros_like(self.prompt_evidence)
        self.predictor_self = torch.zeros_like(self.prompt_evidence)
        self.unknown = torch.zeros_like(self.prompt_evidence)
        topology_shape = (layer_count, len(query_position))
        self.effective_sources = torch.zeros(
            topology_shape, dtype=torch.float32, device=device
        )
        self.effective_head_rank = torch.zeros_like(self.effective_sources)
        self.anchor_source = torch.full(
            route_shape, -1, dtype=torch.long, device=device
        )

    def add_dense(
        self,
        layer: int,
        query_position: Sequence[int] | torch.Tensor,
        capacity: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        """Advance a same-layer chunk from every physical source endpoint.

        Dense positive signed support drives lineage.  Dense capacity separately
        measures the topology of evidence-rooted carriers, so sparse graph
        storage and its unknown tail cannot change the detector state.
        """

        device = self.ancestry.device
        queries = torch.as_tensor(query_position, dtype=torch.long, device=device)
        support = support.float()
        residual = (1.0 - support.sum((1, 2))).clamp_min(0)
        tail = torch.zeros(
            len(queries), self.head_count, dtype=torch.float32, device=device
        )
        self._advance(layer, queries, capacity.float(), support, residual, tail)

    def add(self, row: RouteEdges) -> None:
        """Small-row oracle for :meth:`add_many`."""

        self.add_many((row,))

    def add_many(self, rows: Sequence[RouteEdges]) -> None:
        """Advance sparse rows for small-graph tests and inspection only."""

        rows = tuple(rows)
        if not rows:
            return
        layer = rows[0].layer
        device = self.ancestry.device
        queries = torch.tensor(
            [row.query_position for row in rows], dtype=torch.long, device=device
        )
        counts = torch.tensor(
            [len(row.head) for row in rows], dtype=torch.long, device=device
        )
        edge_row = torch.repeat_interleave(
            torch.arange(len(rows), device=device), counts
        )
        source = torch.cat([row.source for row in rows]).to(device)
        head = torch.cat([row.head for row in rows]).to(device)
        capacity = torch.zeros(
            len(rows), self.head_count, self.token_count, device=device
        )
        support = torch.zeros_like(capacity)
        capacity.index_put_(
            (edge_row, head, source),
            torch.cat([row.capacity for row in rows]).float().to(device),
            accumulate=True,
        )
        support.index_put_(
            (edge_row, head, source),
            torch.cat([row.support for row in rows]).float().to(device),
            accumulate=True,
        )
        residual = torch.stack([row.residual_support for row in rows]).clamp_min(0)
        tail = torch.stack([row.unknown_positive_support for row in rows]).to(device)
        self._advance(layer, queries, capacity, support, residual.to(device), tail)

    def _advance(
        self,
        layer: int,
        queries: torch.Tensor,
        capacity: torch.Tensor,
        support: torch.Tensor,
        residual: torch.Tensor,
        tail: torch.Tensor,
    ) -> None:
        if not self.layer_started[layer]:
            self.ancestry[layer + 1].copy_(self.ancestry[layer])
            self.layer_started[layer] = True

        previous = self.ancestry[layer]
        lineage_source = previous if self.multi_hop else self.ancestry[0]
        sources = capacity.shape[2]
        source_ancestry = lineage_source[:sources]
        positive = support.clamp_min(0)
        positive_by_source = positive.sum(1)
        normalizer = (positive_by_source.sum(1) + residual + tail.sum(1)).clamp_min(
            1e-12
        )

        route_probability = positive_by_source / normalizer[:, None]
        node_ancestry = route_probability @ source_ancestry
        node_ancestry += (
            residual[:, None] / normalizer[:, None] * lineage_source[queries]
        )
        node_ancestry[:, self.unknown_root] += tail.sum(1) / normalizer
        self.ancestry[layer + 1, queries] = node_ancestry

        response = (queries >= self.response_start - 1) & (
            queries < self.token_count - 1
        )
        query = queries[response]
        slot = query - self.response_start + 1
        positive = positive[response]
        capacity = capacity[response]
        scale = normalizer[response, None]
        source = torch.arange(sources, device=queries.device)
        evidence = source_ancestry[:, 1 : self.evidence_unit_count + 1].sum(1)
        response_born = source_ancestry[:, self.response_root]
        unknown = source_ancestry[:, self.unknown_root]
        prompt = (source[None] < self.response_start) & (source[None] != query[:, None])
        history = (source[None] >= self.response_start) & (
            source[None] < query[:, None]
        )
        predictor = source[None] == query[:, None]

        def flow(mask: torch.Tensor, ancestry: torch.Tensor) -> torch.Tensor:
            value = torch.einsum("qhs,qs,s->qh", positive, mask, ancestry)
            return value / scale

        self.prompt_evidence[layer, slot] = flow(prompt, evidence)
        self.grounded_response_relay[layer, slot] = flow(history, evidence)
        self.unrooted_response_feedback[layer, slot] = flow(history, response_born)
        self.predictor_self[layer, slot] = (positive * predictor[:, None]).sum(
            2
        ) / scale
        propagated_unknown = torch.einsum("qhs,s->qh", positive, unknown) / scale
        self.unknown[layer, slot] = tail[response] / scale + propagated_unknown
        self._record_topology(layer, slot, query, capacity, evidence)

    def _record_topology(
        self,
        layer: int,
        slot: torch.Tensor,
        query: torch.Tensor,
        capacity: torch.Tensor,
        evidence_ancestry: torch.Tensor,
    ) -> None:
        source = torch.arange(capacity.shape[2], device=capacity.device)
        carrier = capacity * evidence_ancestry[None, None]
        carrier *= (source[None] < query[:, None])[:, None]
        head_mass = carrier.sum(2)
        active = head_mass > 1e-12
        anchor = carrier.argmax(2).masked_fill(~active, -1)
        carrier /= head_mass.clamp_min(1e-12)[..., None]

        active_count = active.sum(1)
        mixture = (carrier * active[..., None]).sum(1)
        mixture /= active_count.clamp_min(1)[:, None]
        entropy = -(mixture * mixture.clamp_min(1e-12).log()).sum(1)
        effective_sources = entropy.exp().masked_fill(active_count == 0, 0)

        gram = carrier @ carrier.transpose(1, 2)
        trace = gram.diagonal(dim1=1, dim2=2).sum(1)
        energy = gram.square().sum((1, 2))
        effective_rank = trace.square() / energy.clamp_min(1e-12)
        effective_rank.masked_fill_(energy == 0, 0)

        self.effective_sources[layer, slot] = effective_sources
        self.effective_head_rank[layer, slot] = effective_rank
        self.anchor_source[layer, slot] = anchor

    def finish(self) -> RouteLineage:
        """Return accumulated response routes and compact boundary ancestry."""

        for layer in range(self.layer_count):
            if not self.layer_started[layer]:
                self.ancestry[layer + 1].copy_(self.ancestry[layer])
        return RouteLineage(
            query_position=self.query_position,
            prediction_position=self.prediction_position,
            history_valid=self.history_valid,
            ancestry=self.ancestry,
            prompt_evidence=self.prompt_evidence,
            grounded_response_relay=self.grounded_response_relay,
            unrooted_response_feedback=self.unrooted_response_feedback,
            predictor_self=self.predictor_self,
            unknown=self.unknown,
            effective_sources=self.effective_sources,
            effective_head_rank=self.effective_head_rank,
            anchor_source=self.anchor_source,
        )


def propagate_lineage(
    rows: Sequence[RouteEdges],
    token_root_unit_id: torch.Tensor,
    response_start: int,
    evidence_unit_count: int,
) -> RouteLineage:
    """Small-graph oracle implemented by the same streaming tracker."""

    rows = tuple(rows)
    tracker = LineageTracker(
        token_root_unit_id,
        response_start,
        evidence_unit_count,
        max(row.layer for row in rows) + 1,
        rows[0].head_count,
        device=rows[0].capacity.device,
    )
    chunk: list[RouteEdges] = []
    for row in sorted(rows, key=lambda item: (item.layer, item.query_position)):
        contiguous = chunk and row.layer == chunk[-1].layer
        contiguous = contiguous and row.query_position == chunk[-1].query_position + 1
        if chunk and not contiguous:
            tracker.add_many(chunk)
            chunk = []
        chunk.append(row)
    tracker.add_many(chunk)
    return tracker.finish()
