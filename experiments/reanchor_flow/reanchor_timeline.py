"""Head-resolved discovery of generation-time reanchoring change points.

This module reads the already captured route ledger.  It does not run an
intervention for every route.  For each ``(layer, head, response destination)``
it contrasts transport from recent response tokens with transport from either
prompt tokens or a remote response source, then finds temporal changes in that
balance. Evidence association is measured only after structural selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .flow import PairedFlow, SourceLocationBuckets
from .native_world import NativeWorld
from .route_model import EVIDENCE, RouteDynamics

SOURCE_KIND_NAMES = (
    "none",
    "prompt_evidence",
    "other_prompt",
    "remote_response",
)
PROMPT_BUCKET = 0
OTHER_PROMPT_BUCKET = 1
REMOTE_BUCKET = 2
LOCAL_BUCKET = 3
REANCHOR_NMS_RADIUS = 1


@dataclass(frozen=True)
class StructuralReanchorEvent:
    """One strongest head at a temporally non-max-suppressed switch row."""

    position: int
    row_index: int
    layer: int
    head: int
    anchor_bucket: int
    score: float
    support: int


@dataclass(frozen=True)
class ReanchorCandidate:
    """One head-specific local-to-anchor change point on the response timeline."""

    layer: int
    head: int
    position: int
    source_kind: str
    source_position: int
    source_unit_id: int
    source_evidence_fraction: float
    previous_local_source_position: int
    previous_local_source_unit_id: int
    score: float
    switch_delta: float
    previous_anchor_fraction: float
    anchor_fraction: float
    prompt_transport: float
    other_prompt_transport: float
    remote_response_transport: float
    previous_local_transport: float
    local_transport: float
    long_range_downstream_action: float
    bucket_downstream_action: float
    source_downstream_action: float
    local_downstream_action: float
    current_target_match: bool
    long_range_immediate_action: float
    bucket_immediate_action: float
    source_immediate_action: float
    selected_root_integration_budget: float
    selected_root_integration_coherence: float
    selected_root_integration_action: float
    dominance_flip: bool
    relative_anchor_rise: float
    relative_local_fall: float
    support: int


@dataclass(frozen=True)
class StructuralReanchorTrace:
    """Transport-only local-to-long-range switch geometry.

    All dense fields except ``row_position`` have shape ``[L,H,P]``.  No
    target gradient, label, sparse edge, or intervention enters this trace.
    """

    row_position: Tensor
    anchor_transport: Tensor
    local_transport: Tensor
    anchor_fraction: Tensor
    switch_delta: Tensor
    relative_anchor_rise: Tensor
    relative_local_fall: Tensor
    dominance_flip: Tensor
    score: Tensor
    anchor_bucket: Tensor


def structural_reanchor_trace(
    source_location: SourceLocationBuckets,
    row_position: Tensor,
) -> StructuralReanchorTrace:
    """Compute the single canonical structural switch definition."""

    transport = source_location.transport.float()
    row_position = torch.as_tensor(row_position, dtype=torch.long).flatten().cpu()
    if transport.ndim != 4 or transport.shape[-1] != 4:
        raise ValueError("source-location transport must have shape [L,H,P,4]")
    if transport.shape[2] != len(row_position):
        raise ValueError("source-location rows do not match row positions")
    if not bool(torch.isfinite(transport).all()) or bool((transport < 0).any()):
        raise ValueError("source-location transport must be finite and non-negative")

    anchor_transport = transport[..., :LOCAL_BUCKET].sum(-1)
    local_transport = transport[..., LOCAL_BUCKET]
    total = anchor_transport + local_transport
    anchor_fraction = torch.where(
        total > 0,
        anchor_transport / total,
        torch.zeros_like(total),
    )
    previous_fraction = torch.zeros_like(anchor_fraction)
    previous_fraction[..., 1:] = anchor_fraction[..., :-1]
    switch_delta = torch.zeros_like(anchor_fraction)
    switch_delta[..., 1:] = anchor_fraction[..., 1:] - anchor_fraction[..., :-1]
    previous_local = torch.zeros_like(local_transport)
    previous_local[..., 1:] = local_transport[..., :-1]
    previous_anchor = torch.zeros_like(anchor_transport)
    previous_anchor[..., 1:] = anchor_transport[..., :-1]
    anchor_rise = (anchor_transport - previous_anchor).clamp_min(0)
    relative_anchor_rise = torch.where(
        anchor_transport + previous_anchor > 0,
        anchor_rise / (anchor_transport + previous_anchor),
        torch.zeros_like(anchor_rise),
    )
    local_fall = (previous_local - local_transport).clamp_min(0)
    relative_local_fall = torch.where(
        previous_local + local_transport > 0,
        local_fall / (previous_local + local_transport),
        torch.zeros_like(local_fall),
    )
    valid_previous = torch.zeros_like(total, dtype=torch.bool)
    valid_previous[..., 1:] = total[..., :-1] > 0
    dominance_flip = (
        valid_previous
        & (previous_fraction < 0.5)
        & (anchor_fraction >= 0.5)
        & (anchor_rise > 0)
        & (local_fall > 0)
    )
    score = torch.where(
        dominance_flip,
        (relative_anchor_rise * relative_local_fall).sqrt(),
        torch.zeros_like(anchor_fraction),
    )
    return StructuralReanchorTrace(
        row_position=row_position,
        anchor_transport=anchor_transport,
        local_transport=local_transport,
        anchor_fraction=anchor_fraction,
        switch_delta=switch_delta,
        relative_anchor_rise=relative_anchor_rise,
        relative_local_fall=relative_local_fall,
        dominance_flip=dominance_flip,
        score=score,
        anchor_bucket=transport[..., :LOCAL_BUCKET].argmax(-1),
    )


def structural_reanchor_peak_mask(
    trace: StructuralReanchorTrace,
    response_start: int,
) -> Tensor:
    """Return the canonical per-head temporal peaks used by all selectors."""

    eligible = (
        trace.dominance_flip
        & (trace.score > 0)
        & (trace.row_position >= response_start)[None, None, :]
    )
    left = torch.zeros_like(trace.score)
    right = torch.zeros_like(trace.score)
    left[..., 1:] = trace.score[..., :-1]
    right[..., :-1] = trace.score[..., 1:]
    return eligible & (trace.score >= left) & (trace.score >= right)


def rank_reanchor_peak_events(
    trace: StructuralReanchorTrace,
    response_start: int,
    *,
    limit: int,
    nms_radius: int = REANCHOR_NMS_RADIUS,
) -> tuple[StructuralReanchorEvent, ...]:
    """Freeze one exact head per switch time, then apply temporal NMS."""

    if limit < 0:
        raise ValueError("event limit must be non-negative")
    if nms_radius < 0:
        raise ValueError("temporal NMS radius cannot be negative")
    if limit == 0:
        return ()
    peak = structural_reanchor_peak_mask(trace, response_start)
    layers, heads, rows = peak.shape
    flat_peak = peak.view(layers * heads, rows)
    flat_score = trace.score.view(layers * heads, rows)
    eligible_score = torch.where(flat_peak, flat_score, -1)
    strongest_score, strongest_head = eligible_score.max(dim=0)
    support = peak.sum(dim=(0, 1))
    events = []
    for row_index in (
        torch.nonzero(strongest_score > 0, as_tuple=False).flatten().tolist()
    ):
        flat_head = int(strongest_head[row_index])
        layer, head = divmod(flat_head, heads)
        events.append(
            StructuralReanchorEvent(
                position=int(trace.row_position[row_index]),
                row_index=row_index,
                layer=layer,
                head=head,
                anchor_bucket=int(trace.anchor_bucket[layer, head, row_index]),
                score=float(strongest_score[row_index]),
                support=int(support[row_index]),
            )
        )
    events.sort(key=lambda event: (-event.score, -event.support, event.position))
    selected: list[StructuralReanchorEvent] = []
    for event in events:
        if any(abs(event.position - keep.position) <= nms_radius for keep in selected):
            continue
        selected.append(event)
        if len(selected) == limit:
            break
    return tuple(selected)


@dataclass(frozen=True)
class ReanchorTimeline:
    """Non-averaged temporal audit tensors and ranked change points.

    Every dense tensor has shape ``[layer, head, represented destination]``.
    Target action is ``gradient · W_O(A V)`` for the audited output margin; it
    is a screening measure of downstream use, not an independently generated
    next-token label for every intermediate destination.
    """

    row_position: Tensor
    prompt_transport: Tensor
    other_prompt_transport: Tensor
    remote_response_transport: Tensor
    local_transport: Tensor
    prompt_downstream_action: Tensor
    other_prompt_downstream_action: Tensor
    remote_response_downstream_action: Tensor
    local_downstream_action: Tensor
    anchor_fraction: Tensor
    switch_delta: Tensor
    relative_anchor_rise: Tensor
    relative_local_fall: Tensor
    score: Tensor
    dominance_flip: Tensor
    anchor_source_kind: Tensor
    anchor_source_position: Tensor
    anchor_source_unit_id: Tensor
    anchor_source_transport: Tensor
    anchor_source_downstream_action: Tensor
    anchor_source_evidence_fraction: Tensor
    local_source_position: Tensor
    local_source_unit_id: Tensor
    candidates: tuple[ReanchorCandidate, ...]


class ReanchorTimelineAuditor:
    """Discover local-to-long-range switches without route-wise ablation."""

    @staticmethod
    def audit(
        flow: PairedFlow,
        dynamics: RouteDynamics,
        world: NativeWorld,
        *,
        limit: int = 32,
    ) -> ReanchorTimeline:
        if limit < 0:
            raise ValueError("candidate limit must be non-negative")
        layers, heads, rows = dynamics.head_transport.shape[:3]
        location = flow.source_location
        if location is None:
            raise ValueError("reanchor audit requires full-row source-location buckets")
        if location.local_window != dynamics.local_window:
            raise ValueError("source-location and route local windows do not match")
        shape = (layers, heads, rows, 4)
        if location.transport.shape != shape:
            raise ValueError("source-location buckets do not match the route rows")
        if (
            location.downstream_action is None
            or location.source_downstream_action is None
        ):
            raise ValueError("target audit requires source-location action buckets")

        structural = structural_reanchor_trace(location, dynamics.row_position)
        prompt_transport = location.transport[..., PROMPT_BUCKET].float()
        other_prompt_transport = location.transport[..., OTHER_PROMPT_BUCKET].float()
        local_transport = location.transport[..., LOCAL_BUCKET].float()
        remote_response_transport = location.transport[..., REMOTE_BUCKET].float()
        prompt_downstream_action = location.downstream_action[
            ..., PROMPT_BUCKET
        ].float()
        other_prompt_downstream_action = location.downstream_action[
            ..., OTHER_PROMPT_BUCKET
        ].float()
        local_downstream_action = location.downstream_action[..., LOCAL_BUCKET].float()
        remote_response_downstream_action = location.downstream_action[
            ..., REMOTE_BUCKET
        ].float()
        anchor_transport = structural.anchor_transport
        anchor_action = (
            prompt_downstream_action
            + other_prompt_downstream_action
            + remote_response_downstream_action
        )
        anchor_fraction = structural.anchor_fraction
        previous_fraction = torch.zeros_like(anchor_fraction)
        previous_fraction[..., 1:] = anchor_fraction[..., :-1]
        switch_delta = structural.switch_delta
        previous_local = torch.zeros_like(local_transport)
        previous_local[..., 1:] = local_transport[..., :-1]
        relative_anchor_rise = structural.relative_anchor_rise
        relative_local_fall = structural.relative_local_fall
        structural_events = rank_reanchor_peak_events(
            structural,
            world.response_start,
            limit=limit,
        )
        score = structural.score
        dominance_flip = structural.dominance_flip

        winner_bucket = structural.anchor_bucket
        gather = winner_bucket[..., None]
        anchor_source_position = (
            location.source_position[..., :LOCAL_BUCKET]
            .gather(-1, gather)[..., 0]
            .to(torch.int32)
        )
        anchor_source_unit = (
            location.source_unit_id[..., :LOCAL_BUCKET]
            .gather(-1, gather)[..., 0]
            .to(torch.int32)
        )
        anchor_source_transport = (
            location.source_transport[..., :LOCAL_BUCKET]
            .gather(-1, gather)[..., 0]
            .float()
        )
        anchor_source_action = (
            location.source_downstream_action[..., :LOCAL_BUCKET]
            .gather(-1, gather)[..., 0]
            .float()
        )
        anchor_bucket_action = (
            location.downstream_action[..., :LOCAL_BUCKET]
            .gather(-1, gather)[..., 0]
            .float()
        )
        anchor_source_kind = torch.where(
            anchor_transport > 0,
            winner_bucket + 1,
            0,
        ).to(torch.int8)
        present_anchor = anchor_source_position >= 0
        local_source_position = location.source_position[..., LOCAL_BUCKET].to(
            torch.int32
        )
        local_source_unit = location.source_unit_id[..., LOCAL_BUCKET].to(torch.int32)

        anchor_source_evidence = torch.zeros_like(anchor_source_transport)
        if bool(present_anchor.any()):
            coordinate = torch.nonzero(present_anchor, as_tuple=False)
            source_layer = coordinate[:, 0]
            source_position = anchor_source_position[present_anchor].long()
            anchor_source_evidence[present_anchor] = dynamics.node_register[
                source_layer,
                source_position,
                EVIDENCE,
            ]

        candidates: list[ReanchorCandidate] = []
        if structural_events:
            for event in structural_events:
                layer_index = event.layer
                head_index = event.head
                row_index = event.row_index
                source_kind = int(
                    anchor_source_kind[layer_index, head_index, row_index]
                )
                integration = dynamics.head_integration[
                    layer_index, head_index, row_index
                ]
                source_matches_root = (
                    source_kind == 1
                    and int(anchor_source_unit[layer_index, head_index, row_index])
                    == dynamics.root_unit_id
                )
                candidates.append(
                    ReanchorCandidate(
                        layer=layer_index,
                        head=head_index,
                        position=int(dynamics.row_position[row_index]),
                        source_kind=SOURCE_KIND_NAMES[source_kind],
                        source_position=int(
                            anchor_source_position[layer_index, head_index, row_index]
                        ),
                        source_unit_id=int(
                            anchor_source_unit[layer_index, head_index, row_index]
                        ),
                        source_evidence_fraction=float(
                            anchor_source_evidence[layer_index, head_index, row_index]
                        ),
                        previous_local_source_position=int(
                            local_source_position[
                                layer_index, head_index, row_index - 1
                            ]
                        ),
                        previous_local_source_unit_id=int(
                            local_source_unit[layer_index, head_index, row_index - 1]
                        ),
                        score=float(score[layer_index, head_index, row_index]),
                        switch_delta=float(
                            switch_delta[layer_index, head_index, row_index]
                        ),
                        previous_anchor_fraction=float(
                            previous_fraction[layer_index, head_index, row_index]
                        ),
                        anchor_fraction=float(
                            anchor_fraction[layer_index, head_index, row_index]
                        ),
                        prompt_transport=float(
                            prompt_transport[layer_index, head_index, row_index]
                        ),
                        other_prompt_transport=float(
                            other_prompt_transport[layer_index, head_index, row_index]
                        ),
                        remote_response_transport=float(
                            remote_response_transport[
                                layer_index, head_index, row_index
                            ]
                        ),
                        previous_local_transport=float(
                            previous_local[layer_index, head_index, row_index]
                        ),
                        local_transport=float(
                            local_transport[layer_index, head_index, row_index]
                        ),
                        long_range_downstream_action=float(
                            anchor_action[layer_index, head_index, row_index]
                        ),
                        bucket_downstream_action=float(
                            anchor_bucket_action[layer_index, head_index, row_index]
                        ),
                        source_downstream_action=float(
                            anchor_source_action[layer_index, head_index, row_index]
                        ),
                        local_downstream_action=float(
                            local_downstream_action[layer_index, head_index, row_index]
                        ),
                        current_target_match=(
                            int(dynamics.row_position[row_index])
                            == int(flow.target.query_position)
                        ),
                        long_range_immediate_action=(
                            float(anchor_action[layer_index, head_index, row_index])
                            if int(dynamics.row_position[row_index])
                            == int(flow.target.query_position)
                            else float("nan")
                        ),
                        bucket_immediate_action=(
                            float(
                                anchor_bucket_action[layer_index, head_index, row_index]
                            )
                            if int(dynamics.row_position[row_index])
                            == int(flow.target.query_position)
                            else float("nan")
                        ),
                        source_immediate_action=(
                            float(
                                anchor_source_action[layer_index, head_index, row_index]
                            )
                            if int(dynamics.row_position[row_index])
                            == int(flow.target.query_position)
                            else float("nan")
                        ),
                        selected_root_integration_budget=(
                            float(integration[0])
                            if source_matches_root
                            else float("nan")
                        ),
                        selected_root_integration_coherence=(
                            float(integration[2])
                            if source_matches_root
                            else float("nan")
                        ),
                        selected_root_integration_action=(
                            float(integration[3])
                            if source_matches_root
                            else float("nan")
                        ),
                        dominance_flip=bool(
                            dominance_flip[layer_index, head_index, row_index]
                        ),
                        relative_anchor_rise=float(
                            relative_anchor_rise[layer_index, head_index, row_index]
                        ),
                        relative_local_fall=float(
                            relative_local_fall[layer_index, head_index, row_index]
                        ),
                        support=event.support,
                    )
                )

        return ReanchorTimeline(
            row_position=dynamics.row_position.clone(),
            prompt_transport=prompt_transport,
            other_prompt_transport=other_prompt_transport,
            remote_response_transport=remote_response_transport,
            local_transport=local_transport,
            prompt_downstream_action=prompt_downstream_action,
            other_prompt_downstream_action=other_prompt_downstream_action,
            remote_response_downstream_action=remote_response_downstream_action,
            local_downstream_action=local_downstream_action,
            anchor_fraction=anchor_fraction,
            switch_delta=switch_delta,
            relative_anchor_rise=relative_anchor_rise,
            relative_local_fall=relative_local_fall,
            score=score,
            dominance_flip=dominance_flip,
            anchor_source_kind=anchor_source_kind,
            anchor_source_position=anchor_source_position,
            anchor_source_unit_id=anchor_source_unit,
            anchor_source_transport=anchor_source_transport,
            anchor_source_downstream_action=anchor_source_action,
            anchor_source_evidence_fraction=anchor_source_evidence,
            local_source_position=local_source_position,
            local_source_unit_id=local_source_unit,
            candidates=tuple(candidates),
        )
