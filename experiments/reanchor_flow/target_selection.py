"""Label-free target contrasts and structural re-anchor selection."""

from __future__ import annotations

from collections.abc import Callable

import torch

from experiments.common.llama_message_intervention import baseline_forward
from .flow import SourceLocationBuckets
from .native_flow import capture_source_location_buckets
from .native_world import TargetReanchorSelection
from .reanchor_timeline import (
    SOURCE_KIND_NAMES,
    StructuralReanchorEvent,
    StructuralReanchorTrace,
    rank_reanchor_peak_events,
    structural_reanchor_trace,
)
from .sample_scan import SampleScan
from .worlds import SourceUnits, TargetContrast

REANCHOR_POLICIES = ("reanchor", "reanchor-window")
REANCHOR_NMS_RADIUS = 1


def reanchor_target_positions(
    events: tuple[StructuralReanchorEvent, ...],
    available_positions: torch.Tensor,
    *,
    count: int,
    policy: str,
) -> tuple[tuple[int, StructuralReanchorEvent, int], ...]:
    """Allocate a hard target-row budget to event centers or event windows.

    ``reanchor`` spends one row per ranked event center. ``reanchor-window``
    reserves at most ``ceil(count / 3)`` centers, then fills their ``-1`` and
    ``+1`` neighbors in event-rank order.  The returned offset records whether
    a row is the center or temporal context for that event.
    """

    if policy not in REANCHOR_POLICIES:
        raise ValueError("re-anchor target policy is invalid")
    if count < 1:
        raise ValueError("target count must be positive")
    available = tuple(
        int(position)
        for position in torch.as_tensor(available_positions, dtype=torch.long)
        .flatten()
        .tolist()
    )
    available_set = set(available)
    if policy == "reanchor":
        centers = events[:count]
    else:
        event_budget = max(1, (count + 2) // 3)
        centers = events[:event_budget]

    selected: list[tuple[int, StructuralReanchorEvent, int]] = []
    used: set[int] = set()

    def add(event: StructuralReanchorEvent, offset: int) -> None:
        position = event.position + offset
        if len(selected) >= count or position not in available_set or position in used:
            return
        selected.append((position, event, offset))
        used.add(position)

    for event in centers:
        add(event, 0)
    if policy == "reanchor-window":
        for offset in (-1, 1):
            for event in centers:
                add(event, offset)
    return tuple(selected)


def target_slots(cache, *, count: int, policy: str) -> tuple[int, ...]:
    """Select response decisions using clean-model quantities only."""

    available = len(cache.query)
    if available < 1:
        raise ValueError("teacher-forced sample has no response target")
    if policy == "all" or count >= available:
        return tuple(range(available))
    if policy == "evenly-spaced":
        return tuple(
            min(available - 1, int((index + 0.5) * available / count))
            for index in range(count)
        )
    margin = cache.full_margin.float()
    if policy == "uncertain":
        order = sorted(
            range(available),
            key=lambda index: (abs(float(margin[index])), index),
        )
    elif policy == "low-margin":
        order = sorted(
            range(available),
            key=lambda index: (float(margin[index]), index),
        )
    else:
        raise ValueError(
            "margin target policy must be uncertain, low-margin, evenly-spaced, or all"
        )
    return tuple(order[:count])


def _reanchor_origin(
    policy: str,
    event: StructuralReanchorEvent,
    offset: int,
) -> str:
    anchor_kind = SOURCE_KIND_NAMES[event.anchor_bucket + 1]
    return (
        f"label_free_{policy}_exact_full_row_"
        f"l{event.layer}_h{event.head}_{anchor_kind}_"
        f"center_q{event.position}_offset{offset:+d}_"
        f"score{event.score:.8g}_support{event.support}_"
        "observed_token_vs_native_runner"
    )


def _event_selection(
    policy: str,
    query_position: int,
    event: StructuralReanchorEvent,
    offset: int,
    source_location: SourceLocationBuckets,
    trace: StructuralReanchorTrace,
) -> TargetReanchorSelection:
    row = event.row_index
    bucket = event.anchor_bucket
    previous_fraction = (
        0.0
        if row == 0
        else float(trace.anchor_fraction[event.layer, event.head, row - 1])
    )
    return TargetReanchorSelection(
        query_position=query_position,
        policy=policy,
        has_event=True,
        fallback=False,
        center_position=event.position,
        window_offset=offset,
        layer=event.layer,
        head=event.head,
        source_kind=SOURCE_KIND_NAMES[bucket + 1],
        source_position=int(
            source_location.source_position[event.layer, event.head, row, bucket]
        ),
        source_unit_id=int(
            source_location.source_unit_id[event.layer, event.head, row, bucket]
        ),
        score=event.score,
        support=event.support,
        previous_anchor_fraction=previous_fraction,
        anchor_fraction=float(trace.anchor_fraction[event.layer, event.head, row]),
        relative_anchor_rise=float(
            trace.relative_anchor_rise[event.layer, event.head, row]
        ),
        relative_local_fall=float(
            trace.relative_local_fall[event.layer, event.head, row]
        ),
    )


def _non_event_selection(
    query_position: int,
    policy: str,
    *,
    fallback: bool,
) -> TargetReanchorSelection:
    return TargetReanchorSelection(
        query_position=query_position,
        policy=policy,
        has_event=False,
        fallback=fallback,
        center_position=-1,
        window_offset=0,
        layer=-1,
        head=-1,
        source_kind="none",
        source_position=-1,
        source_unit_id=-1,
        score=0.0,
        support=0,
        previous_anchor_fraction=0.0,
        anchor_fraction=0.0,
        relative_anchor_rise=0.0,
        relative_local_fall=0.0,
    )


def freeze_target_plan(
    model,
    token_ids: torch.Tensor,
    response_start: int,
    *,
    count: int,
    policy: str,
    query_chunk: int,
    units: SourceUnits | None = None,
    evidence_unit_id: tuple[int, ...] = (),
    local_window: int = 10,
    on_scan: Callable[[SampleScan], None] | None = None,
) -> tuple[tuple[TargetContrast, ...], tuple[TargetReanchorSelection, ...]]:
    """Freeze target contrasts and structured re-anchor provenance.

    Re-anchor policies reuse the clean baseline's layer inputs to scan exact
    current-model full rows.  They do not interpret absent entries in the
    historical sparse attention cache as zeros, so cache censoring and an
    observer-model mismatch cannot manufacture a target-selection event.
    """

    reanchor_policy = policy in REANCHOR_POLICIES
    needs_reanchor_scan = reanchor_policy or on_scan is not None
    if needs_reanchor_scan and (units is None or not evidence_unit_id):
        raise ValueError("re-anchor target selection requires source-unit evidence")
    cache = baseline_forward(
        model,
        token_ids,
        response_start,
        checkpoint_layers=(
            range(len(model.model.layers)) if needs_reanchor_scan else (0,)
        ),
        attention_query_chunk=query_chunk,
    )
    origin_by_slot: dict[int, str] = {}
    selection_by_slot: dict[int, TargetReanchorSelection] = {}
    if needs_reanchor_scan:
        response_positions = cache.query
        source_location = capture_source_location_buckets(
            model,
            cache,
            units,
            response_positions,
            response_start=response_start,
            evidence_unit_id=evidence_unit_id,
            local_window=local_window,
            query_chunk=query_chunk,
        )
        trace = structural_reanchor_trace(source_location, response_positions)
        if on_scan is not None:
            on_scan(SampleScan(source_location, trace))
    if reanchor_policy:
        events: tuple[StructuralReanchorEvent, ...] = ()
        if len(response_positions):
            events = rank_reanchor_peak_events(
                trace,
                response_start,
                limit=len(response_positions),
                nms_radius=REANCHOR_NMS_RADIUS,
            )
        if count >= len(cache.query):
            slots = tuple(range(len(cache.query)))
            event_by_position = {event.position: event for event in events}
            for slot in slots:
                position = int(cache.query[slot])
                event = event_by_position.get(position)
                if event is None:
                    origin_by_slot[slot] = (
                        f"label_free_{policy}_budget_covers_all_exact_full_row_"
                        "observed_token_vs_native_runner"
                    )
                    selection_by_slot[slot] = _non_event_selection(
                        position, policy, fallback=False
                    )
                else:
                    origin_by_slot[slot] = _reanchor_origin(policy, event, 0)
                    selection_by_slot[slot] = _event_selection(
                        policy,
                        position,
                        event,
                        0,
                        source_location,
                        trace,
                    )
        else:
            selected = reanchor_target_positions(
                events,
                cache.query,
                count=count,
                policy=policy,
            )
            if selected:
                slot_by_position = {
                    int(position): slot for slot, position in enumerate(cache.query)
                }
                slots = tuple(slot_by_position[position] for position, _, _ in selected)
                origin_by_slot = {
                    slot_by_position[position]: _reanchor_origin(policy, event, offset)
                    for position, event, offset in selected
                }
                selection_by_slot = {
                    slot_by_position[position]: _event_selection(
                        policy,
                        position,
                        event,
                        offset,
                        source_location,
                        trace,
                    )
                    for position, event, offset in selected
                }
            else:
                slots = target_slots(cache, count=count, policy="evenly-spaced")
                for slot in slots:
                    origin_by_slot[slot] = (
                        f"label_free_reanchor_fallback_{policy}_no_event_"
                        "evenly_spaced_"
                        "exact_full_row_observed_token_vs_native_runner"
                    )
                    selection_by_slot[slot] = _non_event_selection(
                        int(cache.query[slot]), policy, fallback=True
                    )
    else:
        slots = target_slots(cache, count=count, policy=policy)
    targets = tuple(
        TargetContrast(
            int(cache.query[slot]),
            int(cache.target[slot]),
            int(cache.runner[slot]),
            origin_by_slot.get(
                slot,
                f"label_free_{policy}_observed_token_vs_native_runner",
            ),
        )
        for slot in slots
    )
    selections = (
        tuple(selection_by_slot[slot] for slot in slots) if reanchor_policy else ()
    )
    del cache
    return targets, selections
