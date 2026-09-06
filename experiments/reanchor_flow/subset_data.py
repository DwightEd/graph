"""Label-free cohort, sample, source-unit, and target construction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.common.llama_message_intervention import baseline_forward
from experiments.common.ragtruth_alignment import canonical_task_type

from .flow import SourceLocationBuckets
from .native_flow import capture_source_location_buckets
from .native_world import NativeWorld, TargetReanchorSelection
from .reanchor_timeline import (
    SOURCE_KIND_NAMES,
    StructuralReanchorEvent,
    StructuralReanchorTrace,
    rank_reanchor_peak_events,
    structural_reanchor_trace,
)
from .units import build_source_units
from .worlds import SourceUnits, TargetContrast

REANCHOR_POLICIES = ("reanchor", "reanchor-window")
REANCHOR_NMS_RADIUS = 1


@dataclass(frozen=True)
class SampleRecord:
    """Metadata allowed to select a sample before labels are opened."""

    sample_id: str
    source_id: str
    task_type: str
    generator_model: str


def _hash_rank(seed: int, *parts: str) -> bytes:
    value = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).digest()


def safe_sample_key(sample_id: str) -> str:
    if (
        sample_id
        and Path(sample_id).name == sample_id
        and "\\" not in sample_id
        and sample_id not in {".", ".."}
    ):
        return sample_id
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return f"sample-{digest}"


def _record_task(
    sample_id: str,
    cached_task,
    source: Mapping | None,
) -> str:
    cache_value = None
    if cached_task is not None and str(cached_task).strip():
        cache_value = canonical_task_type(cached_task)
    source_value = None if source is None else canonical_task_type(source["task_type"])
    if (
        cache_value is not None
        and source_value is not None
        and cache_value != source_value
    ):
        raise ValueError(f"sample {sample_id} and source_info disagree on task type")
    task = source_value or cache_value
    if task is None:
        raise ValueError(
            f"sample {sample_id} has no task type in formal metadata or source_info"
        )
    return task


def inspect_records(
    dataset,
    *,
    sample_ids: Iterable[str] | None = None,
    source_info: Mapping[str, Mapping] | None = None,
) -> tuple[SampleRecord, ...]:
    """Read allow-listed metadata, optionally for explicit IDs only.

    Formal datasets provide a memory-mapped metadata path that does not
    dereference co-located attention or label tensors. ``source_info`` is the
    authoritative task fallback; when both sources contain a task, they must
    agree.
    """

    records = []
    available = tuple(map(str, dataset.sample_ids))
    selected = available if sample_ids is None else tuple(map(str, sample_ids))
    available_set = set(available)
    missing = [sample_id for sample_id in selected if sample_id not in available_set]
    if missing:
        raise ValueError(f"sample IDs not found: {', '.join(missing)}")
    metadata_reader = getattr(dataset, "metadata", None)
    for sample_id in selected:
        if callable(metadata_reader):
            metadata = metadata_reader(sample_id)
            source_id = str(metadata["source_id"])
            cached_task = metadata.get("task_type")
            generator_model = str(metadata.get("generator_model") or "")
        else:
            sample = dataset[sample_id]
            try:
                source_id = str(sample.source_id)
                cached_task = sample.task_type
                generator_model = str(getattr(sample, "generator_model", "") or "")
            finally:
                sample.release_attention()
        source = None if source_info is None else source_info.get(source_id)
        records.append(
            SampleRecord(
                sample_id,
                source_id,
                _record_task(sample_id, cached_task, source),
                generator_model,
            )
        )
    return tuple(records)


def select_records(
    records: Iterable[SampleRecord],
    *,
    tasks: tuple[str, ...],
    samples_per_task: int,
    seed: int,
    sample_ids: tuple[str, ...] = (),
) -> tuple[SampleRecord, ...]:
    """Choose a deterministic source-diverse cohort without labels."""

    records = tuple(records)
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("dataset contains duplicate sample IDs")
    if sample_ids:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs not found: {', '.join(missing)}")
        selected = tuple(by_id[sample_id] for sample_id in sample_ids)
        invalid = [item.sample_id for item in selected if item.task_type not in tasks]
        if invalid:
            raise ValueError(
                "explicit samples fall outside --task: " + ", ".join(invalid)
            )
        return selected

    selected: list[SampleRecord] = []
    for task in tasks:
        candidates = [record for record in records if record.task_type == task]
        if len(candidates) < samples_per_task:
            raise ValueError(
                f"task {task} has {len(candidates)} available samples; "
                f"{samples_per_task} requested"
            )
        by_source: dict[str, list[SampleRecord]] = {}
        for record in candidates:
            by_source.setdefault(record.source_id, []).append(record)
        primary = []
        for source_id, source_records in by_source.items():
            representative = min(
                source_records,
                key=lambda item: _hash_rank(seed, task, source_id, item.sample_id),
            )
            primary.append(representative)
        primary.sort(key=lambda item: _hash_rank(seed, task, item.source_id))
        chosen = primary[:samples_per_task]
        if len(chosen) < samples_per_task:
            chosen_ids = {record.sample_id for record in chosen}
            remaining = [
                record for record in candidates if record.sample_id not in chosen_ids
            ]
            remaining.sort(
                key=lambda item: _hash_rank(seed, task, item.source_id, item.sample_id)
            )
            chosen.extend(remaining[: samples_per_task - len(chosen)])
        selected.extend(chosen)
    if not selected:
        raise ValueError("no samples match the requested task subset")
    return tuple(selected)


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
) -> tuple[tuple[TargetContrast, ...], tuple[TargetReanchorSelection, ...]]:
    """Freeze target contrasts and structured re-anchor provenance.

    Re-anchor policies reuse the clean baseline's layer inputs to scan exact
    current-model full rows.  They do not interpret absent entries in the
    historical sparse attention cache as zeros, so cache censoring and an
    observer-model mismatch cannot manufacture a target-selection event.
    """

    reanchor_policy = policy in REANCHOR_POLICIES
    needs_reanchor_scan = reanchor_policy and len(token_ids) - response_start > 1
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
    if reanchor_policy:
        response_positions = cache.query[cache.query >= response_start]
        events: tuple[StructuralReanchorEvent, ...] = ()
        if len(response_positions):
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


def load_world_from_dataset(
    dataset,
    record: SampleRecord,
    source: dict,
    tokenizer,
    model,
    *,
    max_response_tokens: int | None,
    targets_per_sample: int,
    target_policy: str,
    query_chunk: int,
    local_window: int = 10,
) -> NativeWorld:
    """Detach cache token IDs, align units, and freeze native targets."""

    sample = dataset[record.sample_id]
    try:
        cached = sample.attention()
        token_ids = cached.token_ids.detach().cpu().long().clone()
        response_start = int(cached.response_idx)
    finally:
        sample.release_attention()
    if max_response_tokens is not None:
        token_ids = token_ids[: response_start + max_response_tokens]
    if len(token_ids) <= response_start:
        raise ValueError(f"sample {record.sample_id} has an empty response")
    if canonical_task_type(source["task_type"]) != record.task_type:
        raise ValueError(
            f"sample {record.sample_id} and source_info disagree on task type"
        )
    units = build_source_units(source, tokenizer, token_ids, response_start)
    evidence_units = tuple(
        unit_id
        for unit_id, kind in enumerate(units.kind)
        if kind not in {"other_prompt", "response"}
        and bool((units.token_unit_id == unit_id).any())
    )
    targets, target_selection = freeze_target_plan(
        model,
        token_ids,
        response_start,
        count=targets_per_sample,
        policy=target_policy,
        query_chunk=query_chunk,
        units=units,
        evidence_unit_id=evidence_units,
        local_window=local_window,
    )
    return NativeWorld(
        safe_sample_key(record.sample_id),
        Path(tokenizer.name_or_path).name,
        token_ids,
        response_start,
        units,
        evidence_units,
        targets,
        target_selection,
    ).check()
