"""Label-free cohort, sample, source-unit, and target construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from experiments.common.llama_message_intervention import baseline_forward
from experiments.common.ragtruth_alignment import canonical_task_type

from .native_world import NativeWorld
from .units import build_source_units
from .worlds import TargetContrast


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            "target policy must be uncertain, low-margin, evenly-spaced, or all"
        )
    return tuple(order[:count])


def freeze_targets(
    model,
    token_ids: torch.Tensor,
    response_start: int,
    *,
    count: int,
    policy: str,
    query_chunk: int,
) -> tuple[TargetContrast, ...]:
    """Freeze observed-token versus native-runner contrasts before any cut."""

    cache = baseline_forward(
        model,
        token_ids,
        response_start,
        checkpoint_layers=(0,),
        attention_query_chunk=query_chunk,
    )
    slots = target_slots(cache, count=count, policy=policy)
    targets = tuple(
        TargetContrast(
            int(cache.query[slot]),
            int(cache.target[slot]),
            int(cache.runner[slot]),
            f"label_free_{policy}_observed_token_vs_native_runner",
        )
        for slot in slots
    )
    del cache
    return targets


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
) -> NativeWorld:
    """Detach one cache sample, align units, and freeze native targets."""

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
    targets = freeze_targets(
        model,
        token_ids,
        response_start,
        count=targets_per_sample,
        policy=target_policy,
        query_chunk=query_chunk,
    )
    return NativeWorld(
        safe_sample_key(record.sample_id),
        Path(tokenizer.name_or_path).name,
        token_ids,
        response_start,
        units,
        evidence_units,
        targets,
    ).check()
