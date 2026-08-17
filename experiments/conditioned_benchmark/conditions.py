"""Condition grids, prevalence controls, and response aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .types import BenchmarkFrame, MethodScore


@dataclass(frozen=True)
class Condition:
    identifier: str
    task_type: str | None
    data_source: str | None
    generator_model: str | None
    target_positive_rate: float | None


def parse_positive_rate(value):
    if value is None or str(value).lower() == "native":
        return None
    rate = float(value)
    if not 0.0 < rate < 1.0:
        raise ValueError("positive rates must be in (0,1) or 'native'")
    return rate


def _expand(values, observed):
    values = [str(value) for value in values]
    result: list[str | None] = []
    for value in values:
        lowered = value.lower()
        if lowered == "all":
            result.append(None)
        elif lowered == "each":
            result.extend(sorted(set(map(str, observed))))
        else:
            result.append(value)
    deduplicated = []
    for value in result:
        if value not in deduplicated:
            deduplicated.append(value)
    return deduplicated


def condition_grid(
    frame: BenchmarkFrame,
    *,
    tasks,
    data_sources,
    generator_models,
    positive_rates,
):
    tasks = _expand(tasks, frame.task_type)
    data_sources = _expand(data_sources, frame.data_source)
    generators = _expand(generator_models, frame.generator_model)
    rates = [parse_positive_rate(value) for value in positive_rates]
    conditions = []
    for index, (task, source, generator, rate) in enumerate(
        product(tasks, data_sources, generators, rates)
    ):
        parts = [
            f"task={task or 'ALL'}",
            f"source={source or 'ALL'}",
            f"generator={generator or 'ALL'}",
            f"positive_rate={'native' if rate is None else f'{rate:.6g}'}",
        ]
        conditions.append(
            Condition(
                identifier=f"c{index:04d}|" + "|".join(parts),
                task_type=task,
                data_source=source,
                generator_model=generator,
                target_positive_rate=rate,
            )
        )
    return conditions


def condition_mask(frame: BenchmarkFrame, condition: Condition):
    mask = np.ones(len(frame.labels), dtype=bool)
    for value, observed in (
        (condition.task_type, frame.task_type),
        (condition.data_source, frame.data_source),
        (condition.generator_model, frame.generator_model),
    ):
        if value is not None:
            mask &= observed == value
    return mask


def prevalence_weights(labels, target_positive_rate):
    labels = np.asarray(labels, dtype=np.int8)
    weights = np.ones(len(labels), dtype=np.float64)
    if target_positive_rate is None:
        return weights
    native = float(labels.mean())
    if not 0.0 < native < 1.0:
        raise ValueError("prevalence control requires both label classes")
    target = float(target_positive_rate)
    weights[labels == 1] = target / native
    weights[labels == 0] = (1.0 - target) / (1.0 - native)
    return weights


def stratified_subsample(labels, target_positive_rate, *, seed):
    labels = np.asarray(labels, dtype=np.int8)
    if target_positive_rate is None:
        return np.arange(len(labels), dtype=np.int64)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("subsampling requires both label classes")
    target = float(target_positive_rate)
    if len(positive) / len(labels) > target:
        negative_count = len(negative)
        positive_count = min(
            len(positive), max(1, round(negative_count * target / (1 - target)))
        )
    else:
        positive_count = len(positive)
        negative_count = min(
            len(negative), max(1, round(positive_count * (1 - target) / target))
        )
    generator = np.random.default_rng(seed)
    selected = np.concatenate(
        (
            generator.choice(positive, positive_count, replace=False),
            generator.choice(negative, negative_count, replace=False),
        )
    )
    return np.sort(selected)


def aggregate_responses(
    frame: BenchmarkFrame,
    *,
    aggregation="max",
    top_fraction=0.10,
) -> BenchmarkFrame:
    if aggregation not in {"max", "mean", "topk_mean"}:
        raise ValueError("response aggregation must be max, mean, or topk_mean")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")
    groups = []
    seen = set()
    for sample in frame.sample_id:
        if sample not in seen:
            seen.add(sample)
            groups.append(str(sample))

    def reduce(values, rows):
        selected = np.asarray(values)[rows]
        if aggregation == "max":
            return float(np.max(selected))
        if aggregation == "mean":
            return float(np.mean(selected))
        count = max(1, int(np.ceil(len(selected) * float(top_fraction))))
        return float(np.mean(np.partition(selected, -count)[-count:]))

    rows_by_group = {name: np.flatnonzero(frame.sample_id == name) for name in groups}
    methods = {
        name: MethodScore(
            name=name,
            values=np.asarray(
                [reduce(method.values, rows_by_group[group]) for group in groups]
            ),
            direction="higher",
            protocol=method.protocol,
            source_field=method.source_field,
            source_direction=method.source_direction,
        )
        for name, method in frame.methods.items()
    }
    first = np.asarray([rows_by_group[group][0] for group in groups])
    return BenchmarkFrame(
        sample_id=np.asarray(groups, dtype=str),
        token_index=np.zeros(len(groups), dtype=np.int64),
        methods=methods,
        source_id=frame.source_id[first],
        task_type=frame.task_type[first],
        data_source=frame.data_source[first],
        generator_model=frame.generator_model[first],
        response_length=frame.response_length[first],
        relative_position=np.ones(len(groups), dtype=np.float64),
        labels=np.asarray(
            [int(frame.labels[rows_by_group[group]].max()) for group in groups],
            dtype=np.int8,
        ),
    ).validate()
