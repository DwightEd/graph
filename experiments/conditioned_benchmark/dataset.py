"""Build one canonical benchmark frame from fully evaluated artifacts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import BenchmarkFrame, EvaluatedArtifact, MethodScore


def _row_map(artifact: EvaluatedArtifact):
    score = artifact.score
    return {
        (str(sample_id), int(token_index)): index
        for index, (sample_id, token_index) in enumerate(
            zip(score.sample_id, score.token_index, strict=True)
        )
    }


def _aligned_facts(artifact: EvaluatedArtifact, selected):
    labels = artifact.labels
    length = len(artifact.score.sample_id)
    facts = {
        "labels": np.asarray(labels.token_label),
        "response_positive": np.asarray(labels.response_positive),
        "source_id": np.asarray(labels.source_id).astype(str),
        "response_length": np.asarray(labels.response_length),
    }
    for name, values in facts.items():
        if values.ndim != 1 or len(values) != length:
            raise ValueError(f"evaluated artifact has invalid {name} rows")
    return {name: values[selected] for name, values in facts.items()}


def build_benchmark_frame(
    evaluated_artifacts: list[EvaluatedArtifact], dataset
) -> BenchmarkFrame:
    """Intersect scores after canonical full-artifact facts have been obtained."""

    if not evaluated_artifacts:
        raise ValueError("at least one evaluated artifact is required")
    maps = [_row_map(artifact) for artifact in evaluated_artifacts]
    common = set(maps[0])
    for mapping in maps[1:]:
        common.intersection_update(mapping)
    if not common:
        raise ValueError("score artifacts have no common token rows")

    first_score = evaluated_artifacts[0].score
    ordered = [
        (str(sample_id), int(token_index))
        for sample_id, token_index in zip(
            first_score.sample_id, first_score.token_index, strict=True
        )
        if (str(sample_id), int(token_index)) in common
    ]
    sample_id = np.asarray([sample for sample, _ in ordered], dtype=str)
    token_index = np.asarray([token for _, token in ordered], dtype=np.int64)
    methods: dict[str, MethodScore] = {}
    canonical = None

    for artifact, mapping in zip(evaluated_artifacts, maps, strict=True):
        selected = np.asarray([mapping[key] for key in ordered], dtype=np.int64)
        facts = _aligned_facts(artifact, selected)
        if canonical is None:
            canonical = facts
        else:
            for name in canonical:
                if not np.array_equal(canonical[name], facts[name]):
                    raise ValueError(
                        f"canonical {name} disagrees across common artifact rows"
                    )
        for name, method in artifact.score.methods.items():
            if name in methods:
                raise ValueError(f"duplicate method name: {name}")
            methods[name] = replace(
                method,
                values=method.values[selected],
                direction="higher",
            )

    metadata = {
        "task_type": np.empty(len(sample_id), dtype=object),
        "data_source": np.empty(len(sample_id), dtype=object),
        "generator_model": np.empty(len(sample_id), dtype=object),
    }
    for current in np.unique(sample_id):
        sample = dataset[current]
        selected = sample_id == current
        for name, values in metadata.items():
            value = getattr(sample, name)
            values[selected] = "" if value is None else str(value)

    response_length = canonical["response_length"].astype(np.int64)
    relative_position = token_index.astype(np.float64) / np.maximum(
        response_length - 1, 1
    )
    return BenchmarkFrame(
        sample_id=sample_id,
        token_index=token_index,
        methods=methods,
        source_id=canonical["source_id"].astype(str),
        task_type=metadata["task_type"].astype(str),
        data_source=metadata["data_source"].astype(str),
        generator_model=metadata["generator_model"].astype(str),
        response_length=response_length,
        relative_position=relative_position,
        labels=canonical["labels"].astype(np.int8),
        response_positive=canonical["response_positive"].astype(np.int8),
    ).validate()
