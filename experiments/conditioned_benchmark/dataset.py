"""Alignment and evaluation-only label access through ResearchDataset."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, **_):
        return iterable

from .types import BenchmarkFrame, MethodScore, ScoreArtifact


METADATA_NAMES = ("source_id", "task_type", "data_source", "generator_model")


def _row_map(artifact: ScoreArtifact):
    return {
        (str(sample_id), int(token_index)): index
        for index, (sample_id, token_index) in enumerate(
            zip(artifact.sample_id, artifact.token_index, strict=True)
        )
    }


def align_artifacts(artifacts: list[ScoreArtifact]):
    """Align all methods on the exact same token rows for fair comparison."""

    if not artifacts:
        raise ValueError("at least one artifact is required")
    maps = [_row_map(artifact) for artifact in artifacts]
    common = set(maps[0])
    for mapping in maps[1:]:
        common.intersection_update(mapping)
    if not common:
        raise ValueError("score artifacts have no common token rows")
    ordered = [
        key
        for key in zip(
            artifacts[0].sample_id.tolist(),
            artifacts[0].token_index.tolist(),
            strict=True,
        )
        if (str(key[0]), int(key[1])) in common
    ]
    sample_id = np.asarray([str(key[0]) for key in ordered], dtype=str)
    token_index = np.asarray([int(key[1]) for key in ordered], dtype=np.int64)
    methods: dict[str, MethodScore] = {}
    metadata: dict[str, np.ndarray] = {}

    for artifact, mapping in zip(artifacts, maps, strict=True):
        selected = np.asarray(
            [mapping[(str(sample), int(token))] for sample, token in ordered],
            dtype=np.int64,
        )
        for name, method in artifact.methods.items():
            if name in methods:
                raise ValueError(f"duplicate method name: {name}")
            methods[name] = MethodScore(
                name=name,
                values=method.values[selected],
                direction="higher",
                protocol=method.protocol,
                source_field=method.source_field,
                source_direction=method.source_direction,
            )
        for name, values in artifact.metadata.items():
            candidate = values[selected].astype(str)
            if name not in metadata:
                metadata[name] = candidate
                continue
            left, right = metadata[name], candidate
            conflict = (left != "") & (right != "") & (left != right)
            if bool(conflict.any()):
                raise ValueError(
                    f"artifact metadata conflict for {name} on aligned rows"
                )
            metadata[name] = np.where(left != "", left, right)

    return sample_id, token_index, methods, metadata


def _unlock_labels(dataset):
    try:
        return dataset.labels()
    except RuntimeError as error:
        if "every attention sample" not in str(error):
            raise
    for sample_id in tqdm(
        dataset.sample_ids,
        desc="unlock evaluation labels",
        unit="sample",
    ):
        sample = dataset[sample_id]
        sample.attention()
        sample.release_attention()
    return dataset.labels()


def attach_dataset_evaluation(
    sample_id,
    token_index,
    methods,
    artifact_metadata,
    split_root,
    *,
    device="cpu",
) -> BenchmarkFrame:
    """Read labels only after all frozen score artifacts are loaded and aligned."""

    from research_dataset import open_research_dataset

    dataset = open_research_dataset(
        split_root,
        device=device,
        retain_embedded_labels=True,
    )
    labels = _unlock_labels(dataset)
    row_groups = defaultdict(list)
    for row, value in enumerate(sample_id):
        row_groups[str(value)].append(row)

    length = len(sample_id)
    y = np.empty(length, dtype=np.int8)
    response_length = np.empty(length, dtype=np.int32)
    canonical = {
        name: np.full(length, "", dtype=object) for name in METADATA_NAMES
    }
    for current_id, rows_list in tqdm(
        row_groups.items(), desc="align evaluation labels", unit="sample"
    ):
        if current_id not in dataset:
            raise ValueError(f"score sample {current_id!r} is absent from split")
        rows = np.asarray(rows_list, dtype=np.int64)
        sample = dataset[current_id]
        current_labels = labels.response_labels(sample).detach().cpu().numpy()
        positions = token_index[rows]
        if bool((positions >= len(current_labels)).any()):
            raise ValueError(f"token index exceeds response length for {current_id}")
        y[rows] = current_labels[positions].astype(np.int8)
        response_length[rows] = len(current_labels)
        values = {
            "source_id": sample.source_id,
            "task_type": sample.task_type,
            "data_source": sample.data_source,
            "generator_model": sample.generator_model,
        }
        for name, value in values.items():
            canonical[name][rows] = "" if value is None else str(value)
        sample.release_attention()

    for name in METADATA_NAMES:
        if name not in artifact_metadata:
            continue
        observed = np.asarray(artifact_metadata[name]).astype(str)
        expected = np.asarray(canonical[name]).astype(str)
        conflict = (observed != "") & (expected != "") & (observed != expected)
        if bool(conflict.any()):
            raise ValueError(f"artifact {name} disagrees with ResearchDataset")

    relative_position = token_index.astype(np.float64) / np.maximum(
        response_length - 1, 1
    )
    return BenchmarkFrame(
        sample_id=np.asarray(sample_id).astype(str),
        token_index=np.asarray(token_index).astype(np.int64),
        methods=methods,
        source_id=np.asarray(canonical["source_id"]).astype(str),
        task_type=np.asarray(canonical["task_type"]).astype(str),
        data_source=np.asarray(canonical["data_source"]).astype(str),
        generator_model=np.asarray(canonical["generator_model"]).astype(str),
        response_length=response_length,
        relative_position=relative_position,
        labels=y,
    ).validate()
