"""Artifact contracts for GroundedRoute.

The training attention cache remains the source of truth.  ``GraphSpec`` only
freezes a dataset selection and graph configuration; it does not copy the
training graphs.  Encoded test or calibration graphs are stored per sample so
that the learned node representation and its exact typed topology stay
together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiment_protocol import file_sha256, validate_complete_token_rows


ARTIFACT_VERSION = 1
GRAPH_SPEC_SCHEMA = "grounded-route-graph-spec"
CHECKPOINT_SCHEMA = "grounded-route-checkpoint"
ENCODED_GRAPH_SCHEMA = "grounded-route-encoded-token-graph"
EMBEDDING_INDEX_SCHEMA = "grounded-route-embedding-index"
SCORE_SCHEMA = "grounded-route-token-score"


def sha256(path: str | Path) -> str:
    """Return the content identity of one artifact."""

    return file_sha256(path)


@dataclass(frozen=True)
class GraphSpec:
    """A lightweight, label-free selection over a canonical attention split."""

    dataset_root: str
    dataset_manifest_sha256: str
    split: str
    task: str
    sample_ids: tuple[str, ...]
    layer_count: int
    head_count: int
    graph_config: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("graph spec must select at least one sample")
        if self.layer_count < 1 or self.head_count < 1:
            raise ValueError("graph spec must have positive layer/head geometry")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_SPEC_SCHEMA,
            "version": ARTIFACT_VERSION,
            "labels_included": False,
            **asdict(self),
            "sample_ids": list(self.sample_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GraphSpec":
        _require_schema(payload, GRAPH_SPEC_SCHEMA)
        if bool(payload.get("labels_included", True)):
            raise ValueError("graph spec must not contain labels")
        return cls(
            dataset_root=str(payload["dataset_root"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            split=str(payload["split"]),
            task=str(payload["task"]),
            sample_ids=tuple(map(str, payload["sample_ids"])),
            layer_count=int(payload["layer_count"]),
            head_count=int(payload["head_count"]),
            graph_config=dict(payload["graph_config"]),
        )


@dataclass(frozen=True)
class EncodedTokenGraph:
    """One token graph with learned node states and one mechanism tensor.

    ``lineage[..., 0:3]`` is the prompt / response-closed / unresolved (P/C/U)
    path decomposition.  It is the only mechanism-specific tensor persisted
    beside the reusable node embedding.
    """

    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    layer_count: int
    head_count: int
    attention_floor: float
    token_ids: torch.Tensor
    node_embedding: torch.Tensor
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_head: torch.Tensor
    edge_weight: torch.Tensor
    diagonal: torch.Tensor
    unresolved: torch.Tensor
    lineage: torch.Tensor

    def __post_init__(self) -> None:
        tokens = int(self.token_ids.numel())
        responses = tokens - int(self.response_start)
        edges = int(self.edge_weight.numel())
        geometry = (responses, self.layer_count, self.head_count)
        if self.token_ids.ndim != 1 or self.node_embedding.ndim != 2:
            raise ValueError("token IDs and node embeddings must be [N] and [N,D]")
        if len(self.node_embedding) != tokens or not 0 < self.response_start < tokens:
            raise ValueError("node embeddings must align with prompt-response tokens")
        if self.edge_index.shape != (2, edges):
            raise ValueError("edge_index must be [2,E]")
        if any(
            value.shape != (edges,)
            for value in (self.edge_layer, self.edge_head, self.edge_weight)
        ):
            raise ValueError("typed edge vectors must be aligned")
        if self.diagonal.shape != geometry or self.unresolved.shape != geometry:
            raise ValueError("diagonal and unresolved mass must be [R,L,H]")
        if self.lineage.shape != (*geometry, 3):
            raise ValueError("lineage must be [R,L,H,3]")
        if edges:
            source, target = self.edge_index
            if bool(
                (
                    (source < 0)
                    | (target >= tokens)
                    | (source >= target)
                    | (target < self.response_start)
                    | (self.edge_layer < 0)
                    | (self.edge_layer >= self.layer_count)
                    | (self.edge_head < 0)
                    | (self.edge_head >= self.head_count)
                ).any()
            ):
                raise ValueError("encoded graph contains an invalid typed causal edge")

    @property
    def response_count(self) -> int:
        return int(self.token_ids.numel()) - self.response_start

    @property
    def response_embedding(self) -> torch.Tensor:
        return self.node_embedding[self.response_start :]

    @classmethod
    def from_output(cls, graph, output) -> "EncodedTokenGraph":
        """Pack the public ``TokenGraph`` and ``EncoderOutput`` contracts."""

        return cls(
            sample_id=graph.sample_id,
            source_id=graph.source_id,
            task_type=graph.task_type,
            response_start=graph.response_start,
            layer_count=graph.layer_count,
            head_count=graph.head_count,
            attention_floor=graph.attention_floor,
            token_ids=graph.token_ids,
            node_embedding=output.node_embedding,
            edge_index=torch.stack((graph.edges.source, graph.edges.target)),
            edge_layer=graph.edges.layer,
            edge_head=graph.edges.head,
            edge_weight=graph.edges.weight,
            diagonal=graph.diagonal,
            unresolved=graph.unresolved,
            lineage=output.lineage,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": ENCODED_GRAPH_SCHEMA,
            "version": ARTIFACT_VERSION,
            "labels_included": False,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "task_type": self.task_type,
            "response_start": self.response_start,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "attention_floor": self.attention_floor,
            "token_ids": _cpu(self.token_ids, torch.int32),
            "node_embedding": _cpu(self.node_embedding, torch.float32),
            "edge_index": _cpu(self.edge_index, torch.int32),
            "edge_layer": _cpu(self.edge_layer, torch.int16),
            "edge_head": _cpu(self.edge_head, torch.int16),
            "edge_weight": _cpu(self.edge_weight, torch.float16),
            "diagonal": _cpu(self.diagonal, torch.float16),
            "unresolved": _cpu(self.unresolved, torch.float16),
            "lineage": _cpu(self.lineage, torch.float16),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EncodedTokenGraph":
        _require_schema(payload, ENCODED_GRAPH_SCHEMA)
        if bool(payload.get("labels_included", True)):
            raise ValueError("encoded token graph must not contain labels")
        return cls(
            sample_id=str(payload["sample_id"]),
            source_id=str(payload["source_id"]),
            task_type=str(payload["task_type"]),
            response_start=int(payload["response_start"]),
            layer_count=int(payload["layer_count"]),
            head_count=int(payload["head_count"]),
            attention_floor=float(payload["attention_floor"]),
            token_ids=torch.as_tensor(payload["token_ids"]).long(),
            node_embedding=torch.as_tensor(payload["node_embedding"]).float(),
            edge_index=torch.as_tensor(payload["edge_index"]).long(),
            edge_layer=torch.as_tensor(payload["edge_layer"]).long(),
            edge_head=torch.as_tensor(payload["edge_head"]).long(),
            edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
            diagonal=torch.as_tensor(payload["diagonal"]).float(),
            unresolved=torch.as_tensor(payload["unresolved"]).float(),
            lineage=torch.as_tensor(payload["lineage"]).float(),
        )


@dataclass(frozen=True)
class EmbeddingIndex:
    """Merged response-token embeddings consumed by one-class detection."""

    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    response_token_id: np.ndarray
    embedding: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.sample_id)
        vectors = (
            self.sample_id,
            self.source_id,
            self.task_type,
            self.token_index,
            self.response_length,
            self.response_token_id,
        )
        if self.embedding.ndim != 2 or any(value.ndim != 1 for value in vectors):
            raise ValueError("embedding index columns must be row-aligned vectors")
        if len(self.embedding) != rows or any(len(value) != rows for value in vectors):
            raise ValueError("embedding index columns have different row counts")
        validate_complete_token_rows(
            self.sample_id,
            self.source_id,
            self.token_index,
            self.response_length,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "sample_id": np.asarray(self.sample_id).astype(str),
            "source_id": np.asarray(self.source_id).astype(str),
            "task_type": np.asarray(self.task_type).astype(str),
            "token_index": np.asarray(self.token_index, dtype=np.int32),
            "response_length": np.asarray(self.response_length, dtype=np.int32),
            "response_token_id": np.asarray(self.response_token_id, dtype=np.int64),
            "embedding": np.asarray(self.embedding, dtype=np.float32),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "EmbeddingIndex":
        return cls(
            sample_id=np.asarray(arrays["sample_id"]).astype(str),
            source_id=np.asarray(arrays["source_id"]).astype(str),
            task_type=np.asarray(arrays["task_type"]).astype(str),
            token_index=np.asarray(arrays["token_index"], dtype=np.int32),
            response_length=np.asarray(arrays["response_length"], dtype=np.int32),
            response_token_id=np.asarray(arrays["response_token_id"], dtype=np.int64),
            embedding=np.asarray(arrays["embedding"], dtype=np.float32),
        )


def merge_embedding_index(graphs: Iterable[EncodedTokenGraph]) -> EmbeddingIndex:
    """Concatenate response-node embeddings without adding derived features."""

    blocks = list(graphs)
    if not blocks:
        raise ValueError("embedding index needs at least one encoded graph")
    dimensions = {int(graph.node_embedding.shape[1]) for graph in blocks}
    if len(dimensions) != 1:
        raise ValueError("encoded graphs use different embedding dimensions")

    return EmbeddingIndex(
        sample_id=np.concatenate(
            [np.repeat(graph.sample_id, graph.response_count) for graph in blocks]
        ),
        source_id=np.concatenate(
            [np.repeat(graph.source_id, graph.response_count) for graph in blocks]
        ),
        task_type=np.concatenate(
            [np.repeat(graph.task_type, graph.response_count) for graph in blocks]
        ),
        token_index=np.concatenate(
            [np.arange(graph.response_count, dtype=np.int32) for graph in blocks]
        ),
        response_length=np.concatenate(
            [
                np.full(graph.response_count, graph.response_count, dtype=np.int32)
                for graph in blocks
            ]
        ),
        response_token_id=np.concatenate(
            [
                graph.token_ids[graph.response_start :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                for graph in blocks
            ]
        ),
        embedding=np.concatenate(
            [graph.response_embedding.detach().cpu().numpy() for graph in blocks]
        ).astype(np.float32),
    )


def save_graph_spec(path: str | Path, spec: GraphSpec) -> None:
    _save_json(path, spec.payload())


def load_graph_spec(path: str | Path) -> GraphSpec:
    return GraphSpec.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    content = dict(payload)
    content.update(
        schema=CHECKPOINT_SCHEMA,
        version=ARTIFACT_VERSION,
        labels_included=False,
    )
    _save_torch(path, content)


def load_checkpoint(path: str | Path, *, map_location="cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    _require_schema(payload, CHECKPOINT_SCHEMA)
    if bool(payload.get("labels_included", True)):
        raise ValueError("checkpoint must not contain labels")
    return payload


def save_encoded_graph(path: str | Path, graph: EncodedTokenGraph) -> None:
    _save_torch(path, graph.payload())


def load_encoded_graph(path: str | Path) -> EncodedTokenGraph:
    return EncodedTokenGraph.from_payload(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def save_embedding_index(
    path: str | Path,
    index: EmbeddingIndex,
    **metadata: Any,
) -> None:
    _validate_encoded_graph_metadata(metadata)
    arrays = index.arrays()
    arrays.update(_metadata_arrays(metadata))
    arrays.update(
        schema=np.asarray(EMBEDDING_INDEX_SCHEMA),
        version=np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
    )
    save_npz(path, **arrays)


def load_embedding_index(
    path: str | Path,
) -> tuple[EmbeddingIndex, dict[str, np.ndarray]]:
    arrays = load_npz(path)
    _require_npz_schema(arrays, EMBEDDING_INDEX_SCHEMA)
    if bool(arrays["labels_included"].item()):
        raise ValueError("embedding index must not contain labels")
    index = EmbeddingIndex.from_arrays(arrays)
    row_fields = set(index.arrays())
    metadata = {
        name: value
        for name, value in arrays.items()
        if name not in row_fields | {"schema", "version", "labels_included"}
    }
    _validate_encoded_graph_metadata(metadata)
    return index, metadata


def save_scores(
    path: str | Path,
    index: EmbeddingIndex,
    score: np.ndarray,
    **metadata: Any,
) -> None:
    score = np.asarray(score, dtype=np.float32)
    if score.ndim != 1 or len(score) != len(index.sample_id):
        raise ValueError("one token score is required for every embedding row")
    arrays = index.arrays()
    arrays.pop("embedding")
    arrays["score"] = score
    arrays.update(_metadata_arrays(metadata))
    arrays.update(
        schema=np.asarray(SCORE_SCHEMA),
        version=np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
    )
    save_npz(path, **arrays)


def load_scores(path: str | Path) -> dict[str, np.ndarray]:
    arrays = load_npz(path)
    _require_npz_schema(arrays, SCORE_SCHEMA)
    if bool(arrays["labels_included"].item()):
        raise ValueError("score artifact must not contain labels")
    validate_complete_token_rows(
        arrays["sample_id"],
        arrays["source_id"],
        arrays["token_index"],
        arrays["response_length"],
    )
    score = np.asarray(arrays["score"])
    if score.ndim != 1 or len(score) != len(arrays["sample_id"]):
        raise ValueError("score artifact rows are not aligned")
    return arrays


def save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_torch(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=dtype).contiguous()


def _metadata_arrays(metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in metadata.items():
        if isinstance(value, Path):
            value = str(value)
        result[name] = np.asarray(value)
    return result


def _validate_encoded_graph_metadata(metadata: Mapping[str, Any]) -> None:
    names = {
        "encoded_graph_sample_ids",
        "encoded_graph_paths",
        "encoded_graph_sha256",
    }
    present = names.intersection(metadata)
    if not present:
        return
    if present != names:
        raise ValueError("encoded graph metadata must include IDs, paths and SHA-256")
    arrays = {name: np.asarray(metadata[name]) for name in names}
    if any(value.ndim != 1 for value in arrays.values()) or len(
        {len(value) for value in arrays.values()}
    ) != 1:
        raise ValueError("encoded graph metadata vectors must be aligned")
    hashes = arrays["encoded_graph_sha256"].astype(str).tolist()
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
        for value in hashes
    ):
        raise ValueError("encoded graph metadata contains an invalid SHA-256")


def _require_schema(payload: Mapping[str, Any], schema: str) -> None:
    if payload.get("schema") != schema or int(payload.get("version", -1)) != ARTIFACT_VERSION:
        raise ValueError(f"unsupported {schema} artifact")


def _require_npz_schema(arrays: Mapping[str, np.ndarray], schema: str) -> None:
    if (
        "schema" not in arrays
        or "version" not in arrays
        or str(np.asarray(arrays["schema"]).item()) != schema
        or int(np.asarray(arrays["version"]).item()) != ARTIFACT_VERSION
    ):
        raise ValueError(f"unsupported {schema} artifact")
