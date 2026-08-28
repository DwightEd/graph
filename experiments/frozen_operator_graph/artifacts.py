"""Atomic, provenance-bound storage for frozen operator graph artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import torch

from .schema import GRAPH_SCHEMA, GRAPH_VERSION, OperatorGraphArtifact


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError("output dtype must be float32, float16, or bfloat16")
    return mapping[name]


def _payload(artifact: OperatorGraphArtifact, *, output_dtype: str) -> dict[str, Any]:
    artifact.validate()
    dtype = _float_dtype(output_dtype)
    return {
        "schema": GRAPH_SCHEMA,
        "version": GRAPH_VERSION,
        "sample_id": artifact.sample_id,
        "source_id": artifact.source_id,
        "metadata": dict(artifact.metadata),
        "token_ids": artifact.token_ids.detach().cpu().long(),
        "response_start": int(artifact.response_start),
        "edge_index": artifact.edge_index.detach().cpu().long(),
        "edge_layer": artifact.edge_layer.detach().cpu().long(),
        "edge_role": artifact.edge_role.detach().cpu().long(),
        "edge_attention_code": artifact.edge_attention_code.detach().cpu().to(dtype),
        "edge_features": artifact.edge_features.detach().cpu().to(dtype),
        "edge_feature_names": tuple(artifact.edge_feature_names),
        "remainder_features": artifact.remainder_features.detach().cpu().to(dtype),
        "remainder_feature_names": tuple(artifact.remainder_feature_names),
        "route_features": artifact.route_features.detach().cpu().to(dtype),
        "route_feature_names": tuple(artifact.route_feature_names),
        "layer_features": artifact.layer_features.detach().cpu().to(dtype),
        "layer_feature_names": tuple(artifact.layer_feature_names),
        "temporal_features": artifact.temporal_features.detach().cpu().to(dtype),
        "temporal_feature_names": tuple(artifact.temporal_feature_names),
        "final_hidden": artifact.final_hidden.detach().cpu().to(dtype),
        "node_embedding": artifact.node_embedding.detach().cpu().to(dtype),
        "node_feature_names": tuple(artifact.node_feature_names),
        "audit": dict(artifact.audit),
        "provenance": dict(artifact.provenance),
    }


def save_graph_artifact(
    path: str | Path,
    artifact: OperatorGraphArtifact,
    *,
    output_dtype: str = "float32",
) -> dict[str, Any]:
    """Atomically save one validated artifact and return its index record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(artifact, output_dtype=output_dtype)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "sample_id": artifact.sample_id,
        "source_id": artifact.source_id,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "response_count": artifact.response_count,
        "node_width": int(artifact.node_embedding.shape[1]),
        "exposed_edges": int(artifact.edge_index.shape[1]),
    }


def _artifact_from_payload(payload: Mapping[str, Any]) -> OperatorGraphArtifact:
    if (
        payload.get("schema") != GRAPH_SCHEMA
        or int(payload.get("version", -1)) != GRAPH_VERSION
    ):
        raise ValueError("unsupported frozen operator graph artifact")
    return OperatorGraphArtifact(
        sample_id=str(payload["sample_id"]),
        source_id=str(payload["source_id"]),
        metadata=dict(payload["metadata"]),
        token_ids=torch.as_tensor(payload["token_ids"]).long(),
        response_start=int(payload["response_start"]),
        edge_index=torch.as_tensor(payload["edge_index"]).long(),
        edge_layer=torch.as_tensor(payload["edge_layer"]).long(),
        edge_role=torch.as_tensor(payload["edge_role"]).long(),
        edge_attention_code=torch.as_tensor(payload["edge_attention_code"]).float(),
        edge_features=torch.as_tensor(payload["edge_features"]).float(),
        edge_feature_names=tuple(payload["edge_feature_names"]),
        remainder_features=torch.as_tensor(payload["remainder_features"]).float(),
        remainder_feature_names=tuple(payload["remainder_feature_names"]),
        route_features=torch.as_tensor(payload["route_features"]).float(),
        route_feature_names=tuple(payload["route_feature_names"]),
        layer_features=torch.as_tensor(payload["layer_features"]).float(),
        layer_feature_names=tuple(payload["layer_feature_names"]),
        temporal_features=torch.as_tensor(payload["temporal_features"]).float(),
        temporal_feature_names=tuple(payload["temporal_feature_names"]),
        final_hidden=torch.as_tensor(payload["final_hidden"]).float(),
        node_embedding=torch.as_tensor(payload["node_embedding"]).float(),
        node_feature_names=tuple(payload["node_feature_names"]),
        audit=dict(payload["audit"]),
        provenance=dict(payload["provenance"]),
    ).validate()


def load_graph_artifact(
    path: str | Path,
    *,
    verify_sha256: str | None = None,
) -> OperatorGraphArtifact:
    path = Path(path)
    if verify_sha256 is not None and sha256(path) != str(verify_sha256):
        raise ValueError("graph artifact SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("graph artifact must contain a dictionary payload")
    return _artifact_from_payload(payload)


def write_split_manifest(
    output_root: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    checkpoint: str,
    dataset_manifest_sha256: str,
    configuration: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    source_dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic JSONL index and manifest after all samples succeed."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_rows = [dict(row) for row in rows]
    normalized_rows.sort(key=lambda row: str(row["sample_id"]))
    index_path = root / "index.jsonl"
    index_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in normalized_rows
    )
    index_path.write_text(index_text, encoding="utf-8")
    manifest = {
        "schema": GRAPH_SCHEMA,
        "version": GRAPH_VERSION,
        "count": len(normalized_rows),
        "checkpoint": str(checkpoint),
        "dataset_manifest_sha256": str(dataset_manifest_sha256),
        "configuration": dict(configuration),
        "feature_contract": dict(feature_contract),
        "feature_contract_sha256": canonical_json_sha256(feature_contract),
        "index_sha256": sha256(index_path),
        "labels_read_during_construction": False,
    }
    if source_dataset is not None:
        manifest["source_dataset"] = dict(source_dataset)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "canonical_json_sha256",
    "load_graph_artifact",
    "save_graph_artifact",
    "sha256",
    "write_split_manifest",
]
