"""Verified access to GroundedRoute's saved attributed graphs.

This module is the feature boundary of the graph-effectiveness audit.  It
loads only an embedding index and the content-addressed graph sidecars named
by that index.  Canonical dataset access is used solely by
``load_aligned_test_labels`` after the complete bundle has been frozen and
reverified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch

from experiment_protocol import FrozenEvaluation, EvaluationLabels, FrozenFile, scalar_text
from research_dataset import open_research_dataset

from ..artifacts import (
    EmbeddingIndex,
    EncodedTokenGraph,
    load_embedding_index,
    load_encoded_graph,
    load_npz,
    sha256,
)


INTEGRITY_SCHEMA = "grounded-route-graph-effectiveness-integrity"
INTEGRITY_VERSION = 1
DEFAULT_CONSERVATION_TOLERANCE = 1e-2


@dataclass(frozen=True)
class GraphRecord:
    """One lazy graph sidecar and its response rows in ``index.npz``."""

    sample_id: str
    source_id: str
    path: Path
    sha256: str
    row_index: np.ndarray
    response_start: int
    response_count: int
    token_count: int
    edge_count: int
    layer_count: int
    head_count: int

    def load(self, *, verify_hash: bool = False) -> EncodedTokenGraph:
        """Load the graph without reopening its original attention cache."""

        if verify_hash and sha256(self.path) != self.sha256:
            raise ValueError(f"encoded graph SHA-256 changed for {self.sample_id}")
        graph = load_encoded_graph(self.path)
        if graph.sample_id != self.sample_id or graph.source_id != self.source_id:
            raise ValueError("encoded graph identity differs from its frozen record")
        geometry = (
            graph.response_start,
            graph.response_count,
            int(graph.token_ids.numel()),
            int(graph.edge_weight.numel()),
            graph.layer_count,
            graph.head_count,
        )
        expected = (
            self.response_start,
            self.response_count,
            self.token_count,
            self.edge_count,
            self.layer_count,
            self.head_count,
        )
        if geometry != expected:
            raise ValueError("encoded graph geometry differs from its frozen record")
        return graph


@dataclass(frozen=True)
class GraphBundle:
    """A frozen embedding index and its verified, lazily loaded sidecars."""

    index_path: Path
    index_sha256: str
    index: EmbeddingIndex
    metadata: Mapping[str, np.ndarray]
    records: tuple[GraphRecord, ...]
    integrity: Mapping[str, object]

    def iter_graphs(self, *, verify_hash: bool = False) -> Iterator[EncodedTokenGraph]:
        for record in self.records:
            yield record.load(verify_hash=verify_hash)

    def reverify(self) -> None:
        """Recheck every content identity before the label boundary is crossed."""

        if sha256(self.index_path) != self.index_sha256:
            raise ValueError("embedding index changed after bundle verification")
        for record in self.records:
            if sha256(record.path) != record.sha256:
                raise ValueError(f"encoded graph changed for {record.sample_id}")


def verify_bundle(
    index_path: str | Path,
    *,
    conservation_tolerance: float = DEFAULT_CONSERVATION_TOLERANCE,
) -> tuple[GraphBundle, dict[str, object]]:
    """Verify one saved graph bundle and return its label-free audit report."""

    index_path = Path(index_path).resolve()
    index_digest = sha256(index_path)
    index, metadata = load_embedding_index(index_path)
    graph_ids = _text_vector(metadata, "encoded_graph_sample_ids")
    graph_paths = _text_vector(metadata, "encoded_graph_paths")
    graph_hashes = _text_vector(metadata, "encoded_graph_sha256")
    if len(set(graph_ids)) != len(graph_ids):
        raise ValueError("encoded graph sample IDs must be unique")
    if set(graph_ids) != set(index.sample_id.astype(str).tolist()):
        raise ValueError("encoded graph sample IDs differ from embedding-index rows")

    records: list[GraphRecord] = []
    total_nodes = 0
    response_nodes = 0
    total_edges = 0
    maximum_row_error = 0.0
    maximum_lineage_error = 0.0
    embedding_dimensions: set[int] = set()
    geometries: set[tuple[int, int]] = set()

    for sample_id, relative, expected_hash in zip(
        graph_ids,
        graph_paths,
        graph_hashes,
        strict=True,
    ):
        path = _sidecar_path(index_path.parent, relative)
        if sha256(path) != expected_hash:
            raise ValueError(f"encoded graph SHA-256 differs for {sample_id}")
        graph = load_encoded_graph(path)
        rows = np.flatnonzero(index.sample_id.astype(str) == sample_id)
        rows = rows[np.argsort(index.token_index[rows], kind="stable")]
        row_error, lineage_error = _verify_graph(
            graph,
            sample_id,
            index,
            rows,
            conservation_tolerance,
        )
        records.append(
            GraphRecord(
                sample_id=graph.sample_id,
                source_id=graph.source_id,
                path=path,
                sha256=expected_hash,
                row_index=rows,
                response_start=graph.response_start,
                response_count=graph.response_count,
                token_count=int(graph.token_ids.numel()),
                edge_count=int(graph.edge_weight.numel()),
                layer_count=graph.layer_count,
                head_count=graph.head_count,
            )
        )
        total_nodes += int(graph.token_ids.numel())
        response_nodes += graph.response_count
        total_edges += int(graph.edge_weight.numel())
        maximum_row_error = max(maximum_row_error, row_error)
        maximum_lineage_error = max(maximum_lineage_error, lineage_error)
        embedding_dimensions.add(int(graph.node_embedding.shape[1]))
        geometries.add((graph.layer_count, graph.head_count))

    if len(embedding_dimensions) != 1 or len(geometries) != 1:
        raise ValueError("encoded graph bundle has inconsistent geometry")
    layer_count, head_count = next(iter(geometries))
    report: dict[str, object] = {
        "schema": INTEGRITY_SCHEMA,
        "version": INTEGRITY_VERSION,
        "labels_read": False,
        "index": str(index_path),
        "index_sha256": index_digest,
        "split": scalar_text(metadata, "split"),
        "scope": scalar_text(metadata, "scope"),
        "graphs": len(records),
        "nodes": total_nodes,
        "response_nodes": response_nodes,
        "edges": total_edges,
        "embedding_dimension": next(iter(embedding_dimensions)),
        "layer_count": layer_count,
        "head_count": head_count,
        "maximum_row_mass_error": maximum_row_error,
        "maximum_lineage_mass_error": maximum_lineage_error,
        "sidecar_hashes_verified": True,
    }
    bundle = GraphBundle(
        index_path=index_path,
        index_sha256=index_digest,
        index=index,
        metadata=metadata,
        records=tuple(records),
        integrity=report,
    )
    return bundle, report


def load_bundle(
    index_path: str | Path,
    *,
    conservation_tolerance: float = DEFAULT_CONSERVATION_TOLERANCE,
) -> GraphBundle:
    """Load and fully verify a saved graph bundle."""

    bundle, _ = verify_bundle(
        index_path,
        conservation_tolerance=conservation_tolerance,
    )
    return bundle


def load_aligned_test_labels(
    bundle: GraphBundle,
    test_root: str | Path,
) -> EvaluationLabels:
    """Open canonical test labels only after all saved features are frozen."""

    if scalar_text(bundle.metadata, "split") != "test":
        raise ValueError("the supervised audit requires a test graph bundle")
    bundle.reverify()
    frozen = FrozenEvaluation.capture(bundle.index_path, expected_split="test")
    if frozen.artifact.sha256 != bundle.index_sha256:
        raise ValueError("embedding index changed before labels were opened")
    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    rows = {**bundle.index.arrays(), **bundle.metadata}
    return frozen.align_loaded(dataset, rows)


def load_aligned_artifact_labels(
    bundle: GraphBundle,
    test_root: str | Path,
    artifact_path: str | Path,
) -> tuple[EvaluationLabels, FrozenFile, Mapping[str, np.ndarray]]:
    """Freeze a derived row artifact, then open its canonical test labels."""

    bundle.reverify()
    evaluation = FrozenEvaluation.capture(artifact_path, expected_split="test")
    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    rows, labels = evaluation.load_and_align(dataset, load_npz)
    return labels, evaluation.artifact, rows


def _verify_graph(
    graph: EncodedTokenGraph,
    sample_id: str,
    index: EmbeddingIndex,
    rows: np.ndarray,
    tolerance: float,
) -> tuple[float, float]:
    if graph.sample_id != sample_id:
        raise ValueError("encoded graph sample ID differs from index metadata")
    if len(rows) != graph.response_count:
        raise ValueError("encoded graph response count differs from index rows")
    if set(index.source_id[rows].astype(str).tolist()) != {graph.source_id}:
        raise ValueError("encoded graph source ID differs from index rows")
    if not np.array_equal(
        index.response_token_id[rows],
        graph.token_ids[graph.response_start :].numpy(),
    ):
        raise ValueError("encoded graph response token IDs differ from index rows")
    if not np.array_equal(
        index.embedding[rows],
        graph.response_embedding.numpy(),
    ):
        raise ValueError("encoded graph node embeddings differ from index rows")

    tensors = (
        graph.node_embedding,
        graph.edge_weight,
        graph.diagonal,
        graph.unresolved,
        graph.lineage,
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("encoded graph contains non-finite values")
    if any(
        bool((value < 0).any())
        for value in (
            graph.edge_weight,
            graph.diagonal,
            graph.unresolved,
            graph.lineage,
        )
    ):
        raise ValueError("encoded graph contains negative mass")
    _verify_unique_typed_edges(graph)

    retained = torch.zeros_like(graph.diagonal, dtype=torch.float32)
    if graph.edge_weight.numel():
        retained.index_put_(
            (
                graph.edge_index[1] - graph.response_start,
                graph.edge_layer,
                graph.edge_head,
            ),
            graph.edge_weight.float(),
            accumulate=True,
        )
    row_mass = retained + graph.diagonal.float() + graph.unresolved.float()
    row_error = float((row_mass - 1.0).abs().max().item())
    if row_error > tolerance:
        raise ValueError("encoded graph retained/diagonal/unresolved mass is not conserved")
    lineage_error = float((graph.lineage.float().sum(dim=-1) - 1.0).abs().max().item())
    if lineage_error > tolerance:
        raise ValueError("encoded graph lineage mass is not conserved")
    return row_error, lineage_error


def _verify_unique_typed_edges(
    graph: EncodedTokenGraph,
    chunk_size: int = 1_000_000,
) -> None:
    if not graph.edge_weight.numel():
        return
    source, target = graph.edge_index
    token_count = int(graph.token_ids.numel())
    previous: int | None = None
    for start in range(0, int(graph.edge_weight.numel()), chunk_size):
        stop = min(start + chunk_size, int(graph.edge_weight.numel()))
        row = (
            (graph.edge_layer[start:stop] * graph.head_count + graph.edge_head[start:stop])
            * graph.response_count
            + target[start:stop]
            - graph.response_start
        )
        key = row * token_count + source[start:stop]
        if previous is not None and int(key[0].item()) <= previous:
            raise ValueError("encoded graph typed endpoints are not unique and canonical")
        if len(key) > 1 and bool((key[1:] <= key[:-1]).any()):
            raise ValueError("encoded graph typed endpoints are not unique and canonical")
        previous = int(key[-1].item())


def _text_vector(mapping: Mapping[str, np.ndarray], name: str) -> tuple[str, ...]:
    value = np.asarray(mapping[name])
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"artifact field {name!r} must be a text vector")
    return tuple(map(str, value.tolist()))


def _sidecar_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("encoded graph path must be relative to its index")
    path = (root / relative_path).resolve()
    if path.parent != root and root not in path.parents:
        raise ValueError("encoded graph path leaves its index directory")
    if not path.is_file():
        raise ValueError(f"encoded graph sidecar is missing: {relative}")
    return path
