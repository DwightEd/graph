"""Load saved GroundedRoute graphs referenced by one embedding index."""

from dataclasses import dataclass
from pathlib import Path

from experiments.grounded_route.artifacts import (
    load_embedding_index,
    load_encoded_graph,
    sha256,
)


@dataclass(frozen=True)
class GraphRecord:
    sample_id: str
    source_id: str
    path: Path

    def load(self):
        return load_encoded_graph(self.path)


@dataclass(frozen=True)
class GraphBundle:
    index_path: Path
    index_sha256: str
    index: object
    metadata: dict
    records: tuple[GraphRecord, ...]

    def iter_graphs(self):
        for record in self.records:
            yield record.load()

    def reverify(self) -> None:
        return None


def load_bundle(index_path) -> GraphBundle:
    index_path = Path(index_path).resolve()
    index, metadata = load_embedding_index(index_path)
    source_by_sample = dict(
        zip(index.sample_id.astype(str), index.source_id.astype(str), strict=True)
    )
    sample_ids = metadata["encoded_graph_sample_ids"].astype(str).tolist()
    graph_paths = metadata["encoded_graph_paths"].astype(str).tolist()
    records = tuple(
        GraphRecord(
            sample_id=sample_id,
            source_id=source_by_sample[sample_id],
            path=index_path.parent / relative_path,
        )
        for sample_id, relative_path in zip(sample_ids, graph_paths, strict=True)
    )
    return GraphBundle(
        index_path=index_path,
        index_sha256=sha256(index_path),
        index=index,
        metadata=metadata,
        records=records,
    )
