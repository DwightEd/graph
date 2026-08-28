"""Lazy verified access to constructed frozen operator graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .artifacts import load_graph_artifact, sha256
from .schema import GRAPH_SCHEMA, GRAPH_VERSION, OperatorGraphArtifact


class OperatorGraphDataset:
    """Read-only graph split used by downstream evaluation code.

    The class exposes only graph artifacts and construction metadata.  It does
    not know where hallucination labels are stored and therefore cannot leak
    labels into representation construction or graph loading.
    """

    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "index.jsonl"
        if not manifest_path.is_file() or not index_path.is_file():
            raise ValueError("operator graph split requires manifest.json and index.jsonl")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema") != GRAPH_SCHEMA
            or int(self.manifest.get("version", -1)) != GRAPH_VERSION
        ):
            raise ValueError("unsupported operator graph split")
        if sha256(index_path) != self.manifest.get("index_sha256"):
            raise ValueError("operator graph index SHA256 mismatch")
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(self.manifest.get("count", -1)):
            raise ValueError("operator graph index count differs from manifest")
        self.rows = {str(row["sample_id"]): row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("operator graph sample IDs must be unique")
        self.verify_hashes = bool(verify_hashes)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[OperatorGraphArtifact]:
        for sample_id in self.sample_ids:
            yield self[sample_id]

    def __contains__(self, sample_id: object) -> bool:
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id: object) -> OperatorGraphArtifact:
        key = str(sample_id)
        if key not in self.rows:
            raise KeyError(key)
        row = self.rows[key]
        path = self.root / row["path"]
        expected = str(row["sha256"]) if self.verify_hashes else None
        artifact = load_graph_artifact(path, verify_sha256=expected)
        if artifact.sample_id != key:
            raise ValueError("artifact sample ID differs from index")
        return artifact

    @property
    def sample_ids(self) -> list[str]:
        return list(self.rows)


__all__ = ["OperatorGraphDataset"]
