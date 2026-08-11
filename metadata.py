"""Enrich an existing canonical archive with lightweight RAGTruth metadata."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from cache import sha256, verify_split


RESEARCH_INDEX_FIELDS = (
    "sample_id",
    "source_id",
    "split",
    "task_type",
    "data_source",
    "generator_model",
    "temperature",
    "quality",
    "path",
    "sha256",
    "bytes",
)


@dataclass
class _EnrichmentPlan:
    split_root: Path
    index_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    enriched: list[dict[str, Any]]
    split: str | None
    task_types: set[str]
    data_sources: set[str]
    generator_models: set[str]
    qualities: set[str]
    old_manifest_sha256: str
    old_index_sha256: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write(path: Path, text: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _split_roots(root: Path) -> list[Path]:
    if (root / "manifest.json").is_file():
        return [root]
    roots = [root / split for split in ("train", "test") if (root / split / "manifest.json").is_file()]
    if not roots:
        raise ValueError("canonical_root must be a canonical split or a root containing train/test")
    return roots


def _graph_split_root(graph_root: Path, split: str) -> Path:
    if (graph_root / "manifest.json").is_file():
        return graph_root
    candidate = graph_root / split
    if not (candidate / "manifest.json").is_file():
        raise ValueError(f"graph root has no {split} split: {graph_root}")
    return candidate


def _graph_preflight(graph_root: Path, split: str, old_manifest_sha256: str,
                     old_index_sha256: str, canonical_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    graph_split = _graph_split_root(graph_root, split)
    graph_manifest_path = graph_split / "manifest.json"
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    graph_index = graph_split / "index.jsonl"
    graph_rows = _read_jsonl(graph_index)
    graph_ids = {str(row["sample_id"]) for row in graph_rows}
    canonical_ids = {str(row["sample_id"]) for row in canonical_rows}
    if graph_manifest.get("input_manifest_sha256") != old_manifest_sha256 or graph_manifest.get("input_index_sha256") != old_index_sha256:
        raise ValueError(f"graph provenance does not match canonical archive for {split}")
    if graph_manifest.get("index_sha256") != sha256(graph_index):
        raise ValueError(f"graph index hash mismatch for {split}")
    if graph_manifest.get("count") != len(graph_rows) or len(graph_ids) != len(graph_rows):
        raise ValueError(f"graph manifest count mismatch for {split}")
    if graph_ids != canonical_ids or len(graph_rows) != len(canonical_rows):
        raise ValueError(f"graph/canonical sample set mismatch for {split}")
    return graph_manifest_path, graph_manifest


def enrich_ragtruth_indices(
    canonical_root: str | Path,
    dataset_path: str | Path,
    graph_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rewrite canonical index/manifest JSON only; NPZ/PT artifacts are never touched."""
    canonical_root, dataset_path = Path(canonical_root), Path(dataset_path)
    split_roots = _split_roots(canonical_root)
    for split_root in split_roots:
        verify_split(split_root)
    sources = {str(row["source_id"]): row for row in _read_jsonl(dataset_path / "source_info.jsonl")}
    responses = {str(row["id"]): row for row in _read_jsonl(dataset_path / "response.jsonl")}

    plans = []
    for split_root in split_roots:
        index_path = split_root / "index.jsonl"
        manifest_path = split_root / "manifest.json"
        rows = _read_jsonl(index_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_split = manifest.get("split")
        if expected_split is None and split_root.name in ("train", "test"):
            expected_split = split_root.name

        enriched = []
        task_types, data_sources, generator_models, qualities = set(), set(), set(), set()
        resolved_split = expected_split
        for row in rows:
            sample_id, source_id = str(row["sample_id"]), str(row["source_id"])
            response = responses.get(sample_id)
            source = sources.get(source_id)
            if response is None or source is None:
                raise ValueError(f"RAGTruth metadata missing for sample {sample_id}")
            if str(response["source_id"]) != source_id:
                raise ValueError(f"source_id mismatch for sample {sample_id}")

            split = str(response["split"])
            if resolved_split is None:
                resolved_split = split
            if split != resolved_split:
                raise ValueError(f"split mismatch for sample {sample_id}: {split} != {resolved_split}")

            task_type = str(source["task_type"])
            data_source = str(source["source"])
            generator_model = str(response["model"])
            quality = str(response["quality"])
            temperature = response.get("temperature")
            if manifest.get("generator_model") is not None and generator_model != manifest["generator_model"]:
                raise ValueError(f"generator_model mismatch for sample {sample_id}")
            if quality != "good":
                raise ValueError(f"quality must be good for sample {sample_id}")

            enriched.append({
                "sample_id": sample_id,
                "source_id": source_id,
                "split": split,
                "task_type": task_type,
                "data_source": data_source,
                "generator_model": generator_model,
                "temperature": temperature,
                "quality": quality,
                "path": row["path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            })
            task_types.add(task_type)
            data_sources.add(data_source)
            generator_models.add(generator_model)
            qualities.add(quality)

        plans.append(_EnrichmentPlan(
            split_root, index_path, manifest_path, manifest, rows, enriched, resolved_split,
            task_types, data_sources, generator_models, qualities,
            sha256(manifest_path), sha256(index_path),
        ))

    graph_root_path = Path(graph_root) if graph_root is not None else None
    graph_plans = {}
    if graph_root_path is not None:
        for plan in plans:
            split = str(plan.split or plan.split_root.name)
            graph_plans[split] = _graph_preflight(
                graph_root_path, split, plan.old_manifest_sha256, plan.old_index_sha256, plan.rows
            )

    summary: dict[str, Any] = {"canonical_root": str(canonical_root), "splits": {}}
    for plan in plans:
        _atomic_write(plan.index_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan.enriched))
        plan.manifest.update({
            "dataset": "RAGTruth",
            "split": plan.split,
            "index_sha256": sha256(plan.index_path),
            "index_fields": list(RESEARCH_INDEX_FIELDS),
            "task_types": sorted(plan.task_types),
            "data_sources": sorted(plan.data_sources),
            "generator_models": sorted(plan.generator_models),
            "qualities": sorted(plan.qualities),
        })
        _atomic_write(plan.manifest_path, json.dumps(plan.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        split = str(plan.split or plan.split_root.name)
        summary["splits"][split] = len(plan.enriched)
        if split in graph_plans:
            graph_manifest_path, graph_manifest = graph_plans[split]
            graph_manifest["input_manifest_sha256"] = sha256(plan.manifest_path)
            graph_manifest["input_index_sha256"] = sha256(plan.index_path)
            _atomic_write(
                graph_manifest_path,
                json.dumps(graph_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )

    summary["count"] = sum(summary["splits"].values())
    if graph_root_path is not None:
        summary["graphs"] = {split: len(_read_jsonl(path.parent / "index.jsonl")) for split, (path, _) in graph_plans.items()}
    return summary
