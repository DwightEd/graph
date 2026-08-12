"""Enrich canonical split indices with RAGTruth metadata only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from cache import sha256, verify_split

RESEARCH_INDEX_FIELDS = (
    "sample_id", "source_id", "split", "task_type", "data_source",
    "generator_model", "temperature", "quality", "path", "sha256", "bytes",
)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write(path, text):
    path = Path(path)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _split_roots(root):
    root = Path(root)
    if (root / "manifest.json").is_file():
        return [root]
    roots = [root / split for split in ("train", "test") if (root / split / "manifest.json").is_file()]
    if not roots:
        raise ValueError("canonical_root must be a split or contain train/test splits")
    return roots


def enrich_ragtruth_indices(canonical_root, dataset_path):
    """Update JSON metadata only; attention NPZs and labels are untouched."""
    dataset_path = Path(dataset_path)
    sources = {str(row["source_id"]): row for row in _read_jsonl(dataset_path / "source_info.jsonl")}
    responses = {str(row["id"]): row for row in _read_jsonl(dataset_path / "response.jsonl")}
    summary = {"canonical_root": str(canonical_root), "splits": {}}

    for root in _split_roots(canonical_root):
        verify_split(root)
        index_path = root / "index.jsonl"
        manifest_path = root / "manifest.json"
        rows = _read_jsonl(index_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_split = manifest.get("split") or root.name
        enriched = []
        task_types, data_sources, generator_models = set(), set(), set()
        for row in rows:
            sample_id, source_id = str(row["sample_id"]), str(row["source_id"])
            response = responses.get(sample_id)
            source = sources.get(source_id)
            if response is None or source is None or str(response["source_id"]) != source_id:
                raise ValueError(f"RAGTruth metadata mismatch for sample {sample_id}")
            split = str(response["split"])
            if split != expected_split:
                raise ValueError(f"split mismatch for sample {sample_id}")
            quality = str(response.get("quality", ""))
            if quality.casefold() != "good":
                raise ValueError(f"sample {sample_id} is not quality=good")
            task_type = str(source["task_type"])
            data_source = str(source["source"])
            generator_model = str(response["model"])
            enriched.append({
                "sample_id": sample_id,
                "source_id": source_id,
                "split": split,
                "task_type": task_type,
                "data_source": data_source,
                "generator_model": generator_model,
                "temperature": response.get("temperature"),
                "quality": quality,
                "path": row["path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            })
            task_types.add(task_type)
            data_sources.add(data_source)
            generator_models.add(generator_model)

        _atomic_write(index_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched))
        manifest.update({
            "dataset": "RAGTruth",
            "split": expected_split,
            "index_sha256": sha256(index_path),
            "index_fields": list(RESEARCH_INDEX_FIELDS),
            "task_types": sorted(task_types),
            "data_sources": sorted(data_sources),
            "generator_models": sorted(generator_models),
        })
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_split(root)
        summary["splits"][expected_split] = len(enriched)
    summary["count"] = sum(summary["splits"].values())
    return summary
