"""Enrich an existing canonical archive with lightweight RAGTruth metadata."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from cache import sha256


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


def enrich_ragtruth_indices(canonical_root: str | Path, dataset_path: str | Path) -> dict[str, Any]:
    """Rewrite index/manifest JSON only; attention NPZ files are never touched."""
    canonical_root, dataset_path = Path(canonical_root), Path(dataset_path)
    sources = {str(row["source_id"]): row for row in _read_jsonl(dataset_path / "source_info.jsonl")}
    responses = {str(row["id"]): row for row in _read_jsonl(dataset_path / "response.jsonl")}

    summary: dict[str, Any] = {"canonical_root": str(canonical_root), "splits": {}}
    for split_root in _split_roots(canonical_root):
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

        _atomic_write(index_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched))
        manifest.update({
            "dataset": "RAGTruth",
            "split": resolved_split,
            "index_sha256": sha256(index_path),
            "index_fields": list(RESEARCH_INDEX_FIELDS),
            "task_types": sorted(task_types),
            "data_sources": sorted(data_sources),
            "generator_models": sorted(generator_models),
            "qualities": sorted(qualities),
        })
        _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summary["splits"][str(resolved_split or split_root.name)] = len(enriched)

    summary["count"] = sum(summary["splits"].values())
    return summary
