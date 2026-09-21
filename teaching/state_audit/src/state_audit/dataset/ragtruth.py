"""Official RAGTruth to the common Example format."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from .jsonl import read_jsonl
from .schema import Example


def evidence_texts(source: dict) -> list[str]:
    info = source["source_info"]
    if source["task_type"] == "Summary":
        return [info]
    if source["task_type"] == "QA":
        parts = re.split(r"(?:^|\n)\s*passage\s+\d+\s*:", info["passages"])
        return [part.strip() for part in parts if part.strip()]
    if source["task_type"] == "Data2txt":
        return [str(info)]  # The official prompt embeds Python's dict representation.
    raise ValueError(f"Unsupported RAGTruth task: {source['task_type']}")


def locate_evidence(prompt: str, texts: list[str]) -> list[dict]:
    spans = []
    for number, text in enumerate(texts):
        if prompt.count(text) != 1:
            raise ValueError("Evidence must match the prompt exactly once; supply explicit spans")
        start = prompt.index(text)
        spans.append(dict(id=str(number), start=start, end=start + len(text)))
    return spans


def convert_ragtruth(directory: Path, output: Path, split: str | None, limit: int | None):
    sources = {str(row["source_id"]): row for row in read_jsonl(directory / "source_info.jsonl")}
    rows = read_jsonl(directory / "response.jsonl")
    rows = [row for row in rows if split is None or row["split"] == split][:limit]
    examples = []
    for row in rows:
        source = sources[str(row["source_id"])]
        spans = locate_evidence(source["prompt"], evidence_texts(source))
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "source_id", "response", "labels"}
        }
        metadata.update(dataset="RAGTruth", task=source["task_type"])
        example = Example(
            str(row["id"]),
            str(row["source_id"]),
            source["prompt"],
            spans,
            row["response"],
            row["labels"],
            metadata,
        )
        examples.append(asdict(example))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )
