"""Dataset adapters stop here. Downstream code sees the same Example schema."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Example:
    id: str
    source_id: str
    prompt: str
    evidence: list[dict]  # {id, start, end}, half-open character spans in prompt
    response: str | None = None
    labels: list[dict] | None = None  # None = unreviewed; [] = reviewed negative
    metadata: dict = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate(example: Example) -> None:
    if not example.prompt or not example.source_id or not example.id:
        raise ValueError("id, source_id and prompt must be nonempty")
    seen = set()
    for span in example.evidence:
        if span["id"] in seen or not 0 <= span["start"] < span["end"] <= len(example.prompt):
            raise ValueError(f"{example.id}: invalid or duplicate evidence span")
        seen.add(span["id"])
    ordered = sorted(example.evidence, key=lambda span: span["start"])
    if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"{example.id}: evidence spans must be disjoint")
    for span in example.labels or []:
        if example.response is None or not 0 <= span["start"] < span["end"] <= len(
            example.response
        ):
            raise ValueError(f"{example.id}: invalid response label")
        if "text" in span and example.response[span["start"] : span["end"]] != span["text"]:
            raise ValueError(f"{example.id}: label text disagrees with its offsets")


def load_examples(path: Path) -> list[Example]:
    examples = [Example(**row) for row in read_jsonl(path)]
    if not examples or len({example.id for example in examples}) != len(examples):
        raise ValueError("Dataset must be nonempty and have unique example IDs")
    for example in examples:
        validate(example)
    return examples


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
        validate(example)
        examples.append(asdict(example))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )
