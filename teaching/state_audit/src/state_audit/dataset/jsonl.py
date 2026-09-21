"""Dataset adapters stop here. Downstream code sees the same Example schema."""

import json
from pathlib import Path

from .schema import Example


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate(example: Example) -> None:
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
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Dataset must have unique example IDs")
    for example in examples:
        validate(example)
    return examples
