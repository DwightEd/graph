"""RAGTruth preparation and strict annotation-free extraction input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSONL objects: {path}")
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_responses(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    fields = {
        "schema",
        "id",
        "source_id",
        "split",
        "task",
        "generator",
        "prompt",
        "source_span",
        "response",
        "response_sha256",
    }
    for row in rows:
        if set(row) != fields or row["schema"] != "route-graph/response@1":
            raise ValueError("response input has unexpected fields or schema")
        if any(
            not isinstance(row[key], str) or not row[key]
            for key in fields - {"source_span"}
        ):
            raise ValueError("response string fields must be nonempty")
        span = row["source_span"]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1] <= len(row["prompt"])
        ):
            raise ValueError("source_span must be a nonempty prompt character interval")
        if row["response_sha256"] != text_digest(row["response"]):
            raise ValueError("response digest mismatch")
        if row["split"] not in {"train", "test"}:
            raise ValueError("split must be train or test")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate response IDs")
    return rows


class RagtruthPreparer:
    """Select by task/generator only; split source IDs before any feature fit."""

    def __init__(
        self,
        dataset: Path,
        output: Path,
        task: str,
        generator: str,
        max_sources: int = 32,
        seed: int = 20260911,
    ) -> None:
        self.dataset, self.output = dataset, output
        self.task, self.generator = task, generator
        self.max_sources, self.seed = max_sources, seed

    def run(self) -> dict:
        if self.task not in {"QA", "Summary"} or self.max_sources < 2:
            raise ValueError("select QA or Summary and at least two sources")
        sources = {
            str(r["source_id"]): r
            for r in read_jsonl(self.dataset / "source_info.jsonl")
            if r["task_type"] == self.task
        }
        # Raw annotations are in the dataset file but are never inspected or copied.
        responses = [
            {
                "id": str(r["id"]),
                "source_id": str(r["source_id"]),
                "response": r["response"],
            }
            for r in read_jsonl(self.dataset / "response.jsonl")
            if r["model"] == self.generator and str(r["source_id"]) in sources
        ]
        selected = sorted(
            {r["source_id"] for r in responses},
            key=lambda sid: text_digest(f"{self.seed}:{sid}"),
        )[: self.max_sources]
        if len(selected) < 2:
            raise ValueError("selection needs at least two distinct sources")
        train = set(selected[: len(selected) // 2])
        selected = set(selected)
        rows = []
        for response in responses:
            sid = response["source_id"]
            if sid not in selected:
                continue
            source = sources[sid]
            evidence = (
                source["source_info"]["passages"]
                if self.task == "QA"
                else source["source_info"]
            ).strip()
            prompt = source["prompt"]
            start = prompt.find(evidence)
            if not evidence or start < 0 or prompt.find(evidence, start + 1) >= 0:
                raise ValueError(
                    f"source {sid}: evidence must occur exactly once in original prompt"
                )
            rows.append(
                {
                    "schema": "route-graph/response@1",
                    **response,
                    "split": "train" if sid in train else "test",
                    "task": self.task,
                    "generator": self.generator,
                    "prompt": prompt,
                    "source_span": [start, start + len(evidence)],
                    "response_sha256": text_digest(response["response"]),
                }
            )
        write_jsonl(self.output, rows)
        return {
            "responses": len(rows),
            "sources": len(selected),
            "fit_sources": len(train),
            "selection": "task_and_generator_only",
            "split_seed": self.seed,
            "output": str(self.output),
        }
