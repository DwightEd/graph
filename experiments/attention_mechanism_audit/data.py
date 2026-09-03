"""Prompt reconstruction and external-evidence token alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SYSTEM_PROMPT = "You are a helpful assistant."
TASK_TYPES = ("QA", "Summary", "Data2txt")


def load_sources(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        return {str(row["source_id"]): row for row in rows}


def task_name(value: object) -> str:
    name = str(value).casefold()
    return next(task for task in TASK_TYPES if task.casefold() == name)


def evidence_mask(source: dict, tokenizer, cached_ids, response_start: int) -> np.ndarray:
    """Locate the exact external-evidence substring in the cached prompt."""

    prompt = str(source["prompt"])
    evidence = source["source_info"]
    if task_name(source["task_type"]) == "QA":
        evidence = evidence["passages"]
    evidence = str(evidence).removesuffix("\n")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    cached = np.asarray(cached_ids, dtype=np.int64)[:response_start]
    if not np.array_equal(ids, cached):
        raise ValueError("rendered prompt and attention-cache tokens differ")
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    start = rendered.index(prompt) + prompt.index(evidence)
    stop = start + len(evidence)
    return (offsets[:, 0] < stop) & (offsets[:, 1] > start)
