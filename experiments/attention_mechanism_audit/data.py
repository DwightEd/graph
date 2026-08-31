"""RAGTruth prompts and their exact external-evidence tokens."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

HISTORICAL_SYSTEM_PROMPT = "You are a helpful assistant."
TASK_TYPES = ("QA", "Summary", "Data2txt")


def canonical_task_type(value: object) -> str:
    for task in TASK_TYPES:
        if str(value).casefold() == task.casefold():
            return task
    raise ValueError(f"unsupported task type: {value}")


def load_source_info(path: str | Path) -> dict[str, dict]:
    """Load label-free ``source_info.jsonl`` rows keyed by source ID."""

    sources: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row["source_id"])
            if source_id in sources:
                raise ValueError(f"duplicate source_id: {source_id}")
            sources[source_id] = row
    return sources


def render_historical_prompt(tokenizer, prompt: str) -> str:
    """Render the system/user prefix used to create the attention cache."""

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": HISTORICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _cpu_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def build_evidence_mask(
    source: Mapping,
    tokenizer,
    cached_token_ids,
    response_start: int,
) -> np.ndarray:
    """Mark external evidence in a cached QA, Summary, or Data2txt prompt."""

    prompt = str(source["prompt"])
    task = canonical_task_type(source["task_type"])
    evidence = source["source_info"]
    if task == "QA":
        evidence = str(evidence["passages"]).removesuffix("\n")
    evidence = str(evidence)

    rendered = render_historical_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    rebuilt_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    cached_prefix = _cpu_array(cached_token_ids)[:response_start]
    if rebuilt_ids.shape != (response_start,) or not np.array_equal(
        rebuilt_ids, cached_prefix
    ):
        raise ValueError("rebuilt prompt does not exactly match cached token prefix")

    evidence_start = rendered.index(prompt) + prompt.index(evidence)
    evidence_stop = evidence_start + len(evidence)
    return (offsets[:, 0] < evidence_stop) & (offsets[:, 1] > evidence_start)
