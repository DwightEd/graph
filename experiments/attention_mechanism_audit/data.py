"""RAGTruth QA inputs and exact prompt-source roles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROLE_NAMES = ("evidence", "question", "constraint", "other_prompt")
EVIDENCE, QUESTION, CONSTRAINT, OTHER_PROMPT = range(len(ROLE_NAMES))
HISTORICAL_SYSTEM_PROMPT = "You are a helpful assistant."


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


def _qa_character_roles(source: Mapping) -> np.ndarray:
    prompt = str(source["prompt"])
    source_info = source["source_info"]
    question = str(source_info["question"])

    question_start = prompt.index(question)
    evidence_start = prompt.index("passages:\n") + len("passages:\n")
    evidence_stops = [
        position
        for marker in ("\nIn case the passages", "\noutput:")
        if (position := prompt.find(marker, evidence_start)) >= 0
    ]
    if not evidence_stops:
        raise ValueError("QA prompt has no evidence closing marker")

    roles = np.full(len(prompt), CONSTRAINT, dtype=np.int8)
    roles[question_start : question_start + len(question)] = QUESTION
    roles[evidence_start : min(evidence_stops)] = EVIDENCE
    return roles


def _cpu_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def build_prompt_role_ids(
    source: Mapping,
    tokenizer,
    cached_token_ids,
    response_start: int,
) -> np.ndarray:
    """Return exact QA source roles for cached positions ``[0, response_start)``."""

    if str(source["task_type"]).casefold() != "qa":
        raise ValueError("attention mechanism audit currently supports QA only")

    prompt = str(source["prompt"])
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

    prompt_start = rendered.index(prompt)
    prompt_stop = prompt_start + len(prompt)
    character_roles = _qa_character_roles(source)
    role_ids = np.full(response_start, OTHER_PROMPT, dtype=np.int8)
    for token, (start, stop) in enumerate(offsets):
        overlap_start = max(int(start), prompt_start)
        overlap_stop = min(int(stop), prompt_stop)
        if overlap_start >= overlap_stop:
            continue
        covered = character_roles[
            overlap_start - prompt_start : overlap_stop - prompt_start
        ]
        for role in (EVIDENCE, QUESTION, CONSTRAINT):
            if np.any(covered == role):
                role_ids[token] = role
                break
    return role_ids
