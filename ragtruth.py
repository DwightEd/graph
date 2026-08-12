"""RAGTruth adapter: sample metadata, tokenization, and evaluation-label alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class RagTruthSample:
    source_id: str
    response_id: str
    prompt: str
    response: str
    split: str
    task_type: str
    data_source: str
    generator_model: str
    temperature: float | None
    quality: str
    positive_char_spans: tuple[tuple[int, int], ...]


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _label_spans(response_record):
    spans = []
    for label in response_record.get("labels") or []:
        try:
            start, end = int(label["start"]), int(label["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(str(response_record.get("response", ""))):
            spans.append((start, end))
    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def load_ragtruth_samples(dataset_path, *, split, generator_model, task_type="all"):
    dataset_path = Path(dataset_path)
    sources = {
        str(row["source_id"]): row
        for row in _read_jsonl(dataset_path / "source_info.jsonl")
    }
    requested_model = _normalized_name(generator_model)
    samples = []
    for response in _read_jsonl(dataset_path / "response.jsonl"):
        if str(response.get("split", "")).casefold() != split.casefold():
            continue
        if _normalized_name(str(response.get("model", ""))) != requested_model:
            continue
        if str(response.get("quality", "")).casefold() != "good":
            continue
        source_id = str(response["source_id"])
        source = sources[source_id]
        if task_type.casefold() != "all" and str(source["task_type"]).casefold() != task_type.casefold():
            continue
        samples.append(RagTruthSample(
            source_id=source_id,
            response_id=str(response["id"]),
            prompt=str(source["prompt"]),
            response=str(response["response"]),
            split=str(response["split"]),
            task_type=str(source["task_type"]),
            data_source=str(source["source"]),
            generator_model=str(response["model"]),
            temperature=response.get("temperature"),
            quality=str(response["quality"]),
            positive_char_spans=_label_spans(response),
        ))
    return samples


def _token_runs(offsets, spans):
    positive = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            positive.append(index)
    runs = []
    for index in positive:
        if not runs or index != runs[-1][1]:
            runs.append([index, index + 1])
        else:
            runs[-1][1] += 1
    return runs


def tokenize_ragtruth_sample(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    positive_char_spans=(),
):
    """Return exact concatenated tokens, response boundary, and response-relative labels."""
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(
        rendered_prompt + response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if len(input_ids) != len(offsets):
        raise ValueError("input_ids and offset_mapping must align")
    boundary = len(rendered_prompt)
    if any(start < boundary < end for start, end in offsets):
        raise ValueError("a token crosses the prompt/response boundary")
    response_idx = next(
        (index for index, (start, end) in enumerate(offsets) if start >= boundary and end > boundary),
        None,
    )
    if response_idx is None or not 0 < response_idx < len(input_ids):
        raise ValueError("response does not form a token suffix")
    response_offsets = [
        (max(0, start - boundary), max(0, end - boundary))
        for start, end in offsets[response_idx:]
    ]
    positive_runs = _token_runs(response_offsets, positive_char_spans)
    return torch.tensor(input_ids, dtype=torch.int64), response_idx, positive_runs
