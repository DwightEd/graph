"""Label-blind RAGTruth sample loading and tokenization."""

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


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_ragtruth_samples(
    dataset_path: str | Path,
    *,
    split: str,
    generator_model: str,
    task_type: str = "all",
) -> list[RagTruthSample]:
    dataset_path = Path(dataset_path)
    sources = {str(row["source_id"]): row for row in _read_jsonl(dataset_path / "source_info.jsonl")}
    requested_model = _normalized_name(generator_model)
    samples: list[RagTruthSample] = []
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
            source_id=source_id, response_id=str(response["id"]), prompt=str(source["prompt"]),
            response=str(response["response"]), split=str(response["split"]),
        ))
    return samples


def tokenize_ragtruth_sample(tokenizer: Any, *, prompt: str, response: str) -> tuple[torch.Tensor, int]:
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    encoding = tokenizer(rendered_prompt + response, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    boundary = len(rendered_prompt)
    response_idx = next((index for index, (start, end) in enumerate(offsets) if start >= boundary and end > boundary), None)
    if response_idx is None or response_idx == 0 or response_idx >= len(input_ids):
        raise ValueError("response does not form an aligned token suffix")
    if any(start < boundary < end for start, end in offsets):
        raise ValueError("a token crosses the prompt/response boundary")
    return torch.tensor(input_ids, dtype=torch.int64), response_idx
