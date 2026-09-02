"""Label-free RAGTruth samples and their prompt evidence units."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research_dataset import open_research_dataset

HISTORICAL_SYSTEM_PROMPT = "You are a helpful assistant."
TASK_TYPES = ("QA", "Summary", "Data2txt")
OTHER_PROMPT = 0


@dataclass(frozen=True)
class PromptUnits:
    """Prompt-token roots: zero is other prompt and 1..E are evidence units."""

    token_unit_id: torch.Tensor
    evidence_name: tuple[str, ...]
    evidence_char_span: torch.Tensor

    @property
    def evidence_mask(self) -> torch.Tensor:
        return self.token_unit_id > OTHER_PROMPT

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_name)

    @property
    def unit_count(self) -> int:
        return self.evidence_count + 1


@dataclass(frozen=True)
class RouteSample:
    """One teacher-forced sequence with label-free information roots."""

    sample_id: str
    source_id: str
    split: str
    task_type: str
    data_source: str | None
    generator_model: str | None
    token_ids: torch.Tensor
    response_start: int
    prompt_units: PromptUnits

    @property
    def response_count(self) -> int:
        return len(self.token_ids) - self.response_start

    @property
    def response_root_start(self) -> int:
        return self.prompt_units.unit_count

    @property
    def evidence_unit_count(self) -> int:
        return self.prompt_units.evidence_count

    @property
    def root_count(self) -> int:
        return self.response_root_start + self.response_count

    @property
    def token_root_unit_id(self) -> torch.Tensor:
        response_roots = torch.arange(
            self.response_root_start,
            self.root_count,
            dtype=torch.long,
        )
        return torch.cat((self.prompt_units.token_unit_id, response_roots))


def canonical_task_type(value: object) -> str:
    for task in TASK_TYPES:
        if str(value).casefold() == task.casefold():
            return task
    raise ValueError(f"unsupported task type: {value}")


def load_source_info(path: str | Path) -> dict[str, dict]:
    """Read the label-free RAGTruth source rows once, keyed by source ID."""

    sources: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            source_id = str(source["source_id"])
            if source_id in sources:
                raise ValueError(f"duplicate source_id: {source_id}")
            sources[source_id] = source
    return sources


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Recreate the chat prefix used when the attention cache was produced."""

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": HISTORICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def paragraph_units(text: str) -> list[tuple[str, int, int]]:
    """Use retrieved QA passages, rather than their individual words, as roots."""

    starts = [0]
    starts.extend(match.end() for match in re.finditer(r"\n[ \t]*\n+", text))
    anchors = []
    for number, (start, stop) in enumerate(
        zip(starts, (*starts[1:], len(text)), strict=True), start=1
    ):
        left = start
        right = stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            anchors.append((f"passage:{number}", left, right))
    return cover_evidence(text, anchors)


ABBREVIATIONS = {
    "dr.",
    "e.g.",
    "etc.",
    "fig.",
    "i.e.",
    "inc.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "prof.",
    "sr.",
    "st.",
    "u.k.",
    "u.s.",
    "vs.",
}


def sentence_units(text: str) -> list[tuple[str, int, int]]:
    """Split summary evidence into deterministic sentence-sized roots."""

    anchors = []
    start = 0
    for whitespace in re.finditer(r"\s+", text):
        prefix = text[start : whitespace.start()].rstrip()
        if not prefix:
            start = whitespace.end()
            continue
        tail = prefix.rstrip("\"'\u201d\u2019)]}")
        word = tail.rsplit(maxsplit=1)[-1].casefold() if tail else ""
        is_initial = bool(re.fullmatch(r"[a-z]\.", word))
        paragraph_end = "\n\n" in whitespace.group()
        sentence_end = (
            tail.endswith((".", "!", "?"))
            and word not in ABBREVIATIONS
            and not is_initial
        )
        if paragraph_end or sentence_end:
            anchors.append((f"sentence:{len(anchors) + 1}", start, whitespace.start()))
            start = whitespace.end()
    if text[start:].strip():
        anchors.append(
            (
                f"sentence:{len(anchors) + 1}",
                start,
                len(text.rstrip()),
            )
        )
    return cover_evidence(text, anchors)


def data_units(text: str) -> list[tuple[str, int, int]]:
    """Use structured leaf fields and whole review records as Data2txt roots."""

    root = ast.parse(text, mode="eval").body
    if not isinstance(root, ast.Dict):
        raise TypeError("Data2txt source_info must render as a dictionary")
    byte_offsets = [0]
    for character in text:
        byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))

    def character_offset(byte_offset: int) -> int:
        return int(np.searchsorted(byte_offsets, byte_offset))

    anchors: list[tuple[str, int, int]] = []

    def node_start(node: ast.AST) -> int:
        return character_offset(node.col_offset)

    def node_stop(node: ast.AST) -> int:
        return character_offset(node.end_col_offset)

    def visit(mapping: ast.Dict, path: tuple[str, ...], prefix: int | None = None):
        for index, (key, value) in enumerate(
            zip(mapping.keys, mapping.values, strict=True)
        ):
            name = str(ast.literal_eval(key))
            field_path = (*path, name)
            start = prefix if index == 0 and prefix is not None else node_start(key)
            if isinstance(value, ast.Dict) and value.keys:
                visit(value, field_path, start)
            elif isinstance(value, ast.List) and value.elts:
                for item_index, item in enumerate(value.elts):
                    item_start = start if item_index == 0 else node_start(item)
                    anchors.append(
                        (
                            ".".join((*field_path, str(item_index))),
                            item_start,
                            node_stop(item),
                        )
                    )
            else:
                anchors.append((".".join(field_path), start, node_stop(value)))

    visit(root, ())
    return cover_evidence(text, anchors)


def cover_evidence(
    text: str, anchors: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """Extend semantic anchors over separators so every evidence char has one root."""

    if not anchors:
        raise ValueError("external evidence contains no units")
    anchors.sort(key=lambda item: item[1])
    return [
        (
            name,
            0 if index == 0 else anchors[index - 1][2],
            stop if index + 1 < len(anchors) else len(text),
        )
        for index, (name, _start, stop) in enumerate(anchors)
    ]


def evidence_units(source: Mapping[str, Any]) -> tuple[str, list[tuple[str, int, int]]]:
    """Return the exact prompt substring and its task-specific unit partition."""

    task = canonical_task_type(source["task_type"])
    content = source["source_info"]
    if task == "QA":
        evidence = str(content["passages"]).removesuffix("\n")
        return evidence, paragraph_units(evidence)
    if task == "Summary":
        evidence = str(content)
        return evidence, sentence_units(evidence)
    evidence = str(content)
    return evidence, data_units(evidence)


def build_prompt_units(
    source: Mapping[str, Any],
    tokenizer: Any,
    cached_token_ids: Any,
    response_start: int,
) -> PromptUnits:
    """Align evidence-unit character spans to the exact cached prompt tokens."""

    prompt = str(source["prompt"])
    rendered = render_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    rebuilt_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    cached_prefix = torch.as_tensor(cached_token_ids).cpu().numpy()[:response_start]
    if rebuilt_ids.shape != (response_start,) or not np.array_equal(
        rebuilt_ids, cached_prefix
    ):
        raise ValueError("rebuilt prompt does not exactly match cached token prefix")

    evidence, units = evidence_units(source)
    evidence_start = rendered.index(prompt) + prompt.index(evidence)
    spans = np.asarray(
        [
            (evidence_start + start, evidence_start + stop)
            for _name, start, stop in units
        ],
        dtype=np.int64,
    )
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    overlap = np.maximum(
        0,
        np.minimum(offsets[:, None, 1], spans[None, :, 1])
        - np.maximum(offsets[:, None, 0], spans[None, :, 0]),
    )
    best = overlap.argmax(axis=1)
    token_unit = np.where(overlap.max(axis=1) > 0, best + 1, OTHER_PROMPT)
    return PromptUnits(
        token_unit_id=torch.from_numpy(token_unit).long(),
        evidence_name=tuple(name for name, _start, _stop in units),
        evidence_char_span=torch.from_numpy(spans).long(),
    )


def read_route_sample(
    sample: Any,
    source: Mapping[str, Any],
    tokenizer: Any,
) -> RouteSample:
    """Copy one canonical sample into the method and release its old cache tensor."""

    attention = sample.attention()
    try:
        token_ids = attention.token_ids.detach().cpu().long().clone()
        response_start = int(attention.response_idx)
        units = build_prompt_units(source, tokenizer, token_ids, response_start)
        return RouteSample(
            sample_id=str(sample.sample_id),
            source_id=str(sample.source_id),
            split=str(sample.split),
            task_type=canonical_task_type(source["task_type"]),
            data_source=source.get("source"),
            generator_model=sample.generator_model,
            token_ids=token_ids,
            response_start=response_start,
            prompt_units=units,
        )
    finally:
        sample.release_attention()


def iter_route_samples(
    split_root: str | Path,
    source_info: str | Path,
    tokenizer: Any,
    *,
    task_type: str | None = None,
    limit: int | None = None,
) -> Iterator[RouteSample]:
    """Yield CPU-only route samples without exposing labels or retained attention."""

    dataset = open_research_dataset(
        split_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    sources = load_source_info(source_info)
    selected_task = None if task_type is None else canonical_task_type(task_type)
    yielded = 0
    for sample_id in dataset.sample_ids:
        if limit is not None and yielded >= limit:
            return
        sample = dataset[sample_id]
        source = sources[str(sample.source_id)]
        source_task = canonical_task_type(source["task_type"])
        if selected_task is not None and source_task != selected_task:
            sample.release_attention()
            continue
        yield read_route_sample(sample, source, tokenizer)
        yielded += 1
