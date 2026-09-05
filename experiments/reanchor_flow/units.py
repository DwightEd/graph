"""Passage, sentence, field, and response units used as ETCC graph roots."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from experiments.common.ragtruth_alignment import (
    canonical_task_type,
    render_historical_prompt,
)

from .worlds import SourceUnits


@dataclass(frozen=True)
class UnitSpan:
    name: str
    kind: str
    start: int
    stop: int


def cover_spans(text: str, spans: list[UnitSpan]) -> list[UnitSpan]:
    """Assign separators to adjacent semantic units without creating gaps."""

    spans = sorted(spans, key=lambda span: span.start)
    if not spans:
        raise ValueError("external evidence contains no source units")
    return [
        UnitSpan(
            span.name,
            span.kind,
            0 if index == 0 else spans[index - 1].stop,
            span.stop if index + 1 < len(spans) else len(text),
        )
        for index, span in enumerate(spans)
    ]


def passage_spans(text: str) -> list[UnitSpan]:
    starts = [0, *(match.end() for match in re.finditer(r"\n[ \t]*\n+", text))]
    spans = []
    for number, (start, stop) in enumerate(
        zip(starts, (*starts[1:], len(text)), strict=True), start=1
    ):
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append(UnitSpan(f"passage:{number}", "passage", left, right))
    return cover_spans(text, spans)


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


def sentence_spans(text: str) -> list[UnitSpan]:
    spans = []
    start = 0
    for whitespace in re.finditer(r"\s+", text):
        prefix = text[start : whitespace.start()].rstrip()
        if not prefix:
            start = whitespace.end()
            continue
        tail = prefix.rstrip("\"'\u201d\u2019)]}")
        word = tail.rsplit(maxsplit=1)[-1].casefold() if tail else ""
        boundary = "\n\n" in whitespace.group() or (
            tail.endswith((".", "!", "?"))
            and word not in ABBREVIATIONS
            and not re.fullmatch(r"[a-z]\.", word)
        )
        if boundary:
            spans.append(
                UnitSpan(
                    f"sentence:{len(spans) + 1}",
                    "sentence",
                    start,
                    whitespace.start(),
                )
            )
            start = whitespace.end()
    if text[start:].strip():
        spans.append(
            UnitSpan(
                f"sentence:{len(spans) + 1}",
                "sentence",
                start,
                len(text.rstrip()),
            )
        )
    return cover_spans(text, spans)


def field_spans(text: str) -> list[UnitSpan]:
    """Use structured leaf values and list items as Data2txt source units."""

    root = ast.parse(text, mode="eval").body
    if not isinstance(root, ast.Dict):
        raise TypeError("Data2txt source_info must render as a dictionary")
    byte_offset = [0]
    for character in text:
        byte_offset.append(byte_offset[-1] + len(character.encode("utf-8")))
    line_byte_offset = [0]
    for line in text.splitlines(keepends=True):
        line_byte_offset.append(
            line_byte_offset[-1] + len(line.encode("utf-8"))
        )

    def character_offset(offset: int) -> int:
        return int(np.searchsorted(byte_offset, offset))

    def start(node: ast.AST) -> int:
        absolute = line_byte_offset[node.lineno - 1] + node.col_offset
        return character_offset(absolute)

    def stop(node: ast.AST) -> int:
        absolute = line_byte_offset[node.end_lineno - 1] + node.end_col_offset
        return character_offset(absolute)

    spans: list[UnitSpan] = []

    def visit(mapping: ast.Dict, path: tuple[str, ...], prefix: int | None = None):
        for index, (key, value) in enumerate(
            zip(mapping.keys, mapping.values, strict=True)
        ):
            name = str(ast.literal_eval(key))
            field_path = (*path, name)
            left = prefix if index == 0 and prefix is not None else start(key)
            if isinstance(value, ast.Dict) and value.keys:
                visit(value, field_path, left)
            elif isinstance(value, ast.List) and value.elts:
                for item_index, item in enumerate(value.elts):
                    item_start = left if item_index == 0 else start(item)
                    spans.append(
                        UnitSpan(
                            ".".join((*field_path, str(item_index))),
                            "field",
                            item_start,
                            stop(item),
                        )
                    )
            else:
                spans.append(
                    UnitSpan(".".join(field_path), "field", left, stop(value))
                )

    visit(root, ())
    return cover_spans(text, spans)


def evidence_spans(source: Mapping[str, Any]) -> tuple[str, list[UnitSpan]]:
    task = canonical_task_type(source["task_type"])
    information = source["source_info"]
    if task == "QA":
        evidence = str(information["passages"]).removesuffix("\n")
        return evidence, passage_spans(evidence)
    if task == "Summary":
        evidence = str(information)
        return evidence, sentence_spans(evidence)
    evidence = str(information)
    return evidence, field_spans(evidence)


def build_source_units(
    source: Mapping[str, Any],
    tokenizer,
    token_ids,
    response_start: int,
) -> SourceUnits:
    """Align semantic prompt units and strict response-token carrier units."""

    prompt = str(source["prompt"])
    rendered = render_historical_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    rebuilt = np.asarray(encoded["input_ids"], dtype=np.int64)
    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    if rebuilt.shape != (response_start,) or not np.array_equal(
        rebuilt, token_ids[:response_start].numpy()
    ):
        raise ValueError("rebuilt prompt does not match the teacher-forced prefix")

    evidence, spans = evidence_spans(source)
    prompt_start = rendered.rfind(prompt)
    evidence_in_prompt = prompt.find(evidence)
    if prompt_start < 0 or evidence_in_prompt < 0:
        raise ValueError("rendered prompt does not contain the declared evidence")
    evidence_start = prompt_start + evidence_in_prompt
    absolute = np.asarray(
        [
            (evidence_start + span.start, evidence_start + span.stop)
            for span in spans
        ],
        dtype=np.int64,
    )
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    overlap = np.maximum(
        0,
        np.minimum(offsets[:, None, 1], absolute[None, :, 1])
        - np.maximum(offsets[:, None, 0], absolute[None, :, 0]),
    )
    prompt_unit = np.where(overlap.max(axis=1) > 0, overlap.argmax(axis=1) + 1, 0)

    source_count = len(token_ids) - 1
    response_sources = max(source_count - response_start, 0)
    response_ids = np.arange(
        len(spans) + 1,
        len(spans) + 1 + response_sources,
        dtype=np.int64,
    )
    unit_ids = np.concatenate((prompt_unit, response_ids))
    names = (
        "other_prompt",
        *(span.name for span in spans),
        *(f"response:{position}" for position in range(response_start, source_count)),
    )
    kinds = (
        "other_prompt",
        *(span.kind for span in spans),
        *("response" for _ in range(response_sources)),
    )
    return SourceUnits(torch.from_numpy(unit_ids).long(), names, kinds).check(
        source_count
    )
