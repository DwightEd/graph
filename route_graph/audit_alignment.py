"""Exact replay coordinates and text views; no semantic or gold decisions."""

import re

from route_graph.evidence_anchor import text_units
from route_graph.frozen_reader import digest


def align_row(row, tokenizer):
    prompt = tokenizer(
        row["prompt"], add_special_tokens=False, return_offsets_mapping=True
    )
    response = tokenizer(
        row["response"], add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [tokenizer.bos_token_id] + prompt["input_ids"] + response["input_ids"]
    if ids != row["token_ids"] or len(prompt["input_ids"]) + 1 != row["prompt_length"]:
        raise ValueError("input IDs differ from frozen observer tokenization")
    offsets = [(0, 0)] + [tuple(p) for p in prompt["offset_mapping"]]
    source_start, source_end = row["source_span"]
    source_offsets = [
        (max(0, a - source_start), min(source_end - source_start, b - source_start))
        if a < source_end and b > source_start
        else (0, 0)
        for a, b in offsets
    ]
    source_keys = [i for i, present in enumerate(row["source_mask"]) if present]
    expected = [
        i
        for i, (a, b) in enumerate(offsets)
        if b > a and a < source_end and b > source_start
    ]
    if source_keys != expected:
        raise ValueError("source role mask differs from exact source span")
    return {
        "source_offsets": source_offsets,
        "response_offsets": response["offset_mapping"],
        "prompt_length": row["prompt_length"],
        "source_keys": source_keys,
    }


def span_keys(alignment, role, span, prefix_length=None):
    offsets = alignment["source_offsets" if role == "source" else "response_offsets"]
    shift = 0 if role == "source" else alignment["prompt_length"]
    a, b = span
    return [
        i + shift
        for i, (left, right) in enumerate(offsets)
        if left < b
        and right > a
        and right > left
        and (prefix_length is None or i + shift < prefix_length)
    ]


def key_view(row, alignment, role, keys):
    offsets = alignment["source_offsets" if role == "source" else "response_offsets"]
    shift = 0 if role == "source" else alignment["prompt_length"]
    text = (
        row["prompt"][slice(*row["source_span"])]
        if role == "source"
        else row["response"]
    )
    spans = []
    for key in sorted(keys):
        a, b = offsets[key - shift]
        if b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return {
        "role": role,
        "keys": list(keys),
        "spans": spans,
        "segments": [text[a:b] for a, b in spans],
    }


def text_nodes(row, alignment, prefix_length):
    """Punctuation leaves, with <=64-token subdivisions; full context stays visible."""
    result = []
    for role, text in (
        ("source", row["prompt"][slice(*row["source_span"])]),
        ("history", row["response"]),
    ):
        for unit in text_units(text):
            keys = span_keys(
                alignment, role, [unit["start"], unit["end"]], prefix_length
            )
            for start in range(0, len(keys), 64):
                view = key_view(row, alignment, role, keys[start : start + 64])
                view["id"] = digest(view)[:20]
                result.append(view)
    return result


def slot_masks(contrast, metadata, tokenizer):
    """Partition complete B/A log probabilities into edited-slot and context terms."""
    result = []
    for branch, text in enumerate((metadata["original"], metadata["alternative"])):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        expected = list(
            contrast.prefix[contrast.prompt_length :] + contrast.continuations[branch]
        )
        if encoded["input_ids"] != expected:
            raise ValueError("slot metadata differs from complete event token IDs")
        offsets = encoded["offset_mapping"]
        relative_start = len(contrast.prefix) - contrast.prompt_length
        a, b = metadata["answer_spans"][branch]
        mask = [left < b and right > a for left, right in offsets[relative_start:]]
        if len(mask) != len(contrast.continuations[branch]):
            raise ValueError("slot/log probability alignment mismatch")
        result.append(mask)
    return result


def assertion_words(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
