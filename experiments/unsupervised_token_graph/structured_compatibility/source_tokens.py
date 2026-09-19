"""Exact source_info-to-prompt token mapping; no hallucination labels."""

import numpy as np


def source_strings(record):
    strings = []

    def visit(value):
        if isinstance(value, str) and len(value.strip()) >= 3:
            strings.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record.get("source_info"))
    return strings


def prompt_text_offsets(tokenizer, prompt_ids):
    text = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    np.testing.assert_array_equal(encoded["input_ids"], prompt_ids)
    return text, np.asarray(encoded["offset_mapping"])


def source_token_positions(tokenizer, prompt_ids, source_record):
    text, offsets = prompt_text_offsets(tokenizer, prompt_ids)
    positions = []
    for string in source_strings(source_record):
        if text.count(string) != 1:
            continue
        start = text.index(string)
        stop = start + len(string)
        hits = (offsets[:, 0] < stop) & (offsets[:, 1] > start)
        positions.extend(np.flatnonzero(hits).tolist())
    return np.unique(positions).astype(int)
