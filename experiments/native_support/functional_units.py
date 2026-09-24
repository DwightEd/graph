"""Deterministic original-answer intervals; every selected token is retained."""

import re


def answer_units(response):
    """Punctuation boundaries are objective intervals, not semantic judgments."""
    pieces = response["token_text"][response["prompt_length"]:]
    start, unit = 0, ""
    for stop, piece in enumerate(pieces, 1):
        unit += piece
        if re.fullmatch(r'\s*(?:\d+[.)]|[-*•])\s*', unit):
            continue
        if "\n" in piece or re.search(r'[.!?][\s\"\u201d\')\]]*$', piece):
            yield start, stop
            start, unit = stop, ""
    if start < len(pieces):
        yield start, len(pieces)


def capture_units(response, start_target, stop_target, max_tokens):
    """Clip the explicit selection and split long units instead of skipping them."""
    answer_length = len(response["token_ids"]) - response["prompt_length"]
    stop_target = answer_length if stop_target is None else min(stop_target, answer_length)
    for unit_start, unit_stop in answer_units(response):
        start, stop = max(unit_start, start_target), min(unit_stop, stop_target)
        for chunk_start in range(start, stop, max_tokens):
            yield dict(start=chunk_start, stop=min(chunk_start + max_tokens, stop),
                       punctuation_start=unit_start, punctuation_stop=unit_stop)
