"""Punctuation boundaries used only as visual references."""

from __future__ import annotations

import re

import numpy as np

_END = re.compile(r"(?:[.!?;。！？；]+[\"'”’\)\]]*\s*$|\n+\s*$)")


def sentence_boundaries(tokenizer, token_ids, response_start: int) -> np.ndarray:
    """Return absolute starts after punctuation; never used to discover peaks."""

    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    pieces = [
        tokenizer.decode(
            [int(token)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token in ids[response_start:]
    ]
    boundary = [response_start]
    for offset, piece in enumerate(pieces[:-1]):
        if _END.search(piece):
            boundary.append(response_start + offset + 1)
    return np.asarray(sorted(set(boundary)), dtype=np.int64)
