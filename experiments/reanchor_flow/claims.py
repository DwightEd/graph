"""Label-free claim-like span boundaries for re-anchor discovery."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_END = re.compile(r"(?:[.!?;。！？；]+[\"'”’\)\]]*\s*$|\n+\s*$)")

RESPONSE_START = 0
NATURAL_BOUNDARY = 1
FORCED_CHUNK = 2


@dataclass(frozen=True)
class ClaimSpan:
    """Half-open absolute token span inside a response."""

    start: int
    stop: int
    boundary_kind: int


def split_claims(
    tokenizer,
    token_ids,
    response_start: int,
    *,
    min_tokens: int = 2,
    max_tokens: int = 96,
) -> list[ClaimSpan]:
    """Split a response into deterministic sentence-like claim proxies.

    The boundaries use decoded punctuation only. They are independent of model
    routes and hallucination labels, but they are not semantic claim labels.
    """

    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if not 0 <= response_start < len(ids):
        return []
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("claim length bounds are inconsistent")

    pieces = [
        tokenizer.decode(
            [int(token)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token in ids[response_start:]
    ]
    spans: list[ClaimSpan] = []
    start = response_start
    boundary_kind = RESPONSE_START
    for offset, piece in enumerate(pieces):
        position = response_start + offset
        length = position + 1 - start
        natural_end = length >= min_tokens and bool(_END.search(piece))
        forced_end = length >= max_tokens
        if natural_end or forced_end:
            spans.append(ClaimSpan(start, position + 1, boundary_kind))
            start = position + 1
            boundary_kind = NATURAL_BOUNDARY if natural_end else FORCED_CHUNK

    response_stop = response_start + len(pieces)
    if response_stop - start >= min_tokens:
        spans.append(ClaimSpan(start, response_stop, boundary_kind))
    return spans
