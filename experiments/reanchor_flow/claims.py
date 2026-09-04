"""Label-free claim-like span boundaries for re-anchor discovery."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_END = re.compile(r"(?:[.!?;。！？；]+[\"'”’\)\]]*\s*$|\n+\s*$)")


@dataclass(frozen=True)
class ClaimSpan:
    """Half-open absolute token span inside the response."""

    start: int
    stop: int

    @property
    def sink(self) -> int:
        return self.stop - 1

    @property
    def length(self) -> int:
        return self.stop - self.start


def token_pieces(tokenizer, token_ids) -> list[str]:
    ids = np.asarray(token_ids, dtype=np.int64).tolist()
    return [
        tokenizer.decode(
            [int(token)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token in ids
    ]


def split_claims(
    tokenizer,
    token_ids,
    response_start: int,
    *,
    min_tokens: int = 2,
    max_tokens: int = 96,
) -> list[ClaimSpan]:
    """Split a response into deterministic sentence-like claim proxies.

    These spans are deliberately independent of attention and hallucination
    labels. They are a pilot boundary proxy, not semantic claim annotation.
    """

    ids = np.asarray(token_ids, dtype=np.int64)
    if not 0 <= response_start < len(ids):
        return []
    pieces = token_pieces(tokenizer, ids[response_start:])
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("claim length bounds are inconsistent")

    spans: list[ClaimSpan] = []
    start = response_start
    for offset, piece in enumerate(pieces):
        position = response_start + offset
        length = position + 1 - start
        boundary = bool(_END.search(piece))
        forced = length >= max_tokens
        if (boundary and length >= min_tokens) or forced:
            spans.append(ClaimSpan(start, position + 1))
            start = position + 1

    response_stop = response_start + len(pieces)
    if start < response_stop:
        tail = ClaimSpan(start, response_stop)
        if tail.length >= min_tokens:
            spans.append(tail)
        elif spans:
            previous = spans[-1]
            spans[-1] = ClaimSpan(previous.start, tail.stop)

    return spans
