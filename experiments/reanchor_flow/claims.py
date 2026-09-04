"""Label-free claim-like span boundaries for re-anchor discovery."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_END = re.compile(r"(?:[.!?;。！？；]+[\"'”’\)\]]*\s*$|\n+\s*$)")


@dataclass(frozen=True)
class ClaimSpan:
    """Half-open absolute token span inside a response."""

    start: int
    stop: int

    @property
    def sink(self) -> int:
        """Last generated token in the span."""

        return self.stop - 1

    @property
    def length(self) -> int:
        return self.stop - self.start


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
    for offset, piece in enumerate(pieces):
        position = response_start + offset
        length = position + 1 - start
        if (length >= min_tokens and _END.search(piece)) or length >= max_tokens:
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
