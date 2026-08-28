"""Causal alignment shared by every attention-mechanism audit branch.

For a prompt of length ``P`` followed by ``R`` response tokens, response token
``t`` lives at absolute position ``P + t`` and is predicted by the logit at
``P + t - 1``.  The sparse attention cache contains response-query rows only,
so its row ``t - 1`` can be attached to response token ``t`` only for
``t >= 1``.  In particular, the first response token is available to a full
model replay but has no cached routing row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _integer_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True)
class PredictorAlignment:
    """The exact predecessor-logit view of one prompt/response sequence.

    All arrays have one row per factual response token.  ``cached_query_index``
    is ``-1`` where the formal response-only attention cache cannot provide the
    predictor query; consumers must use ``cached_route_available`` rather than
    treating that row as an observed zero.
    """

    response_start: int
    token_count: int
    predictor_position: np.ndarray
    target_position: np.ndarray
    target_token_id: np.ndarray
    cached_query_index: np.ndarray
    cached_route_available: np.ndarray

    @property
    def prompt_length(self) -> int:
        return self.response_start

    @property
    def response_length(self) -> int:
        return self.token_count - self.response_start

    @property
    def predictor_positions(self) -> np.ndarray:
        """Plural alias used by array-oriented callers."""

        return self.predictor_position

    @property
    def target_positions(self) -> np.ndarray:
        return self.target_position

    @property
    def target_token_ids(self) -> np.ndarray:
        return self.target_token_id

    @property
    def cached_query_indices(self) -> np.ndarray:
        return self.cached_query_index

    def validate(self) -> "PredictorAlignment":
        length = self.response_length
        if not 0 < self.response_start < self.token_count:
            raise ValueError("response_start must split a non-empty sequence")
        fields = {
            "predictor_position": self.predictor_position,
            "target_position": self.target_position,
            "target_token_id": self.target_token_id,
            "cached_query_index": self.cached_query_index,
            "cached_route_available": self.cached_route_available,
        }
        if any(np.asarray(value).shape != (length,) for value in fields.values()):
            raise ValueError("alignment arrays must have one row per response token")
        expected_target = np.arange(
            self.response_start, self.token_count, dtype=np.int64
        )
        expected_predictor = expected_target - 1
        expected_query = np.arange(-1, length - 1, dtype=np.int64)
        expected_available = np.arange(length) >= 1
        if not np.array_equal(self.target_position, expected_target):
            raise ValueError("target positions are not the response suffix")
        if not np.array_equal(self.predictor_position, expected_predictor):
            raise ValueError("predictors must be the immediate predecessor positions")
        if not np.array_equal(self.cached_query_index, expected_query):
            raise ValueError("cached query indices are not predecessor aligned")
        if not np.array_equal(self.cached_route_available, expected_available):
            raise ValueError("cached route availability must be false only at token zero")
        if np.asarray(self.cached_route_available).dtype != np.bool_:
            raise ValueError("cached_route_available must be boolean")
        return self


def predecessor_alignment(
    token_ids,
    response_start: int,
    *,
    cached_query_count: int | None = None,
) -> PredictorAlignment:
    """Build the unique valid teacher-forcing alignment for a cached sample.

    Args:
        token_ids: Full ``prompt + factual response`` token sequence.
        response_start: Absolute index ``P`` of the first response token.
        cached_query_count: Number of cached response-query rows.  The formal
            cache has exactly ``R`` rows, including a final query that predicts
            a token outside the recorded response.  Supplying a different
            count is rejected instead of partially filling an audit trace.
    """

    tokens = _integer_vector(token_ids, "token_ids")
    response_start = int(response_start)
    if not 0 < response_start < tokens.size:
        raise ValueError("response_start must split prompt and response tokens")
    response_length = tokens.size - response_start
    if cached_query_count is not None and int(cached_query_count) != response_length:
        raise ValueError("cached_query_count must equal the response length")

    target = np.arange(response_start, tokens.size, dtype=np.int64)
    predictor = target - 1
    cached_query = np.arange(-1, response_length - 1, dtype=np.int64)
    available = np.arange(response_length) >= 1
    return PredictorAlignment(
        response_start=response_start,
        token_count=int(tokens.size),
        predictor_position=predictor,
        target_position=target,
        target_token_id=tokens[target].copy(),
        cached_query_index=cached_query,
        cached_route_available=available,
    ).validate()


build_predecessor_alignment = predecessor_alignment


def gather_chosen_logits(logits, alignment: PredictorAlignment) -> np.ndarray:
    """Gather factual-token logits from their predecessor positions.

    ``logits`` must be ``[sequence, vocabulary]``.  Requiring the sequence
    axis explicitly prevents accidental gathering from same-token rows.
    """

    values = np.asarray(logits)
    alignment.validate()
    if values.ndim != 2 or values.shape[0] < alignment.token_count:
        raise ValueError("logits must be [sequence, vocabulary]")
    targets = alignment.target_token_id
    if bool(((targets < 0) | (targets >= values.shape[1])).any()):
        raise ValueError("a target token ID is outside the vocabulary")
    return values[alignment.predictor_position, targets]


def align_cached_queries(
    cached_values,
    alignment: PredictorAlignment,
    *,
    unavailable_value=np.nan,
) -> np.ndarray:
    """Shift response-query rows onto the tokens they actually predict.

    Cache query ``i`` is written to response-token row ``i + 1``.  The final
    cache query is intentionally unused because it predicts the unrecorded next
    token.  The first output row receives ``unavailable_value``.
    """

    values = np.asarray(cached_values)
    alignment.validate()
    if values.ndim < 1 or values.shape[0] != alignment.response_length:
        raise ValueError("cached_values must have one row per response query")
    dtype = np.result_type(values.dtype, np.asarray(unavailable_value).dtype)
    output = np.full(
        (alignment.response_length, *values.shape[1:]),
        unavailable_value,
        dtype=dtype,
    )
    if alignment.response_length > 1:
        output[1:] = values[:-1]
    return output
