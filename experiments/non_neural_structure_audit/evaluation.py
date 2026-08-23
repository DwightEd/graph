"""Label-aware tests run only after structure scores are frozen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedRows:
    score: np.ndarray
    labels: np.ndarray
    token_index: np.ndarray


def align_query_to_next_token(
    score: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
) -> AlignedRows:
    """Align a post-token query at t with the token predicted at t + 1."""

    score = np.asarray(score)
    labels = np.asarray(labels)
    eligible = np.asarray(eligible, dtype=bool)
    if score.shape[0] != labels.shape[0] or labels.shape != eligible.shape:
        raise ValueError("score, labels, and eligible mask must share token rows")
    selected = eligible[1:]
    return AlignedRows(
        score=score[:-1][selected],
        labels=labels[1:][selected],
        token_index=np.arange(1, len(labels), dtype=np.int32)[selected],
    )
