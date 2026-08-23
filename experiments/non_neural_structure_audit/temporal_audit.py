"""Causally aligned pre-onset and post-onset structure summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EvaluationConfig
from .evaluation_data import FrozenSample
from .features import RELATION_NAMES


@dataclass(frozen=True)
class AlignedRows:
    score: np.ndarray
    labels: np.ndarray
    token_index: np.ndarray


def align_query_to_next_token(
    score: np.ndarray, labels: np.ndarray, eligible: np.ndarray
) -> AlignedRows:
    """Align the query after token t with the model target at token t + 1."""

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


def pre_onset_slope(
    score: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    window: int,
) -> float | None:
    """Slope of scores available before the first positive token is emitted."""

    positive = np.flatnonzero(
        (np.asarray(labels) == 1) & np.asarray(eligible, dtype=bool)
    )
    if not len(positive):
        return None
    aligned = align_query_to_next_token(score, labels, eligible)
    values = aligned.score[aligned.token_index <= int(positive[0])][-window:]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _slope_before_cutoff(
    score: np.ndarray, eligible: np.ndarray, cutoff: int, window: int
) -> float | None:
    target = np.arange(1, len(score))[np.asarray(eligible, dtype=bool)[1:]]
    values = score[:-1][np.asarray(eligible, dtype=bool)[1:]][target <= cutoff][
        -window:
    ]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _post_onset_change(
    score: np.ndarray, eligible: np.ndarray, onset: int, window: int
) -> float | None:
    target = np.arange(1, len(score))[np.asarray(eligible, dtype=bool)[1:]]
    values = score[:-1][np.asarray(eligible, dtype=bool)[1:]][target > onset]
    if len(values) < 2 * window:
        return None
    return float(values[-window:].mean() - values[:window].mean())


def _effect(values: list[float]) -> tuple[int, float | None, float | None]:
    if not values:
        return 0, None, None
    array = np.asarray(values, dtype=np.float64)
    deviation = array.std(ddof=1) if len(array) > 1 else 0.0
    return (
        len(array),
        float(array.mean()),
        float(array.mean() / deviation) if deviation > 0 else None,
    )


def temporal_rows(
    samples: list[FrozenSample], config: EvaluationConfig
) -> list[dict[str, object]]:
    hallucinating = [
        sample for sample in samples if (sample.labels & sample.eligible).any()
    ]
    correct = [
        sample for sample in samples if not (sample.labels & sample.eligible).any()
    ]
    fractions = [
        float(
            np.flatnonzero(sample.labels & sample.eligible)[0]
            / max(len(sample.labels) - 1, 1)
        )
        for sample in hallucinating
    ]
    rows = []
    for relation, name in enumerate(RELATION_NAMES):
        pre = []
        for sample in hallucinating:
            value = pre_onset_slope(
                sample.relation[:, relation],
                sample.labels,
                sample.eligible,
                window=config.onset_window,
            )
            if value is not None:
                pre.append(value)
        pseudo = []
        for index, sample in enumerate(correct):
            if not fractions:
                break
            cutoff = round(fractions[index % len(fractions)] * (len(sample.labels) - 1))
            value = _slope_before_cutoff(
                sample.relation[:, relation],
                sample.eligible,
                cutoff,
                config.onset_window,
            )
            if value is not None:
                pseudo.append(value)
        lockin = []
        for sample in hallucinating:
            onset = int(np.flatnonzero(sample.labels & sample.eligible)[0])
            value = _post_onset_change(
                sample.relation[:, relation],
                sample.eligible,
                onset,
                config.onset_window,
            )
            if value is not None:
                lockin.append(value)

        pre_count, pre_mean, pre_dz = _effect(pre)
        pseudo_count, pseudo_mean, _ = _effect(pseudo)
        lockin_count, lockin_mean, lockin_dz = _effect(lockin)
        rows.append(
            {
                "relation": name,
                "pre_onset_responses": pre_count,
                "pre_onset_slope": pre_mean,
                "pre_onset_standardized_mean": pre_dz,
                "pseudo_onset_responses": pseudo_count,
                "pseudo_onset_slope": pseudo_mean,
                "pre_minus_pseudo_slope": None
                if pre_mean is None or pseudo_mean is None
                else pre_mean - pseudo_mean,
                "lockin_responses": lockin_count,
                "late_minus_early": lockin_mean,
                "lockin_standardized_mean": lockin_dz,
            }
        )
    return rows
