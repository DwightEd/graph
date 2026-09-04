"""Automatic internal transition, prompt-revisit and future-anchor discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RhythmSignals:
    route_change: np.ndarray
    prompt_share: np.ndarray
    evidence_share: np.ndarray
    history_share: np.ndarray
    prompt_lift: np.ndarray
    evidence_lift: np.ndarray
    history_lift: np.ndarray
    nonlocality: np.ndarray
    prompt_breadth: np.ndarray
    predictor_reuse: np.ndarray
    future_influence: np.ndarray
    prompt_delta: np.ndarray
    evidence_delta: np.ndarray
    nonlocal_delta: np.ndarray
    transition_peaks: np.ndarray
    prompt_peaks: np.ndarray
    review_peaks: np.ndarray
    anchor_peaks: np.ndarray
    prompt_paired_anchor: np.ndarray
    review_paired_anchor: np.ndarray
    prompt_coupling_rate: float
    prompt_null_rate: float
    prompt_median_lag: float
    review_coupling_rate: float
    review_null_rate: float
    review_median_lag: float


def finite_layer_mean(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = np.isfinite(values).sum(axis=0)
    total = np.nansum(values, axis=0)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def rolling_delta(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    for index in range(window, len(values)):
        history = values[index - window : index]
        history = history[np.isfinite(history)]
        if len(history) and np.isfinite(values[index]):
            result[index] = values[index] - np.median(history)
    return result


def robust_z(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    center = np.median(values[finite])
    scale = 1.4826 * np.median(np.abs(values[finite] - center))
    if scale <= 1e-12:
        scale = np.std(values[finite])
    result[finite] = 0.0 if scale <= 1e-12 else (values[finite] - center) / scale
    return result


def local_peaks(values, quantile: float, min_gap: int = 2) -> np.ndarray:
    score = np.asarray(values, dtype=np.float64)
    selected = np.zeros(len(score), dtype=bool)
    finite = score[np.isfinite(score)]
    if not len(finite):
        return selected
    threshold = max(0.0, float(np.quantile(finite, quantile)))
    candidates = [
        index
        for index, value in enumerate(score)
        if np.isfinite(value)
        and value >= threshold
        and value > 0
        and (index == 0 or value >= score[index - 1] or not np.isfinite(score[index - 1]))
        and (index + 1 == len(score) or value >= score[index + 1] or not np.isfinite(score[index + 1]))
    ]
    for index in sorted(candidates, key=lambda item: score[item], reverse=True):
        if not selected[max(0, index - min_gap) : index + min_gap + 1].any():
            selected[index] = True
    return selected


def pair_peaks(event, anchor, max_lag: int) -> np.ndarray:
    event_index = np.flatnonzero(event)
    anchor_index = np.flatnonzero(anchor)
    paired = np.full(len(event), -1, dtype=np.int64)
    for start in event_index:
        candidate = anchor_index[(anchor_index >= start) & (anchor_index <= start + max_lag)]
        if len(candidate):
            paired[start] = int(candidate[0])
    return paired


def coupling_rate(event, anchor, max_lag: int) -> float:
    count = int(np.count_nonzero(event))
    if not count:
        return float("nan")
    return float(np.count_nonzero(pair_peaks(event, anchor, max_lag) >= 0) / count)


def circular_null(event, anchor, max_lag: int) -> float:
    if len(event) < 3 or not np.any(event) or not np.any(anchor):
        return float("nan")
    rates = np.asarray(
        [coupling_rate(event, np.roll(anchor, shift), max_lag) for shift in range(1, len(anchor))],
        dtype=np.float64,
    )
    return float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")


def coupling_summary(event, anchor, max_lag: int) -> tuple[np.ndarray, float, float, float]:
    paired = pair_peaks(event, anchor, max_lag)
    rate = coupling_rate(event, anchor, max_lag)
    null = circular_null(event, anchor, max_lag)
    matched = np.flatnonzero(paired >= 0)
    lag = paired[matched] - matched
    return paired, rate, null, float(np.median(lag)) if len(lag) else float("nan")


def build_rhythm(
    trace,
    *,
    revisit_window: int = 4,
    peak_quantile: float = 0.9,
    max_lag: int = 3,
) -> RhythmSignals:
    """Find generic route transitions before asking whether they re-anchor."""

    route_change = finite_layer_mean(trace.route_change)
    prompt_share = finite_layer_mean(trace.prompt_share)
    evidence_share = finite_layer_mean(trace.evidence_share)
    history_share = finite_layer_mean(trace.history_share)
    prompt_lift = finite_layer_mean(trace.prompt_lift)
    evidence_lift = finite_layer_mean(trace.evidence_lift)
    history_lift = finite_layer_mean(trace.history_lift)
    nonlocality = finite_layer_mean(trace.nonlocality)
    prompt_breadth = finite_layer_mean(trace.prompt_breadth)
    predictor_reuse = finite_layer_mean(trace.predictor_reuse)
    future_influence = finite_layer_mean(trace.future_influence)

    prompt_delta = rolling_delta(prompt_share, revisit_window)
    evidence_delta = rolling_delta(evidence_share, revisit_window)
    nonlocal_delta = rolling_delta(nonlocality, revisit_window)
    transition_peaks = local_peaks(route_change, peak_quantile)
    prompt_peaks = local_peaks(prompt_delta, peak_quantile)
    review_peaks = local_peaks(nonlocal_delta, peak_quantile)
    anchor_peaks = local_peaks(future_influence, peak_quantile)

    prompt_paired, prompt_rate, prompt_null, prompt_lag = coupling_summary(
        prompt_peaks, anchor_peaks, max_lag
    )
    review_paired, review_rate, review_null, review_lag = coupling_summary(
        review_peaks, anchor_peaks, max_lag
    )
    return RhythmSignals(
        route_change=route_change,
        prompt_share=prompt_share,
        evidence_share=evidence_share,
        history_share=history_share,
        prompt_lift=prompt_lift,
        evidence_lift=evidence_lift,
        history_lift=history_lift,
        nonlocality=nonlocality,
        prompt_breadth=prompt_breadth,
        predictor_reuse=predictor_reuse,
        future_influence=future_influence,
        prompt_delta=prompt_delta,
        evidence_delta=evidence_delta,
        nonlocal_delta=nonlocal_delta,
        transition_peaks=transition_peaks,
        prompt_peaks=prompt_peaks,
        review_peaks=review_peaks,
        anchor_peaks=anchor_peaks,
        prompt_paired_anchor=prompt_paired,
        review_paired_anchor=review_paired,
        prompt_coupling_rate=prompt_rate,
        prompt_null_rate=prompt_null,
        prompt_median_lag=prompt_lag,
        review_coupling_rate=review_rate,
        review_null_rate=review_null,
        review_median_lag=review_lag,
    )
