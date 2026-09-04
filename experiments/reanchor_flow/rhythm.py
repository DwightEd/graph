"""Automatic prompt-revisit and future-anchor discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RhythmSignals:
    route_change: np.ndarray
    prompt_revisit: np.ndarray
    evidence_revisit: np.ndarray
    history_share: np.ndarray
    prompt_breadth: np.ndarray
    future_influence: np.ndarray
    revisit_delta: np.ndarray
    evidence_delta: np.ndarray
    revisit_peaks: np.ndarray
    anchor_peaks: np.ndarray
    peak_kind: np.ndarray
    paired_anchor: np.ndarray
    coupling_rate: float
    null_rate: float
    median_lag: float


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
    if scale <= 1e-12:
        result[finite] = 0.0
    else:
        result[finite] = (values[finite] - center) / scale
    return result


def local_peaks(values, quantile: float, min_gap: int = 2) -> np.ndarray:
    """Return sparse positive local maxima above a within-sequence quantile."""

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


def pair_peaks(revisit, anchor, max_lag: int) -> np.ndarray:
    revisit_index = np.flatnonzero(revisit)
    anchor_index = np.flatnonzero(anchor)
    paired = np.full(len(revisit), -1, dtype=np.int64)
    for start in revisit_index:
        candidate = anchor_index[(anchor_index >= start) & (anchor_index <= start + max_lag)]
        if len(candidate):
            paired[start] = int(candidate[0])
    return paired


def coupling_rate(revisit, anchor, max_lag: int) -> float:
    count = int(np.count_nonzero(revisit))
    if not count:
        return float("nan")
    return float(np.count_nonzero(pair_peaks(revisit, anchor, max_lag) >= 0) / count)


def circular_null(revisit, anchor, max_lag: int) -> float:
    if len(revisit) < 3 or not np.any(revisit) or not np.any(anchor):
        return float("nan")
    rates = [
        coupling_rate(revisit, np.roll(anchor, shift), max_lag)
        for shift in range(1, len(anchor))
    ]
    rates = np.asarray(rates, dtype=np.float64)
    return float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")


def build_rhythm(
    trace,
    *,
    revisit_window: int = 4,
    peak_quantile: float = 0.9,
    max_lag: int = 3,
) -> RhythmSignals:
    """Collapse only the layer axis after retaining the token trajectory."""

    route_change = finite_layer_mean(trace.route_change)
    prompt_revisit = finite_layer_mean(trace.far_prompt_share)
    evidence_revisit = finite_layer_mean(trace.evidence_share)
    history_share = finite_layer_mean(trace.history_share)
    prompt_breadth = finite_layer_mean(trace.prompt_breadth)
    future_influence = finite_layer_mean(trace.future_influence)

    revisit_delta = rolling_delta(prompt_revisit, revisit_window)
    evidence_delta = rolling_delta(evidence_revisit, revisit_window)
    revisit_peaks = local_peaks(revisit_delta, peak_quantile)
    anchor_peaks = local_peaks(future_influence, peak_quantile)
    paired = pair_peaks(revisit_peaks, anchor_peaks, max_lag)

    peak_kind = np.zeros(len(revisit_peaks), dtype=np.int8)
    peak_index = np.flatnonzero(revisit_peaks)
    if len(peak_index):
        breadth = prompt_breadth[peak_index]
        peak_kind[peak_index[breadth <= 0.35]] = 1
        peak_kind[peak_index[breadth >= 0.65]] = 2
        peak_kind[peak_index[(breadth > 0.35) & (breadth < 0.65)]] = 3

    rate = coupling_rate(revisit_peaks, anchor_peaks, max_lag)
    null = circular_null(revisit_peaks, anchor_peaks, max_lag)
    matched = np.flatnonzero(paired >= 0)
    lags = paired[matched] - matched
    median_lag = float(np.median(lags)) if len(lags) else float("nan")
    return RhythmSignals(
        route_change=route_change,
        prompt_revisit=prompt_revisit,
        evidence_revisit=evidence_revisit,
        history_share=history_share,
        prompt_breadth=prompt_breadth,
        future_influence=future_influence,
        revisit_delta=revisit_delta,
        evidence_delta=evidence_delta,
        revisit_peaks=revisit_peaks,
        anchor_peaks=anchor_peaks,
        peak_kind=peak_kind,
        paired_anchor=paired,
        coupling_rate=rate,
        null_rate=null,
        median_lag=median_lag,
    )
