"""Population summaries for the registered re-anchor mechanism stages."""

from __future__ import annotations

import numpy as np

from .events import match_onsets
from .stats import curve_ci, mean_ci, source_means

BROAD_SIGNALS = (
    "prompt_delta",
    "nonlocal_delta",
    "evidence_delta",
    "route_change",
    "future_influence",
)
MECHANISM_SIGNALS = (
    "evidence_effect",
    "other_prompt_effect",
    "evidence_prompt_interaction",
    "history_effect",
    "evidence_readout_gain",
    "evidence_late_control_loss",
)


def centered(values, center: int, radius: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    if center - radius < 0 or center + radius >= len(values):
        return None
    return values[center - radius : center + radius + 1]


def slope(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return float("nan")
    position = np.linspace(0.0, 1.0, len(values))
    return float(np.polyfit(position[finite], values[finite], 1)[0])


def circular_event_lift(signal, event) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if len(signal) < 3 or not event.any():
        return float("nan")
    actual = np.nanmean(signal[event])
    null = []
    for shift in range(1, len(event)):
        selected = np.roll(event, shift)
        value = np.nanmean(signal[selected])
        if np.isfinite(value):
            null.append(value)
    return float(actual - np.mean(null)) if null and np.isfinite(actual) else float("nan")


def coupling_population(rows: list[dict], kind: str, *, bootstrap: int, seed: int) -> dict:
    peak_name = f"{kind}_peak"
    paired_name = f"{kind}_paired_anchor"
    rate_name = f"{kind}_coupling_rate"
    null_name = f"{kind}_coupling_null_rate"
    lag_name = f"{kind}_median_anchor_lag"
    peaks = coupled = 0
    weighted_null = 0.0
    null_weight = 0
    sample_lift, sample_source, lags = [], [], []
    for row in rows:
        peak = np.asarray(row[peak_name], dtype=bool)
        paired = np.asarray(row[paired_name], dtype=np.int64)
        count = int(peak.sum())
        peaks += count
        coupled += int((paired >= 0).sum())
        null = float(row[null_name])
        rate = float(row[rate_name])
        if count and np.isfinite(null):
            weighted_null += count * null
            null_weight += count
        if np.isfinite(rate) and np.isfinite(null):
            sample_lift.append(rate - null)
            sample_source.append(row["source_id"])
        lag = float(row[lag_name])
        if np.isfinite(lag):
            lags.append(lag)
    pooled = coupled / peaks if peaks else None
    null = weighted_null / null_weight if null_weight else None
    grouped = source_means(sample_lift, sample_source)
    return {
        "event_peaks": peaks,
        "coupled_events": coupled,
        "pooled_rate": pooled,
        "pooled_null": null,
        "pooled_lift": None if pooled is None or null is None else pooled - null,
        "sample_lift": mean_ci(sample_lift, sample_source, repeats=bootstrap, seed=seed),
        "positive_source_fraction": float(np.mean(grouped > 0)) if len(grouped) else None,
        "median_anchor_lag": float(np.median(lags)) if lags else None,
    }


def normal_summary(rows: list[dict], *, bootstrap: int, seed: int) -> dict:
    correct = [row for row in rows if not np.asarray(row["label"], dtype=bool).any()]
    prompt_slope, history_slope, sources = [], [], []
    transition = {name: [] for name in ("prompt_delta", "evidence_delta", "future_influence")}
    transition_sources = {name: [] for name in transition}
    transition_peaks = 0
    for row in correct:
        source = row["source_id"]
        p = slope(row["prompt_lift"])
        h = slope(row["history_lift"])
        if np.isfinite(p) and np.isfinite(h):
            prompt_slope.append(p)
            history_slope.append(h)
            sources.append(source)
        peak = np.asarray(row["transition_peak"], dtype=bool)
        transition_peaks += int(peak.sum())
        for name in transition:
            value = circular_event_lift(row[name], peak)
            if np.isfinite(value):
                transition[name].append(value)
                transition_sources[name].append(source)
    return {
        "correct_samples": len(correct),
        "direct_route_shift": {
            "prompt_lift_slope": mean_ci(prompt_slope, sources, repeats=bootstrap, seed=seed),
            "history_lift_slope": mean_ci(history_slope, sources, repeats=bootstrap, seed=seed + 1),
        },
        "internal_transition": {
            "peaks": transition_peaks,
            **{
                name: mean_ci(
                    transition[name],
                    transition_sources[name],
                    repeats=bootstrap,
                    seed=seed + 10 + index,
                )
                for index, name in enumerate(transition)
            },
        },
    }


def mechanism_summary(rows: list[dict], *, bootstrap: int, seed: int) -> dict:
    rows = [row for row in rows if row.get("mechanism")]
    effects = {name: [] for name in ("evidence_entry", *MECHANISM_SIGNALS)}
    sources = {name: [] for name in effects}
    curves = {
        "entry_onset": [],
        "entry_clean": [],
        "presence_onset": [],
        "presence_clean": [],
        "control_onset": [],
        "control_clean": [],
    }
    curve_sources = {name: [] for name in curves}
    pairs = 0
    layer_count = 0
    for row in rows:
        source = row["source_id"]
        label = np.asarray(row["label"], dtype=bool)
        entry = np.nanmean(np.asarray(row["evidence_share_layer"], dtype=float), axis=0)
        presence = np.asarray(row["evidence_state_presence"], dtype=float)
        control = np.asarray(row["evidence_state_control"], dtype=float)
        layer_count = max(layer_count, presence.shape[0])
        for onset, clean in match_onsets(label, row["target_token_id"]):
            pairs += 1
            values = {"evidence_entry": entry}
            values.update({name: np.asarray(row[name], dtype=float) for name in MECHANISM_SIGNALS})
            for name, series in values.items():
                difference = series[onset] - series[clean]
                if np.isfinite(difference):
                    effects[name].append(float(difference))
                    sources[name].append(source)
            candidate = {
                "entry_onset": np.asarray(row["evidence_share_layer"], dtype=float)[:, onset],
                "entry_clean": np.asarray(row["evidence_share_layer"], dtype=float)[:, clean],
                "presence_onset": presence[:, onset],
                "presence_clean": presence[:, clean],
                "control_onset": control[:, onset],
                "control_clean": control[:, clean],
            }
            for name, curve in candidate.items():
                if np.isfinite(curve).all():
                    curves[name].append(curve)
                    curve_sources[name].append(source)
    return {
        "samples": len(rows),
        "onset_pairs": pairs,
        "onset_minus_clean": {
            name: mean_ci(
                effects[name], sources[name], repeats=bootstrap, seed=seed + index
            )
            for index, name in enumerate(effects)
        },
        "layer": list(range(layer_count)),
        "layer_curves": {
            name: curve_ci(
                curves[name], curve_sources[name], repeats=bootstrap, seed=seed + 30 + index
            )
            for index, name in enumerate(curves)
        },
    }


def task_summary(rows: list[dict], *, bootstrap: int, seed: int, radius: int) -> dict:
    tokens = sum(len(row["label"]) for row in rows)
    positives = sum(int(np.asarray(row["label"], dtype=bool).sum()) for row in rows)
    broad_effect = {name: [] for name in BROAD_SIGNALS}
    broad_source = {name: [] for name in BROAD_SIGNALS}
    onset_pairs = 0
    onset_curves, clean_curves, curve_sources = [], [], []
    for row in rows:
        source = row["source_id"]
        label = np.asarray(row["label"], dtype=bool)
        pairs = match_onsets(label, row["target_token_id"])
        onset_pairs += len(pairs)
        for onset, clean in pairs:
            for name in BROAD_SIGNALS:
                series = np.asarray(row[name], dtype=float)
                difference = series[onset] - series[clean]
                if np.isfinite(difference):
                    broad_effect[name].append(float(difference))
                    broad_source[name].append(source)
            positive = centered(row["evidence_delta"], onset, radius)
            negative = centered(row["evidence_delta"], clean, radius)
            if positive is not None and negative is not None:
                if np.isfinite(positive).all() and np.isfinite(negative).all():
                    onset_curves.append(positive)
                    clean_curves.append(negative)
                    curve_sources.append(source)
    offset = np.arange(-radius, radius + 1)
    return {
        "samples": len(rows),
        "tokens": tokens,
        "positive_tokens": positives,
        "prevalence": positives / tokens if tokens else None,
        "onset_pairs": onset_pairs,
        "normal": normal_summary(rows, bootstrap=bootstrap, seed=seed),
        "prompt_to_anchor": coupling_population(rows, "prompt", bootstrap=bootstrap, seed=seed + 50),
        "nonlocal_to_anchor": coupling_population(rows, "review", bootstrap=bootstrap, seed=seed + 51),
        "onset_minus_matched_clean": {
            name: mean_ci(
                broad_effect[name], broad_source[name], repeats=bootstrap, seed=seed + 60 + index
            )
            for index, name in enumerate(BROAD_SIGNALS)
        },
        "onset_evidence_curve": {
            "offset": offset.tolist(),
            "hallucination": curve_ci(
                onset_curves, curve_sources, repeats=bootstrap, seed=seed + 80
            ),
            "matched_clean": curve_ci(
                clean_curves, curve_sources, repeats=bootstrap, seed=seed + 81
            ),
        },
        "mechanism": mechanism_summary(rows, bootstrap=bootstrap, seed=seed + 100),
    }
