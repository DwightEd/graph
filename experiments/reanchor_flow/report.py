"""Aggregate population rhythm and matched hallucination-onset effects."""

from __future__ import annotations

import numpy as np

from .events import match_onsets
from .rhythm import robust_z
from .stats import curve_ci, mean_ci, source_means

KEY_SIGNALS = (
    "prompt_delta",
    "nonlocal_delta",
    "evidence_delta",
    "route_change",
    "future_influence",
    "prompt_breadth",
)


def centered(values, center: int, radius: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    if center - radius < 0 or center + radius >= len(values):
        return None
    return values[center - radius : center + radius + 1]


def coupling_population(
    rows: list[dict],
    kind: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    peak_name = f"{kind}_peak"
    paired_name = f"{kind}_paired_anchor"
    rate_name = f"{kind}_coupling_rate"
    null_name = f"{kind}_coupling_null_rate"
    lag_name = f"{kind}_median_anchor_lag"

    peaks = coupled = 0
    weighted_null = 0.0
    null_weight = 0
    sample_lift: list[float] = []
    sample_source: list[str] = []
    lags: list[float] = []
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

    pooled_rate = coupled / peaks if peaks else None
    pooled_null = weighted_null / null_weight if null_weight else None
    grouped = source_means(sample_lift, sample_source)
    return {
        "event_peaks": peaks,
        "coupled_events": coupled,
        "pooled_rate": pooled_rate,
        "pooled_null": pooled_null,
        "pooled_lift": (
            None
            if pooled_rate is None or pooled_null is None
            else pooled_rate - pooled_null
        ),
        "sample_lift": mean_ci(
            sample_lift,
            sample_source,
            repeats=bootstrap,
            seed=seed,
        ),
        "positive_source_fraction": (
            float(np.mean(grouped > 0)) if len(grouped) else None
        ),
        "median_sample_lift": (
            float(np.median(grouped)) if len(grouped) else None
        ),
        "median_anchor_lag": float(np.median(lags)) if lags else None,
    }


def task_summary(rows: list[dict], *, bootstrap: int, seed: int, radius: int) -> dict:
    counts = {
        "tokens": 0,
        "positive_tokens": 0,
        "anchor_peaks": 0,
        "onset_pairs": 0,
    }
    effects = {name: [] for name in KEY_SIGNALS}
    effect_sources = {name: [] for name in KEY_SIGNALS}
    centered_curves = {
        kind: {
            name: []
            for name in (
                "prompt_delta",
                "nonlocal_delta",
                "route_change",
                "future_influence",
            )
        }
        for kind in ("prompt", "review")
    }
    centered_sources = {
        kind: {name: [] for name in centered_curves[kind]}
        for kind in centered_curves
    }
    onset_prompt: list[np.ndarray] = []
    clean_prompt: list[np.ndarray] = []
    onset_nonlocal: list[np.ndarray] = []
    clean_nonlocal: list[np.ndarray] = []
    onset_sources: list[str] = []

    for row in rows:
        source_id = row["source_id"]
        label = row["label"]
        counts["tokens"] += len(label)
        counts["positive_tokens"] += int(label.sum())
        counts["anchor_peaks"] += int(np.asarray(row["anchor_peak"], dtype=bool).sum())

        z = {
            name: robust_z(row[name])
            for name in (
                "prompt_delta",
                "nonlocal_delta",
                "route_change",
                "future_influence",
            )
        }
        for kind, peak_name in (("prompt", "prompt_peak"), ("review", "review_peak")):
            for peak in np.flatnonzero(np.asarray(row[peak_name], dtype=bool)):
                for name, series in z.items():
                    curve = centered(series, int(peak), radius)
                    if curve is not None and np.isfinite(curve).all():
                        centered_curves[kind][name].append(curve)
                        centered_sources[kind][name].append(source_id)

        pairs = match_onsets(label, row["target_token_id"])
        counts["onset_pairs"] += len(pairs)
        for onset, control in pairs:
            for name in KEY_SIGNALS:
                values = np.asarray(row[name], dtype=np.float64)
                difference = values[onset] - values[control]
                if np.isfinite(difference):
                    effects[name].append(float(difference))
                    effect_sources[name].append(source_id)
            positive_prompt = centered(robust_z(row["prompt_delta"]), onset, radius)
            negative_prompt = centered(robust_z(row["prompt_delta"]), control, radius)
            positive_nonlocal = centered(robust_z(row["nonlocal_delta"]), onset, radius)
            negative_nonlocal = centered(robust_z(row["nonlocal_delta"]), control, radius)
            if all(
                curve is not None and np.isfinite(curve).all()
                for curve in (
                    positive_prompt,
                    negative_prompt,
                    positive_nonlocal,
                    negative_nonlocal,
                )
            ):
                onset_prompt.append(positive_prompt)
                clean_prompt.append(negative_prompt)
                onset_nonlocal.append(positive_nonlocal)
                clean_nonlocal.append(negative_nonlocal)
                onset_sources.append(source_id)

    offset = np.arange(-radius, radius + 1)
    return {
        "samples": len(rows),
        **counts,
        "prevalence": (
            counts["positive_tokens"] / counts["tokens"]
            if counts["tokens"]
            else None
        ),
        "prompt_to_anchor": coupling_population(
            rows, "prompt", bootstrap=bootstrap, seed=seed
        ),
        "nonlocal_to_anchor": coupling_population(
            rows, "review", bootstrap=bootstrap, seed=seed + 1
        ),
        "onset_minus_matched_clean": {
            name: mean_ci(
                effects[name],
                effect_sources[name],
                repeats=bootstrap,
                seed=seed + 10 + index,
            )
            for index, name in enumerate(KEY_SIGNALS)
        },
        "curves": {
            "offset": offset.tolist(),
            "prompt_centered": {
                name: curve_ci(
                    centered_curves["prompt"][name],
                    centered_sources["prompt"][name],
                    repeats=bootstrap,
                    seed=seed + 30 + index,
                )
                for index, name in enumerate(centered_curves["prompt"])
            },
            "review_centered": {
                name: curve_ci(
                    centered_curves["review"][name],
                    centered_sources["review"][name],
                    repeats=bootstrap,
                    seed=seed + 40 + index,
                )
                for index, name in enumerate(centered_curves["review"])
            },
            "hallucination_onset_prompt": curve_ci(
                onset_prompt,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 50,
            ),
            "matched_clean_prompt": curve_ci(
                clean_prompt,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 51,
            ),
            "hallucination_onset_nonlocal": curve_ci(
                onset_nonlocal,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 52,
            ),
            "matched_clean_nonlocal": curve_ci(
                clean_nonlocal,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 53,
            ),
        },
    }
