"""Aggregate peak coupling and matched hallucination-onset effects."""

from __future__ import annotations

import numpy as np

from .events import match_onsets
from .rhythm import robust_z
from .stats import curve_ci, mean_ci

KEY_SIGNALS = ("revisit_delta", "route_change", "future_influence", "prompt_breadth")


def centered(values, center: int, radius: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    if center - radius < 0 or center + radius >= len(values):
        return None
    return values[center - radius : center + radius + 1]


def task_summary(rows: list[dict], *, bootstrap: int, seed: int, radius: int) -> dict:
    counts = {
        "tokens": 0,
        "positive_tokens": 0,
        "revisit_peaks": 0,
        "anchor_peaks": 0,
        "coupled_revisits": 0,
        "selective_revisits": 0,
        "global_reviews": 0,
        "mixed_revisits": 0,
        "onset_pairs": 0,
    }
    null_sum = 0.0
    null_weight = 0
    lags: list[int] = []
    effects = {name: [] for name in KEY_SIGNALS}
    effect_sources = {name: [] for name in KEY_SIGNALS}
    peak_curves = {
        name: []
        for name in ("revisit_delta", "route_change", "future_influence")
    }
    peak_sources = {name: [] for name in peak_curves}
    onset_curves: list[np.ndarray] = []
    clean_curves: list[np.ndarray] = []
    onset_sources: list[str] = []

    for row in rows:
        source_id = row["source_id"]
        label = row["label"]
        counts["tokens"] += len(label)
        counts["positive_tokens"] += int(label.sum())
        revisit = np.asarray(row["revisit_peak"], dtype=bool)
        anchor = np.asarray(row["anchor_peak"], dtype=bool)
        paired = np.asarray(row["paired_anchor"], dtype=np.int64)
        kind = np.asarray(row["revisit_peak_kind"], dtype=np.int8)
        revisit_count = int(revisit.sum())
        counts["revisit_peaks"] += revisit_count
        counts["anchor_peaks"] += int(anchor.sum())
        counts["coupled_revisits"] += int((paired >= 0).sum())
        counts["selective_revisits"] += int((kind == 1).sum())
        counts["global_reviews"] += int((kind == 2).sum())
        counts["mixed_revisits"] += int((kind == 3).sum())

        null = row["coupling_null_rate"]
        if revisit_count and np.isfinite(null):
            null_sum += null * revisit_count
            null_weight += revisit_count
        matched = np.flatnonzero(paired >= 0)
        lags.extend((paired[matched] - matched).tolist())

        z = {name: robust_z(row[name]) for name in peak_curves}
        for peak in np.flatnonzero(revisit):
            for name, series in z.items():
                curve = centered(series, int(peak), radius)
                if curve is not None and np.isfinite(curve).all():
                    peak_curves[name].append(curve)
                    peak_sources[name].append(source_id)

        pairs = match_onsets(label, row["target_token_id"])
        counts["onset_pairs"] += len(pairs)
        for onset, control in pairs:
            for name in KEY_SIGNALS:
                values = np.asarray(row[name], dtype=np.float64)
                difference = values[onset] - values[control]
                if np.isfinite(difference):
                    effects[name].append(float(difference))
                    effect_sources[name].append(source_id)
            positive = centered(robust_z(row["revisit_delta"]), onset, radius)
            negative = centered(robust_z(row["revisit_delta"]), control, radius)
            if (
                positive is not None
                and negative is not None
                and np.isfinite(positive).all()
                and np.isfinite(negative).all()
            ):
                onset_curves.append(positive)
                clean_curves.append(negative)
                onset_sources.append(source_id)

    revisit_count = counts["revisit_peaks"]
    coupling = counts["coupled_revisits"] / revisit_count if revisit_count else None
    null = null_sum / null_weight if null_weight else None
    offset = np.arange(-radius, radius + 1)
    return {
        "samples": len(rows),
        **counts,
        "coupling_rate": coupling,
        "circular_null_rate": null,
        "coupling_lift": None if coupling is None or null is None else coupling - null,
        "median_anchor_lag": float(np.median(lags)) if lags else None,
        "onset_minus_matched_clean": {
            name: mean_ci(
                effects[name],
                effect_sources[name],
                repeats=bootstrap,
                seed=seed + index,
            )
            for index, name in enumerate(KEY_SIGNALS)
        },
        "curves": {
            "offset": offset.tolist(),
            "revisit_centered": {
                name: curve_ci(
                    peak_curves[name],
                    peak_sources[name],
                    repeats=bootstrap,
                    seed=seed + 20 + index,
                )
                for index, name in enumerate(peak_curves)
            },
            "hallucination_onset_revisit": curve_ci(
                onset_curves,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 30,
            ),
            "matched_clean_revisit": curve_ci(
                clean_curves,
                onset_sources,
                repeats=bootstrap,
                seed=seed + 31,
            ),
        },
    }
