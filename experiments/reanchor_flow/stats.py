"""Small source-cluster summaries used by the rhythm audit."""

from __future__ import annotations

import numpy as np


def source_means(values, sources) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sources = np.asarray(sources)
    kept = np.isfinite(values)
    values, sources = values[kept], sources[kept]
    return np.asarray(
        [values[sources == source].mean() for source in np.unique(sources)],
        dtype=np.float64,
    )


def mean_ci(values, sources, repeats: int = 1000, seed: int = 2026) -> dict:
    grouped = source_means(values, sources)
    if not len(grouped):
        return {"mean": None, "ci95": [None, None], "sources": 0, "events": 0}
    estimate = float(grouped.mean())
    if repeats <= 0 or len(grouped) < 2:
        interval = [None, None]
    else:
        random = np.random.default_rng(seed)
        draw = random.choice(grouped, size=(repeats, len(grouped)), replace=True).mean(1)
        interval = [float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))]
    return {
        "mean": estimate,
        "ci95": interval,
        "sources": int(len(grouped)),
        "events": int(np.isfinite(values).sum()),
    }


def curve_ci(curves, sources, repeats: int = 1000, seed: int = 2026) -> dict:
    curves = np.asarray(curves, dtype=np.float64)
    sources = np.asarray(sources)
    if curves.ndim != 2 or not len(curves):
        return {"mean": [], "ci95_low": [], "ci95_high": [], "sources": 0, "events": 0}
    unique = np.unique(sources)
    grouped = np.stack([np.nanmean(curves[sources == source], axis=0) for source in unique])
    mean = np.nanmean(grouped, axis=0)
    if repeats <= 0 or len(unique) < 2:
        low = high = np.full_like(mean, np.nan)
    else:
        random = np.random.default_rng(seed)
        sampled = np.stack(
            [np.nanmean(grouped[random.integers(0, len(grouped), len(grouped))], axis=0) for _ in range(repeats)]
        )
        low = np.nanquantile(sampled, 0.025, axis=0)
        high = np.nanquantile(sampled, 0.975, axis=0)
    return {
        "mean": mean.tolist(),
        "ci95_low": low.tolist(),
        "ci95_high": high.tolist(),
        "sources": int(len(unique)),
        "events": int(len(curves)),
    }
