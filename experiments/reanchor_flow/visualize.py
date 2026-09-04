"""Concise population rhythm figure; sample plotting lives in sample_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sample_plot import save_sample_figure


def draw_curve(axis, offset, summary: dict, label: str) -> None:
    mean = np.asarray(summary.get("mean", []), dtype=float)
    if len(mean) != len(offset):
        return
    low = np.asarray(summary.get("ci95_low", []), dtype=float)
    high = np.asarray(summary.get("ci95_high", []), dtype=float)
    axis.plot(offset, mean, label=f"{label} (n={summary.get('events', 0)})")
    if len(low) == len(offset) and np.isfinite(low).any():
        axis.fill_between(offset, low, high, alpha=0.15)


def save_population_figure(path: str | Path, curves: dict) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(curves["offset"], dtype=int)
    centered = curves["revisit_centered"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    try:
        draw_curve(
            axes[0],
            offset,
            centered["revisit_delta"],
            "far-prompt revisit",
        )
        draw_curve(
            axes[0],
            offset,
            centered["route_change"],
            "route change",
        )
        axes[0].set_title(
            "A  Incoming transition",
            loc="left",
            fontweight="semibold",
        )
        draw_curve(
            axes[1],
            offset,
            centered["future_influence"],
            "future influence",
        )
        axes[1].set_title(
            "B  Anchor after revisit",
            loc="left",
            fontweight="semibold",
        )
        draw_curve(
            axes[2],
            offset,
            curves["hallucination_onset_revisit"],
            "hallucination onset",
        )
        draw_curve(
            axes[2],
            offset,
            curves["matched_clean_revisit"],
            "matched clean token",
        )
        axes[2].set_title(
            "C  Onset comparison",
            loc="left",
            fontweight="semibold",
        )
        for axis in axes:
            axis.axhline(0, linewidth=0.6)
            axis.axvline(0, linewidth=0.7, linestyle="--")
            axis.set_xlabel("Token offset")
            axis.set_ylabel("robust z-score")
            axis.legend(frameon=False, fontsize=8)
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
