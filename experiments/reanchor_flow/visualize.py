"""Concise population rhythm figures; sample plotting lives in sample_plot."""

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


def add_legend(axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)


def save_population_figure(path: str | Path, curves: dict) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(curves["offset"])
    prompt = curves["prompt_centered"]
    review = curves["review_centered"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    try:
        draw_curve(axes[0], offset, prompt["prompt_delta"], "prompt revisit")
        draw_curve(axes[0], offset, prompt["route_change"], "route change")
        draw_curve(axes[0], offset, prompt["future_influence"], "future influence")
        axes[0].set_title("A  Prompt-revisit rhythm", loc="left")

        draw_curve(axes[1], offset, review["nonlocal_delta"], "nonlocal review")
        draw_curve(axes[1], offset, review["route_change"], "route change")
        draw_curve(axes[1], offset, review["future_influence"], "future influence")
        axes[1].set_title("B  Nonlocal-review rhythm", loc="left")

        draw_curve(
            axes[2],
            offset,
            curves["hallucination_onset_prompt"],
            "hallucination: prompt",
        )
        draw_curve(
            axes[2],
            offset,
            curves["matched_clean_prompt"],
            "clean: prompt",
        )
        draw_curve(
            axes[2],
            offset,
            curves["hallucination_onset_nonlocal"],
            "hallucination: nonlocal",
        )
        draw_curve(
            axes[2],
            offset,
            curves["matched_clean_nonlocal"],
            "clean: nonlocal",
        )
        axes[2].set_title("C  Hallucination onset", loc="left")

        for axis in axes:
            axis.axhline(0, linewidth=0.6)
            axis.axvline(0, linewidth=0.7, linestyle="--")
            axis.set_xlabel("Token offset")
            axis.set_ylabel("robust z-score")
            add_legend(axis)
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
