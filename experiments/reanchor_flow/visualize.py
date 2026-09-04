"""Figures for layer-resolved re-anchor observations and event-time tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .capture import SampleCapture
from .claims import FORCED_CHUNK


def routing_lift(observed, availability):
    observed = np.asarray(observed, dtype=float)
    availability = np.asarray(availability, dtype=float)
    value = np.log((observed + 1e-8) / (availability + 1e-8))
    value[availability <= 0] = np.nan
    return value


def save_sample_figure(
    path: str | Path,
    capture: SampleCapture,
    *,
    response_labels: list[str] | None = None,
    title: str = "",
) -> Path:
    from matplotlib import pyplot as plt

    arrays = capture.arrays
    positions = np.asarray(arrays["prediction_position"], dtype=int)
    functional = np.asarray(arrays["functional_role_share"], dtype=float)
    attention = np.asarray(arrays["attention_role_share"], dtype=float)
    functional_lift = routing_lift(
        functional, arrays["functional_availability_null"]
    )
    attention_lift = routing_lift(
        attention, arrays["attention_availability_null"]
    )
    functional_specificity = functional_lift[:, :, 0] - functional_lift[:, :, 1]
    attention_specificity = attention_lift[:, :, 0] - attention_lift[:, :, 1]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    try:
        if title:
            figure.suptitle(title)
        for axis, trace, panel in (
            (axes[0, 0], functional_specificity, "A  Functional AVWₒ specificity"),
            (axes[0, 1], attention_specificity, "B  Attention-only specificity"),
        ):
            mean = np.nanmean(trace, axis=0)
            axis.plot(positions, mean, label="evidence − other prompt")
            for start, kind in zip(
                arrays["claim_start"], arrays["claim_boundary_kind"], strict=True
            ):
                if int(kind) != FORCED_CHUNK:
                    axis.axvline(start, linewidth=0.7, alpha=0.25)
            axis.axhline(0, color="black", linewidth=0.6)
            axis.set_title(panel, loc="left", fontweight="semibold")
            axis.set_xlabel("Predicted token p (observed at q=p-1)")
            axis.set_ylabel("log-lift difference")
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.2)

        finite_heat = np.abs(functional_specificity[np.isfinite(functional_specificity)])
        heat_limit = float(np.quantile(finite_heat, 0.99)) if len(finite_heat) else 1.0
        heat_limit = max(heat_limit, 1e-6)
        heat = axes[1, 0].imshow(
            functional_specificity,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-heat_limit,
            vmax=heat_limit,
        )
        axes[1, 0].set_title(
            "C  Functional specificity by layer", loc="left", fontweight="semibold"
        )
        axes[1, 0].set_xlabel("Response event")
        axes[1, 0].set_ylabel("Layer")
        figure.colorbar(heat, ax=axes[1, 0], label="evidence − other prompt log lift")

        for name, index in zip(
            ("early", "middle", "late"),
            np.array_split(np.arange(functional_specificity.shape[0]), 3),
            strict=True,
        ):
            if len(index):
                axes[1, 1].plot(
                    positions,
                    np.nanmean(functional_specificity[index], axis=0),
                    label=name,
                )
        axes[1, 1].set_title(
            "D  Functional specificity by stage", loc="left", fontweight="semibold"
        )
        axes[1, 1].set_xlabel("Predicted token p")
        axes[1, 1].set_ylabel("evidence − other prompt log lift")
        axes[1, 1].legend(frameon=False)
        axes[1, 1].grid(axis="y", alpha=0.2)

        if response_labels and len(response_labels) == len(positions):
            tick = np.linspace(0, len(positions) - 1, min(8, len(positions)))
            tick = tick.round().astype(int)
            labels = [
                str(response_labels[index]).replace("\n", "↵")[:12]
                for index in tick
            ]
            axes[0, 0].set_xticks(positions[tick], labels)
            axes[0, 1].set_xticks(positions[tick], labels)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination


def draw_curve(axis, offset, summary: dict, label: str) -> None:
    mean = np.asarray(summary["mean"], dtype=float)
    low = np.asarray(summary["ci95_low"], dtype=float)
    high = np.asarray(summary["ci95_high"], dtype=float)
    axis.plot(offset, mean, label=f"{label} (n={summary['events']})")
    if np.isfinite(low).any() and np.isfinite(high).any():
        axis.fill_between(offset, low, high, alpha=0.16)


def save_population_figure(path: str | Path, curves: dict) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(curves["offset"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    try:
        draw_curve(
            axes[0],
            offset,
            curves["correct_boundary_evidence"],
            "correct boundary",
        )
        draw_curve(
            axes[0],
            offset,
            curves["within_claim_control_evidence"],
            "within-claim control",
        )
        axes[0].set_title("A  Clean boundary re-read", loc="left", fontweight="semibold")

        draw_curve(
            axes[1],
            offset,
            curves["hallucination_onset_evidence"],
            "hallucination onset",
        )
        draw_curve(
            axes[1],
            offset,
            curves["matched_token_evidence"],
            "matched token",
        )
        axes[1].set_title("B  Onset association", loc="left", fontweight="semibold")

        draw_curve(
            axes[2],
            offset,
            curves["correct_boundary_history_release"],
            "correct boundary",
        )
        draw_curve(
            axes[2],
            offset,
            curves["hallucination_onset_history_release"],
            "hallucination onset",
        )
        axes[2].set_title("C  History release", loc="left", fontweight="semibold")

        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.7)
            axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
            axis.set_xlabel("Offset from event (p; query q=p-1)")
            axis.set_ylabel("Availability-adjusted change")
            axis.legend(frameon=False, fontsize=8)
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
