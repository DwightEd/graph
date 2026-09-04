"""Population and sample figures for the complete re-anchor audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sample_plot import save_sample_figure


def _curve(axis, x, summary: dict, label: str) -> None:
    mean = np.asarray(summary.get("mean", []), dtype=float)
    if len(mean) != len(x):
        return
    axis.plot(x, mean, label=f"{label} (n={summary.get('events', 0)})")
    low = np.asarray(summary.get("ci95_low", []), dtype=float)
    high = np.asarray(summary.get("ci95_high", []), dtype=float)
    if len(low) == len(x) and np.isfinite(low).any():
        axis.fill_between(x, low, high, alpha=0.15)


def _point(axis, x: int, summary: dict, label: str) -> None:
    mean = summary.get("mean")
    low, high = summary.get("ci95", [None, None])
    if mean is None:
        return
    error = None if low is None or high is None else [[mean - low], [high - mean]]
    axis.errorbar([x], [mean], yerr=error, fmt="o", capsize=3)
    axis.text(x, mean, label, ha="center", va="bottom", fontsize=8)


def _legend(axis) -> None:
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)


def save_population_figure(path: str | Path, report: dict) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    try:
        shift = report["normal"]["direct_route_shift"]
        _point(axes[0, 0], 0, shift["prompt_lift_slope"], "prompt")
        _point(axes[0, 0], 1, shift["history_lift_slope"], "history")
        axes[0, 0].axhline(0, linewidth=0.7)
        axes[0, 0].set_xticks([0, 1], ["prompt", "history"])
        axes[0, 0].set_title("A  Normal direct-route drift")
        axes[0, 0].set_ylabel("Exposure-adjusted slope")

        transition = report["normal"]["internal_transition"]
        for index, name in enumerate(("prompt_delta", "evidence_delta", "future_influence")):
            _point(axes[0, 1], index, transition[name], name.replace("_", " "))
        axes[0, 1].axhline(0, linewidth=0.7)
        axes[0, 1].set_xticks(range(3), ["prompt", "evidence", "future"])
        axes[0, 1].set_title("B  At internal route transitions")
        axes[0, 1].set_ylabel("Peak minus circular null")

        for index, key in enumerate(("prompt_to_anchor", "nonlocal_to_anchor")):
            summary = report[key]["sample_lift"]
            _point(axes[1, 0], index, summary, key.replace("_to_anchor", ""))
        axes[1, 0].axhline(0, linewidth=0.7)
        axes[1, 0].set_xticks([0, 1], ["prompt revisit", "nonlocal review"])
        axes[1, 0].set_title("C  Event → future-anchor coupling")
        axes[1, 0].set_ylabel("Per-source lift over circular null")

        curve = report["onset_evidence_curve"]
        offset = np.asarray(curve["offset"], dtype=int)
        _curve(axes[1, 1], offset, curve["hallucination"], "hallucination onset")
        _curve(axes[1, 1], offset, curve["matched_clean"], "matched clean")
        axes[1, 1].axhline(0, linewidth=0.7)
        axes[1, 1].axvline(0, linewidth=0.7, linestyle="--")
        axes[1, 1].set_title("D  Evidence re-entry around onset")
        axes[1, 1].set_xlabel("Token offset")
        axes[1, 1].set_ylabel("Evidence-share change")
        _legend(axes[1, 1])

        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination


def save_mechanism_figure(path: str | Path, mechanism: dict) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    curves = mechanism["layer_curves"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    try:
        entry_layer = np.arange(len(curves["entry_onset"].get("mean", [])))
        _curve(axes[0, 0], entry_layer, curves["entry_onset"], "hallucination onset")
        _curve(axes[0, 0], entry_layer, curves["entry_clean"], "matched clean")
        axes[0, 0].set_title("A  Direct evidence entry")
        axes[0, 0].set_ylabel("A||W_OV|| share")

        depth = np.arange(len(curves["presence_onset"].get("mean", [])))
        _curve(axes[0, 1], depth, curves["presence_onset"], "hallucination onset")
        _curve(axes[0, 1], depth, curves["presence_clean"], "matched clean")
        axes[0, 1].set_title("B  Evidence-conditioned state presence")
        axes[0, 1].set_ylabel("Relative residual difference")

        _curve(axes[1, 0], depth, curves["control_onset"], "hallucination onset")
        _curve(axes[1, 0], depth, curves["control_clean"], "matched clean")
        axes[1, 0].axhline(0, linewidth=0.7)
        axes[1, 0].set_title("C  State control under fixed readout")
        axes[1, 0].set_ylabel("Target-runner margin contribution")

        effects = mechanism["onset_minus_clean"]
        keys = (
            "evidence_entry",
            "evidence_effect",
            "evidence_prompt_interaction",
            "history_effect",
            "evidence_readout_gain",
            "evidence_late_control_loss",
        )
        for index, key in enumerate(keys):
            _point(axes[1, 1], index, effects[key], key.replace("evidence_", "").replace("_", " "))
        axes[1, 1].axhline(0, linewidth=0.7)
        axes[1, 1].set_xticks(range(len(keys)), [key.replace("evidence_", "") for key in keys], rotation=30, ha="right")
        axes[1, 1].set_title("D  Hallucination onset − matched clean")

        for axis in axes.flat:
            axis.set_xlabel("Layer" if axis is not axes[1, 1] else "")
            axis.grid(alpha=0.2)
            _legend(axis)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination


def save_sample_mechanism_figure(
    path: str | Path,
    result: dict,
    label,
    *,
    title: str = "",
) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = np.asarray(label, dtype=bool)
    onset = np.flatnonzero(label & ~np.r_[False, label[:-1]])
    event = int(onset[0]) if len(onset) else int(np.nanargmax(result["route_change"]))
    entry = np.asarray(result["evidence_share_layer"], dtype=float)[:, event]
    presence = np.asarray(result["evidence_state_presence"], dtype=float)[:, event]
    control = np.asarray(result["evidence_state_control"], dtype=float)[:, event]
    effects = [
        float(np.asarray(result[name])[event])
        for name in (
            "evidence_effect",
            "other_prompt_effect",
            "evidence_prompt_interaction",
            "history_effect",
        )
    ]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    try:
        if title:
            figure.suptitle(f"{title}; event={event}")
        axes[0, 0].plot(np.arange(len(entry)), entry)
        axes[0, 0].set_title("A  Direct evidence entry")
        axes[0, 0].set_ylabel("A||W_OV|| share")
        axes[0, 1].plot(np.arange(len(presence)), presence)
        axes[0, 1].set_title("B  Evidence-conditioned state presence")
        axes[0, 1].set_ylabel("Relative residual difference")
        axes[1, 0].plot(np.arange(len(control)), control)
        axes[1, 0].axhline(0, linewidth=0.7)
        axes[1, 0].set_title("C  Fixed-readout control")
        axes[1, 0].set_ylabel("Target-runner margin contribution")
        names = ["evidence", "other prompt", "interaction", "history"]
        axes[1, 1].bar(np.arange(4), effects)
        axes[1, 1].axhline(0, linewidth=0.7)
        axes[1, 1].set_xticks(np.arange(4), names, rotation=20)
        axes[1, 1].set_title("D  Grouped causal effects")
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
