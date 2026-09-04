"""Figures for claim-boundary routing and graph-necessity experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .capture import SampleCapture


def save_sample_figure(
    path: str | Path,
    capture: SampleCapture,
    *,
    response_labels: list[str] | None = None,
    title: str = "",
) -> Path:
    from matplotlib import pyplot as plt

    arrays = capture.arrays
    response_start = int(np.asarray(arrays["response_start"]).item())
    positions = np.asarray(arrays["prediction_position"], dtype=int)
    starts = np.asarray(arrays["claim_start"], dtype=int)
    claim_x = np.arange(len(starts))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    try:
        if title:
            figure.suptitle(title)
        for axis, prefix, panel in (
            (axes[0, 0], "functional", "A  Functional message routing"),
            (axes[0, 1], "attention", "B  Attention-only control"),
        ):
            axis.plot(positions, arrays[f"{prefix}_evidence_inflow"], label="evidence")
            axis.plot(
                positions,
                arrays[f"{prefix}_other_prompt_inflow"],
                label="other prompt",
            )
            axis.plot(
                positions,
                arrays[f"{prefix}_history_inflow"],
                label="response history",
            )
            for start in starts:
                axis.axvline(start, linewidth=0.7, alpha=0.25)
            axis.set_title(panel, loc="left", fontweight="semibold")
            axis.set_xlabel("Predicted token position")
            axis.set_ylabel("Incoming route share")
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.2)

        claim_axis = axes[1, 0]
        width = 0.2
        if len(claim_x):
            for offset, field, label in (
                (-1.5, "functional_evidence_reanchor_flow", "functional global flow"),
                (-0.5, "attention_evidence_reanchor_flow", "attention flow"),
                (0.5, "rewired_evidence_reanchor_flow", "role/lag rewire"),
                (1.5, "functional_direct_evidence_sink", "direct sink edge"),
            ):
                claim_axis.bar(
                    claim_x + offset * width,
                    arrays[field],
                    width,
                    label=label,
                )
        claim_axis.set_title(
            "C  Evidence flow through claim boundary",
            loc="left",
            fontweight="semibold",
        )
        claim_axis.set_xlabel("Claim proxy index")
        claim_axis.set_ylabel("Path mass")
        claim_axis.legend(frameon=False, fontsize=8)
        claim_axis.grid(axis="y", alpha=0.2)

        graph_axis = axes[1, 1]
        matrix = capture.functional_transition[:, response_start:]
        image = graph_axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        graph_axis.set_title(
            "D  Full token DAG (source → predicted token)",
            loc="left",
            fontweight="semibold",
        )
        graph_axis.set_xlabel("Response target index")
        graph_axis.set_ylabel("Absolute source position")
        for start in starts:
            graph_axis.axvline(start - response_start, linewidth=0.7, alpha=0.35)
        edge_source = np.asarray(arrays["audit_edge_source"], dtype=int)
        edge_target = np.asarray(arrays["audit_edge_target"], dtype=int)
        if len(edge_source):
            graph_axis.scatter(
                edge_target - response_start,
                edge_source,
                s=18,
                facecolors="none",
                edgecolors="white",
                linewidths=0.7,
                label="audited backbone",
            )
            graph_axis.legend(frameon=False, fontsize=8)
        figure.colorbar(
            image,
            ax=graph_axis,
            label="Incoming-normalized functional capacity",
        )

        if response_labels and len(response_labels) == len(positions):
            tick = np.linspace(0, len(positions) - 1, min(8, len(positions))).round().astype(int)
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


def save_population_figure(path: str | Path, curves: dict[str, list[float]]) -> Path:
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = np.asarray(curves["offset"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    try:
        for axis, name, ylabel, panel in (
            (axes[0], "evidence", "Evidence incoming share", "A  Claim-boundary re-read"),
            (axes[1], "history", "Response-history incoming share", "B  History takeover"),
        ):
            axis.plot(offset, curves[f"correct_{name}"], label="correct claim")
            axis.plot(
                offset,
                curves[f"hallucinated_{name}"],
                label="hallucinated claim",
            )
            axis.axvline(0, linewidth=1.0, linestyle="--")
            axis.set_title(panel, loc="left", fontweight="semibold")
            axis.set_xlabel("Token offset from claim start")
            axis.set_ylabel(ylabel)
            axis.legend(frameon=False)
            axis.grid(alpha=0.2)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
