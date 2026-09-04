"""One-sample routing figure; the deep state audit has a separate figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph_plot import draw_top_graph
from .rhythm import robust_z


def add_legend(axis, **kwargs) -> None:
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(**kwargs)


def save_sample_figure(
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
    response_start = int(np.asarray(result["response_start"]).item())
    edge = np.asarray(result["detail_edge_map"], dtype=float)
    transition = np.asarray(result["transition_peak"], dtype=bool)
    prompt_peak = np.asarray(result["prompt_peak"], dtype=bool)
    review_peak = np.asarray(result["review_peak"], dtype=bool)
    anchor_peak = np.asarray(result["anchor_peak"], dtype=bool)
    boundaries = np.asarray(result["sentence_boundary_position"], dtype=int) - response_start
    event = np.arange(edge.shape[0])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    try:
        if title:
            figure.suptitle(title)

        image = axes[0, 0].imshow(
            np.log1p(1000 * edge.T),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, 0].axhline(response_start - 0.5, linewidth=0.8, linestyle="--")
        for peak in np.flatnonzero(transition):
            axes[0, 0].axvline(peak, linewidth=0.7, alpha=0.3)
        axes[0, 0].set_title("A  Source → predicted-token routes", loc="left")
        axes[0, 0].set_xlabel("Response token event")
        axes[0, 0].set_ylabel("Source token position")
        figure.colorbar(image, ax=axes[0, 0], label="log functional share")

        curves = {
            "route change": robust_z(result["route_change"]),
            "prompt re-entry": robust_z(result["prompt_delta"]),
            "evidence re-entry": robust_z(result["evidence_delta"]),
            "nonlocal review": robust_z(result["nonlocal_delta"]),
            "predictor reuse": robust_z(
                result.get("predictor_reuse", np.full(len(event), np.nan))
            ),
            "emitted-token anchor": robust_z(
                result.get("emitted_token_anchor", result["future_influence"])
            ),
        }
        for name, values in curves.items():
            axes[0, 1].plot(event, values, label=name)
        for position in boundaries:
            if 0 <= position < len(event):
                axes[0, 1].axvline(position, linewidth=0.5, linestyle=":", alpha=0.25)
        for position in np.flatnonzero(label):
            axes[0, 1].axvspan(position - 0.5, position + 0.5, alpha=0.08)
        axes[0, 1].scatter(
            np.flatnonzero(transition),
            curves["route change"][transition],
            marker="o",
            label="transition peak",
        )
        axes[0, 1].scatter(
            np.flatnonzero(prompt_peak),
            curves["prompt re-entry"][prompt_peak],
            marker="D",
            label="prompt peak",
        )
        axes[0, 1].scatter(
            np.flatnonzero(review_peak),
            curves["nonlocal review"][review_peak],
            marker="P",
            label="review peak",
        )
        axes[0, 1].scatter(
            np.flatnonzero(anchor_peak),
            curves["emitted-token anchor"][anchor_peak],
            marker="^",
            label="emitted-anchor peak",
        )
        axes[0, 1].axhline(0, linewidth=0.6)
        axes[0, 1].set_title("B  Internal transition and re-entry", loc="left")
        axes[0, 1].set_xlabel("Response token event")
        axes[0, 1].set_ylabel("robust z-score")
        add_legend(axes[0, 1], frameon=False, fontsize=7, ncol=2)
        axes[0, 1].grid(axis="y", alpha=0.2)

        head = np.asarray(result["detail_evidence_head"], dtype=float)
        flattened = head.reshape(-1, head.shape[-1])
        ranking = np.argsort(-np.nanmax(flattened, axis=1))[: min(64, len(flattened))]
        head_image = axes[1, 0].imshow(
            flattened[ranking],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[1, 0].set_title("C  Head-resolved RAG-evidence reads", loc="left")
        axes[1, 0].set_xlabel("Response token event")
        axes[1, 0].set_ylabel("top layer-head rows")
        figure.colorbar(head_image, ax=axes[1, 0], label="evidence functional share")

        draw_top_graph(axes[1, 1], result, label)
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
