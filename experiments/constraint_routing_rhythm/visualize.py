"""Per-sample figures for functional routes and intervention effects."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def numeric_array(
    name: str,
    values,
    dimensions: int,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    """Convert one plotting input and check only its visible array contract."""

    array = np.asarray(values, dtype=float)
    if array.ndim != dimensions or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {dimensions}D array")
    finite = np.isfinite(array)
    if np.isinf(array).any() or (not allow_nan and not finite.all()):
        raise ValueError(f"{name} must contain finite values")
    return array


def sparse_ticks(length: int, maximum: int = 8) -> np.ndarray:
    """Choose readable position ticks without labeling every long-sequence token."""

    count = min(length, maximum)
    return np.linspace(0, length - 1, count).round().astype(int)


def event_tick_labels(
    positions: np.ndarray,
    indices: np.ndarray,
    token_labels: Sequence[str] | None,
) -> list[str]:
    labels = [f"{positions[index]:g}" for index in indices]
    if token_labels is not None:
        labels = [
            f"{labels[i]}\n{compact_token(token_labels[index])}"
            for i, index in enumerate(indices)
        ]
    return labels


def compact_token(token: object, maximum: int = 12) -> str:
    """Keep decoded token labels readable on sparse axes."""

    label = str(token).replace("\n", "↵").replace("\t", "⇥")
    return label if len(label) <= maximum else f"{label[: maximum - 1]}…"


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs for lightly shading evidence source spans."""

    padded = np.pad(mask.astype(np.int8), (1, 1))
    change = np.flatnonzero(np.diff(padded))
    return list(zip(change[::2], change[1::2], strict=True))


def save_sample_figure(
    path: str | Path,
    *,
    local_route,
    global_route,
    functional_reach,
    relay_capacity,
    constraint_deficit,
    response_positions=None,
    token_labels: Sequence[str] | None = None,
    source_token_labels: Sequence[str] | None = None,
    response_start: int | None = None,
    evidence_mask=None,
    carrier_mask=None,
    title: str | None = None,
) -> Path:
    """Save a four-panel diagnostic figure for one sample.

    ``local_route`` and ``global_route`` have shape ``[response, source]``.
    The two rhythm inputs have one value per response event. The primary
    ``constraint_deficit`` is the unnormalized cut-minus-baseline margin. It is
    plotted separately and is never combined with the route diagnostics.
    """

    local = numeric_array("local_route", local_route, 2)
    global_ = numeric_array("global_route", global_route, 2)
    if global_.shape != local.shape:
        raise ValueError("local_route and global_route must have the same shape")

    event_count, source_count = local.shape
    diagnostics = {
        "functional reach": numeric_array("functional_reach", functional_reach, 1),
        "relay capacity": numeric_array(
            "relay_capacity", relay_capacity, 1, allow_nan=True
        ),
    }
    primary_values = numeric_array(
        "constraint_deficit", constraint_deficit, 1, allow_nan=True
    )
    for name, values in (*diagnostics.items(), ("constraint deficit", primary_values)):
        if len(values) != event_count:
            raise ValueError(f"{name} must have one value per response event")

    if response_positions is None:
        positions = np.arange(event_count)
    else:
        positions = numeric_array("response_positions", response_positions, 1)
        if len(positions) != event_count:
            raise ValueError(
                "response_positions must have one value per response event"
            )

    if token_labels is not None and len(token_labels) != event_count:
        raise ValueError("token_labels must have one label per response event")
    if source_token_labels is not None and len(source_token_labels) != source_count:
        raise ValueError("source_token_labels must have one label per source")

    evidence = None
    if evidence_mask is not None:
        evidence = np.asarray(evidence_mask, dtype=bool)
        if evidence.shape != (source_count,):
            raise ValueError("evidence_mask must have one value per source")
    carriers = None
    if carrier_mask is not None:
        carriers = np.asarray(carrier_mask, dtype=bool)
        if carriers.shape != (event_count,):
            raise ValueError("carrier_mask must have one value per response event")

    # Importing pyplot only when a figure is requested keeps analysis modules
    # independent of the plotting backend.
    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response_ticks = sparse_ticks(event_count)
    source_ticks = sparse_ticks(source_count)
    tick_labels = event_tick_labels(positions, response_ticks, token_labels)
    source_tick_labels = event_tick_labels(
        np.arange(source_count), source_ticks, source_token_labels
    )
    route_max = max(float(local.max()), float(global_.max()))
    route_min = min(0.0, float(local.min()), float(global_.min()))
    if route_max == route_min:
        route_max = route_min + 1.0

    with plt.rc_context(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "figure.dpi": 120,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
        try:
            if title:
                figure.suptitle(title, fontsize=12, fontweight="semibold")

            images = []
            for axis, route, panel_title in (
                (axes[0, 0], local, "A  Local functional routes"),
                (axes[0, 1], global_, "B  Global functional routes"),
            ):
                images.append(
                    axis.imshow(
                        route,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                        cmap="magma",
                        vmin=route_min,
                        vmax=route_max,
                    )
                )
                axis.set_title(panel_title, loc="left", fontweight="semibold")
                axis.set_xlabel("Source position")
                axis.set_ylabel("Response position")
                axis.set_xticks(source_ticks, source_tick_labels)
                axis.set_yticks(response_ticks, tick_labels)
                if response_start is not None:
                    axis.axvline(
                        response_start - 0.5,
                        color="white",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.8,
                    )
                if evidence is not None:
                    for start, stop in true_runs(evidence):
                        axis.axvspan(
                            start - 0.5,
                            stop - 0.5,
                            color="#66CCEE",
                            alpha=0.12,
                            linewidth=0,
                        )
            figure.colorbar(
                images[0],
                ax=[axes[0, 0], axes[0, 1]],
                label="Mean functional route share",
                shrink=0.9,
            )

            timeline = axes[1, 0]
            timeline.set_title(
                "C  Response timeline", loc="left", fontweight="semibold"
            )
            diagnostic_colors = ("#4477AA", "#EE7733")
            for (name, values), color in zip(
                diagnostics.items(), diagnostic_colors, strict=True
            ):
                timeline.plot(positions, values, color=color, linewidth=1.8, label=name)
            if carriers is not None and carriers.any():
                timeline.scatter(
                    positions[carriers],
                    diagnostics["relay capacity"][carriers],
                    marker="*",
                    s=65,
                    color="#AA3377",
                    label="proposed carrier",
                    zorder=4,
                )
            timeline.set_xlabel("Response position")
            timeline.set_ylabel("Route diagnostic")
            timeline.grid(axis="y", color="#DDDDDD", linewidth=0.6)

            primary = timeline.twinx()
            primary.spines["right"].set_visible(True)
            primary.axhline(0, color="#333333", linewidth=0.7, alpha=0.5)
            primary.plot(
                positions,
                primary_values,
                color="#111111",
                linestyle="--",
                linewidth=2.0,
                label="constraint deficit (primary)",
            )
            primary.set_ylabel("Signed margin change (primary)")
            handles, labels = timeline.get_legend_handles_labels()
            primary_handles, primary_labels = primary.get_legend_handles_labels()
            timeline.legend(
                handles + primary_handles,
                labels + primary_labels,
                frameon=False,
                fontsize=8,
                ncol=2,
                loc="best",
            )

            control = axes[1, 1]
            control.set_title(
                "D  Relay proposal vs total evidence control",
                loc="left",
                fontweight="semibold",
            )
            finite = np.isfinite(primary_values) & np.isfinite(
                diagnostics["relay capacity"]
            )
            if finite.any():
                points = control.scatter(
                    diagnostics["relay capacity"][finite],
                    -primary_values[finite],
                    c=positions[finite],
                    cmap="viridis",
                    s=24,
                    alpha=0.8,
                )
                figure.colorbar(
                    points,
                    ax=control,
                    label="Response position",
                    shrink=0.8,
                )
            if carriers is not None and (carriers & finite).any():
                selected = carriers & finite
                control.scatter(
                    diagnostics["relay capacity"][selected],
                    -primary_values[selected],
                    marker="*",
                    s=80,
                    facecolors="none",
                    edgecolors="#AA3377",
                    linewidths=1.2,
                    label="proposed carrier",
                )
                control.legend(frameon=False, fontsize=8, loc="best")
            control.axhline(0, color="#333333", linewidth=0.7, alpha=0.5)
            control.set_xlabel("Relay capacity (ordered route proposal)")
            control.set_ylabel("Evidence support = −constraint deficit")
            control.grid(color="#DDDDDD", linewidth=0.6)

            timeline.set_xticks(positions[response_ticks], tick_labels)

            figure.savefig(destination, format="png", dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)

    return destination
