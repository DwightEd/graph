"""Figures for sample-level and population-level mechanism audits."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .reporting import KEY_METRICS, ONSET_METRICS


def _hallucination_spans(label: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(label.astype(bool), (1, 1))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, stops))


def _shade_hallucinations(axis, spans: list[tuple[int, int]]) -> None:
    for start, stop in spans:
        axis.axvspan(start - 0.5, stop - 0.5, color="crimson", alpha=0.14, lw=0)


def _token_ticks(axis, token_text: list[str]) -> None:
    stride = max(1, int(np.ceil(len(token_text) / 24)))
    ticks = np.arange(0, len(token_text), stride)
    labels = [token_text[index].replace("\n", "\\n") for index in ticks]
    axis.set_xticks(ticks, labels, rotation=60, ha="right", fontsize=7)


def plot_sample_dashboard(
    record: dict,
    layer_metrics: dict[str, np.ndarray],
    output: Path,
) -> None:
    """Plot routes, causal effects, and role shares for one generated response."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token_text = list(record["token_text"])
    label = np.asarray(record["label"], dtype=bool)
    token_metrics = record["token_metrics"]
    evidence_effect = np.asarray(token_metrics["evidence_message_effect"], dtype=float)
    response_effect = np.asarray(token_metrics["response_message_effect"], dtype=float)
    routing = np.asarray(layer_metrics["routing_imbalance"], dtype=float)
    dispersion = np.asarray(layer_metrics["source_dispersion"], dtype=float)
    evidence_share = np.asarray(layer_metrics["evidence_share"], dtype=float)
    response_share = np.asarray(layer_metrics["response_share"], dtype=float)
    spans = _hallucination_spans(label)
    x = np.arange(len(token_text))

    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    figure.suptitle(f"Sample {record['sample_id']}: attention mechanism audit", fontsize=14)

    image = axes[0].imshow(
        routing,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
    )
    axes[0].set(title="Response − evidence message routing", ylabel="Layer")
    figure.colorbar(image, ax=axes[0], label="Routing imbalance")

    image = axes[1].imshow(
        dispersion,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set(title="Source-message dispersion", ylabel="Layer")
    figure.colorbar(image, ax=axes[1], label="Normalized entropy")

    axes[2].axhline(0, color="0.55", lw=0.8)
    axes[2].plot(x, evidence_effect, label="Evidence effect", color="#2166ac", lw=1.5)
    axes[2].plot(x, response_effect, label="Response effect", color="#b2182b", lw=1.5)
    axes[2].set(title="Observed-token causal effects", ylabel="Δ log p")
    axes[2].legend(frameon=False, ncols=2)

    axes[3].plot(x, evidence_share.mean(0), label="Evidence", color="#1b9e77", lw=1.5)
    axes[3].plot(x, response_share.mean(0), label="Response", color="#d95f02", lw=1.5)
    axes[3].set(
        title="Mean functional-message role share",
        ylabel="Share",
        xlabel="Response token",
    )
    axes[3].set_ylim(0, 1)
    axes[3].legend(frameon=False, ncols=2)

    for axis in axes:
        _shade_hallucinations(axis, spans)
    _token_ticks(axes[-1], token_text)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_population_effects(report: dict, output: Path) -> None:
    if "by_split" in report:
        styles = {
            "train": ("s", "#1b9e77", 0.2),
            "test": ("^", "#d95f02", 0.0),
        }
        split_reports = [
            (name, current, *styles.get(name, ("D", "#7570b3", 0.1)))
            for name, current in report["by_split"].items()
        ]
        split_reports.append(("all", report, "o", "#2166ac", -0.2))
    else:
        split_reports = [(None, report, "o", "#2166ac", 0.0)]

    panels = (
        (
            "Percentage-point effects",
            (KEY_METRICS[0], KEY_METRICS[1], KEY_METRICS[4]),
            "pp",
        ),
        ("Head-role disagreement", (KEY_METRICS[2],), "JS divergence"),
        ("Observed-token evidence support", (KEY_METRICS[3],), "Δ log p"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (title, metrics, unit) in zip(axes, panels):
        y = np.arange(len(metrics))[::-1]
        axis.axvline(0, color="0.4", lw=1)
        for split_name, split_report, marker, color, offset in split_reports:
            for row, metric in zip(y, metrics):
                result = split_report["summaries"][metric.key]
                delta = result["position_matched_source_equal_difference"]
                if delta is None:
                    continue
                effect = delta * metric.scale
                ci = result.get("ci95")
                error = None
                if ci is not None and ci[0] is not None and ci[1] is not None:
                    error = [
                        [effect - ci[0] * metric.scale],
                        [ci[1] * metric.scale - effect],
                    ]
                axis.errorbar(
                    effect,
                    row + offset,
                    xerr=error,
                    fmt=marker,
                    color=color,
                    ecolor=color,
                    capsize=3,
                    ms=6,
                )
            if split_name is not None and axis is axes[0]:
                axis.plot([], [], marker=marker, color=color, ls="none", label=split_name)
        axis.set_yticks(y, [metric.label for metric in metrics])
        axis.set_xlabel(f"Hallucinated − correct ({unit})")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Position-matched population effects (95% source-bootstrap CI)")
    if "by_split" in report:
        axes[0].legend(frameon=False, ncols=3)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_sample_map(sample_records: list[dict], output: Path) -> None:
    routing = np.asarray(
        [
            record["routing_mean"]
            if "routing_mean" in record
            else np.mean(record["token_metrics"]["message_routing_drift_mean"])
            for record in sample_records
        ]
    )
    evidence = np.asarray(
        [
            record["evidence_effect_mean"]
            if "evidence_effect_mean" in record
            else np.mean(record["token_metrics"]["evidence_message_effect"])
            for record in sample_records
        ]
    )
    fraction = np.asarray(
        [
            record["hallucinated_fraction"]
            if "hallucinated_fraction" in record
            else np.mean(record["label"])
            for record in sample_records
        ]
    )

    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    points = axis.scatter(
        routing,
        evidence,
        c=fraction,
        cmap="magma_r",
        vmin=0,
        vmax=1,
        s=34,
        alpha=0.8,
        edgecolors="none",
    )
    axis.axvline(0, color="0.6", lw=0.8)
    axis.axhline(0, color="0.6", lw=0.8)
    axis.set(
        xlabel="Mean response − evidence routing",
        ylabel="Mean evidence effect (Δ log p)",
        title="Sample-level mechanism map",
    )
    figure.colorbar(points, ax=axis, label="Hallucinated-token fraction")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _onset_series(values, length: int, scale: float) -> np.ndarray:
    if values is None:
        return np.full(length, np.nan)
    return np.asarray([np.nan if value is None else value for value in values]) * scale


def _plot_onset(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True, constrained_layout=True)
    panels = (
        (axes[0], ONSET_METRICS[:2], "Matched DiD (pp)", "Routing and dispersion"),
        (axes[1], ONSET_METRICS[2:], "Matched DiD (Δ log p)", "Evidence support"),
    )
    for axis, metrics, ylabel, title in panels:
        for metric in metrics:
            result = report["matched_onset"][metric.key]
            offset = np.asarray(result["offset"])
            value = _onset_series(
                result.get("difference_in_difference"), len(offset), metric.scale
            )
            low = _onset_series(result.get("ci95_low"), len(offset), metric.scale)
            high = _onset_series(result.get("ci95_high"), len(offset), metric.scale)
            axis.plot(offset, value, marker="o", ms=3, label=metric.label)
            axis.fill_between(offset, low, high, alpha=0.15)
        axis.axhline(0, color="0.45", lw=0.8)
        axis.axvline(0, color="crimson", lw=1, ls="--")
        axis.set(title=title, ylabel=ylabel)
        axis.legend(frameon=False)
    axes[1].set_xlabel("Token offset from hallucination-span onset")
    figure.suptitle("Mechanism dynamics around hallucination-span onsets")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_population(report: dict, sample_records: list[dict], output_root: Path) -> None:
    """Plot population effects, sample heterogeneity, and available onset dynamics."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _plot_population_effects(report, output_root / "population_effects.png")
    _plot_sample_map(sample_records, output_root / "sample_map.png")

    onset = report["matched_onset"][ONSET_METRICS[0].key]
    if any(onset["events"]):
        _plot_onset(report, output_root / "onset_dynamics.png")


__all__ = ["plot_population", "plot_sample_dashboard"]
