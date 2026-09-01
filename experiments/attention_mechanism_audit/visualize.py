"""Label-free detector and mechanism-state visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .detect import SCORE_NAMES

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PRIMARY_SCORE = SCORE_NAMES[0]
SCORE_LABELS = {
    "unsupported_history_takeover": "Unsupported-history takeover",
    "evidence_bypass": "Evidence bypass",
    "evidence_route_contraction": "Evidence-route contraction",
    "history_route_contraction": "History-route contraction",
    "confidence": "Token surprisal",
}
COLORS = {
    PRIMARY_SCORE: "#b2182b",
    "evidence_bypass": "#ef8a62",
    "evidence_route_contraction": "#2166ac",
    "history_route_contraction": "#762a83",
    "confidence": "#1b7837",
}


def _style(name: str) -> tuple[str, str]:
    return SCORE_LABELS.get(name, name.replace("_", " ").title()), COLORS.get(
        name, "#555555"
    )


def _detection_curves(
    label: np.ndarray,
    scores: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="0.65", ls="--", label="Chance")
    axes[1].axhline(label.mean(), color="0.65", ls="--", label="Prevalence")
    for name, score in scores.items():
        display, color = _style(name)
        fpr, tpr, _ = roc_curve(label, score)
        precision, recall, _ = precision_recall_curve(label, score)
        width = 2.5 if name == PRIMARY_SCORE else 1.4
        axes[0].plot(fpr, tpr, color=color, lw=width, label=display)
        axes[1].plot(recall, precision, color=color, lw=width, label=display)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Frozen out-of-fold token scores")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _ecdf(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(value)
    return x, np.arange(1, len(x) + 1) / len(x)


def _score_distributions(
    scores: dict[str, np.ndarray],
    output: Path,
) -> None:
    names = tuple(scores)
    figure, axes = plt.subplots(
        1, len(names), figsize=(4.4 * len(names), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        score = scores[name]
        low, high = np.quantile(score, [0.01, 0.99])
        x, y = _ecdf(np.clip(score, low, high))
        display, color = _style(name)
        axis.plot(x, y, color=color, lw=2 if name == PRIMARY_SCORE else 1.5)
        axis.set(title=display, xlabel="Token score", ylabel="Empirical CDF")
        axis.grid(alpha=0.18)
    figure.suptitle("Frozen score distributions (1st–99th percentile)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _scores_by_position(
    scores: dict[str, np.ndarray],
    token_index: np.ndarray,
    response_length: np.ndarray,
    output: Path,
) -> None:
    names = tuple(scores)
    position = np.minimum(
        (10 * (token_index + 0.5) / response_length).astype(np.int8), 9
    )
    x = (np.arange(10) + 0.5) / 10
    figure, axes = plt.subplots(
        1, len(names), figsize=(4.4 * len(names), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        mean = [
            scores[name][position == block].mean()
            if np.any(position == block)
            else np.nan
            for block in range(10)
        ]
        display, color = _style(name)
        axis.plot(x, mean, marker="o", ms=3, color=color)
        axis.set(
            title=display,
            xlabel="Relative response position",
            ylabel="Mean frozen score",
        )
        axis.grid(alpha=0.18)
    figure.suptitle("Position diagnostics (not used to choose score direction)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_population(
    label: np.ndarray,
    scores: dict[str, np.ndarray],
    token_index: np.ndarray,
    response_length: np.ndarray,
    report: dict,
    output_root: Path,
) -> None:
    """Render pooled frozen-score diagnostics."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generated = (
        output_root / "roc_pr.png",
        output_root / "score_distributions.png",
        output_root / "scores_by_position.png",
    )
    unavailable = output_root / "DETECTION_UNAVAILABLE.txt"
    for path in (*generated, unavailable):
        path.unlink(missing_ok=True)
    detector = report.get("detector", {})
    if not len(label) or not detector.get("mechanism_scores_available", True):
        reason = detector.get("reason") or (
            "no response token has strict-history support yet"
            if not len(label)
            else "mechanism scores are unavailable"
        )
        unavailable.write_text(
            str(reason) + "\n",
            encoding="utf-8",
        )
        return
    if all(result["auroc"] is not None for result in report["detection"].values()):
        _detection_curves(label, scores, generated[0])
    _score_distributions(scores, generated[1])
    _scores_by_position(
        scores,
        token_index,
        response_length,
        generated[2],
    )


def plot_sample_dashboard(
    record: dict,
    layers: dict[str, np.ndarray],
    graph,
    output: Path,
) -> None:
    """Plot one response's explicit routes and attention/MLP pathway state."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token_text = list(record["token_text"])
    x = np.arange(len(token_text))
    figure, axes = plt.subplots(
        6,
        1,
        figsize=(14, 18),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(f"Sample {record['sample_id']}")

    graph_layer = np.asarray(graph.layer)
    row_layer = np.asarray(graph.row_layer)
    last_layer = int(row_layer.max()) if len(row_layer) else 0
    selected = graph_layer == last_layer
    magnitude = np.asarray(graph.magnitude)[selected]
    scale = (
        magnitude / max(float(magnitude.max()), 1e-12) if len(magnitude) else magnitude
    )
    scatter = axes[0].scatter(
        np.asarray(graph.target)[selected] - int(record["predictor_position"][0]),
        np.asarray(graph.source)[selected],
        c=np.asarray(graph.head)[selected],
        s=2 + 18 * np.sqrt(scale),
        alpha=0.35,
        cmap="turbo",
        linewidths=0,
    )
    row_selected = row_layer == last_layer
    remainder = np.asarray(graph.remainder)[row_selected]
    remainder_mean = float(remainder.mean()) if len(remainder) else 0.0
    axes[0].set(
        title=(
            f"Explicit evidence/history source-head routes at layer {last_layer} "
            f"(mean adaptive-cover remainder {remainder_mean:.3f})"
        ),
        ylabel="Absolute source token",
    )
    if len(magnitude):
        figure.colorbar(scatter, ax=axes[0], label="Head")

    for axis, key, title in (
        (axes[1], "edge_evidence_head_entropy", "Evidence route entropy by head"),
        (axes[2], "edge_history_head_entropy", "History route entropy by head"),
    ):
        image = axis.imshow(
            layers[key][-1].T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
        )
        axis.set(title=title, ylabel="Head")
        figure.colorbar(image, ax=axis)

    pathway = layers["pathway_mlp_projection"]
    pathway_image = np.concatenate(
        [pathway[..., index] for index in range(pathway.shape[-1])], axis=0
    )
    finite = np.abs(pathway_image[np.isfinite(pathway_image)])
    limit = max(float(finite.max()), 1e-12) if len(finite) else 1.0
    image = axes[3].imshow(
        pathway_image,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    layer_count = pathway.shape[0]
    for boundary in (layer_count - 0.5, 2 * layer_count - 0.5):
        axes[3].axhline(boundary, color="black", lw=0.5)
    axes[3].set(
        title="MLP projection on evidence / history / interaction pathways",
        ylabel="Contrast × layer",
    )
    figure.colorbar(image, ax=axes[3])

    axes[4].axhline(0, color="0.55", lw=0.8)
    for name, color, label in (
        ("evidence_support", "#2166ac", "Evidence support"),
        ("history_support", "#b2182b", "History support"),
        ("route_interaction", "#762a83", "E×history interaction"),
    ):
        axes[4].plot(x, record[name], color=color, label=label)
    axes[4].set(title="Symmetric factorial deletion effects", ylabel="Δ log p")
    axes[4].legend(frameon=False, ncols=3)

    for name, color, label in (
        ("evidence_route_contraction", "#2166ac", "Evidence routes"),
        ("history_route_contraction", "#762a83", "History routes"),
    ):
        axes[5].plot(x, record[name], color=color, label=label)
    axes[5].axhline(0, color="0.55", lw=0.8)
    axes[5].set(title="Fixed-window route contraction", ylabel="Contraction")
    axes[5].legend(frameon=False, ncols=2)

    stride = max(1, int(np.ceil(len(token_text) / 24)))
    ticks = np.arange(0, len(token_text), stride)
    axes[-1].set_xticks(
        ticks,
        [token_text[index].replace("\n", "\\n") for index in ticks],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    axes[-1].set_xlabel("Response token")
    figure.savefig(output, dpi=180)
    plt.close(figure)
