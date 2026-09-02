"""Label-free detector and mechanism-state visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .capture import REGISTER_NAMES
from .detect import SCORE_NAMES

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PRIMARY_SCORE = SCORE_NAMES[0]
SCORE_LABELS = {
    "provenance_takeover": "Autonomous-history provenance takeover",
    "evidence_bypass": "Evidence bypass",
    "symmetric_route_capture": "Symmetric route capture",
    "unsupported_history_takeover": "Unsupported-history takeover",
    "confidence": "Token surprisal",
}
COLORS = {
    "evidence_bypass": "#b2182b",
    "symmetric_route_capture": "#2166ac",
    "unsupported_history_takeover": "#762a83",
    "provenance_takeover": "#ef8a62",
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
    figure.suptitle("Frozen raw token scores")
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
    """Plot explicit provenance edges and both layer-wise register states."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token_text = list(record["token_text"])
    x = np.arange(len(token_text))
    figure, axes = plt.subplots(6, 1, figsize=(14, 19), constrained_layout=True)
    figure.suptitle(f"Sample {record['sample_id']}")

    graph_layer = np.asarray(graph.layer)
    row_layer = np.asarray(graph.row_layer)
    last_layer = int(row_layer.max()) if len(row_layer) else 0
    selected = graph_layer == last_layer
    magnitude = np.asarray(graph.magnitude)[selected]
    contribution = np.asarray(graph.contribution)[selected]
    scale = (
        magnitude / max(float(magnitude.max()), 1e-12) if len(magnitude) else magnitude
    )
    limit = max(float(np.abs(contribution).max()), 1e-12) if len(contribution) else 1
    scatter = None
    edge_register = np.asarray(graph.register)[selected]
    edge_head = np.asarray(graph.head)[selected]
    head_scale = max(int(edge_head.max()), 1) if len(edge_head) else 1
    source_with_head = np.asarray(graph.source)[selected] + 0.3 * (
        edge_head / head_scale - 0.5
    )
    for register, marker in zip(REGISTER_NAMES, ("o", "^")):
        current = edge_register == register
        if not current.any():
            continue
        scatter = axes[0].scatter(
            np.asarray(graph.target)[selected][current]
            - int(record["predictor_position"][0]),
            source_with_head[current],
            c=contribution[current],
            s=3 + 22 * np.sqrt(scale[current]),
            marker=marker,
            alpha=0.5,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            linewidths=0,
            label=register.replace("_", " "),
        )
    row_selected = row_layer == last_layer
    remainder = np.asarray(graph.remainder_magnitude)[row_selected]
    remainder_mean = float(remainder.mean()) if len(remainder) else 0.0
    axes[0].set(
        title=(
            f"Explicit head-resolved provenance edges at layer {last_layer}; "
            f"omitted tail remains endpoint-free (mean magnitude {remainder_mean:.3f})"
        ),
        xlabel="Response target token",
        ylabel="Absolute source token (head jitter)",
    )
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[0], label="Signed register contribution")
        axes[0].legend(frameon=False, ncols=2)

    def register_image(axis, statistic: str, title: str, *, signed: bool = False):
        values = np.concatenate(
            [layers[f"register_{register}_{statistic}"] for register in REGISTER_NAMES]
        )
        kwargs = {"cmap": "viridis", "vmin": 0.0}
        if signed:
            bound = max(float(np.abs(values).max()), 1e-12)
            kwargs = {"cmap": "coolwarm", "vmin": -bound, "vmax": bound}
        image = axis.imshow(values, aspect="auto", origin="lower", **kwargs)
        layer_count = values.shape[0] // len(REGISTER_NAMES)
        axis.axhline(
            layer_count - 0.5,
            color="white" if not signed else "black",
            lw=0.8,
        )
        axis.set(title=title, xlabel="Response token", ylabel="Register × layer")
        figure.colorbar(image, ax=axis)

    register_image(axes[1], "attention_norm", "Attention-stage register norm")
    register_image(axes[2], "mlp_norm", "MLP-stage register norm")
    register_image(axes[3], "output_norm", "Output register norm")
    register_image(
        axes[4], "mlp_alignment", "MLP alignment with each register", signed=True
    )

    axes[5].axhline(0, color="0.55", lw=0.8)
    for name, color, label in (
        ("evidence_support", "#2166ac", "Evidence support"),
        ("history_support", "#b2182b", "History support"),
        ("route_interaction", "#762a83", "E×history interaction"),
        ("evidence_bypass", "#ef8a62", "Evidence bypass"),
        ("symmetric_route_capture", "#1b7837", "Symmetric capture"),
    ):
        axes[5].plot(x, record[name], color=color, label=label)
    conservation = np.asarray(layers["register_conservation_error"])
    edge_error = np.asarray(layers["register_attention_edge_error"])
    axes[5].set(
        title=(
            "Causal branch controls "
            f"(max register/edge errors {np.abs(conservation).max():.2e}/"
            f"{np.abs(edge_error).max():.2e})"
        ),
        ylabel="Δ log p",
    )
    axes[5].legend(frameon=False, ncols=3)

    stride = max(1, int(np.ceil(len(token_text) / 24)))
    ticks = np.arange(0, len(token_text), stride)
    axes[5].set_xticks(
        ticks,
        [token_text[index].replace("\n", "\\n") for index in ticks],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    axes[5].set_xlabel("Response token")
    figure.savefig(output, dpi=180)
    plt.close(figure)
