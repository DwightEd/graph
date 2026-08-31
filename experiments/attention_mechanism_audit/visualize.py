"""Population figures and one explicitly requested sample dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

SCORE_LABELS = {
    "causal_route_capture": "Causal route capture",
    "routing_imbalance": "Response − evidence routing",
    "source_dispersion": "Source dispersion",
    "message_independent_preference": "Message-independent preference",
}
COLORS = {
    "causal_route_capture": "#b2182b",
    "routing_imbalance": "#2166ac",
    "source_dispersion": "#762a83",
    "message_independent_preference": "#1b7837",
}


def _detection_curves(
    label: np.ndarray,
    scores: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], color="0.65", ls="--", label="Chance")
    axes[1].axhline(label.mean(), color="0.65", ls="--", label="Prevalence")
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(label, score)
        precision, recall, _ = precision_recall_curve(label, score)
        width = 2.5 if name == "causal_route_capture" else 1.4
        axes[0].plot(
            fpr,
            tpr,
            color=COLORS[name],
            lw=width,
            label=SCORE_LABELS[name],
        )
        axes[1].plot(
            recall,
            precision,
            color=COLORS[name],
            lw=width,
            label=SCORE_LABELS[name],
        )
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Pooled captured response tokens")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _ecdf(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(value)
    return x, np.arange(1, len(x) + 1) / len(x)


def _mechanism_distributions(
    scores: dict[str, np.ndarray],
    output: Path,
) -> None:
    names = (
        "routing_imbalance",
        "source_dispersion",
        "message_independent_preference",
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, name in zip(axes, names):
        score = scores[name]
        low, high = np.quantile(score, [0.01, 0.99])
        x, y = _ecdf(np.clip(score, low, high))
        axis.plot(x, y, color=COLORS[name], lw=1.8)
        axis.set(
            title=SCORE_LABELS[name],
            xlabel="Token score",
            ylabel="Empirical CDF",
        )
        axis.grid(alpha=0.18)
    figure.suptitle("Pooled-token distributions (clipped at 1st/99th percentiles)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _mechanisms_by_position(
    scores: dict[str, np.ndarray],
    token_index: np.ndarray,
    response_length: np.ndarray,
    output: Path,
) -> None:
    names = (
        "routing_imbalance",
        "source_dispersion",
        "message_independent_preference",
    )
    position = np.minimum(
        (10 * (token_index + 0.5) / response_length).astype(np.int8),
        9,
    )
    x = (np.arange(10) + 0.5) / 10
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, name in zip(axes, names):
        mean = [
            scores[name][position == block].mean()
            if np.any(position == block)
            else np.nan
            for block in range(10)
        ]
        axis.plot(x, mean, marker="o", ms=3, color=COLORS[name])
        axis.set(
            title=SCORE_LABELS[name],
            xlabel="Relative response position",
            ylabel="Mean token score",
        )
        axis.grid(alpha=0.18)
    figure.suptitle("Mechanism dynamics over pooled captured responses")
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
    """Generate only pooled population figures."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if all(result["auroc"] is not None for result in report["detection"].values()):
        _detection_curves(label, scores, output_root / "roc_pr.png")
    _mechanism_distributions(scores, output_root / "mechanism_distributions.png")
    _mechanisms_by_position(
        scores,
        token_index,
        response_length,
        output_root / "mechanism_by_position.png",
    )


def plot_sample_dashboard(
    record: dict,
    layers: dict[str, np.ndarray],
    output: Path,
) -> None:
    """Plot the exact saved route dynamics for one requested response."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    token_text = list(record["token_text"])
    x = np.arange(len(token_text))
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(14, 15),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(f"Sample {record['sample_id']}")

    image = axes[0].imshow(
        layers["routing_imbalance"],
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    axes[0].set(title="Response − evidence functional routing", ylabel="Layer")
    figure.colorbar(image, ax=axes[0])

    image = axes[1].imshow(
        layers["source_dispersion"],
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1].set(title="Source-message dispersion", ylabel="Layer")
    figure.colorbar(image, ax=axes[1])

    image = axes[2].imshow(
        record["source_flow"],
        aspect="auto",
        origin="upper",
        cmap="magma",
        vmin=0,
    )
    axes[2].set(
        title="Retained top-k source-token message share",
        ylabel="Source token",
    )
    axes[2].set_yticks(
        np.arange(len(record["source_token_text"])),
        record["source_token_text"],
        fontsize=7,
    )
    figure.colorbar(image, ax=axes[2])

    axes[3].axhline(0, color="0.55", lw=0.8)
    axes[3].plot(
        x,
        record["evidence_effect"],
        color="#2166ac",
        label="Evidence effect",
    )
    axes[3].plot(
        x,
        record["response_effect"],
        color="#b2182b",
        label="Response effect",
    )
    axes[3].set(title="Frozen-model message-deletion effects", ylabel="Δ log p")
    axes[3].legend(frameon=False, ncols=2)

    axes[4].plot(
        x,
        layers["evidence_share"].mean(0),
        color="#1b9e77",
        label="Evidence",
    )
    axes[4].plot(
        x,
        layers["response_share"].mean(0),
        color="#d95f02",
        label="Response",
    )
    axes[4].set(title="Functional-message role share", ylabel="Share")
    axes[4].legend(frameon=False, ncols=2)

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
