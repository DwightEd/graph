"""Layer/head identity is preserved; the plotted ledger is not an intermediate prediction."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ledger(trace, path):
    scores = trace["residual_scores"]
    curve = np.r_[scores[0, 0], scores[:, 1:].reshape(-1)]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(np.arange(len(curve)) / 2, curve, marker=".", color="#245fa5")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(
        xlabel="Layer progress: attention at .5, FFN at next integer",
        ylabel="Projection on final candidate contrast",
        title="Actual residual trajectory",
    )
    layer = np.arange(len(scores))
    axes[1].bar(
        layer - 0.18, trace["attention_score"], width=0.36, label="Attention write"
    )
    axes[1].bar(layer + 0.18, trace["mlp_score"], width=0.36, label="FFN write")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(xlabel="Physical layer (zero based)", ylabel="Direct logit write")
    axes[1].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_heads(trace, path):
    groups = [
        ("Scope", [0]),
        ("Supported value", [1]),
        ("Other value source", [2]),
        ("History", [4, 5, 6]),
    ]
    arrays = [trace["group_route_mass"], trace["group_logit_write"]]
    sensitivities = np.stack(
        [
            trace["edge_margin_sensitivity"][..., trace["group_ids"] == group].sum(-1)
            for group in range(len(trace["group_names"]))
        ],
        -1,
    )
    arrays.append(sensitivities)
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    labels = ("Route mass", "Direct logit write", "Local final-margin sensitivity")
    for row, (array, label) in enumerate(zip(arrays, labels)):
        panels = [array[..., indices].sum(-1) for _, indices in groups]
        limit = max(float(np.abs(panel).max()) for panel in panels) or 1.0
        for column, ((name, _), panel) in enumerate(zip(groups, panels)):
            axis = axes[row, column]
            plot = axis.imshow(
                panel,
                origin="lower",
                aspect="auto",
                cmap="viridis" if row == 0 else "RdBu_r",
                vmin=0 if row == 0 else -limit,
                vmax=limit,
            )
            axis.set(
                title=f"{name}\n{label}",
                xlabel="Physical head",
                ylabel="Physical layer",
            )
            axis.set_xticks(np.arange(0, panel.shape[1], max(1, panel.shape[1] // 8)))
            axis.set_yticks(np.arange(0, panel.shape[0], max(1, panel.shape[0] // 8)))
            figure.colorbar(plot, ax=axis, shrink=0.65)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_timeline(traces, path):
    offsets = [int(trace["decision_offset"]) for trace in traces]
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    axes[0].plot(
        offsets, [float(t["entropy"]) for t in traces], marker="o", label="Entropy"
    )
    axes[0].plot(
        offsets,
        [-float(t["observed_logp"]) for t in traces],
        marker="o",
        label="Observed surprisal",
    )
    axes[0].legend()
    axes[0].set(ylabel="Nats", title="Observed native tokens; no forced candidate tail")
    axes[1].plot(offsets, [float(t["logit_gap"]) for t in traces], marker="o")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        xlabel="Query offset from reviewed decision",
        ylabel="Candidate logit gap",
        title="Earlier points are future-candidate diagnostics, not factual decisions",
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
