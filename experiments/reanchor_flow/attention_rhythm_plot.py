"""Raw maps, full-response timelines and native relay inspection figures."""
from pathlib import Path

import numpy as np

from .attention_relay import relay_examples

def plot_sample(trace, audit, path, title=""):
    """Individual raw heads, same query/source coordinates; no pooled main view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "attention_maps" not in trace:
        return
    selected, rows = trace["map_heads"], trace["map_query_position"]
    heads, p = trace["waad"].shape[1], int(trace["response_start"])
    columns = len(selected)
    fig, axes = plt.subplots(4, columns, figsize=(5 * columns, 12), squeeze=False)
    for column, flat in enumerate(selected):
        l, h = divmod(int(flat), heads)
        raw = trace["attention_maps"][column]
        # This is an exact crop, not a renormalization or a sparse reconstruction.
        left, right = max(0, int(rows[0]) - 32), min(raw.shape[-1], int(rows[-1]) + 1)
        image = axes[0, column].imshow(raw[:, left:right], aspect="auto", origin="upper",
                                      extent=(left - .5, right - .5, rows[-1] + .5, rows[0] - .5))
        fig.colorbar(image, ax=axes[0, column], fraction=.04)
        axes[0, column].set_title(f"L{l} H{h}; mean distance={trace['distance_mean'][l,h]:.2f}")
        axes[0, column].set_xlabel("absolute source position (exact crop)")
        axes[0, column].set_ylabel("query position")
        slot = rows - (p - 1)
        axes[1, column].plot(rows, trace["waad"][l, h, slot], label="attention WAAD")
        axes[1, column].plot(rows, trace["message_waad"][l, h, slot], label="message-weighted WAAD")
        valid = rows >= p
        peak = audit["waad_peaks"][l, h, rows[valid] - p]
        axes[1, column].scatter(rows[valid][peak], trace["waad"][l, h, slot[valid]][peak], marker="x")
        axes[1, column].set_ylabel("clipped lookback distance")
        axes[1, column].legend(fontsize=8)
        bucket = trace["attention_buckets"][l, h, slot]
        for k, name in enumerate(("evidence", "other prompt", "older response", "recent response")):
            axes[2, column].plot(rows, bucket[:, k], label=name)
        axes[2, column].set_ylabel("attention mass; old bucket view")
        axes[2, column].legend(fontsize=7, ncol=2)
        axes[3, column].plot(rows, trace["fai"][l, h, slot], label="FAI (offline)")
        axes[3, column].set_ylabel("future incoming attention")
        axes[3, column].set_xlabel("absolute source position; NOT predictor label")
        axes[3, column].legend(fontsize=8)
    if "token_text" in trace:
        for axis in axes[0]:
            xmin, xmax = axis.get_xlim()
            positions = np.linspace(max(0, int(xmin + .5)), min(len(trace["token_text"])-1, int(xmax-.5)), 8, dtype=int)
            axis.set_xticks(positions)
            axis.set_xticklabels([f"{i}: {str(trace['token_text'][i]).strip()[:10]}" for i in positions],
                                 rotation=55, ha="right", fontsize=7)
    fig.suptitle(title + "\nIndividual-head observations; peaks are candidates, not factual mechanisms.")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # Full source panel preserves prompt stripes hidden by the near-diagonal crop.
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4), squeeze=False)
    for column, flat in enumerate(selected):
        l, h = divmod(int(flat), heads)
        axes[0, column].imshow(trace["attention_maps"][column], aspect="auto", origin="upper",
                              extent=(-.5, len(trace["token_ids"]) - .5, rows[-1] + .5, rows[0] - .5))
        axes[0, column].axvline(p - .5, linestyle="--")
        axes[0, column].set_title(f"L{l} H{h}: all source positions")
        axes[0, column].set_xlabel("source; dashed line = response start")
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".full_sources.png"), dpi=130)
    plt.close(fig)
    if "paper_group_maps" in trace:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for k, name in enumerate(("local", "global")):
            axes[k].imshow(trace["paper_group_maps"][k], aspect="auto", origin="upper")
            axes[k].set_title(name + " group MEAN: paper-reference panel ONLY")
        fig.tight_layout()
        fig.savefig(Path(path).with_suffix(".paper_reference.png"), dpi=130)
        plt.close(fig)


def plot_relays(trace, path, labels=None):
    """Two-hop topology plus every receiver head and the actual MLP/readout path."""
    if not len(trace.get("relay_paths", [])):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = relay_examples(trace)
    fig, axes = plt.subplots(len(examples), 3, figsize=(19, 4.8 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        s, wl, wh, b, rl, rh, q = map(int, trace["relay_paths"][row])
        axis = axes[row, 0]
        outcome = "unlabeled"
        if labels is not None:
            label = int(labels[q + 1 - int(trace["response_start"])])
            outcome = {0: "nonhallucinated", 1: "hallucinated", -1: "unknown"}[label]
        nodes = [(0, 2, f"source {s}\n{example['tokens']['source'][:18]}"),
                 (1.7, 2, f"carrier {b}\nL{wl} H{wh} write"),
                 (1.7, .9, f"carrier {b}\nMLP + residual"),
                 (3.4, .9, f"carrier {b}\nL{rl} input K/V"),
                 (5.1, .9, f"query {q}\nL{rl} H{rh} read"),
                 (5.1, -.3, f"predict {q+1}\n{example['predicted_token'][:18]}\n{outcome}")]
        for x, y, label in nodes:
            axis.text(x, y, label, ha="center", va="center", fontsize=8,
                      bbox=dict(boxstyle="round,pad=.4", facecolor="#edf4fa", edgecolor="#4b6780"))
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
            axis.annotate("", xy=nodes[right][:2], xytext=nodes[left][:2],
                          arrowprops=dict(arrowstyle="->", color="#4b6780", shrinkA=24, shrinkB=24))
        axis.set(xlim=(-.9, 6), ylim=(-.9, 2.7), title="Same carrier; strictly increasing layers")
        axis.axis("off")
        slot = int(np.flatnonzero(trace["relay_node_position"] == q)[0])
        margins = trace["relay_head_margin"][..., slot]
        bound = max(float(np.abs(margins).max()), 1e-8)
        image = axes[row, 1].imshow(margins, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        axes[row, 1].set(xlabel="head", ylabel="layer", title=f"Receiver {q}: signed head writes (all heads)")
        axes[row, 1].set_xticks(np.unique(np.linspace(0, margins.shape[1]-1, min(8, margins.shape[1]), dtype=int)))
        axes[row, 1].set_yticks(np.unique(np.linspace(0, margins.shape[0]-1, min(8, margins.shape[0]), dtype=int)))
        fig.colorbar(image, ax=axes[row, 1], fraction=.04)
        stage = trace["relay_stage_margin"][:, slot]
        axes[row, 2].plot(np.arange(len(stage)), stage, label="residual readout")
        axes[row, 2].bar(np.arange(len(stage)-1)+.5, trace["relay_mlp_margin"][:, slot],
                         alpha=.45, label="MLP write")
        axes[row, 2].axhline(0, color="grey", linewidth=.7)
        axes[row, 2].set(xlabel="layer boundary", ylabel="observed minus native runner",
                         title=f"{example['predicted_token'][:16]!r}: module/readout accounting")
        axes[row, 2].legend(fontsize=8)
    fig.suptitle("Native relay inspection: signed readout is not factual support or a causal effect")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_timeline(trace, labels, path):
    """Whole response, with predictor/created-carrier coordinates distinguished."""
    from .attention_rhythm import representative_heads
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = trace.get("map_heads", representative_heads(trace))
    if not len(selected):
        return
    p, heads = int(trace["response_start"]), trace["waad"].shape[1]
    token = np.arange(p, len(trace["token_ids"]))
    fig, axes = plt.subplots(3, len(selected), figsize=(5 * len(selected), 10), squeeze=False)
    for column, flat in enumerate(selected):
        layer, head = divmod(int(flat), heads)
        axes[0, column].plot(token, trace["waad"][layer, head, :-1], label="attention WAAD at b-1")
        axes[0, column].plot(token, trace["message_waad"][layer, head, :-1], label="magnitude-weighted control", alpha=.65)
        axes[0, column].set_title(f"L{layer} H{head}: full response")
        axes[0, column].set_ylabel("reading before predicting b")
        axes[1, column].plot(token, trace["fai"][layer, head, 1:], label="FAI of carrier b (offline)")
        axes[1, column].set_ylabel("later attention to b")
        buckets = trace["attention_buckets"][layer, head, :-1]
        axes[2, column].plot(token, buckets[:, :2].sum(-1), label="prompt mass at b-1")
        axes[2, column].plot(token, buckets[:, 2:].sum(-1), label="history mass at b-1")
        axes[2, column].plot(token, p / token, linestyle=":", color="grey", label="uniform-source prompt control")
        axes[2, column].set(xlabel="absolute response token b", ylabel="attention mass")
        for axis in axes[:, column]:
            if labels is not None:
                boundaries = np.flatnonzero(np.diff(np.r_[False, np.asarray(labels) == 1, False]))
                for left, right in boundaries.reshape(-1, 2):
                    axis.axvspan(token[left]-.5, token[right-1]+.5, color="#d62728", alpha=.12)
            axis.legend(fontsize=7)
    fig.suptitle("Red: hallucinated token b; reading is at b-1, FAI is for b. No head averaging.")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_population(name, group, pairs, output, evaluated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["waad_peak_rate", "bucket_missed_fraction", "attention_message_waad_mae",
            "prompt_slope_all", "history_slope_all", "prompt_excess_uniform_slope_all"]
    if evaluated:
        keys += ["prompt_slope_nonhallucinated", "waad_peak_matched_h_minus_n",
                 "fai_carrier_matched_h_minus_n"]
    rows = (len(keys) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
    for axis, key in zip(axes.ravel(), keys):
        values = np.asarray(group[key]["mean"], float)
        signed = "slope" in key or "minus" in key
        finite = values[np.isfinite(values)]
        bound = max(float(np.abs(finite).max()), 1e-8) if finite.size else 1
        scale = dict(cmap="RdBu_r", vmin=-bound, vmax=bound) if signed else dict(cmap="viridis")
        image = axis.imshow(values, aspect="auto", **scale)
        axis.set(title=key.replace("_", " "), xlabel="head", ylabel="layer")
        axis.title.set_fontsize(9)
        fig.colorbar(image, ax=axis, fraction=.04)
    fig.suptitle(f"{name}: source-balanced estimates for EACH head; no causal classification")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / f"population_{name}.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    values = pairs["mean_lift"]
    finite = values[np.isfinite(values)]
    bound = max(float(np.abs(finite).max()), 1e-8) if finite.size else 1
    image = axes[0].imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    fig.colorbar(image, ax=axes[0], fraction=.04)
    image = axes[1].imshow(pairs["valid_sources"], aspect="auto")
    fig.colorbar(image, ax=axes[1], fraction=.04)
    axes[0].set_title("Same-carrier coupling minus occupancy control")
    axes[1].set_title("Eligible sources for that exact head pair")
    for axis in axes:
        axis.set(xlabel="reading head: layer * heads + head", ylabel="writing head: layer * heads + head")
    fig.suptitle(f"{name}: reading layer > writing layer; missing pairs stay missing")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / f"head_pairs_{name}.png", dpi=130)
    plt.close(fig)
