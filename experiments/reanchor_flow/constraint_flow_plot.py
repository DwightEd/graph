"""Fixed-target views of constraint response; no head averaging or causal labels."""
from __future__ import annotations

import numpy as np


def _heat(ax, values, title, xlabel, ylabel):
    limit = max(float(np.max(np.abs(values))), 1e-12)
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(0, values.shape[1], max(1, values.shape[1]//12)))
    ax.set_yticks(np.arange(0, values.shape[0], max(1, values.shape[0]//12)))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return im


def plot_pair(trace, confirmation, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    names = [*map(str, trace["ledger_names"]), "rounding"]
    terms = np.r_[trace["ledger_terms"], trace["ledger_rounding"]]
    axes[0, 0].bar(np.arange(len(terms)), terms, color=np.where(terms >= 0, "#b64f43", "#32789c"))
    axes[0, 0].set_xticks(np.arange(len(terms)), names, rotation=20)
    axes[0, 0].axhline(0, color="black", lw=.6)
    axes[0, 0].set(title=f"Target change = {float(trace['target_margin_delta']):.4g}",
                   ylabel="positive - negative logit change")
    for column, field in enumerate(("head_content_projection", "head_routing_projection")):
        image = _heat(axes[1, column], trace[field][..., -1],
                      "Content term at target q" if column == 0 else "Routing term at target q", "head", "layer")
        fig.colorbar(image, ax=axes[1, column], shrink=.8)
    state_slots = np.searchsorted(trace["state_position"], trace["row_position"])
    states = trace["state_projection"][:, state_slots]
    image = _heat(axes[0, 1], states, "State response in the SAME target basis\n(not a mediated effect)",
                  "absolute token position (includes P-1)", "residual stage")
    ticks = axes[0, 1].get_xticks().astype(int)
    axes[0, 1].set_xticklabels(trace["row_position"][ticks])
    fig.colorbar(image, ax=axes[0, 1], shrink=.8)
    axes[2, 0].plot(states[:, -1], marker=".", label="residual response at q")
    axes[2, 0].bar(np.arange(len(states)-1)+.25, trace["attention_projection"][:, -1], width=.25, label="attention delta")
    axes[2, 0].bar(np.arange(len(states)-1)+.5, trace["mlp_projection"][:, -1], width=.25, label="MLP delta")
    axes[2, 0].axhline(0, color="black", lw=.6)
    axes[2, 0].set(xlabel="layer / residual stage", ylabel="fixed-target projection", title="Native updates; final norm shown separately above")
    axes[2, 0].legend(fontsize=8)
    ax = axes[2, 1]
    ax.axis("off")
    ax.set_title("Frozen value-path intervention")
    if not len(trace["paths"]):
        ax.text(.05, .7, "No eligible two-edge witness.\nThis pair remains in the cohort denominator.", transform=ax.transAxes)
    else:
        for i, (s, ell, wh, b, k, h, q) in enumerate(trace["paths"]):
            y = .82 - i * .43
            text = trace["token_text"]
            root_text = repr(str(text[s])[:12])
            if "token_text_pair" in trace:
                root_text += " / " + repr(str(trace["token_text_pair"][1, s])[:12])
            labels = [f"source {s}\n{root_text}", f"carrier {b}\n{str(text[b])[:16]!r}", f"query {q}\nnext-token contrast"]
            for x, label in zip((.12, .5, .88), labels):
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        bbox={"boxstyle": "round", "fc": "#eef3f6", "ec": "#b4c3cc"}, transform=ax.transAxes)
            for left, right, label in ((.24, .39, f"L{ell}H{wh}"), (.62, .77, f"L{k}H{h}")):
                ax.annotate("", xy=(right, y), xytext=(left, y), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
                ax.text((left+right)/2, y+.095, label, ha="center", fontsize=8, transform=ax.transAxes)
            effect = "not run" if confirmation is None else f"{float(confirmation['path_effect'][i]):+.5g}"
            ax.text(.02, y-.15, f"{trace['path_roles'][i]}: exact path effect = {effect}", fontsize=9, transform=ax.transAxes)
        ax.text(.02, .03, "Incoming edge replacement; outgoing A fixed native.\nOther value paths and all Q/K-mediated paths are excluded.",
                fontsize=8, transform=ax.transAxes)
    contrast = ""
    if "candidate_text" in trace:
        negative, positive = map(str, trace["candidate_text"])
        contrast = f" | fixed next token: {positive!r} minus {negative!r}"
    fig.suptitle(title + contrast + "\nControlled condition response; no hallucination classification", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_gallery(items, path):
    from html import escape

    sections = []
    for item in items:
        m = item["metadata"]
        text = "\n\n".join(f"{name}:\n{m[name]}" for name in ("condition0", "condition1", "response_prefix", "negative", "positive"))
        sections.append(f'<section><h2>{escape(item["title"])}</h2><details><summary>Inputs and fixed candidates</summary>'
                        f'<pre>{escape(text)}</pre></details><img src="{escape(item["path"])}" alt="constraint audit"></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Constraint response audit</title>'
            '<style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 16px}'
            'img{width:100%}pre{white-space:pre-wrap;background:#f2f5f7;padding:16px}section{margin:32px 0}</style>'
            '<h1>Constraint response audit</h1><p>Designed condition comparisons, not natural hallucination classification. '
            'Expand each input before interpreting its graph. Controls reuse the fact comparison’s frozen paths.</p>'
            '<p><a href="summary.md">Report</a> · <a href="summary.json">Numbers</a></p><img src="cohort.png" alt="paired comparisons">')
    path.write_text(page + "".join(sections) + "</html>", encoding="utf-8")


def plot_cohort(entries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Lines connect the SAME comparison group. Every failure remains visible.
    kinds = [k for k in ("fact_flip", "rename", "irrelevant", "wording") if any(e["kind"] == k for e in entries)]
    kinds += sorted({e["kind"] for e in entries} - set(kinds))
    groups = sorted({e["group_id"] for e in entries})
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for group in groups:
        by_kind = {e["kind"]: e for e in entries if e["group_id"] == group}
        values = [by_kind[k]["delta"] if k in by_kind else np.nan for k in kinds]
        ax.plot(np.arange(len(kinds)), values, "o-", alpha=.65, markersize=4, label=group)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(np.arange(len(kinds)), kinds)
    ax.set(ylabel="registered candidate logit change", title="Fact-specific response versus matched controls\nDesigned pairs; not natural hallucination detection")
    if len(groups) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)
