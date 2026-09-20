"""Inspect exact selected nodes and each physical head's native timeline."""

import html
import json

import numpy as np
import pandas as pd


def token_window(text, position, radius=5):
    return "".join(text[max(0, position - radius):position]) + " ⟦" + text[position] + "⟧ " + "".join(text[position + 1:position + radius + 1])


def timeline(directory, context, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with np.load(directory / "routes.npz", allow_pickle=False) as saved:
        gains = saved["lookback_gain"]
    with np.load(directory / "flow.npz", allow_pickle=False) as saved:
        flow = saved["attention_potential_edges"].sum(axis=-1)
    figure, axes = plt.subplots(len(plan), 2, figsize=(12, 2.5 * len(plan)), squeeze=False)
    start = max(1, context["prompt_length"] - 1)
    positions = np.arange(start, len(context["prefix_ids"]))
    for row, event in enumerate(plan):
        unit = event["entry"]
        layer, head, receiver = [unit[key] for key in ("layer", "head", "receiver")]
        values = [gains[layer, head, start:], flow[layer, head, start:]]
        for axis, series, label in zip(axes[row], values, ["Common-key lookback gain", "Target-conditioned incoming flow"]):
            axis.plot(positions, series, linewidth=1)
            axis.axvline(receiver, color="#b45309", linestyle="--", label=f"selected r={receiver}")
            axis.axvline(positions[-1], color="#555555", linestyle=":", label=f"decision q={positions[-1]}")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set(title=f"E{event['event']} · L{layer} H{head} · {label}", xlabel="Absolute prefix position")
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "route_timeline.png", dpi=140)
    plt.close(figure)


def event_html(event, context, effects):
    unit = event["entry"]
    receiver = unit["receiver"]
    roles = context["reviewed_case"]["source_roles"]
    table = pd.DataFrame([dict(role=role, reviewed_quotes=" | ".join(quotes)) for role, quotes in roles.items()])
    fragments = [f"<h3>Event {event['event']}: L{unit['layer']} H{unit['head']} at {receiver}</h3>",
        "<p>" + html.escape(event["selected"]["selection"]) + "</p>",
        "<pre>" + html.escape(token_window(context["token_text"], receiver)) + "</pre>",
        "<p>Selected source: " + html.escape(token_window(context["token_text"], unit["sources"][0])) + "</p>",
        table.to_html(index=False, escape=True)]
    relay = event["relay"]
    if relay:
        fragments.append(f"<p>Later read: L{relay['layer']} H{relay['head']}, source {receiver} → receiver {relay['receiver']}.</p>")
    else:
        fragments.append("<p>No retained later cross-position read: relay not measured.</p>")
    if not effects.empty:
        columns = [name for name in ["source_group", "dose", "support", "entry_conditional", "interaction", "relay_restoration_gain", "numeric_ok"] if name in effects]
        fragments.append(effects[effects.event == event["event"]][columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.6g}"))
    return "\n".join(fragments)


def write_views(output):
    fragments = ["<!doctype html><meta charset='utf-8'><title>Reanchor audit</title>",
        "<style>body{max-width:1200px;margin:30px auto;font:16px system-ui;color:#19232d}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #cbd5e1}pre{white-space:pre-wrap;background:#f1f5f9;padding:12px}img{max-width:100%}</style>",
        "<h1>Reanchor audit</h1><p>Candidate routes, independent source roles and native intervention effects. Layer/head indices are zero-based. These are mechanism audits, not detector results.</p>"]
    for path in sorted((output / "cases").glob("*/*/plan.json")):
        directory = path.parent
        plan = json.loads(path.read_text())
        if not plan:
            continue
        context = json.loads((directory / "context.json").read_text())
        effects_path = directory / "effects.csv"
        effects = pd.read_csv(effects_path) if effects_path.exists() else pd.DataFrame()
        if (directory / "flow.npz").exists():
            timeline(directory, context, plan)
        fragments.append("<h2>" + html.escape(context["case_id"] + " / " + context["side"]) + "</h2>")
        if (directory / "route_timeline.png").exists():
            fragments.append(f"<img src='{directory.relative_to(output)}/route_timeline.png' alt='Native head timelines'>")
        fragments.extend(event_html(event, context, effects) for event in plan)
    if len(fragments) == 3:
        fragments.append("<p>No native routes measured in this output. See inventory.csv and coverage.csv.</p>")
    (output / "review.html").write_text("\n".join(fragments), encoding="utf-8")
