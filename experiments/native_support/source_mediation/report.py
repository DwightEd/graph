"""Magnitude/position checks accompany detection; neither is a semantic verdict."""

import numpy as np
from scipy.stats import spearmanr
from state_audit.storage import read_arrays, read_json


def diagnostics(output, settings, protocol):
    rows = []
    for index, response in enumerate(settings["responses"]):
        directory = output / "responses" / f"{index:04d}"
        values = read_arrays(directory / "components.npz")
        units = read_json(directory / "views.json")["units"]
        interaction = np.abs(values["interaction"])
        first = np.array([unit["start"] for unit in units])
        total = float(interaction.sum())
        correlation = None
        if len(interaction) > 1 and np.ptp(interaction) > 0:
            correlation = float(spearmanr(np.arange(len(interaction)), interaction).statistic)
        row = dict(id=response["id"], tokens=len(interaction), units=len(units),
            interaction_mean=float(interaction.mean()),
            unit_first_token_share=float(interaction[first].sum()/total) if total > 0 else None,
            interaction_position_spearman=correlation,
            shapley_additivity_max_error=float(np.max(np.abs(
                values["context_shapley"] + values["history_shapley"] - values["total"]))))
        if protocol["mode"] == "cache":
            edges = read_arrays(directory / "head_effects.npz")
            size = np.abs(edges["interaction"])
            order = np.argsort(size)[::-1][:10]
            row.update(coalition_mean=float(np.abs(values["conditional_coalition"]).mean()),
                measured_edge_rows=len(size),
                unintervened_tokens=int((values["selected_edge_count"] == 0).sum()),
                strongest_edges=[{name: float(value[index]) if name in ("with_source", "without_source", "interaction")
                                  else int(value[index]) for name, value in edges.items()} for index in order])
        else:
            row.update(read_json(directory / "capture_complete.json"))
        rows.append(row)
    return dict(responses=rows, labels_used=False,
        interpretation="Signed interactions describe functional dependence, not correct/incorrect facts",
        cached_joint_minus_single="Aggregate higher-order remainder; not identified pairwise interactions",
        native_mediator="All-layer strictly earlier answer K/V; prompt-conditioned current-query bypass retained")
