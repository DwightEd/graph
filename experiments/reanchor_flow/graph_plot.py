"""Draw the sparse source-to-token graph for one detailed sample."""

from __future__ import annotations

import numpy as np

from .rhythm import robust_z


def token_label(value: str) -> str:
    value = str(value).replace("\n", "↵").replace(" ", "·")
    return value[:12] or "∅"


def draw_top_graph(axis, result: dict, label: np.ndarray) -> None:
    from matplotlib.patches import FancyArrowPatch

    edge = np.asarray(result["detail_edge_map"], dtype=float)
    response_start = int(np.asarray(result["response_start"]).item())
    text = np.asarray(result["token_text"]).astype(str)
    prompt_peak = np.asarray(result["prompt_peak"], dtype=bool)
    review_peak = np.asarray(result["review_peak"], dtype=bool)
    anchor_peak = np.asarray(result["anchor_peak"], dtype=bool)
    future = np.asarray(result["future_influence"], dtype=float)
    events, _ = edge.shape

    target_candidates = (
        set(np.flatnonzero(prompt_peak).tolist())
        | set(np.flatnonzero(review_peak).tolist())
        | set(np.flatnonzero(anchor_peak).tolist())
    )
    if len(target_candidates) < 6:
        score = np.nan_to_num(robust_z(result["prompt_delta"]), nan=-np.inf)
        score += np.nan_to_num(robust_z(result["nonlocal_delta"]), nan=-np.inf)
        score += np.nan_to_num(robust_z(future), nan=-np.inf)
        target_candidates.update(np.argsort(-score)[: min(8, events)].tolist())
    target_candidates = sorted(target_candidates)[:12]

    candidates: list[tuple[float, int, int]] = []
    for event in target_candidates:
        for source in np.argsort(-edge[event])[:3]:
            weight = float(edge[event, source])
            if weight > 0:
                candidates.append((weight, int(source), response_start + int(event)))
    candidates = sorted(candidates, reverse=True)[:30]
    nodes = sorted({node for _, source, target in candidates for node in (source, target)})
    prompt_nodes = [node for node in nodes if node < response_start]
    evidence_mask = np.asarray(
        result.get("detail_prompt_evidence_mask", np.zeros(response_start)),
        dtype=bool,
    )
    evidence_nodes = [
        node
        for node in prompt_nodes
        if node < len(evidence_mask) and evidence_mask[node]
    ]
    other_prompt_nodes = [node for node in prompt_nodes if node not in evidence_nodes]
    response_nodes = [node for node in nodes if node >= response_start]

    position: dict[int, tuple[float, float]] = {}
    for index, node in enumerate(prompt_nodes):
        position[node] = (0.05, (index + 1) / (len(prompt_nodes) + 1))
    denominator = max(events - 1, 1)
    for node in response_nodes:
        event = node - response_start
        position[node] = (
            0.28 + 0.68 * event / denominator,
            0.5 + 0.13 * np.sin(event * 1.7),
        )

    future_z = np.nan_to_num(robust_z(future), nan=0.0)

    def response_sizes(items):
        return [
            45
            + 18
            * max(0.0, min(4.0, future_z[node - response_start] + 1.0))
            for node in items
        ]

    maximum = max((weight for weight, _, _ in candidates), default=1.0)
    for weight, source, target in candidates:
        if source not in position or target not in position:
            continue
        axis.add_patch(
            FancyArrowPatch(
                position[source],
                position[target],
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.4 + 2.2 * weight / maximum,
                alpha=0.35 + 0.55 * weight / maximum,
                connectionstyle="arc3,rad=0.08",
            )
        )

    if evidence_nodes:
        axis.scatter(
            [position[node][0] for node in evidence_nodes],
            [position[node][1] for node in evidence_nodes],
            marker="s",
            s=55,
            label="evidence source",
        )
    if other_prompt_nodes:
        axis.scatter(
            [position[node][0] for node in other_prompt_nodes],
            [position[node][1] for node in other_prompt_nodes],
            marker="s",
            s=35,
            label="other prompt",
        )
    ordinary = [
        node
        for node in response_nodes
        if not prompt_peak[node - response_start]
        and not review_peak[node - response_start]
        and not anchor_peak[node - response_start]
    ]
    if ordinary:
        axis.scatter(
            [position[node][0] for node in ordinary],
            [position[node][1] for node in ordinary],
            s=response_sizes(ordinary),
            label="response",
        )
    prompt_nodes_response = [
        node for node in response_nodes if prompt_peak[node - response_start]
    ]
    if prompt_nodes_response:
        axis.scatter(
            [position[node][0] for node in prompt_nodes_response],
            [position[node][1] for node in prompt_nodes_response],
            marker="D",
            s=response_sizes(prompt_nodes_response),
            label="prompt revisit",
        )
    review_nodes = [node for node in response_nodes if review_peak[node - response_start]]
    if review_nodes:
        axis.scatter(
            [position[node][0] for node in review_nodes],
            [position[node][1] for node in review_nodes],
            marker="P",
            s=response_sizes(review_nodes),
            label="nonlocal review",
        )
    anchor_nodes = [node for node in response_nodes if anchor_peak[node - response_start]]
    if anchor_nodes:
        axis.scatter(
            [position[node][0] for node in anchor_nodes],
            [position[node][1] for node in anchor_nodes],
            marker="^",
            s=response_sizes(anchor_nodes),
            label="future anchor",
        )
    hallucinated = [
        node
        for node in response_nodes
        if 0 <= node - response_start < len(label) and label[node - response_start]
    ]
    if hallucinated:
        axis.scatter(
            [position[node][0] for node in hallucinated],
            [position[node][1] for node in hallucinated],
            marker="x",
            s=95,
            linewidths=1.8,
            label="hallucinated",
        )

    for node in nodes:
        x, y = position[node]
        axis.text(
            x,
            y + 0.035,
            token_label(text[node]),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_xlim(-0.02, 1.03)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("D  Top functional routes", loc="left")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=7, ncol=2)
