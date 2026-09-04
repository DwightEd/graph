"""Draw a sparse source-to-token view for one detailed sample."""

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
    transition = np.asarray(result["transition_peak"], dtype=bool)
    prompt_peak = np.asarray(result["prompt_peak"], dtype=bool)
    review_peak = np.asarray(result["review_peak"], dtype=bool)
    anchor_peak = np.asarray(result["anchor_peak"], dtype=bool)
    future = np.asarray(result["future_influence"], dtype=float)
    events, _ = edge.shape

    target_candidates = (
        set(np.flatnonzero(transition).tolist())
        | set(np.flatnonzero(prompt_peak).tolist())
        | set(np.flatnonzero(review_peak).tolist())
        | set(np.flatnonzero(anchor_peak).tolist())
    )
    if len(target_candidates) < 6:
        score = np.nan_to_num(robust_z(result["route_change"]), nan=-np.inf)
        score += np.nan_to_num(robust_z(result["evidence_delta"]), nan=-np.inf)
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

    def scatter(items, **kwargs):
        if items:
            axis.scatter(
                [position[node][0] for node in items],
                [position[node][1] for node in items],
                **kwargs,
            )

    scatter(evidence_nodes, marker="s", s=55, label="evidence source")
    scatter(other_prompt_nodes, marker="s", s=35, label="other prompt")
    ordinary = [
        node
        for node in response_nodes
        if not transition[node - response_start]
        and not prompt_peak[node - response_start]
        and not review_peak[node - response_start]
        and not anchor_peak[node - response_start]
    ]
    scatter(ordinary, s=response_sizes(ordinary), label="response")
    transition_nodes = [node for node in response_nodes if transition[node - response_start]]
    prompt_nodes_response = [node for node in response_nodes if prompt_peak[node - response_start]]
    review_nodes = [node for node in response_nodes if review_peak[node - response_start]]
    anchor_nodes = [node for node in response_nodes if anchor_peak[node - response_start]]
    scatter(transition_nodes, marker="o", s=response_sizes(transition_nodes), label="route transition")
    scatter(prompt_nodes_response, marker="D", s=response_sizes(prompt_nodes_response), label="prompt re-entry")
    scatter(review_nodes, marker="P", s=response_sizes(review_nodes), label="nonlocal review")
    scatter(anchor_nodes, marker="^", s=response_sizes(anchor_nodes), label="future anchor")
    hallucinated = [
        node
        for node in response_nodes
        if 0 <= node - response_start < len(label) and label[node - response_start]
    ]
    scatter(hallucinated, marker="x", s=95, linewidths=1.8, label="hallucinated")

    for node in nodes:
        x, y = position[node]
        axis.text(x, y + 0.035, token_label(text[node]), ha="center", va="bottom", fontsize=7)
    axis.set_xlim(-0.02, 1.03)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("D  Top observed routes", loc="left")
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=7, ncol=2)
