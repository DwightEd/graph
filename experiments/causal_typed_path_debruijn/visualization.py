"""Per-sample causal routing and phase visualizations.

Plotting is deliberately downstream of channel-resolved computation.  Means in
the trajectory panels are display summaries only; they are never fed back into
the detector.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .graph_builder import RP


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _radial_positions(nodes, *, current: int, response_idx: int) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {int(current): (0.0, 0.0)}
    prompt = sorted(node for node in nodes if node < response_idx)
    response = sorted(node for node in nodes if response_idx <= node < current)
    for rank, node in enumerate(prompt):
        angle = 2.0 * np.pi * rank / max(len(prompt), 1)
        positions[node] = (3.0 * np.cos(angle), 3.0 * np.sin(angle))
    maximum_lag = max(current - response_idx, 1)
    for node in response:
        lag = current - node
        radius = 0.8 + 1.6 * lag / maximum_lag
        angle = 2.0 * np.pi * (node - response_idx + 0.5) / max(maximum_lag, 1)
        positions[node] = (radius * np.cos(angle), radius * np.sin(angle))
    return positions


def _draw_radial_graph(
    axis,
    graph,
    *,
    token_index: int,
    channel: int,
    maximum_edges: int,
) -> None:
    layer = channel // graph.num_heads
    head = channel % graph.num_heads
    current = graph.response_idx + token_index
    selected = (
        (graph.layer == layer)
        & (graph.head == head)
        & (graph.target <= current)
    )
    edge_indices = np.flatnonzero(_as_numpy(selected).astype(bool))
    weights = _as_numpy(graph.weight)[edge_indices]
    if len(edge_indices) > maximum_edges:
        keep = np.argsort(weights, kind="stable")[-maximum_edges:]
        edge_indices = edge_indices[keep]
        weights = weights[keep]
    source = _as_numpy(graph.source)[edge_indices].astype(int)
    target = _as_numpy(graph.target)[edge_indices].astype(int)
    relation = _as_numpy(graph.relation)[edge_indices].astype(int)
    nodes = set(source.tolist()) | set(target.tolist()) | {current}
    positions = _radial_positions(nodes, current=current, response_idx=graph.response_idx)
    scale = float(weights.max()) if len(weights) else 1.0
    for source_node, target_node, edge_weight, edge_relation in zip(
        source, target, weights, relation, strict=True
    ):
        if source_node not in positions or target_node not in positions:
            continue
        color = "#2563eb" if edge_relation == RP else "#ea580c"
        arrow = FancyArrowPatch(
            positions[source_node],
            positions[target_node],
            arrowstyle="-|>",
            mutation_scale=7,
            connectionstyle="arc3,rad=0.12",
            linewidth=0.4 + 2.4 * float(edge_weight) / max(scale, 1e-8),
            alpha=0.45,
            color=color,
        )
        axis.add_patch(arrow)
    for node, (x, y) in positions.items():
        if node == current:
            color, size = "#dc2626", 75
        elif node < graph.response_idx:
            color, size = "#2563eb", 34
        else:
            color, size = "#f59e0b", 42
        axis.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        axis.text(x, y, str(node), fontsize=6, ha="center", va="bottom")
    axis.set_title(f"radial prefix graph · layer {layer}, head {head}")
    axis.set_aspect("equal")
    axis.axis("off")


def render_sample_diagnostics(
    graph,
    automaton,
    phase,
    *,
    token_score,
    output_path,
    token_index: int | None = None,
    maximum_edges: int = 80,
) -> Path:
    """Render route states, phase dynamics, and one exact-channel graph."""

    score = np.asarray(token_score, dtype=np.float32)
    if score.shape != (graph.num_response_tokens,) or not np.isfinite(score).all():
        raise ValueError("token_score must be a finite [R] vector")
    if token_index is None:
        token_index = int(np.argmax(score))
    token_index = int(token_index)
    if not 0 <= token_index < graph.num_response_tokens:
        raise ValueError("token_index is outside the response")
    raw = _as_numpy(phase.raw_channel_score)
    channel = int(np.argmax(raw[token_index]))
    route = _as_numpy(automaton.flat_route_distribution)
    prompt_lineage = _as_numpy(automaton.flat_prompt_lineage)
    detached = _as_numpy(automaton.flat_detached)
    rupture = _as_numpy(phase.rupture_memory)
    lockin = _as_numpy(phase.lockin)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    x = np.arange(graph.num_response_tokens)
    for state, name in enumerate(automaton.state_names):
        axes[0].plot(x, route[..., state].mean(axis=1), label=name, linewidth=1.2)
    axes[0].axvline(token_index, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("five-state lineage (display mean over channels)")
    axes[0].set_xlabel("response token")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].plot(x, score, label="final score", color="#7c3aed", linewidth=1.8)
    axes[1].plot(x, rupture.mean(axis=1), label="rupture", color="#dc2626")
    axes[1].plot(x, lockin.mean(axis=1), label="lock-in", color="#ea580c")
    axes[1].plot(
        x,
        prompt_lineage.mean(axis=1),
        label="prompt lineage",
        color="#2563eb",
    )
    axes[1].plot(x, detached.mean(axis=1), label="R_PLUS", color="#059669")
    axes[1].axvline(token_index, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("rupture × lock-in trajectory")
    axes[1].set_xlabel("response token")
    axes[1].legend(fontsize=7)

    _draw_radial_graph(
        axes[2],
        graph,
        token_index=token_index,
        channel=channel,
        maximum_edges=int(maximum_edges),
    )
    figure.suptitle(f"sample {graph.sample_id} · selected token {token_index}", fontsize=11)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


__all__ = ["render_sample_diagnostics"]
