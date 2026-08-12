"""Mechanism-readable visualization of one sparse causal token graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .graph import RP, build_attention_graph


def _edge_display_strength(edge_score, *, attention_floor, num_channels):
    baseline = float(attention_floor) / int(num_channels)
    values = np.asarray(edge_score, dtype=np.float32)
    strength = np.log1p(values / baseline) / np.log1p(1.0 / baseline)
    return np.clip(strength, 0.0, 1.0), baseline


def _sample_scores(score_path, sample_id, source_id, response_count):
    with np.load(score_path, allow_pickle=False) as data:
        selected = data["sample_id"].astype(str) == str(sample_id)
        token_index = data["token_index"][selected]
        scores = data["score"][selected]
        sources = data["source_id"][selected].astype(str)
        quantiles = np.quantile(data["score"], [0.01, 0.99]).astype(np.float32)
    expected = np.arange(response_count, dtype=token_index.dtype)
    valid = (
        len(token_index) == response_count
        and np.array_equal(np.sort(token_index), expected)
        and np.all(sources == str(source_id))
    )
    if not valid:
        raise ValueError("frozen score rows do not exactly match the selected canonical sample")
    return scores[np.argsort(token_index)].astype(np.float32), quantiles


def _visible_graph(graph, *, start, end, display_top_k):
    target_in_window = (
        (graph.edge_index[1] >= graph.response_idx + start)
        & (graph.edge_index[1] < graph.response_idx + end)
    )
    candidate = torch.nonzero(target_in_window, as_tuple=False).flatten()
    group = graph.edge_index[1, candidate] * 2 + graph.edge_type[candidate]
    kept = []
    for group_id in torch.unique(group, sorted=True):
        ids = torch.nonzero(group == group_id, as_tuple=False).flatten()
        ranked = ids[torch.argsort(graph.edge_score[candidate[ids]], descending=True, stable=True)]
        kept.append(candidate[ranked[:display_top_k]])
    edge_ids = torch.cat(kept) if kept else torch.empty(0, dtype=torch.long, device=group.device)
    window_nodes = torch.arange(graph.response_idx + start, graph.response_idx + end, device=group.device)
    nodes = torch.unique(torch.cat((graph.edge_index[0, edge_ids], window_nodes)), sorted=True)
    node_to_local = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=group.device)
    node_to_local[nodes] = torch.arange(len(nodes), device=group.device)
    edge_index = node_to_local[graph.edge_index[:, edge_ids]]
    edge_to_local = torch.full((graph.num_edges,), -1, dtype=torch.long, device=group.device)
    edge_to_local[edge_ids] = torch.arange(len(edge_ids), device=group.device)
    trace_edge_id = edge_to_local[graph.trace_edge_id]
    trace_keep = trace_edge_id >= 0
    return nodes, edge_ids, edge_index, trace_edge_id[trace_keep], trace_keep


def _save_graph_data(output, graph, nodes, edge_ids, edge_index, trace_edge_id, trace_keep,
                     scores, score_quantiles, labels, start, end, display_top_k):
    absolute = nodes.detach().cpu().numpy().astype(np.int32)
    relative = absolute - graph.response_idx
    response = relative >= 0
    in_window = response & (relative >= start) & (relative < end)
    role = np.where(~response, "prompt", np.where(in_window, "response", "history"))
    node_score = np.full(len(nodes), np.nan, dtype=np.float32)
    node_score[response] = scores[relative[response]]
    node_label = np.full(len(nodes), -1, dtype=np.int8)
    node_label[response] = labels[relative[response]]
    channels = graph.trace_channel[trace_keep]
    edge_score = graph.edge_score[edge_ids].detach().cpu().numpy().astype(np.float32)
    edge_strength, baseline = _edge_display_strength(
        edge_score, attention_floor=graph.attention_floor, num_channels=graph.num_channels
    )
    np.savez_compressed(
        output,
        schema=np.asarray("attention-graph-view-v1"),
        sample_id=np.asarray(graph.sample_id), source_id=np.asarray(graph.source_id),
        response_idx=np.asarray(graph.response_idx, dtype=np.int32),
        attention_floor=np.asarray(graph.attention_floor, dtype=np.float32),
        selection=np.asarray(graph.build_config.selection),
        mass_cover=np.asarray(graph.build_config.mass_cover, dtype=np.float32),
        graph_top_k=np.asarray(graph.build_config.top_k, dtype=np.int32),
        graph_threshold=np.asarray(
            np.nan if graph.build_config.threshold is None else graph.build_config.threshold,
            dtype=np.float32,
        ),
        display_top_k=np.asarray(display_top_k, dtype=np.int32),
        labels_read_during=np.asarray("rendering_only"),
        window=np.asarray([start, end], dtype=np.int32),
        score_vmin=np.asarray(score_quantiles[0], dtype=np.float32),
        score_vmax=np.asarray(score_quantiles[1], dtype=np.float32),
        node_absolute_index=absolute,
        node_token_id=graph.token_ids[nodes].detach().cpu().numpy().astype(np.int32),
        node_role=role, node_score=node_score, node_label=node_label,
        edge_source_node=edge_index[0].detach().cpu().numpy().astype(np.int32),
        edge_target_node=edge_index[1].detach().cpu().numpy().astype(np.int32),
        edge_source_absolute_index=absolute[edge_index[0].detach().cpu().numpy()],
        edge_target_absolute_index=absolute[edge_index[1].detach().cpu().numpy()],
        edge_type=np.where(graph.edge_type[edge_ids].detach().cpu().numpy() == RP, "RP", "RR"),
        edge_score=edge_score, edge_display_strength=edge_strength,
        edge_display_baseline=np.asarray(baseline, dtype=np.float32),
        trace_edge_id=trace_edge_id.detach().cpu().numpy().astype(np.int32),
        trace_layer=(channels // graph.num_heads).detach().cpu().numpy().astype(np.int16),
        trace_head=(channels % graph.num_heads).detach().cpu().numpy().astype(np.int16),
        trace_value=graph.trace_value[trace_keep].detach().cpu().numpy().astype(np.float32),
    )


def _draw_graph(output, graph, nodes, edge_ids, edge_index, scores, score_quantiles,
                labels, start, end):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    absolute = nodes.detach().cpu().numpy()
    relative = absolute - graph.response_idx
    response = relative >= 0
    in_window = response & (relative >= start) & (relative < end)
    prompt, history = ~response, response & ~in_window
    x = relative.astype(np.float32)
    for mask in (prompt, history):
        indices = np.flatnonzero(mask)
        if len(indices) == 1:
            x[indices] = (start + end - 1) / 2
        elif len(indices) > 1:
            x[indices] = np.linspace(start, end - 1, len(indices))
    y = np.where(prompt, 1.0, np.where(history, -1.0, 0.0))
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    edge_strength, _ = _edge_display_strength(
        graph.edge_score[edge_ids].detach().cpu().numpy(),
        attention_floor=graph.attention_floor, num_channels=graph.num_channels,
    )
    for full in nodes.detach().cpu().tolist():
        if full >= graph.response_idx and labels[full - graph.response_idx]:
            axis.axvspan(full - graph.response_idx - .45, full - graph.response_idx + .45,
                         color="tab:red", alpha=.08, zorder=0)
    for local, full_edge in enumerate(edge_ids.detach().cpu().tolist()):
        source, target = edge_index[:, local].detach().cpu().tolist()
        relation = int(graph.edge_type[full_edge])
        score = float(edge_strength[local])
        axis.add_patch(FancyArrowPatch(
            (x[source], y[source]), (x[target], y[target]), arrowstyle="->",
            mutation_scale=7, linewidth=.4 + 3.6 * score,
            color="tab:blue" if relation == RP else "tab:orange", alpha=.1 + .8 * score,
            connectionstyle=f"arc3,rad={.15 if relation == RP else -.22}", zorder=1,
        ))
    axis.scatter(x[prompt], y[prompt], s=28, c=".35", label="prompt source", zorder=3)
    axis.scatter(x[history], y[history], s=28, c=".5", label="response history", zorder=3)
    plot = axis.scatter(
        x[in_window], y[in_window], s=42, c=scores[relative[in_window]], cmap="viridis",
        vmin=float(score_quantiles[0]), vmax=float(score_quantiles[1]),
        edgecolors=np.where(labels[relative[in_window]] == 1, "tab:red", "white"),
        linewidths=np.where(labels[relative[in_window]] == 1, 1.3, .5),
        label="response token", zorder=3,
    )
    figure.colorbar(plot, ax=axis, label="frozen anomaly score")
    axis.plot([], [], color="tab:blue", label="RP: prompt -> response")
    axis.plot([], [], color="tab:orange", label="RR: response -> response")
    axis.set(
        title=f"Causal attention graph: {graph.sample_id}, response tokens [{start}, {end})",
        xlabel="response-relative position; prompt/history rows are source-order normalized",
        xlim=(start - 1, end), ylim=(-1.35, 1.55),
    )
    axis.set_yticks((-1, 0, 1), ("response history", "response timeline", "prompt sources"))
    axis.legend(loc="upper right")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def visualize_graph(dataset, *, score_path, sample_id, output_dir, graph_config,
                    center_token=None, window=48, display_top_k=4):
    """Build, export, and render one label-free selected attention graph."""
    sample = dataset[sample_id]
    attention = sample.attention()
    scores, score_quantiles = _sample_scores(
        score_path, sample.sample_id, sample.source_id, attention.num_response_tokens
    )
    center = int(np.argmax(scores)) if center_token is None else int(center_token)
    if not 0 <= center < attention.num_response_tokens:
        raise ValueError("center_token must be a valid response-relative index")
    if int(window) < 1 or int(display_top_k) < 1:
        raise ValueError("window and display_top_k must be positive")
    start = max(0, center - int(window))
    end = min(attention.num_response_tokens, center + int(window) + 1)
    graph = build_attention_graph(attention, graph_config)
    nodes, edge_ids, edge_index, trace_edge_id, trace_keep = _visible_graph(
        graph, start=start, end=end, display_top_k=int(display_top_k)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = dataset.labels().response_labels(sample).detach().cpu().numpy().astype(np.int8)
    data_path, figure_path = output / "graph_view.npz", output / "graph_view.png"
    _save_graph_data(
        data_path, graph, nodes, edge_ids, edge_index, trace_edge_id, trace_keep,
        scores, score_quantiles, labels, start, end, display_top_k,
    )
    _draw_graph(
        figure_path, graph, nodes, edge_ids, edge_index, scores, score_quantiles,
        labels, start, end,
    )
    sample.release_attention()
    return {
        "figure": str(figure_path), "data": str(data_path), "sample_id": graph.sample_id,
        "center_token": center, "window": [start, end], "visible_nodes": len(nodes),
        "visible_edges": len(edge_ids), "visible_traces": int(trace_keep.sum()),
        "display_top_k": int(display_top_k),
    }
