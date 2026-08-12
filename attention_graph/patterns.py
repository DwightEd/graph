"""Training-free discovery of prompt-provenance patterns.

The representation deliberately contains one mechanism rather than a bag of
handcrafted graph statistics.  Starting from every response-token node, it
walks backwards through the layer-ordered attention graph and can record either:

* cumulative probability mass absorbed by prompt nodes; and
* concentration of the response ancestry that is still alive.

The two curves are separate experiment views and are never concatenated.

Unobserved/censored mass is retained as a separate control curve and never
concatenated into the representation used for clustering or t-SNE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .graph import AttentionGraph, GraphBuildConfig, RP, build_attention_graph


@dataclass(frozen=True)
class PatternDiscoveryConfig:
    signature_view: str = "prompt_absorption"
    checkpoints: int = 8
    min_patterns: int = 2
    max_patterns: int = 6
    fit_reference_size: int = 30_000
    tsne_landmarks: int = 10_000
    perplexity: float = 40.0
    seed: int = 42
    prototype_hops: int = 3
    prototype_max_incoming: int = 6

    def validate(self):
        if self.signature_view not in {
            "prompt_absorption",
            "response_concentration",
        }:
            raise ValueError("unknown provenance signature view")
        if self.checkpoints < 2:
            raise ValueError("checkpoints must be at least two")
        if not 2 <= self.min_patterns <= self.max_patterns:
            raise ValueError("pattern range must satisfy 2 <= min <= max")
        if min(self.fit_reference_size, self.tsne_landmarks) < 10:
            raise ValueError("reference sizes must be at least ten")
        if self.perplexity <= 1:
            raise ValueError("perplexity must exceed one")
        if min(self.prototype_hops, self.prototype_max_incoming) < 1:
            raise ValueError("prototype limits must be positive")


def _layer_transition(graph: AttentionGraph, layer: int):
    """Return prompt absorption, response transition, and censored mass.

    Rows are response targets at layer ``layer``.  Columns of ``response`` are
    response sources at the previous layer.  Attention is averaged over heads
    but exact source endpoints and layer order are preserved.
    """

    response_count = graph.num_nodes - graph.response_idx
    device = graph.node_attr.device
    first_channel = layer * graph.num_heads
    last_channel = first_channel + graph.num_heads
    trace_mask = (
        (graph.trace_channel >= first_channel)
        & (graph.trace_channel < last_channel)
    )
    edge_mass = torch.zeros(
        graph.num_edges, dtype=torch.float32, device=device
    )
    if bool(trace_mask.any()):
        edge_mass.index_add_(
            0,
            graph.trace_edge_id[trace_mask],
            graph.trace_value[trace_mask].float(),
        )
        edge_mass /= float(graph.num_heads)

    prompt = torch.zeros(response_count, dtype=torch.float32, device=device)
    response_rows = []
    response_columns = []
    response_values = []
    if graph.num_edges:
        source, target = graph.edge_index
        row = target - graph.response_idx
        rp = (graph.edge_type == RP) & (edge_mass > 0)
        rr = (graph.edge_type != RP) & (edge_mass > 0)
        if bool(rp.any()):
            prompt.index_add_(0, row[rp], edge_mass[rp])
        if bool(rr.any()):
            response_rows.append(row[rr])
            response_columns.append(source[rr] - graph.response_idx)
            response_values.append(edge_mass[rr])

    channels = slice(first_channel, last_channel)
    diagonal = graph.node_attr[graph.response_idx :, channels].float().mean(dim=1)
    index = torch.arange(response_count, device=device)
    response_rows.append(index)
    response_columns.append(index)
    response_values.append(diagonal)
    rows = torch.cat(response_rows)
    columns = torch.cat(response_columns)
    values = torch.cat(response_values)
    response_mass = torch.zeros_like(prompt)
    response_mass.index_add_(0, rows, values)

    observed = prompt + response_mass
    # Floating-point accumulation or non-standard caches can exceed one by a
    # tiny amount.  Rescale only those rows instead of changing every row.
    raw_normalizer = observed.clamp_min(1.0)
    prompt = prompt / raw_normalizer
    values = values / raw_normalizer[rows]
    raw_response = torch.sparse_coo_tensor(
        torch.stack((rows, columns)),
        values,
        (response_count, response_count),
        device=device,
    ).coalesce()
    normalized_response_mass = torch.zeros_like(prompt)
    normalized_response_mass.index_add_(
        0, raw_response.indices()[0], raw_response.values()
    )
    unresolved = (1.0 - prompt - normalized_response_mass).clamp(0.0, 1.0)

    # The primary representation is conditional on the observed graph so that
    # cache censoring is not silently used as a clustering feature.  Absolute
    # missing mass is propagated separately through ``raw_response``.
    conditional_mass = (prompt + normalized_response_mass).clamp_min(1e-12)
    conditional_prompt = torch.where(
        conditional_mass > 1e-11,
        prompt / conditional_mass,
        torch.zeros_like(prompt),
    )
    conditional_values = raw_response.values() / conditional_mass[
        raw_response.indices()[0]
    ]
    conditional_response = torch.sparse_coo_tensor(
        raw_response.indices(),
        conditional_values,
        raw_response.shape,
        device=device,
    ).coalesce()
    return conditional_prompt, conditional_response, raw_response, unresolved


def _checkpoint_indices(layers: int, checkpoints: int, device):
    # Indices refer to the curves after one or more backwards layer steps.
    values = torch.linspace(1, layers, checkpoints, device=device)
    return values.round().long().clamp(1, layers) - 1


def provenance_curves(
    graph: AttentionGraph,
    *,
    checkpoints: int = 8,
    signature_view: str = "prompt_absorption",
):
    """Return one compact topology signature per response-token node.

    Exactly one curve is returned as ``signature``.  The two allowed views are
    intentionally separate experiments; they are never concatenated.  The
    ``unresolved`` curve is a cache-censoring control and is not part of either
    signature.
    """

    if signature_view not in {"prompt_absorption", "response_concentration"}:
        raise ValueError("unknown provenance signature view")
    if graph.num_layers < 1 or checkpoints < 2:
        raise ValueError("a positive layer count and at least two checkpoints are required")
    response_count = graph.num_nodes - graph.response_idx
    device = graph.node_attr.device
    live = torch.eye(response_count, dtype=torch.float32, device=device)
    control_live = live.clone()
    prompt_total = torch.zeros(response_count, dtype=torch.float32, device=device)
    unresolved_total = torch.zeros_like(prompt_total)
    prompt_curve, unresolved_curve, concentration_curve = [], [], []

    for layer in range(graph.num_layers - 1, -1, -1):
        prompt, transition, raw_transition, unresolved = _layer_transition(
            graph, layer
        )
        prompt_total = prompt_total + live @ prompt
        unresolved_total = unresolved_total + control_live @ unresolved
        # dense@sparse is evaluated as (sparse.T@dense.T).T.  This avoids an
        # O(R^3) dense multiplication while retaining the ancestry of every
        # response-token root simultaneously.
        live = torch.sparse.mm(transition.transpose(0, 1), live.transpose(0, 1)).transpose(0, 1)
        control_live = torch.sparse.mm(
            raw_transition.transpose(0, 1), control_live.transpose(0, 1)
        ).transpose(0, 1)
        live_mass = live.sum(dim=1)
        conditional = live / live_mass.clamp_min(1e-12).unsqueeze(1)
        concentration = conditional.square().sum(dim=1)
        concentration = torch.where(
            live_mass > 1e-12, concentration, torch.zeros_like(concentration)
        )
        prompt_curve.append(prompt_total.clamp(0.0, 1.0))
        unresolved_curve.append(unresolved_total.clamp(0.0, 1.0))
        concentration_curve.append(concentration.clamp(0.0, 1.0))

    prompt_curve = torch.stack(prompt_curve, dim=1)
    unresolved_curve = torch.stack(unresolved_curve, dim=1)
    concentration_curve = torch.stack(concentration_curve, dim=1)
    selected = _checkpoint_indices(graph.num_layers, checkpoints, device)
    prompt_selected = prompt_curve[:, selected]
    concentration_selected = concentration_curve[:, selected]
    unresolved_selected = unresolved_curve[:, selected]
    signature = (
        prompt_selected
        if signature_view == "prompt_absorption"
        else concentration_selected
    )
    return signature, unresolved_selected


def _extract_split(dataset, graph_config, discovery_config, *, split_name):
    signatures, controls = [], []
    metadata = {
        "sample_id": [],
        "source_id": [],
        "token_index": [],
        "relative_position": [],
        "task_type": [],
        "data_source": [],
        "generator_model": [],
    }
    token_total = 0
    progress = tqdm(
        dataset.sample_ids,
        total=len(dataset),
        desc=f"[2/4] {split_name} provenance graphs",
        unit="graph",
        dynamic_ncols=True,
    )
    for sample_id in progress:
        sample = dataset[sample_id]
        graph = build_attention_graph(sample.attention(), graph_config)
        signature, unresolved = provenance_curves(
            graph,
            checkpoints=discovery_config.checkpoints,
            signature_view=discovery_config.signature_view,
        )
        signatures.append(signature.detach().cpu().numpy())
        controls.append(unresolved.detach().cpu().numpy())
        count = len(signature)
        token_total += count
        metadata["sample_id"].extend([sample.sample_id] * count)
        metadata["source_id"].extend([sample.source_id] * count)
        metadata["token_index"].extend(range(count))
        metadata["relative_position"].extend(
            np.arange(count, dtype=np.float32) / max(count - 1, 1)
        )
        metadata["task_type"].extend([str(sample.task_type)] * count)
        metadata["data_source"].extend([str(sample.data_source)] * count)
        metadata["generator_model"].extend([str(sample.generator_model)] * count)
        sample.release_attention()
        progress.set_postfix(tokens=token_total, refresh=False)
    if not signatures:
        raise ValueError(f"{split_name} split contains no samples")
    return (
        np.concatenate(signatures).astype(np.float32),
        np.concatenate(controls).astype(np.float32),
        {name: np.asarray(values) for name, values in metadata.items()},
    )


def _robust_fit(values):
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard = values.std(axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(standard > 1e-8, standard, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def _reference_indices(count, maximum, seed):
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def _fit_patterns(train, config):
    reference = train[
        _reference_indices(len(train), config.fit_reference_size, config.seed)
    ]
    upper = min(config.max_patterns, len(reference) - 1)
    if upper < config.min_patterns:
        raise ValueError("not enough training nodes for pattern discovery")
    candidates, bic = {}, {}
    print(
        f"[3/4] BIC pattern selection on {len(reference)} train nodes",
        flush=True,
    )
    for components in tqdm(
        range(config.min_patterns, upper + 1),
        desc="[3/4] diagonal GMM candidates",
        unit="model",
        dynamic_ncols=True,
    ):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=3,
            random_state=config.seed,
        ).fit(reference)
        candidates[components] = model
        bic[components] = float(model.bic(reference))
    selected = min(bic, key=bic.get)
    return candidates[selected], bic


def _remap_patterns(labels, raw_centroids):
    # Stable label-blind ordering by the final value of the selected curve.
    order = np.argsort(raw_centroids[:, -1], kind="stable")
    old_to_new = np.empty(len(order), dtype=np.int64)
    old_to_new[order] = np.arange(len(order))
    return old_to_new[labels], raw_centroids[order], order


def _landmark_tsne(values, config):
    """Place every node with label-blind landmark t-SNE interpolation."""

    from sklearn.manifold import TSNE, trustworthiness

    landmark_index = _reference_indices(
        len(values), config.tsne_landmarks, config.seed
    )
    landmarks = values[landmark_index]
    print(
        f"[4/4] t-SNE: {len(landmarks)} landmarks for {len(values)} test nodes",
        flush=True,
    )
    perplexity = min(config.perplexity, max(2.0, (len(landmarks) - 1) / 3.0))
    landmark_coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=config.seed,
        verbose=2,
    ).fit_transform(landmarks)
    coordinates = np.empty((len(values), 2), dtype=np.float32)
    coordinates[landmark_index] = landmark_coordinates

    remaining_mask = np.ones(len(values), dtype=bool)
    remaining_mask[landmark_index] = False
    remaining = np.flatnonzero(remaining_mask)
    if len(remaining):
        neighbors = min(8, len(landmarks))
        index = NearestNeighbors(n_neighbors=neighbors).fit(landmarks)
        # Chunking avoids allocating a large all-token distance matrix.
        for start in tqdm(
            range(0, len(remaining), 20_000),
            desc="[4/4] interpolate remaining nodes",
            unit="chunk",
            dynamic_ncols=True,
        ):
            ids = remaining[start : start + 20_000]
            distance, nearest = index.kneighbors(values[ids])
            weights = 1.0 / np.maximum(distance, 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            coordinates[ids] = np.einsum(
                "ij,ijk->ik", weights, landmark_coordinates[nearest]
            )
    quality_neighbors = max(1, min(10, (len(landmarks) - 1) // 2))
    quality = float(
        trustworthiness(
            landmarks,
            landmark_coordinates,
            n_neighbors=quality_neighbors,
        )
    )
    return coordinates, {
        "method": "landmark_tsne_with_knn_interpolation",
        "landmarks": int(len(landmarks)),
        "all_nodes": int(len(values)),
        "perplexity": float(perplexity),
        "trustworthiness_on_landmarks": quality,
    }


def _read_token_labels(dataset):
    store = dataset.labels()
    values = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        values.append(store.response_labels(sample).cpu().numpy())
        sample.release_attention()
    return np.concatenate(values).astype(np.int64)


def _ranking(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _cluster_report(
    labels, patterns, pattern_count, *, relative_position, unresolved
):
    overall = float(labels.mean())
    output = []
    for pattern in range(pattern_count):
        selected = patterns == pattern
        prevalence = float(labels[selected].mean()) if selected.any() else None
        output.append({
            "pattern": int(pattern),
            "nodes": int(selected.sum()),
            "hallucination_nodes": int(labels[selected].sum()),
            "hallucination_prevalence": prevalence,
            "prevalence_enrichment": (
                prevalence / overall
                if prevalence is not None and overall
                else None
            ),
            "mean_relative_position": (
                float(relative_position[selected].mean()) if selected.any() else None
            ),
            "mean_final_unresolved_mass": (
                float(unresolved[selected, -1].mean()) if selected.any() else None
            ),
        })
    return output


def _cluster_report_by_group(
    labels, patterns, pattern_count, metadata, unresolved, field
):
    groups = np.asarray(metadata[field]).astype(str)
    output = {}
    for group in sorted(np.unique(groups).tolist()):
        selected = groups == group
        output[group] = _cluster_report(
            labels[selected],
            patterns[selected],
            pattern_count,
            relative_position=metadata["relative_position"][selected],
            unresolved=unresolved[selected],
        )
    return output


def _plot_patterns(
    output, coordinates, labels, patterns, unresolved, centroids, cluster_rows,
    checkpoints, signature_view,
):
    import matplotlib.pyplot as plt

    output = Path(output)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for pattern in sorted(np.unique(patterns).tolist()):
        selected = patterns == pattern
        axes[0].scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=3, alpha=0.18, color=palette(pattern % 10),
            label=f"Pattern {pattern} (n={int(selected.sum())})", rasterized=True,
        )
    axes[0].set_title("Unsupervised provenance patterns")
    axes[0].legend(frameon=False, markerscale=3, fontsize=8)

    normal = labels == 0
    anomaly = labels == 1
    axes[1].scatter(
        coordinates[normal, 0], coordinates[normal, 1], s=3, alpha=0.07,
        color="#2ca02c", label=f"Normal token (n={int(normal.sum())})",
        rasterized=True, zorder=1,
    )
    axes[1].scatter(
        coordinates[anomaly, 0], coordinates[anomaly, 1], s=13, alpha=0.70,
        color="#d62728", marker="x",
        label=f"Hallucination token (n={int(anomaly.sum())})",
        rasterized=True, zorder=3,
    )
    axes[1].set_title("Same coordinates; labels added post hoc")
    axes[1].legend(frameon=False, fontsize=8)

    final_unresolved = unresolved[:, -1]
    scatter = axes[2].scatter(
        coordinates[:, 0], coordinates[:, 1], c=final_unresolved,
        cmap="viridis", s=3, alpha=0.15, rasterized=True,
    )
    figure.colorbar(scatter, ax=axes[2], label="Final unresolved mass")
    axes[2].set_title("Censoring control (not clustered)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(output / "provenance_pattern_tsne.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    depth = np.linspace(1.0 / checkpoints, 1.0, checkpoints)
    curve_figure, curve_axes = plt.subplots(
        1, 2, figsize=(10, 4.2), constrained_layout=True
    )
    for pattern, centroid in enumerate(centroids):
        row = cluster_rows[pattern]
        prevalence = row["hallucination_prevalence"]
        display = (
            f"Pattern {pattern}: {100 * prevalence:.1f}% H"
            if prevalence is not None
            else f"Pattern {pattern}: no test nodes"
        )
        curve_axes[0].plot(
            depth, centroid, marker="o", ms=3,
            color=palette(pattern % 10), label=display,
        )
        selected = patterns == pattern
        if selected.any():
            curve_axes[1].plot(
                depth, unresolved[selected].mean(axis=0), marker="o", ms=3,
                color=palette(pattern % 10), label=display,
            )
    curve_axes[0].set_title(signature_view.replace("_", " ").title())
    curve_axes[1].set_title("Unresolved-mass control")
    for axis in curve_axes:
        axis.set_xlabel("Normalized backward layer depth")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.20)
    curve_axes[0].set_ylabel("Mass / concentration")
    curve_axes[0].legend(frameon=False, fontsize=8)
    curve_figure.savefig(
        output / "provenance_pattern_curves.png", dpi=240, bbox_inches="tight"
    )
    plt.close(curve_figure)


def _prototype_indices(values, patterns, centroids_scaled):
    selected = []
    for pattern in range(len(centroids_scaled)):
        candidates = np.flatnonzero(patterns == pattern)
        if not len(candidates):
            continue
        distance = np.square(
            values[candidates] - centroids_scaled[pattern]
        ).sum(axis=1)
        selected.append((pattern, int(candidates[int(np.argmin(distance))])))
    return selected


def _render_ego_graph(
    graph, root_token, path, *, pattern, hops, max_incoming,
):
    """Render a readable view; pruning here never changes the representation."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    root = graph.response_idx + int(root_token)
    source, target = graph.edge_index.detach().cpu().numpy()
    edge_type = graph.edge_type.detach().cpu().numpy()
    weight = graph.edge_score.detach().cpu().numpy()
    depth = {root: 0}
    edges = []
    frontier = [root]
    for current_depth in range(hops):
        next_frontier = []
        for node in frontier:
            incoming = np.flatnonzero(target == node)
            if len(incoming) > max_incoming:
                incoming = incoming[
                    np.argsort(weight[incoming])[-max_incoming:]
                ]
            for edge_id in incoming.tolist():
                ancestor = int(source[edge_id])
                edges.append((ancestor, node, int(edge_type[edge_id]), float(weight[edge_id])))
                if ancestor not in depth:
                    depth[ancestor] = current_depth + 1
                    if ancestor >= graph.response_idx:
                        next_frontier.append(ancestor)
        frontier = next_frontier
        if not frontier:
            break

    levels = {}
    for node, value in depth.items():
        levels.setdefault(value, []).append(node)
    position = {}
    for level, nodes in levels.items():
        nodes = sorted(nodes)
        offsets = np.linspace(-1.0, 1.0, len(nodes)) if len(nodes) > 1 else [0.0]
        for node, offset in zip(nodes, offsets):
            position[node] = (-float(level), float(offset))

    figure, axis = plt.subplots(figsize=(max(6.0, hops * 2.2), 4.8), constrained_layout=True)
    maximum = max((value for *_rest, value in edges), default=1.0)
    for ancestor, node, relation, value in edges:
        start, end = position[ancestor], position[node]
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=0.6 + 3.0 * value / max(maximum, 1e-8),
            color="#2ca02c" if relation == RP else "#756bb1", alpha=0.65,
            connectionstyle="arc3,rad=0.06",
        )
        axis.add_patch(arrow)
    for node, (x, y) in position.items():
        is_root = node == root
        is_prompt = node < graph.response_idx
        color = "#ff7f0e" if is_root else ("#8c8c8c" if is_prompt else "#3182bd")
        axis.scatter([x], [y], s=180 if is_root else 105, color=color, zorder=3)
        label = f"P{node}" if is_prompt else f"R{node - graph.response_idx}"
        if is_root:
            label += " (root)"
        axis.text(x, y + 0.10, label, ha="center", va="bottom", fontsize=8)
    axis.set_title(
        f"Pattern {pattern} prototype — visible top-{max_incoming} incoming edges"
    )
    axis.set_axis_off()
    axis.autoscale_view()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _response_graph_report(metadata, patterns, labels, pattern_count):
    """Describe whole-response graph modes after node patterns are frozen."""

    sample_ids = np.asarray(metadata["sample_id"]).astype(str)
    summaries = []
    occupancy = {0: [], 1: []}
    transitions = {
        0: np.zeros((pattern_count, pattern_count), dtype=np.int64),
        1: np.zeros((pattern_count, pattern_count), dtype=np.int64),
    }
    dominant_rows = []
    start = 0
    while start < len(sample_ids):
        end = start + 1
        while end < len(sample_ids) and sample_ids[end] == sample_ids[start]:
            end += 1
        sequence = patterns[start:end]
        response_label = int(bool(labels[start:end].any()))
        counts = np.bincount(sequence, minlength=pattern_count)
        proportions = counts / max(len(sequence), 1)
        dominant = int(np.argmax(counts))
        occupancy[response_label].append(proportions)
        if len(sequence) > 1:
            np.add.at(transitions[response_label], (sequence[:-1], sequence[1:]), 1)
        summaries.append({
            "sample_id": str(sample_ids[start]),
            "source_id": str(metadata["source_id"][start]),
            "response_label": response_label,
            "nodes": int(len(sequence)),
            "dominant_pattern": dominant,
            "pattern_proportions": proportions.tolist(),
            "pattern_sequence": sequence.astype(int).tolist(),
        })
        dominant_rows.append((dominant, response_label))
        start = end

    def mean_occupancy(label):
        values = occupancy[label]
        return (
            np.mean(values, axis=0)
            if values
            else np.zeros(pattern_count, dtype=np.float64)
        )

    def normalized_transition(label):
        matrix = transitions[label].astype(np.float64)
        rows = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows > 0)

    dominant_rows = np.asarray(dominant_rows, dtype=np.int64)
    graph_prevalence = float(dominant_rows[:, 1].mean())
    dominant_report = []
    for pattern in range(pattern_count):
        selected = dominant_rows[:, 0] == pattern
        prevalence = (
            float(dominant_rows[selected, 1].mean()) if selected.any() else None
        )
        dominant_report.append({
            "pattern": pattern,
            "response_graphs": int(selected.sum()),
            "hallucination_graphs": int(dominant_rows[selected, 1].sum()),
            "hallucination_prevalence": prevalence,
            "prevalence_enrichment": (
                prevalence / graph_prevalence
                if prevalence is not None and graph_prevalence
                else None
            ),
        })

    normal_occupancy = mean_occupancy(0)
    hallucination_occupancy = mean_occupancy(1)
    normal_transition = normalized_transition(0)
    hallucination_transition = normalized_transition(1)
    return summaries, {
        "response_graphs": int(len(summaries)),
        "hallucination_graphs": int(dominant_rows[:, 1].sum()),
        "hallucination_prevalence": graph_prevalence,
        "dominant_pattern_enrichment": dominant_report,
        "mean_pattern_occupancy_correct": normal_occupancy.tolist(),
        "mean_pattern_occupancy_hallucination": hallucination_occupancy.tolist(),
        "occupancy_delta_hallucination_minus_correct": (
            hallucination_occupancy - normal_occupancy
        ).tolist(),
        "transition_correct": normal_transition.tolist(),
        "transition_hallucination": hallucination_transition.tolist(),
        "transition_delta_hallucination_minus_correct": (
            hallucination_transition - normal_transition
        ).tolist(),
    }


def _plot_response_graph_report(output, graph_report):
    import matplotlib.pyplot as plt

    normal = np.asarray(graph_report["mean_pattern_occupancy_correct"])
    hallucination = np.asarray(
        graph_report["mean_pattern_occupancy_hallucination"]
    )
    correct_transition = np.asarray(graph_report["transition_correct"])
    hallucination_transition = np.asarray(
        graph_report["transition_hallucination"]
    )
    patterns = np.arange(len(normal))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    width = 0.38
    axes[0].bar(patterns - width / 2, normal, width, label="Correct graph")
    axes[0].bar(
        patterns + width / 2,
        hallucination,
        width,
        label="Hallucination graph",
    )
    axes[0].set(
        title="Mean node-pattern occupancy",
        xlabel="Node pattern",
        ylabel="Proportion of response nodes",
        xticks=patterns,
    )
    axes[0].legend(frameon=False)
    for axis, matrix, title in (
        (axes[1], correct_transition, "Correct graph transitions"),
        (axes[2], hallucination_transition, "Hallucination graph transitions"),
    ):
        image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
        axis.set(
            title=title,
            xlabel="Next node pattern",
            ylabel="Current node pattern",
            xticks=patterns,
            yticks=patterns,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(
        Path(output) / "response_graph_pattern_modes.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def discover_provenance_patterns(
    train_dataset,
    test_dataset,
    label_dataset,
    *,
    output_dir,
    graph_config: GraphBuildConfig | None = None,
    config: PatternDiscoveryConfig | None = None,
):
    """Discover and visualize node patterns without fitting to labels."""

    config = PatternDiscoveryConfig() if config is None else config
    config.validate()
    graph_config = GraphBuildConfig() if graph_config is None else graph_config
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {"pattern": asdict(config), "graph": asdict(graph_config)},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    train_signature, _train_control, _train_metadata = _extract_split(
        train_dataset, graph_config, config, split_name="train"
    )
    test_signature, unresolved, metadata = _extract_split(
        test_dataset, graph_config, config, split_name="test"
    )
    center, scale = _robust_fit(train_signature)
    train_scaled = (train_signature - center) / scale
    test_scaled = (test_signature - center) / scale
    model, bic = _fit_patterns(train_scaled, config)
    raw_pattern = model.predict(test_scaled)
    raw_centroids = model.means_ * scale + center
    patterns, centroids, order = _remap_patterns(raw_pattern, raw_centroids)
    anomaly_score = -model.score_samples(test_scaled)
    coordinates, projection = _landmark_tsne(test_scaled, config)

    # Everything up to this point is label-blind.  Persist it before opening
    # the evaluation sidecar so the experimental boundary is auditable.
    np.savez_compressed(
        output / "node_signatures_label_free.npz",
        schema=np.asarray("layer-provenance-pattern-v2"),
        signature=test_signature,
        unresolved_control=unresolved,
        pattern=patterns.astype(np.int16),
        anomaly_score=anomaly_score.astype(np.float32),
        coordinates=coordinates,
        center=center,
        scale=scale,
        centroids=centroids.astype(np.float32),
        sample_id=metadata["sample_id"].astype(str),
        source_id=metadata["source_id"].astype(str),
        token_index=metadata["token_index"].astype(np.int32),
        relative_position=metadata["relative_position"].astype(np.float32),
        task_type=metadata["task_type"].astype(str),
        data_source=metadata["data_source"].astype(str),
        generator_model=metadata["generator_model"].astype(str),
    )
    label_free_report = {
        "schema": "layer-provenance-pattern-v2",
        "labels_read": False,
        "signature_view": config.signature_view,
        "train_nodes": int(len(train_signature)),
        "test_nodes": int(len(test_signature)),
        "patterns": int(model.n_components),
        "bic_by_pattern_count": {str(key): value for key, value in bic.items()},
        "projection": projection,
        "centroids": [
            {
                "pattern": pattern,
                "signature_curve": centroid.tolist(),
            }
            for pattern, centroid in enumerate(centroids)
        ],
    }
    (output / "patterns_label_free.json").write_text(
        json.dumps(label_free_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Prototype selection is also label-blind.
    ordered_centroids_scaled = model.means_[order]
    prototype_ids = _prototype_indices(test_scaled, patterns, ordered_centroids_scaled)
    prototype_root = output / "prototype_graphs"
    prototype_root.mkdir(exist_ok=True)
    prototype_rows = []
    for pattern, global_index in prototype_ids:
        sample_id = str(metadata["sample_id"][global_index])
        token_index = int(metadata["token_index"][global_index])
        sample = test_dataset[sample_id]
        graph = build_attention_graph(sample.attention(), graph_config)
        figure_path = prototype_root / f"pattern_{pattern}.png"
        _render_ego_graph(
            graph,
            token_index,
            figure_path,
            pattern=pattern,
            hops=config.prototype_hops,
            max_incoming=config.prototype_max_incoming,
        )
        sample.release_attention()
        prototype_rows.append({
            "pattern": pattern,
            "sample_id": sample_id,
            "token_index": token_index,
            "figure": str(figure_path),
        })
    (output / "prototype_graphs.json").write_text(
        json.dumps(prototype_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    labels = _read_token_labels(label_dataset)
    if len(labels) != len(test_signature):
        raise ValueError("test labels and node signatures have different lengths")
    clusters = _cluster_report(
        labels,
        patterns,
        len(centroids),
        relative_position=metadata["relative_position"],
        unresolved=unresolved,
    )
    graph_summaries, graph_report = _response_graph_report(
        metadata, patterns, labels, len(centroids)
    )
    with (output / "response_graph_patterns.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in graph_summaries:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        **label_free_report,
        "labels_read": True,
        "labels_read_during": "post_hoc_pattern_interpretation_only",
        "hallucination_nodes": int(labels.sum()),
        "hallucination_prevalence": float(labels.mean()),
        "novelty_score": _ranking(labels, anomaly_score),
        "cluster_label_enrichment": clusters,
        "cluster_label_enrichment_by_task": _cluster_report_by_group(
            labels,
            patterns,
            len(centroids),
            metadata,
            unresolved,
            "task_type",
        ),
        "cluster_label_enrichment_by_data_source": _cluster_report_by_group(
            labels,
            patterns,
            len(centroids),
            metadata,
            unresolved,
            "data_source",
        ),
        "response_graph_pattern_modes": graph_report,
        "prototypes": prototype_rows,
    }
    (output / "pattern_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "node_pattern_visualization.npz",
        coordinates=coordinates,
        pattern=patterns.astype(np.int16),
        label=labels,
        unresolved_control=unresolved,
        sample_id=metadata["sample_id"].astype(str),
        token_index=metadata["token_index"].astype(np.int32),
        relative_position=metadata["relative_position"].astype(np.float32),
    )
    _plot_patterns(
        output,
        coordinates,
        labels,
        patterns,
        unresolved,
        centroids,
        clusters,
        config.checkpoints,
        config.signature_view,
    )
    _plot_response_graph_report(output, graph_report)
    return {
        "output_dir": str(output),
        "label_free_signatures": str(output / "node_signatures_label_free.npz"),
        "pattern_report": str(output / "pattern_report.json"),
        "tsne": str(output / "provenance_pattern_tsne.png"),
        "pattern_curves": str(output / "provenance_pattern_curves.png"),
        "prototype_graphs": str(prototype_root),
        "response_graph_modes": str(output / "response_graph_pattern_modes.png"),
        "test_nodes": int(len(test_signature)),
        "patterns": int(model.n_components),
    }
