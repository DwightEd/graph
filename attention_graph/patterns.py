"""Unsupervised Lookback-ratio graph patterns and token visualizations.

The only node representation is the Lookback Lens mechanism: mean attention
per prompt token versus mean attention per generated-side token.  Labels are
not available while the representation, position calibration, K-Means model,
or t-SNE coordinates are fitted.  They are opened afterwards for evaluation,
coloring, and optional illustrative-example selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, davies_bouldin_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .graph import GraphBuildConfig, RP, build_attention_graph


REPRESENTATION = "lookback_ratio_layer_trajectory"


@dataclass(frozen=True)
class PatternDiscoveryConfig:
    layer_bins: int = 8
    min_patterns: int = 2
    max_patterns: int = 6
    fit_reference_size: int = 30_000
    tsne_landmarks: int = 10_000
    perplexity: float = 40.0
    position_bins: int = 10
    csr_row_block: int = 4096
    sample_ids: tuple[str, ...] = ()
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    seed: int = 42

    def validate(self):
        if self.layer_bins < 2:
            raise ValueError("layer_bins must be at least two")
        if not 2 <= self.min_patterns <= self.max_patterns:
            raise ValueError("pattern range must satisfy 2 <= min <= max")
        if min(self.fit_reference_size, self.tsne_landmarks) < 10:
            raise ValueError("reference sizes must be at least ten")
        if self.perplexity <= 1.0 or self.position_bins < 2:
            raise ValueError("perplexity and position_bins are too small")
        if self.csr_row_block < 1:
            raise ValueError("csr_row_block must be positive")
        if not 0.0 < self.display_mass_cover <= 1.0:
            raise ValueError("display_mass_cover must be in (0,1]")
        if self.display_edges_per_type < 1:
            raise ValueError("display_edges_per_type must be positive")


def _layer_bin_mean(values: torch.Tensor, bins: int):
    layers = values.shape[1]
    boundaries = torch.linspace(0, layers, min(bins, layers) + 1, device=values.device)
    boundaries = boundaries.round().long()
    return torch.stack(
        [values[:, boundaries[i] : boundaries[i + 1]].mean(dim=1) for i in range(len(boundaries) - 1)],
        dim=1,
    )


def _layer_bin_weights(num_layers: int, bins: int):
    boundaries = np.rint(
        np.linspace(0, num_layers, min(bins, num_layers) + 1)
    ).astype(int)
    return np.diff(boundaries).astype(np.float64) / float(num_layers)


def _lookback_from_masses(prompt_mass, history_mass, diagonal, response_idx, response_count):
    """Apply the exact length-normalized Lookback Lens ratio per channel row."""
    row = torch.arange(len(prompt_mass), device=prompt_mass.device).remainder(response_count)
    context_mean = prompt_mass / float(response_idx)
    generated_mean = (history_mass + diagonal) / (row + 1).float()
    denominator = context_mean + generated_mean
    return torch.where(denominator > 0, context_mean / denominator, torch.zeros_like(denominator))


def lookback_trajectories(attention, *, layer_bins: int = 8, csr_row_block: int = 4096):
    """Compute ``[response token, layer bin]`` Lookback trajectories from CSR.

    Rows are processed in bounded blocks, so peak temporary memory is
    ``O(block nnz + layers*heads*response)`` instead of ``O(sample nnz)``.
    Attention below the cache floor remains zero, while the saved diagonal is
    included on the generated side exactly as in autoregressive attention.
    """
    response_count = attention.num_response_tokens
    if response_count < 1 or attention.response_idx < 1:
        raise ValueError("lookback ratio requires non-empty prompt and response")
    device = attention.response_values.device
    rows_count = attention.num_channels * response_count
    row_ptr = attention.response_row_ptr.long()
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    history_mass = torch.zeros_like(prompt_mass)

    for row_start in range(0, rows_count, csr_row_block):
        row_end = min(row_start + csr_row_block, rows_count)
        starts = row_ptr[row_start:row_end]
        lengths = row_ptr[row_start + 1 : row_end + 1] - starts
        entry_count = int(lengths.sum())
        if not entry_count:
            continue
        repeated_starts = torch.repeat_interleave(starts, lengths)
        prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        positions = repeated_starts + torch.arange(entry_count, device=device) - prefix
        local_row = torch.repeat_interleave(
            torch.arange(row_end - row_start, device=device), lengths
        )
        source = attention.response_column_indices[positions].long()
        value = attention.response_values[positions].float().clamp_min(0.0)
        local_prompt = torch.zeros(row_end - row_start, dtype=torch.float32, device=device)
        local_history = torch.zeros_like(local_prompt)
        is_prompt = source < attention.response_idx
        if bool(is_prompt.any()):
            local_prompt.index_add_(0, local_row[is_prompt], value[is_prompt])
        if bool((~is_prompt).any()):
            local_history.index_add_(0, local_row[~is_prompt], value[~is_prompt])
        prompt_mass[row_start:row_end] = local_prompt
        history_mass[row_start:row_end] = local_history

    diagonal = attention.attention_diagonal.float()[:, :, attention.response_idx :].reshape(-1)
    ratio = _lookback_from_masses(
        prompt_mass, history_mass, diagonal, attention.response_idx, response_count
    )
    ratio = ratio.reshape(attention.num_layers, attention.num_heads, response_count)
    # Ratio is nonlinear: compute it per head first, then average heads.
    layer_trajectory = ratio.permute(2, 0, 1).mean(dim=2)
    binned = _layer_bin_mean(layer_trajectory, layer_bins)

    unresolved = (1.0 - prompt_mass - history_mass - diagonal).clamp(0.0, 1.0)
    unresolved = unresolved.reshape(attention.num_layers, attention.num_heads, response_count)
    unresolved = unresolved.permute(2, 0, 1).mean(dim=2)
    return torch.nan_to_num(binned), torch.nan_to_num(_layer_bin_mean(unresolved, layer_bins))


def graph_lookback_trajectories(graph, *, layer_bins: int = 8, trace_block: int = 1_000_000):
    """The same representation for graph-construction counterfactuals."""
    response_count = graph.num_nodes - graph.response_idx
    device = graph.node_attr.device
    rows_count = graph.num_channels * response_count
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    history_mass = torch.zeros_like(prompt_mass)
    for start in range(0, len(graph.trace_value), trace_block):
        end = min(start + trace_block, len(graph.trace_value))
        edge = graph.trace_edge_id[start:end].long()
        source = graph.edge_index[0, edge]
        target_row = graph.edge_index[1, edge] - graph.response_idx
        row = graph.trace_channel[start:end].long() * response_count + target_row
        value = graph.trace_value[start:end].float().clamp_min(0.0)
        is_prompt = source < graph.response_idx
        if bool(is_prompt.any()):
            prompt_mass.index_add_(0, row[is_prompt], value[is_prompt])
        if bool((~is_prompt).any()):
            history_mass.index_add_(0, row[~is_prompt], value[~is_prompt])
    diagonal = graph.node_attr[graph.response_idx :].reshape(
        response_count, graph.num_layers, graph.num_heads
    ).permute(1, 2, 0).reshape(-1).float()
    ratio = _lookback_from_masses(
        prompt_mass, history_mass, diagonal, graph.response_idx, response_count
    ).reshape(graph.num_layers, graph.num_heads, response_count)
    layer_trajectory = ratio.permute(2, 0, 1).mean(dim=2)
    unresolved = (1.0 - prompt_mass - history_mass - diagonal).clamp(0.0, 1.0)
    unresolved = unresolved.reshape(graph.num_layers, graph.num_heads, response_count)
    unresolved = unresolved.permute(2, 0, 1).mean(dim=2)
    return (
        torch.nan_to_num(_layer_bin_mean(layer_trajectory, layer_bins)),
        torch.nan_to_num(_layer_bin_mean(unresolved, layer_bins)),
    )


def _extract_split(dataset, config, *, split_name):
    trajectories, controls = [], []
    metadata = {
        "sample_id": [], "source_id": [], "token_index": [], "response_count": [],
        "relative_position": [], "task_type": [], "data_source": [], "generator_model": [],
    }
    token_total = 0
    progress = tqdm(dataset.sample_ids, desc=f"[1/5] {split_name} lookback", unit="sample", dynamic_ncols=True)
    for sample_id in progress:
        sample = dataset[sample_id]
        attention = sample.attention()
        ratio, unresolved = lookback_trajectories(
            attention, layer_bins=config.layer_bins, csr_row_block=config.csr_row_block
        )
        trajectories.append(ratio.detach().cpu().numpy())
        controls.append(unresolved.detach().cpu().numpy())
        count = attention.num_response_tokens
        token_total += count
        metadata["sample_id"].extend([str(sample.sample_id)] * count)
        metadata["source_id"].extend([str(sample.source_id)] * count)
        metadata["token_index"].extend(range(count))
        metadata["response_count"].extend([count] * count)
        metadata["relative_position"].extend(np.arange(count) / max(count - 1, 1))
        for field in ("task_type", "data_source", "generator_model"):
            metadata[field].extend([getattr(sample, field, None)] * count)
        sample.release_attention()
        progress.set_postfix(tokens=token_total, refresh=False)
    if not trajectories:
        raise ValueError(f"{split_name} split is empty")
    arrays = {
        key: np.asarray(value, dtype=np.int32 if key in {"token_index", "response_count"} else object)
        for key, value in metadata.items()
    }
    arrays["relative_position"] = np.asarray(metadata["relative_position"], dtype=np.float64)
    return np.concatenate(trajectories).astype(np.float64), np.concatenate(controls).astype(np.float64), arrays


def _sample_rows(count, maximum, rng):
    return np.arange(count) if count <= maximum else np.sort(rng.choice(count, maximum, replace=False))


def _fit_position_calibration(train, position, bins):
    position_bin = np.minimum((position * bins).astype(int), bins - 1)
    global_center = np.median(train, axis=0)
    global_scale = 1.4826 * np.median(np.abs(train - global_center), axis=0)
    global_std = train.std(axis=0)
    global_scale = np.where(global_scale >= 1e-8, global_scale, np.where(global_std >= 1e-8, global_std, 1.0))
    centers = np.empty((bins, train.shape[1]), dtype=np.float64)
    scales = np.empty_like(centers)
    report = []
    for index in range(bins):
        reference = train[position_bin == index]
        if len(reference) < 20:
            center, scale = global_center, global_scale
        else:
            center = np.median(reference, axis=0)
            scale = 1.4826 * np.median(np.abs(reference - center), axis=0)
            local_std = reference.std(axis=0)
            scale = np.where(scale >= 1e-8, scale, np.where(local_std >= 1e-8, local_std, global_scale))
        centers[index], scales[index] = center, scale
        report.append({
            "position_bin": index, "reference_nodes": int(len(reference)),
            "median": center.tolist(), "robust_scale": scale.tolist(),
        })
    return centers, scales, report


def _apply_position_calibration(values, position, centers, scales):
    bins = len(centers)
    index = np.minimum((position * bins).astype(int), bins - 1)
    return (values - centers[index]) / scales[index]


def _fit_patterns(reference, config):
    rng = np.random.default_rng(config.seed)
    values = reference[_sample_rows(len(reference), config.fit_reference_size, rng)]
    unique = len(np.unique(np.round(values, 8), axis=0))
    largest = min(config.max_patterns, unique, len(values) - 1)
    if largest < config.min_patterns:
        raise ValueError("lookback representation has too few distinct rows")
    models, diagnostics = {}, {}
    for count in range(config.min_patterns, largest + 1):
        model = KMeans(n_clusters=count, n_init=20, random_state=config.seed).fit(values)
        if len(np.unique(model.labels_)) != count or np.min(np.bincount(model.labels_)) < 2:
            continue
        score = float(davies_bouldin_score(values, model.labels_))
        if np.isfinite(score):
            models[count], diagnostics[str(count)] = model, score
            print(f"  KMeans K={count}: Davies-Bouldin={score:.4f}", flush=True)
    if not models:
        raise ValueError("all candidate KMeans solutions collapsed")
    selected = min(models, key=lambda count: diagnostics[str(count)])
    return models[selected], diagnostics


def _landmark_tsne(values, config):
    if len(values) < 3:
        raise ValueError("at least three response nodes are needed for t-SNE")
    rng = np.random.default_rng(config.seed)
    landmark_ids = _sample_rows(len(values), config.tsne_landmarks, rng)
    scaler = StandardScaler().fit(values[landmark_ids])
    landmarks = scaler.transform(values[landmark_ids])
    perplexity = min(config.perplexity, max(2.0, len(landmarks) - 1.0))
    coordinates = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
        max_iter=1500, random_state=config.seed, verbose=2,
    ).fit_transform(landmarks)
    output = np.empty((len(values), 2), dtype=np.float32)
    output[landmark_ids] = coordinates
    if len(landmark_ids) < len(values):
        mask = np.ones(len(values), dtype=bool)
        mask[landmark_ids] = False
        count = min(8, len(landmarks))
        distance, neighbors = NearestNeighbors(n_neighbors=count).fit(landmarks).kneighbors(
            scaler.transform(values[mask])
        )
        weight = 1.0 / np.maximum(distance, 1e-6)
        weight /= weight.sum(axis=1, keepdims=True)
        output[mask] = np.sum(coordinates[neighbors] * weight[:, :, None], axis=1)
    return output, {
        "all_nodes": int(len(values)), "landmarks": int(len(landmark_ids)),
        "non_landmarks": int(len(values) - len(landmark_ids)), "perplexity": float(perplexity),
    }


def _read_token_labels(dataset, metadata):
    # Formal caches embed labels in the sample file. This separate evaluation
    # dataset is first traversed only now, after all label-free objects froze.
    for sample_id in tqdm(
        dataset.sample_ids, desc="[4/5] loading evaluation labels",
        unit="sample", dynamic_ncols=True,
    ):
        sample = dataset[sample_id]
        sample.attention()
        sample.release_attention()
    store = dataset.labels()
    by_sample = {}
    for sample_id in tqdm(
        dataset.sample_ids, desc="[4/5] aligning token labels",
        unit="sample", dynamic_ncols=True,
    ):
        sample = dataset[sample_id]
        by_sample[str(sample_id)] = store.response_labels(sample).detach().cpu().numpy().astype(np.int8)
        sample.release_attention()
    output = np.empty(len(metadata["sample_id"]), dtype=np.int8)
    for row, (sample_id, token) in enumerate(zip(metadata["sample_id"], metadata["token_index"])):
        output[row] = by_sample[str(sample_id)][int(token)]
    return output


def _ranking(labels, score):
    labels = np.asarray(labels, dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _safe_correlation(left, right):
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _pattern_report(labels, patterns, score, count):
    prevalence = float(labels.mean())
    rows = []
    for pattern in range(count):
        selected = patterns == pattern
        rate = float(labels[selected].mean()) if selected.any() else None
        rows.append({
            "pattern": pattern, "nodes": int(selected.sum()),
            "hallucination_nodes": int(labels[selected].sum()), "hallucination_rate": rate,
            "enrichment": rate / prevalence if prevalence and rate is not None else None,
            "mean_anomaly_score": float(score[selected].mean()) if selected.any() else None,
        })
    return rows


def _grouped_report(labels, raw, score, metadata, field):
    values = np.asarray(metadata[field]).astype(str)
    output = {}
    for group in sorted(np.unique(values)):
        selected = values == group
        correct = raw[selected & (labels == 0)]
        anomaly = raw[selected & (labels == 1)]
        output[group] = {
            "nodes": int(selected.sum()), "hallucination_nodes": int(labels[selected].sum()),
            "ranking": _ranking(labels[selected], score[selected]),
            "correct_median_lookback": float(np.median(correct)) if correct.size else None,
            "hallucination_median_lookback": float(np.median(anomaly)) if anomaly.size else None,
        }
    return output


def _response_records(metadata, patterns, labels, score, count):
    records = []
    for sample_id in dict.fromkeys(metadata["sample_id"].astype(str)):
        ids = np.flatnonzero(metadata["sample_id"].astype(str) == sample_id)
        occupancy = np.bincount(patterns[ids], minlength=count) / len(ids)
        records.append({
            "sample_id": sample_id, "source_id": str(metadata["source_id"][ids[0]]),
            "response_tokens": int(len(ids)), "hallucinated": int(labels[ids].max()),
            "hallucination_tokens": int(labels[ids].sum()),
            "mean_anomaly_score": float(score[ids].mean()), "max_anomaly_score": float(score[ids].max()),
            "pattern_occupancy": occupancy.tolist(), "pattern_sequence": patterns[ids].astype(int).tolist(),
        })
    response_labels = np.asarray([row["hallucinated"] for row in records])
    return records, {
        "responses": len(records), "hallucinated_responses": int(response_labels.sum()),
        "mean_score": _ranking(response_labels, [row["mean_anomaly_score"] for row in records]),
        "max_score": _ranking(response_labels, [row["max_anomaly_score"] for row in records]),
    }


def _plot_population(
    output, coordinates, patterns, labels, raw, raw_score, calibrated_score
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for pattern in np.unique(patterns):
        selected = patterns == pattern
        axes[0].scatter(coordinates[selected, 0], coordinates[selected, 1], s=4, alpha=.45,
                        label=f"Pattern {pattern}", rasterized=True)
    axes[0].set(title="Label-free Lookback patterns", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axes[0].legend(markerscale=3, frameon=False)
    # Draw the majority class first so rare hallucination tokens remain visible.
    for label, color, name, size, alpha in (
        (0, "#2ca02c", "correct", 3, .16),
        (1, "#d62728", "hallucination", 9, .72),
    ):
        selected = labels == label
        if selected.any():
            axes[1].scatter(
                coordinates[selected, 0], coordinates[selected, 1],
                c=color, s=size, alpha=alpha, label=name, rasterized=True,
            )
    axes[1].set(title="Same frozen coordinates; labels only color nodes", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axes[1].legend(frameon=False, markerscale=2)
    figure.savefig(output / "lookback_embedding_tsne.png", dpi=240)
    plt.close(figure)

    bins = raw.shape[1]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    for label, color, name in ((0, "#2ca02c", "correct"), (1, "#d62728", "hallucination")):
        selected = raw[labels == label]
        if not len(selected):
            continue
        mean = selected.mean(axis=0)
        error = selected.std(axis=0) / np.sqrt(len(selected))
        x = np.arange(1, bins + 1)
        axes[0].plot(x, mean, color=color, label=name)
        axes[0].fill_between(x, mean - error, mean + error, color=color, alpha=.18)
    metric = _ranking(labels, raw_score)
    metric_text = "N/A" if metric["auroc"] is None else f"AUROC={metric['auroc']:.3f}, AUPRC={metric['auprc']:.3f}"
    axes[0].set(title=f"Lookback ratio by layer bin\n{metric_text}", xlabel="layer bin (early → late)", ylabel="lookback ratio")
    axes[0].legend(frameon=False)
    for axis, values, title, xlabel in (
        (axes[1], raw_score, "Direct Lookback baseline", "1 - mean Lookback ratio"),
        (axes[2], calibrated_score, "Train-position-adjusted control", "lower-than-train Lookback z-score"),
    ):
        for label, color, name in (
            (0, "#2ca02c", "correct"), (1, "#d62728", "hallucination")
        ):
            selected = values[labels == label]
            if len(selected):
                axis.hist(
                    selected, bins=60, density=True, alpha=.55,
                    color=color, label=name,
                )
        ranking = _ranking(labels, values)
        subtitle = "N/A" if ranking["auroc"] is None else f"AUROC={ranking['auroc']:.3f}"
        axis.set(title=f"{title}\n{subtitle}", xlabel=xlabel, ylabel="density")
        axis.legend(frameon=False)
    figure.savefig(output / "lookback_separation.png", dpi=240)
    plt.close(figure)


def _display_edge_ids(graph, per_type):
    selected = []
    for target in range(graph.response_idx, graph.num_nodes):
        incoming = torch.nonzero(graph.edge_index[1] == target, as_tuple=False).flatten()
        for relation in (0, 1):
            ids = incoming[graph.edge_type[incoming] == relation]
            if ids.numel():
                ranked = ids[torch.argsort(graph.edge_score[ids], descending=True)]
                selected.extend(ranked[:per_type].detach().cpu().tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _safe_filename(value):
    text = "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value)).strip("._")
    return (text or "sample")[:120]


def _sample_ids(metadata, sample_id):
    return np.flatnonzero(metadata["sample_id"].astype(str) == str(sample_id))


def _render_sample_graph(
    dataset, sample_id, output, raw, calibrated, labels, patterns,
    raw_score, calibrated_score, config,
):
    """One figure: causal token graph plus every node's Lookback trajectory."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sample = dataset[str(sample_id)]
    attention = sample.attention()
    graph = build_attention_graph(
        attention, GraphBuildConfig(selection="typed_mass_cover", mass_cover=config.display_mass_cover)
    )
    edge_ids = _display_edge_ids(graph, config.display_edges_per_type)
    sources = graph.edge_index[0, edge_ids].detach().cpu().numpy() if len(edge_ids) else np.empty(0, dtype=int)
    targets = graph.edge_index[1, edge_ids].detach().cpu().numpy() if len(edge_ids) else np.empty(0, dtype=int)
    relations = graph.edge_type[edge_ids].detach().cpu().numpy() if len(edge_ids) else np.empty(0, dtype=int)
    weights = graph.edge_score[edge_ids].detach().cpu().numpy() if len(edge_ids) else np.empty(0)
    prompt_nodes = np.unique(sources[sources < graph.response_idx])
    response_nodes = np.arange(graph.response_idx, graph.num_nodes)
    response_x = np.arange(len(response_nodes), dtype=float)
    prompt_x = np.linspace(0, max(len(response_nodes) - 1, 1), len(prompt_nodes)) if len(prompt_nodes) else np.empty(0)
    prompt_y = np.full(
        len(prompt_nodes), min(-.25, float(raw_score.min() - .20))
    )
    position = {
        **{int(node): (float(x), float(y)) for node, x, y in zip(prompt_nodes, prompt_x, prompt_y)},
        **{
            int(node): (float(x), float(y))
            for node, x, y in zip(response_nodes, response_x, raw_score)
        },
    }

    width = max(14, min(30, len(response_nodes) * .22 + 9))
    figure, axes = plt.subplots(2, 1, figsize=(width, 10), constrained_layout=True, height_ratios=(3, 2))
    axis = axes[0]
    if len(weights):
        relative = weights / max(float(weights.max()), 1e-8)
        for source, target, relation, strength in zip(sources, targets, relations, relative):
            if int(source) not in position:
                continue
            axis.annotate("", xy=position[int(target)], xytext=position[int(source)], arrowprops={
                "arrowstyle": "->", "color": "#1f77b4" if relation == RP else "#7f7f7f",
                "alpha": .18 + .55 * float(strength), "lw": .4 + 2 * float(strength),
                "connectionstyle": "arc3,rad=.08",
            })
    if len(prompt_nodes):
        axis.scatter(prompt_x, prompt_y, marker="s", s=30, c="#4c78a8", zorder=3)
    colors = np.where(labels == 1, "#d62728", "#2ca02c")
    axis.scatter(
        response_x, raw_score, c=colors, s=52,
        edgecolors="black", linewidths=.4, zorder=4,
    )
    for index, (x, y, pattern) in enumerate(zip(response_x, raw_score, patterns)):
        axis.text(x, y + .018, f"R{index}/P{int(pattern)}", fontsize=6, ha="center")
    within = _ranking(labels, raw_score)
    auc_text = "N/A" if within["auroc"] is None else f"within-sample AUROC={within['auroc']:.3f}"
    axis.set(
        title=f"Sample {sample_id}: Lookback graph ({auc_text})",
        xlabel="response token order", ylabel="1 - mean Lookback ratio",
    )
    axis.grid(alpha=.15)
    axis.legend(handles=[
        Line2D([], [], marker="s", color="none", markerfacecolor="#4c78a8", label="prompt endpoint", markersize=7),
        Line2D([], [], marker="o", color="none", markerfacecolor="#2ca02c", label="correct token", markersize=7),
        Line2D([], [], marker="o", color="none", markerfacecolor="#d62728", label="hallucinated token", markersize=7),
        Line2D([], [], color="#1f77b4", label="prompt → response"),
        Line2D([], [], color="#7f7f7f", label="response → response"),
    ], frameon=False, ncol=5)

    image = axes[1].imshow(raw.T, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    axes[1].set(
        title="Every response node's Lookback ratio trajectory",
        xlabel="response token index", ylabel="layer bin (early → late)",
    )
    figure.colorbar(image, ax=axes[1], label="Lookback ratio", fraction=.02)
    safe_id = _safe_filename(sample_id)
    figure_path = output / f"sample_{safe_id}_lookback_graph.png"
    data_path = output / f"sample_{safe_id}_lookback_graph.npz"
    figure.savefig(figure_path, dpi=240)
    plt.close(figure)
    np.savez_compressed(
        data_path, sample_id=np.asarray(str(sample_id)), token_ids=graph.token_ids.detach().cpu().numpy(),
        response_idx=np.asarray(graph.response_idx, dtype=np.int32), lookback_ratio=raw.astype(np.float32),
        calibrated_lookback=calibrated.astype(np.float32),
        anomaly_score=raw_score.astype(np.float32),
        position_adjusted_anomaly_score=calibrated_score.astype(np.float32),
        label=labels.astype(np.int8), pattern=patterns.astype(np.int16),
        display_edge_source=sources.astype(np.int32), display_edge_target=targets.astype(np.int32),
        display_edge_relation=relations.astype(np.int8), display_edge_score=weights.astype(np.float32),
    )
    sample.release_attention()
    return figure_path, data_path, int(len(response_nodes)), int(len(edge_ids)), within


def _select_samples(config, metadata, labels, score):
    available = set(metadata["sample_id"].astype(str))
    if config.sample_ids:
        missing = [sample_id for sample_id in config.sample_ids if str(sample_id) not in available]
        if missing:
            raise ValueError(f"sample IDs are absent from test split: {missing}")
        return list(dict.fromkeys(map(str, config.sample_ids))), "user_requested"

    candidates = []
    for sample_id in dict.fromkeys(metadata["sample_id"].astype(str)):
        ids = _sample_ids(metadata, sample_id)
        if len(np.unique(labels[ids])) == 2:
            auc = roc_auc_score(labels[ids], score[ids])
            positives = int(labels[ids].sum())
            negatives = int(len(ids) - positives)
            candidates.append(
                (min(positives, negatives), float(auc), len(ids), sample_id)
            )
    if candidates:
        # This is explicitly post-hoc figure selection. It never changes the
        # representation, score, population metrics, or coordinates.
        stable = [row for row in candidates if row[0] >= 2 and row[2] >= 8]
        pool = stable or candidates
        chosen = max(pool, key=lambda row: (row[1], row[0], row[2], row[3]))
        rule = (
            "post_hoc_highest_within_sample_auroc_min_2_per_class_and_8_tokens"
            if stable else "post_hoc_highest_within_sample_auroc_relaxed"
        )
        return [chosen[-1]], rule
    positive = np.flatnonzero(labels == 1)
    node = positive[np.argmax(score[positive])] if len(positive) else int(np.argmax(score))
    return [str(metadata["sample_id"][node])], "post_hoc_highest_positive_token_score"


def _geometry(dataset):
    fields = ("schema", "num_layers", "num_heads", "alignment", "attention_floor", "observer_model")
    return {field: dataset.manifest.get(field) for field in fields}


def discover_lookback_patterns(
    train_dataset,
    test_dataset,
    evaluation_dataset,
    *,
    output_dir,
    config: PatternDiscoveryConfig | None = None,
):
    config = PatternDiscoveryConfig() if config is None else config
    config.validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("lookback output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    if _geometry(train_dataset) != _geometry(test_dataset):
        raise ValueError("train and test attention geometry differ")
    if list(map(str, test_dataset.sample_ids)) != list(map(str, evaluation_dataset.sample_ids)):
        raise ValueError("evaluation dataset does not match the ordered test sample IDs")

    train, _train_control, train_metadata = _extract_split(train_dataset, config, split_name="train")
    test, unresolved, metadata = _extract_split(test_dataset, config, split_name="test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test produce different Lookback layer bins")
    if set(train_metadata["source_id"].astype(str)) & set(metadata["source_id"].astype(str)):
        raise ValueError("train and test source groups overlap")

    print("[2/5] fitting train-only position calibration and Lookback patterns", flush=True)
    center, scale, calibration_report = _fit_position_calibration(
        train, train_metadata["relative_position"], config.position_bins
    )
    calibrated_train = _apply_position_calibration(
        train, train_metadata["relative_position"], center, scale
    )
    calibrated_test = _apply_position_calibration(test, metadata["relative_position"], center, scale)
    model, pattern_diagnostics = _fit_patterns(calibrated_train, config)
    patterns = model.predict(calibrated_test)
    # Primary score: the direct single-feature Lookback baseline.  Lower mean
    # Lookback is more anomalous. Position calibration is a separate control
    # and is used for pattern discovery/projection, never mixed into this score.
    bin_weight = _layer_bin_weights(
        int(train_dataset.manifest["num_layers"]), test.shape[1]
    )
    anomaly_score = 1.0 - test @ bin_weight
    position_adjusted_score = -(calibrated_test @ bin_weight)

    print("[3/5] projecting the frozen Lookback trajectories", flush=True)
    coordinates, projection = _landmark_tsne(calibrated_test, config)
    np.savez_compressed(
        output / "lookback_nodes_label_free.npz",
        schema=np.asarray(REPRESENTATION), lookback_ratio=test.astype(np.float32),
        calibrated_lookback=calibrated_test.astype(np.float32),
        anomaly_score=anomaly_score.astype(np.float32),
        position_adjusted_anomaly_score=position_adjusted_score.astype(np.float32),
        coordinates=coordinates,
        pattern=patterns.astype(np.int16), unresolved_control=unresolved.astype(np.float32),
        calibration_center=center.astype(np.float32), calibration_scale=scale.astype(np.float32),
        sample_id=metadata["sample_id"].astype(str), source_id=metadata["source_id"].astype(str),
        token_index=metadata["token_index"].astype(np.int32),
        relative_position=metadata["relative_position"].astype(np.float32),
    )
    label_free = {
        "schema": REPRESENTATION, "labels_read": False,
        "definition": "mean_attention_per_prompt_token / (mean_attention_per_prompt_token + mean_attention_per_generated_side_token)",
        "generated_side": "response history plus saved current-token diagonal",
        "cache_floor_semantics": "unretained attention is zero; unresolved mass is a separate control",
        "head_aggregation": "compute ratio per layer/head, then average heads",
        "anomaly_direction": "one_minus_mean_lookback_ratio",
        "primary_score_uses_position_calibration": False,
        "layer_bin_weights": bin_weight.tolist(),
        "config": asdict(config), "train_nodes": int(len(train)), "test_nodes": int(len(test)),
        "patterns": int(model.n_clusters), "candidate_davies_bouldin": pattern_diagnostics,
        "position_calibration": calibration_report, "projection": projection,
    }
    (output / "lookback_label_free.json").write_text(
        json.dumps(label_free, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[4/5] opening the separate evaluation dataset for labels", flush=True)
    labels = _read_token_labels(evaluation_dataset, metadata)
    correct = test[labels == 0]
    hallucination = test[labels == 1]
    records, response_report = _response_records(metadata, patterns, labels, anomaly_score, model.n_clusters)
    with (output / "response_patterns.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        **label_free, "labels_read": True,
        "labels_read_during": "post_hoc_evaluation_coloring_and_example_selection_only",
        "hallucination_nodes": int(labels.sum()), "hallucination_prevalence": float(labels.mean()),
        "token_separation": {
            **_ranking(labels, anomaly_score),
            "correct_median_lookback": float(np.median(correct)) if correct.size else None,
            "hallucination_median_lookback": float(np.median(hallucination)) if hallucination.size else None,
        },
        "position_adjusted_token_separation": _ranking(
            labels, position_adjusted_score
        ),
        "separation_by_task_type": _grouped_report(labels, test, anomaly_score, metadata, "task_type"),
        "separation_by_data_source": _grouped_report(labels, test, anomaly_score, metadata, "data_source"),
        "pattern_label_enrichment": _pattern_report(labels, patterns, anomaly_score, model.n_clusters),
        "response_separation": response_report,
        "unresolved_control": {
            "mean": float(unresolved.mean()),
            "anomaly_score_correlation": _safe_correlation(anomaly_score, unresolved.mean(axis=1)),
        },
    }
    # Persist metrics before Matplotlib so an optional rendering failure cannot
    # erase the expensive population result.
    report_path = output / "lookback_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[5/5] rendering population separation and one full token graph", flush=True)
    _plot_population(
        output, coordinates, patterns, labels, test,
        anomaly_score, position_adjusted_score,
    )
    selected_samples, selection_rule = _select_samples(config, metadata, labels, anomaly_score)
    sample_rows = []
    for sample_id in selected_samples:
        ids = _sample_ids(metadata, sample_id)
        figure, data, nodes, edges, within = _render_sample_graph(
            evaluation_dataset, sample_id, output, test[ids], calibrated_test[ids],
            labels[ids], patterns[ids], anomaly_score[ids],
            position_adjusted_score[ids], config,
        )
        sample_rows.append({
            "sample_id": sample_id, "selection_rule": selection_rule,
            "task_type": str(metadata["task_type"][ids[0]]),
            "data_source": str(metadata["data_source"][ids[0]]),
            "response_nodes": nodes, "hallucination_tokens": int(labels[ids].sum()),
            "display_edges": edges, "within_sample_separation": within,
            "figure": str(figure), "data": str(data),
        })
    report["sample_visualizations"] = sample_rows
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "lookback_visualization.npz", coordinates=coordinates, label=labels,
        pattern=patterns.astype(np.int16), anomaly_score=anomaly_score.astype(np.float32),
        position_adjusted_anomaly_score=position_adjusted_score.astype(np.float32),
        sample_id=metadata["sample_id"].astype(str), token_index=metadata["token_index"].astype(np.int32),
    )
    return {
        "output_dir": str(output), "report": str(report_path),
        "population_tsne": str(output / "lookback_embedding_tsne.png"),
        "population_separation": str(output / "lookback_separation.png"),
        "sample_visualizations": sample_rows, "test_nodes": int(len(test)),
        "patterns": int(model.n_clusters),
    }
