"""Label-free token behavior features with post-hoc exploratory plots."""

from __future__ import annotations

from bisect import bisect_left
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from behavior import (
    BEHAVIOR_FEATURE_NAMES,
    align_error_onsets,
    centered_window,
    positive_mask,
    summarize_run_windows,
    token_behavior_features,
    validate_positive_runs,
)
from descriptors import _edge_weights, _field
from research_dataset import ResearchDataset


KEY_FEATURES = (
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "history_edge_share",
)


def token_tsne_perplexity(response_tokens: int) -> int:
    if response_tokens < 4:
        raise ValueError("token t-SNE requires at least four response tokens")
    return min(30, max(2, (response_tokens - 1) // 3))


def token_tsne_coordinates(features: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Embed label-free [response_tokens, 11] behavior features in two dimensions."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(BEHAVIOR_FEATURE_NAMES):
        raise ValueError("token behavior features must have shape [response_tokens, 11]")
    perplexity = token_tsne_perplexity(len(values))
    coordinates = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto").fit_transform(
        StandardScaler().fit_transform(values)
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("token t-SNE produced non-finite coordinates")
    return coordinates


def _feature_index(name: str) -> int:
    return BEHAVIOR_FEATURE_NAMES.index(name)


def _shade_runs(axis, runs, *, horizontal: bool = False) -> None:
    for start, end in runs:
        (axis.axhspan if horizontal else axis.axvspan)(start - 0.5, end - 0.5, alpha=0.12)


def _write_single_csv(path: Path, sample, features: torch.Tensor, runs) -> None:
    mask = positive_mask(len(features), runs).cpu().numpy()
    values = features.cpu().numpy()
    token_ids = sample.token_ids[sample.response_idx:].cpu().numpy()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("response_position", "absolute_position", "token_id", "is_hallucination", *BEHAVIOR_FEATURE_NAMES))
        for position, row in enumerate(values):
            writer.writerow((position, sample.response_idx + position, int(token_ids[position]), int(mask[position]), *row.tolist()))


def _write_run_summary(path: Path, features: torch.Tensor, runs, pre_window: int, post_window: int) -> None:
    summary = summarize_run_windows(features, runs, pre_window=pre_window, post_window=post_window).cpu().numpy()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run_index", "start", "end", "segment", *BEHAVIOR_FEATURE_NAMES))
        for run_index, (start, end) in enumerate(validate_positive_runs(len(features), runs)):
            for segment_index, segment in enumerate(("pre", "error", "post")):
                writer.writerow((run_index, start, end, segment, *summary[run_index, segment_index]))
            writer.writerow((run_index, start, end, "error_minus_pre", *(summary[run_index, 1] - summary[run_index, 0])))


def _plot_single(path: Path, graph: dict, features: torch.Tensor, runs, response_idx: int, num_channels: int) -> None:
    x = np.arange(len(features))
    values = features.cpu().numpy()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name in KEY_FEATURES:
        axes[0, 0].plot(x, values[:, _feature_index(name)], label=name)
    _shade_runs(axes[0, 0], runs)
    axes[0, 0].set(title="Normalized routing/topology behavior", xlabel="Response token position", ylabel="Value")
    axes[0, 0].legend(fontsize=8)
    for name in ("in_degree", "prompt_degree", "history_degree"):
        axes[0, 1].plot(x, values[:, _feature_index(name)], label=name)
    _shade_runs(axes[0, 1], runs)
    axes[0, 1].set(title="Retained incoming edge counts", xlabel="Response token position", ylabel="Edges")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(x, values[:, _feature_index("incoming_mass")])
    _shade_runs(axes[1, 0], runs)
    axes[1, 0].set(title="Incoming attention mass", xlabel="Response token position", ylabel="Mean-channel edge mass")
    edge_index = torch.as_tensor(_field(graph, "edge_index"))
    if edge_index.numel():
        source, target = edge_index.long()
        weights = _edge_weights(graph, num_channels, edge_index.device).cpu().numpy()
        scale = weights / weights.max()
        prompt = (source < response_idx).cpu().numpy()
        target = (target - response_idx).cpu().numpy()
        source = source.cpu().numpy()
        for mask, label in ((prompt, "prompt source"), (~prompt, "response-history source")):
            axes[1, 1].scatter(source[mask], target[mask], s=8 + 36 * scale[mask], alpha=0.5, label=label)
    axes[1, 1].axvline(response_idx - 0.5, linestyle="--", linewidth=1)
    _shade_runs(axes[1, 1], runs, horizontal=True)
    axes[1, 1].set(title="Sparse graph routing map", xlabel="Absolute source token position", ylabel="Response target position")
    axes[1, 1].legend(fontsize=8)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_pair(path: Path, error_features: torch.Tensor, error_runs, control_features: torch.Tensor, error_id: str, control_id: str) -> None:
    figure, axes = plt.subplots(len(KEY_FEATURES), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    error_x = np.linspace(0.0, 1.0, len(error_features))
    control_x = np.linspace(0.0, 1.0, len(control_features))
    for axis, name in zip(axes, KEY_FEATURES):
        axis.plot(error_x, error_features[:, _feature_index(name)].cpu().numpy(), label=f"error sample {error_id}")
        axis.plot(control_x, control_features[:, _feature_index(name)].cpu().numpy(), label=f"correct sample {control_id}")
        for start, end in error_runs:
            axis.axvspan(start / max(len(error_features) - 1, 1), max(start, end - 1) / max(len(error_features) - 1, 1), alpha=0.12)
        axis.set(ylabel=name)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Normalized response position")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_token_tsne(path: Path, coordinates: np.ndarray, labels: np.ndarray) -> None:
    positions = np.arange(len(coordinates))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis in axes:
        axis.plot(coordinates[:, 0], coordinates[:, 1], color="0.75", linewidth=1, zorder=1)
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=np.where(labels, "tab:red", "tab:blue"), s=34, edgecolors="black", linewidths=0.35, zorder=2)
    axes[0].set_title("Post-hoc hallucination labels")
    color = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=positions / max(len(positions) - 1, 1), cmap="viridis", s=34, edgecolors="black", linewidths=0.35, zorder=2)
    axes[1].set_title("Normalized response position")
    figure.colorbar(color, ax=axes[1], label="Normalized response position")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _nan_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix)
    count = valid.sum(axis=0)
    mean = np.divide(np.nansum(matrix, axis=0), count, out=np.full(matrix.shape[1:], np.nan), where=count > 0)
    variance = np.divide(np.nansum(np.where(valid, matrix - mean, 0.0) ** 2, axis=0), count, out=np.full(matrix.shape[1:], np.nan), where=count > 0)
    return mean, np.sqrt(variance), count


def _write_alignment_csv(path: Path, errors: np.ndarray, controls: np.ndarray | None, radius: int) -> None:
    error_mean, error_std, error_count = _nan_stats(errors)
    control_mean, control_std, control_count = (None, None, None) if controls is None else _nan_stats(controls)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["relative_position", "feature", "error_mean", "error_std", "error_count"]
        if controls is not None:
            header += ["control_mean", "control_std", "control_count", "error_minus_control"]
        writer.writerow(header)
        for offset in range(-radius, radius + 1):
            for column, name in enumerate(BEHAVIOR_FEATURE_NAMES):
                row = [offset, name, error_mean[offset + radius, column], error_std[offset + radius, column], int(error_count[offset + radius, column])]
                if controls is not None:
                    row += [control_mean[offset + radius, column], control_std[offset + radius, column], int(control_count[offset + radius, column]), error_mean[offset + radius, column] - control_mean[offset + radius, column]]
                writer.writerow(row)


def _plot_alignment(path: Path, errors: np.ndarray, controls: np.ndarray | None, radius: int) -> None:
    offsets = np.arange(-radius, radius + 1)
    error_mean, error_std, _ = _nan_stats(errors)
    control_mean, control_std, _ = (None, None, None) if controls is None else _nan_stats(controls)
    figure, axes = plt.subplots(len(KEY_FEATURES), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, KEY_FEATURES):
        column = _feature_index(name)
        axis.plot(offsets, error_mean[:, column], label="hallucination onset")
        axis.fill_between(offsets, error_mean[:, column] - error_std[:, column], error_mean[:, column] + error_std[:, column], alpha=0.12)
        if controls is not None:
            axis.plot(offsets, control_mean[:, column], label="matched correct position")
            axis.fill_between(offsets, control_mean[:, column] - control_std[:, column], control_mean[:, column] + control_std[:, column], alpha=0.12)
        axis.axvline(0, linestyle="--", linewidth=1)
        axis.set(ylabel=name)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Token position relative to hallucination onset")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


class BehaviorAnalysis:
    """Analyze one canonical split with original-threshold graph topology only."""

    def __init__(self, split_root, output_dir, graph_root=None, tau=None) -> None:
        graph_roots = {} if graph_root is None else {"original": Path(graph_root)}
        self.dataset = ResearchDataset(split_root, graph_roots)
        if graph_root is not None:
            graph_manifest = self.dataset.graph_manifests["original"]
            if graph_manifest.get("kind") != "original":
                raise ValueError("token behavior topology features require graph manifest kind == original")
            cached_tau = graph_manifest.get("parameters", {}).get("tau")
            if tau is not None and cached_tau != tau:
                raise ValueError("cached original graph tau does not match requested tau")
            tau = cached_tau
        self._labels = None
        self.output_dir = Path(output_dir)
        self.tau = tau
        self.num_channels = int(self.dataset.manifest["num_layers"]) * int(self.dataset.manifest["num_heads"])

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.dataset.labels()
        return self._labels

    def _graph(self, sample):
        return sample.original_graph(self.tau) if not self.dataset.graph_roots else sample.graph("original")

    def _features(self, sample) -> torch.Tensor:
        return token_behavior_features(self._graph(sample), self.num_channels)

    def single(self, sample_id: str, *, control_sample_id: str | None = None, pre_window: int = 8, post_window: int = 8) -> dict:
        sample = self.dataset[sample_id]
        graph = self._graph(sample)
        features = token_behavior_features(graph, self.num_channels)
        coordinates = token_tsne_coordinates(features.cpu().numpy())
        runs = validate_positive_runs(len(features), self.labels.positive_runs(sample_id))
        labels = positive_mask(len(features), runs).cpu().numpy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_single_csv(self.output_dir / "behavior.csv", sample.attention(), features, runs)
        _write_run_summary(self.output_dir / "run_summary.csv", features, runs, pre_window, post_window)
        _plot_single(self.output_dir / "behavior.png", graph, features, runs, sample.attention().response_idx, self.num_channels)
        _plot_token_tsne(self.output_dir / "token_tsne.png", coordinates, labels)
        np.savez_compressed(self.output_dir / "token_tsne.npz", coordinates=coordinates, response_position=np.arange(len(features)), is_hallucination=labels)
        metadata = {"sample_id": sample_id, "source_id": sample.source_id, "response_tokens": len(features), "positive_runs": [list(run) for run in runs], "feature_names": list(BEHAVIOR_FEATURE_NAMES), "tsne_perplexity": token_tsne_perplexity(len(features))}
        if control_sample_id is not None:
            control = self.dataset[control_sample_id]
            control_runs = validate_positive_runs(control.attention().num_response_tokens, self.labels.positive_runs(control_sample_id))
            if control_runs:
                raise ValueError("control_sample_id must be fully correct")
            _plot_pair(self.output_dir / "error_vs_control.png", features, runs, self._features(control), sample_id, control_sample_id)
            metadata.update(control_sample_id=control_sample_id, control_source_id=control.source_id)
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata

    def _control_index(self):
        correct = [sample for sample in self.dataset if not self.labels.positive_runs(sample.sample_id)]
        if not correct:
            raise ValueError("no fully correct samples are available for controls")
        by_source, global_lengths = {}, []
        for sample in correct:
            item = (sample.attention().num_response_tokens, sample.sample_id)
            global_lengths.append(item)
            by_source.setdefault(sample.source_id, []).append(item)
        global_lengths.sort()
        for values in by_source.values():
            values.sort()
        return by_source, global_lengths

    @staticmethod
    def _nearest(lengths, target: int) -> str:
        position = bisect_left([length for length, _ in lengths], target)
        candidates = lengths[max(0, position - 1):position + 1]
        return min(candidates, key=lambda item: (abs(item[0] - target), item[1]))[1]

    def align(self, *, radius: int = 12, run_policy: str = "first", controls: bool = True, max_events: int | None = None) -> dict:
        if radius < 0 or run_policy not in {"first", "all"}:
            raise ValueError("radius must be non-negative and run_policy must be 'first' or 'all'")
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        by_source, global_lengths = self._control_index() if controls else (None, None)
        error_windows, control_windows, rows, cache = [], [], [], {}
        for sample in self.dataset:
            runs = validate_positive_runs(sample.attention().num_response_tokens, self.labels.positive_runs(sample.sample_id))
            if not runs:
                continue
            features = self._features(sample)
            selected = runs[:1] if run_policy == "first" else runs
            aligned, _ = align_error_onsets(features, selected, radius, policy="all")
            for run, window in zip(selected, aligned):
                error_windows.append(window.cpu().numpy())
                row = [sample.sample_id, sample.source_id, len(features), run[0], run[1]]
                if controls:
                    candidates = by_source.get(sample.source_id, global_lengths)
                    control_id = self._nearest(candidates, len(features))
                    control_features = cache.get(control_id)
                    if control_features is None:
                        control_features = self._features(self.dataset[control_id])
                        cache[control_id] = control_features
                    center = round(run[0] * max(len(control_features) - 1, 0) / max(len(features) - 1, 1))
                    control_windows.append(centered_window(control_features, center, radius)[0].cpu().numpy())
                    row += [control_id, "same_source_length" if sample.source_id in by_source else "global_length", len(control_features), center]
                rows.append(row)
                if max_events is not None and len(error_windows) == max_events:
                    break
            if max_events is not None and len(error_windows) == max_events:
                break
        if not error_windows:
            raise ValueError("no hallucination spans were found")
        errors = np.stack(error_windows)
        controls_array = np.stack(control_windows) if controls else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"error": errors, "feature_names": np.asarray(BEHAVIOR_FEATURE_NAMES)}
        if controls_array is not None:
            arrays["control"] = controls_array
        np.savez_compressed(self.output_dir / "onset_alignment.npz", **arrays)
        _write_alignment_csv(self.output_dir / "onset_summary.csv", errors, controls_array, radius)
        _plot_alignment(self.output_dir / "onset_alignment.png", errors, controls_array, radius)
        with (self.output_dir / "matched_events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            header = ["error_sample_id", "error_source_id", "error_response_tokens", "run_start", "run_end"]
            if controls:
                header += ["control_sample_id", "match_type", "control_response_tokens", "control_center"]
            writer.writerow(header)
            writer.writerows(rows)
        metadata = {"events": len(errors), "radius": radius, "run_policy": run_policy, "controls": controls, "feature_names": list(BEHAVIOR_FEATURE_NAMES)}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
