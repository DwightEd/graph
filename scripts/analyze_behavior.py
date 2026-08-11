#!/usr/bin/env python3
"""Analyze token-level correct/error behavior in sparse attention graphs."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from behavior import (
    BEHAVIOR_FEATURE_NAMES,
    align_error_onsets,
    centered_window,
    positive_mask,
    summarize_run_windows,
    token_behavior_features,
    validate_positive_runs,
)
from cache import load_attention_sample, sha256
from descriptors import _edge_weights


KEY_FEATURES = (
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "history_edge_share",
)


class BehaviorStore:
    """Verified view over one canonical attention split and its graph split."""

    def __init__(self, attention_root: Path, graph_root: Path) -> None:
        self.attention_root = attention_root
        self.graph_root = graph_root
        self.attention_manifest = json.loads((attention_root / "manifest.json").read_text(encoding="utf-8"))
        self.graph_manifest = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
        self._verify_provenance()

        self.attention_rows = self._read_index(attention_root / "index.jsonl")
        self.graph_rows = self._read_index(graph_root / "index.jsonl")
        if set(self.attention_rows) != set(self.graph_rows):
            raise ValueError("attention and graph indexes do not contain the same sample IDs")

        labels_path = attention_root / "labels.jsonl"
        if not labels_path.is_file():
            raise ValueError("behavior analysis requires labels.jsonl")
        labels_sha = self.attention_manifest.get("labels_sha256")
        if labels_sha is not None and sha256(labels_path) != labels_sha:
            raise ValueError("labels_sha256 does not match labels.jsonl")
        self.labels = {
            str(row["sample_id"]): row["positive_runs"]
            for row in (json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line)
        }
        if set(self.labels) != set(self.attention_rows):
            raise ValueError("labels.jsonl does not align with the split index")

        self.sample_ids = tuple(self.attention_rows)
        self.num_channels = int(self.attention_manifest["num_layers"]) * int(self.attention_manifest["num_heads"])
        self._shape_cache: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _read_index(path: Path) -> dict[str, dict]:
        return {
            str(row["sample_id"]): row
            for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        }

    def _verify_provenance(self) -> None:
        attention_manifest = self.attention_root / "manifest.json"
        attention_index = self.attention_root / "index.jsonl"
        if (
            self.graph_manifest.get("input_manifest_sha256") != sha256(attention_manifest)
            or self.graph_manifest.get("input_index_sha256") != sha256(attention_index)
        ):
            raise ValueError("graph manifest does not match the canonical attention split")
        graph_index = self.graph_root / "index.jsonl"
        if self.graph_manifest.get("index_sha256") != sha256(graph_index):
            raise ValueError("graph index_sha256 does not match index.jsonl")

    @staticmethod
    def _verified_load(root: Path, row: dict) -> dict:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"artifact byte count does not match index: {path}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"artifact SHA256 does not match index: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_graph(self, sample_id: str) -> dict:
        if sample_id not in self.graph_rows:
            raise KeyError(f"unknown sample_id: {sample_id}")
        return self._verified_load(self.graph_root, self.graph_rows[sample_id])

    def load_attention(self, sample_id: str):
        if sample_id not in self.attention_rows:
            raise KeyError(f"unknown sample_id: {sample_id}")
        row = self.attention_rows[sample_id]
        path = self.attention_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("attention artifact does not match index")
        return load_attention_sample(
            path,
            sample_id=sample_id,
            source_id=str(row["source_id"]),
            attention_floor=float(self.attention_manifest["attention_floor"]),
        )

    def graph_shape(self, sample_id: str) -> tuple[int, int]:
        """Return ``(response_idx, response_count)`` and cache the small metadata."""
        if sample_id not in self._shape_cache:
            graph = self.load_graph(sample_id)
            response_idx = int(torch.as_tensor(graph["response_idx"]).item())
            num_nodes = int(torch.as_tensor(graph["num_nodes"]).item())
            self._shape_cache[sample_id] = (response_idx, num_nodes - response_idx)
        return self._shape_cache[sample_id]

    def source_id(self, sample_id: str) -> str:
        return str(self.attention_rows[sample_id]["source_id"])


def _feature_index(name: str) -> int:
    return BEHAVIOR_FEATURE_NAMES.index(name)


def _shade_runs(axis, runs, *, horizontal: bool = False) -> None:
    for start, end in runs:
        if horizontal:
            axis.axhspan(start - 0.5, end - 0.5, alpha=0.12)
        else:
            axis.axvspan(start - 0.5, end - 0.5, alpha=0.12)


def _write_single_csv(path: Path, sample, features: torch.Tensor, runs) -> None:
    mask = positive_mask(len(features), runs).cpu().numpy()
    values = features.cpu().numpy()
    token_ids = sample.token_ids[sample.response_idx:].cpu().numpy()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("response_position", "absolute_position", "token_id", "is_hallucination", *BEHAVIOR_FEATURE_NAMES))
        for position in range(len(features)):
            writer.writerow((
                position,
                sample.response_idx + position,
                int(token_ids[position]),
                int(mask[position]),
                *[float(value) for value in values[position]],
            ))


def _write_run_summary(path: Path, features: torch.Tensor, runs, pre_window: int, post_window: int) -> None:
    summary = summarize_run_windows(features, runs, pre_window=pre_window, post_window=post_window).cpu().numpy()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run_index", "start", "end", "segment", *BEHAVIOR_FEATURE_NAMES))
        normalized = validate_positive_runs(len(features), runs)
        for run_index, (start, end) in enumerate(normalized):
            for segment_index, segment in enumerate(("pre", "error", "post")):
                writer.writerow((run_index, start, end, segment, *summary[run_index, segment_index].tolist()))
            delta = summary[run_index, 1] - summary[run_index, 0]
            writer.writerow((run_index, start, end, "error_minus_pre", *delta.tolist()))


def _plot_single(path: Path, graph: dict, features: torch.Tensor, runs, response_idx: int, num_channels: int) -> None:
    x = np.arange(len(features))
    values = features.cpu().numpy()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axis = axes[0, 0]
    for name in KEY_FEATURES:
        axis.plot(x, values[:, _feature_index(name)], label=name)
    _shade_runs(axis, runs)
    axis.set(title="Normalized routing/topology behavior", xlabel="Response token position", ylabel="Value")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for name in ("in_degree", "prompt_degree", "history_degree"):
        axis.plot(x, values[:, _feature_index(name)], label=name)
    _shade_runs(axis, runs)
    axis.set(title="Retained incoming edge counts", xlabel="Response token position", ylabel="Edges")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.plot(x, values[:, _feature_index("incoming_mass")])
    _shade_runs(axis, runs)
    axis.set(title="Incoming attention mass", xlabel="Response token position", ylabel="Mean-channel edge mass")

    axis = axes[1, 1]
    edge_index = torch.as_tensor(graph["edge_index"])
    if edge_index.numel():
        source, target = edge_index.long()
        weights = _edge_weights(graph, num_channels, edge_index.device).cpu().numpy()
        weight_scale = weights / max(float(weights.max()), 1e-12)
        prompt = (source < response_idx).cpu().numpy()
        target_relative = (target - response_idx).cpu().numpy()
        source_np = source.cpu().numpy()
        axis.scatter(source_np[prompt], target_relative[prompt], s=8.0 + 36.0 * weight_scale[prompt], alpha=0.5, label="prompt source")
        axis.scatter(source_np[~prompt], target_relative[~prompt], s=8.0 + 36.0 * weight_scale[~prompt], alpha=0.5, label="response-history source")
    axis.axvline(response_idx - 0.5, linestyle="--", linewidth=1)
    _shade_runs(axis, runs, horizontal=True)
    axis.set(title="Sparse graph routing map", xlabel="Absolute source token position", ylabel="Response target position")
    axis.legend(fontsize=8)

    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_pair(path: Path, error_features: torch.Tensor, error_runs, control_features: torch.Tensor, error_id: str, control_id: str) -> None:
    error_x = np.linspace(0.0, 1.0, max(len(error_features), 1))
    control_x = np.linspace(0.0, 1.0, max(len(control_features), 1))
    figure, axes = plt.subplots(len(KEY_FEATURES), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, KEY_FEATURES):
        axis.plot(error_x, error_features[:, _feature_index(name)].cpu().numpy(), label=f"error sample {error_id}")
        axis.plot(control_x, control_features[:, _feature_index(name)].cpu().numpy(), label=f"correct sample {control_id}")
        for start, end in error_runs:
            denominator = max(len(error_features) - 1, 1)
            axis.axvspan(start / denominator, max(start, end - 1) / denominator, alpha=0.12)
        axis.set(ylabel=name)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Normalized response position")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze_single(args) -> None:
    store = BehaviorStore(args.attention_root, args.graph_root)
    graph = store.load_graph(args.sample_id)
    sample = store.load_attention(args.sample_id)
    runs = validate_positive_runs(sample.num_response_tokens, store.labels[args.sample_id])
    features = token_behavior_features(graph, store.num_channels)
    if len(features) != sample.num_response_tokens:
        raise ValueError("graph and attention response lengths disagree")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_single_csv(output / "behavior.csv", sample, features, runs)
    _write_run_summary(output / "run_summary.csv", features, runs, args.pre_window, args.post_window)
    _plot_single(output / "behavior.png", graph, features, runs, sample.response_idx, store.num_channels)

    metadata = {
        "sample_id": args.sample_id,
        "source_id": sample.source_id,
        "response_tokens": sample.num_response_tokens,
        "positive_runs": [list(run) for run in runs],
        "feature_names": list(BEHAVIOR_FEATURE_NAMES),
    }

    if args.control_sample_id is not None:
        control_runs = validate_positive_runs(store.graph_shape(args.control_sample_id)[1], store.labels[args.control_sample_id])
        if control_runs:
            raise ValueError("--control-sample-id must be a fully correct sample with no positive_runs")
        control_graph = store.load_graph(args.control_sample_id)
        control_features = token_behavior_features(control_graph, store.num_channels)
        _plot_pair(output / "error_vs_control.png", features, runs, control_features, args.sample_id, args.control_sample_id)
        metadata["control_sample_id"] = args.control_sample_id
        metadata["control_source_id"] = store.source_id(args.control_sample_id)

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2))


def _nearest_length_candidate(sorted_lengths, target_length: int) -> str:
    lengths_only = [item[0] for item in sorted_lengths]
    position = bisect_left(lengths_only, target_length)
    candidates = []
    if position < len(sorted_lengths):
        candidates.append(sorted_lengths[position])
    if position:
        candidates.append(sorted_lengths[position - 1])
    if not candidates:
        raise ValueError("no fully correct samples are available for controls")
    return min(candidates, key=lambda item: (abs(item[0] - target_length), item[1]))[1]


def _build_control_index(store: BehaviorStore):
    correct_ids = [sample_id for sample_id in store.sample_ids if not store.labels[sample_id]]
    if not correct_ids:
        raise ValueError("no fully correct samples are available for control matching")
    by_source: dict[str, list[tuple[int, str]]] = {}
    global_lengths: list[tuple[int, str]] = []
    for sample_id in correct_ids:
        response_count = store.graph_shape(sample_id)[1]
        item = (response_count, sample_id)
        global_lengths.append(item)
        by_source.setdefault(store.source_id(sample_id), []).append(item)
    global_lengths.sort()
    for values in by_source.values():
        values.sort()
    return by_source, global_lengths


def _select_control(store: BehaviorStore, by_source, global_lengths, error_id: str, error_length: int) -> tuple[str, str]:
    source_candidates = by_source.get(store.source_id(error_id), [])
    if source_candidates:
        return _nearest_length_candidate(source_candidates, error_length), "same_source_length"
    return _nearest_length_candidate(global_lengths, error_length), "global_length"


def _nan_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix)
    count = valid.sum(axis=0)
    total = np.nansum(matrix, axis=0)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    centered = np.where(valid, matrix - mean, 0.0)
    variance = np.divide((centered * centered).sum(axis=0), count, out=np.full_like(total, np.nan), where=count > 0)
    return mean, np.sqrt(variance), count


def _write_alignment_csv(path: Path, error_windows: np.ndarray, control_windows: np.ndarray | None, radius: int) -> None:
    error_mean, error_std, error_count = _nan_stats(error_windows)
    if control_windows is not None:
        control_mean, control_std, control_count = _nan_stats(control_windows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["relative_position", "feature", "error_mean", "error_std", "error_count"]
        if control_windows is not None:
            header.extend(("control_mean", "control_std", "control_count", "error_minus_control"))
        writer.writerow(header)
        for offset in range(-radius, radius + 1):
            row_index = offset + radius
            for feature_index, name in enumerate(BEHAVIOR_FEATURE_NAMES):
                row = [offset, name, error_mean[row_index, feature_index], error_std[row_index, feature_index], int(error_count[row_index, feature_index])]
                if control_windows is not None:
                    row.extend((
                        control_mean[row_index, feature_index],
                        control_std[row_index, feature_index],
                        int(control_count[row_index, feature_index]),
                        error_mean[row_index, feature_index] - control_mean[row_index, feature_index],
                    ))
                writer.writerow(row)


def _plot_alignment(path: Path, error_windows: np.ndarray, control_windows: np.ndarray | None, radius: int) -> None:
    offsets = np.arange(-radius, radius + 1)
    error_mean, error_std, _ = _nan_stats(error_windows)
    if control_windows is not None:
        control_mean, control_std, _ = _nan_stats(control_windows)
    figure, axes = plt.subplots(len(KEY_FEATURES), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, KEY_FEATURES):
        column = _feature_index(name)
        axis.plot(offsets, error_mean[:, column], label="hallucination onset")
        axis.fill_between(offsets, error_mean[:, column] - error_std[:, column], error_mean[:, column] + error_std[:, column], alpha=0.12)
        if control_windows is not None:
            axis.plot(offsets, control_mean[:, column], label="matched correct position")
            axis.fill_between(offsets, control_mean[:, column] - control_std[:, column], control_mean[:, column] + control_std[:, column], alpha=0.12)
        axis.axvline(0, linestyle="--", linewidth=1)
        axis.set(ylabel=name)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Token position relative to hallucination onset")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze_alignment(args) -> None:
    store = BehaviorStore(args.attention_root, args.graph_root)
    by_source = global_lengths = None
    if not args.no_controls:
        by_source, global_lengths = _build_control_index(store)

    error_windows = []
    control_windows = []
    pair_rows = []
    control_feature_cache: dict[str, torch.Tensor] = {}
    events = 0

    for sample_id in store.sample_ids:
        runs = store.labels[sample_id]
        if not runs:
            continue
        graph = store.load_graph(sample_id)
        features = token_behavior_features(graph, store.num_channels)
        normalized_runs = validate_positive_runs(len(features), runs)
        selected_runs = normalized_runs[:1] if args.run_policy == "first" else normalized_runs
        aligned, _ = align_error_onsets(features, selected_runs, args.radius, policy="all")

        for run, window in zip(selected_runs, aligned):
            error_windows.append(window.cpu().numpy())
            control_id = match_type = None
            control_center = None
            if not args.no_controls:
                control_id, match_type = _select_control(store, by_source, global_lengths, sample_id, len(features))
                if control_id not in control_feature_cache:
                    control_feature_cache[control_id] = token_behavior_features(store.load_graph(control_id), store.num_channels)
                control_features = control_feature_cache[control_id]
                fraction = run[0] / max(len(features) - 1, 1)
                control_center = int(round(fraction * max(len(control_features) - 1, 0)))
                control_window, _ = centered_window(control_features, control_center, args.radius)
                control_windows.append(control_window.cpu().numpy())

            pair_rows.append((
                sample_id,
                store.source_id(sample_id),
                len(features),
                run[0],
                run[1],
                control_id or "",
                match_type or "",
                len(control_feature_cache[control_id]) if control_id is not None else "",
                control_center if control_center is not None else "",
            ))
            events += 1
            if args.max_events is not None and events >= args.max_events:
                break
        if args.max_events is not None and events >= args.max_events:
            break

    if not error_windows:
        raise ValueError("no hallucination spans were found")

    error_array = np.stack(error_windows)
    control_array = None if args.no_controls else np.stack(control_windows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"error": error_array, "feature_names": np.asarray(BEHAVIOR_FEATURE_NAMES)}
    if control_array is not None:
        arrays["control"] = control_array
    np.savez_compressed(output / "onset_alignment.npz", **arrays)
    _write_alignment_csv(output / "onset_summary.csv", error_array, control_array, args.radius)
    _plot_alignment(output / "onset_alignment.png", error_array, control_array, args.radius)

    with (output / "matched_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "error_sample_id", "error_source_id", "error_response_tokens", "run_start", "run_end",
            "control_sample_id", "match_type", "control_response_tokens", "control_center",
        ))
        writer.writerows(pair_rows)

    metadata = {
        "events": events,
        "radius": args.radius,
        "run_policy": args.run_policy,
        "controls": not args.no_controls,
        "feature_names": list(BEHAVIOR_FEATURE_NAMES),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser):
        subparser.add_argument("--attention-root", type=Path, required=True, help="Canonical attention split root")
        subparser.add_argument("--graph-root", type=Path, required=True, help="Graph split root built from the same attention split")
        subparser.add_argument("--output-dir", type=Path, required=True)

    single = subparsers.add_parser("single", help="Analyze one sample and optionally overlay one fully correct control")
    common(single)
    single.add_argument("--sample-id", required=True)
    single.add_argument("--control-sample-id")
    single.add_argument("--pre-window", type=int, default=8)
    single.add_argument("--post-window", type=int, default=8)
    single.set_defaults(func=analyze_single)

    align = subparsers.add_parser("align", help="Align hallucinated samples at error onset and compare matched correct positions")
    common(align)
    align.add_argument("--radius", type=int, default=12)
    align.add_argument("--run-policy", choices=("first", "all"), default="first")
    align.add_argument("--no-controls", action="store_true", help="Skip matched fully correct controls")
    align.add_argument("--max-events", type=int, help="Optional deterministic cap for debugging or quick studies")
    align.set_defaults(func=analyze_alignment)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "radius", 0) < 0:
        parser.error("--radius must be non-negative")
    if getattr(args, "pre_window", 0) < 0 or getattr(args, "post_window", 0) < 0:
        parser.error("window sizes must be non-negative")
    if getattr(args, "max_events", None) is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
