#!/usr/bin/env python3
"""Project reconstructed token graphs with t-SNE without reloading attention features.

The projection is graph-only: descriptors are computed from each saved ``.pt``
graph plus metadata already stored in the graph split manifest. A label sidecar
is optional and is used only to color the final scatter plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior import BEHAVIOR_FEATURE_NAMES, token_behavior_features
from cache import sha256
from descriptors import temporal_summary


TOKEN_GRAPH_SCHEMA = "ragtruth-token-graph-v1"
SUMMARY_NAMES = tuple(
    f"{stat}_{feature}"
    for stat in ("mean", "std", "slope")
    for feature in BEHAVIOR_FEATURE_NAMES
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_graph_split(root: Path) -> tuple[dict, list[dict], int]:
    """Load and validate one reconstructed graph split directory."""
    manifest_path = root / "manifest.json"
    index_path = root / "index.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Graph manifest not found: {manifest_path}\n"
            "--graph-root must be the split directory that directly contains "
            "manifest.json and index.jsonl (for example .../relation_topk_channels/test)."
        )
    if not index_path.is_file():
        raise FileNotFoundError(f"Graph index not found: {index_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TOKEN_GRAPH_SCHEMA:
        raise ValueError(
            f"graph-only t-SNE currently expects token graphs ({TOKEN_GRAPH_SCHEMA}); "
            f"got {manifest.get('schema')!r}"
        )
    if manifest.get("kind") not in {"original", "relation_topk", "relation_topk_channels"}:
        raise ValueError(f"unsupported token graph kind: {manifest.get('kind')!r}")
    if sha256(index_path) != manifest.get("index_sha256"):
        raise ValueError("graph index_sha256 does not match manifest.json")

    rows = _jsonl(index_path)
    if len(rows) != int(manifest.get("count", -1)):
        raise ValueError("graph index row count does not match manifest.json")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("graph index contains duplicate sample_id values")

    num_layers = int(manifest["num_layers"])
    num_heads = int(manifest["num_heads"])
    if num_layers <= 0 or num_heads <= 0:
        raise ValueError("graph manifest must contain positive num_layers and num_heads")
    return manifest, rows, num_layers * num_heads


def load_verified_graph(root: Path, row: dict) -> dict:
    """Load one saved graph while checking the graph index contract."""
    path = root / row["path"]
    if not path.is_file():
        raise FileNotFoundError(f"graph file not found: {path}")
    if path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"graph byte count does not match index.jsonl: {path.name}")
    if sha256(path) != row["sha256"]:
        raise ValueError(f"graph SHA256 does not match index.jsonl: {path.name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def graph_descriptor(graph: dict, num_channels: int) -> np.ndarray:
    """Return one fixed 33-D graph behavior descriptor.

    Each response token first receives 11 graph-routing/topology features. Mean,
    population standard deviation, and linear slope over response position then
    summarize the variable-length graph into 33 dimensions for projection.
    """
    token_features = token_behavior_features(graph, num_channels)
    return temporal_summary(token_features).cpu().numpy()


def embed(matrix: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardize descriptors, optionally PCA-reduce, then fit 2-D t-SNE."""
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("t-SNE requires a [samples, features] matrix with at least two samples")
    scaled = StandardScaler().fit_transform(matrix)
    projected_input = scaled
    if scaled.shape[1] > 50:
        components = min(50, scaled.shape[0], scaled.shape[1])
        projected_input = PCA(n_components=components, random_state=random_state).fit_transform(scaled)
    perplexity = min(30.0, float(len(projected_input) - 1))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=random_state,
    ).fit_transform(projected_input)
    return coordinates, projected_input


def load_sample_labels(path: Path, sample_ids: list[str]) -> np.ndarray:
    """Load sample-level hallucination labels solely for visualization colors."""
    if not path.is_file():
        raise FileNotFoundError(f"label sidecar not found: {path}")
    labels = {
        str(row["sample_id"]): int(bool(row.get("positive_runs", [])))
        for row in _jsonl(path)
    }
    missing = [sample_id for sample_id in sample_ids if sample_id not in labels]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"labels are missing {len(missing)} graph samples; first: {preview}")
    return np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.int8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="t-SNE directly from reconstructed graph .pt files; attention features are not required."
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        required=True,
        help="Graph split directory containing manifest.json, index.jsonl, and graph files referenced by the index.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels.jsonl used only to color correct vs hallucinated samples.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/graph_tsne"))
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_root = args.graph_root.resolve()
    manifest, rows, num_channels = load_graph_split(graph_root)

    sample_ids: list[str] = []
    descriptors: list[np.ndarray] = []
    for row in rows:
        graph = load_verified_graph(graph_root, row)
        sample_ids.append(str(row["sample_id"]))
        descriptors.append(graph_descriptor(graph, num_channels))

    matrix = np.stack(descriptors)
    coordinates, projected_input = embed(matrix, args.random_state)
    labels = load_sample_labels(args.labels.resolve(), sample_ids) if args.labels is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "graph_tsne_coordinates.npz",
        sample_id=np.asarray(sample_ids),
        descriptor=matrix,
        descriptor_names=np.asarray(SUMMARY_NAMES),
        tsne=coordinates,
        labels=labels if labels is not None else np.asarray([], dtype=np.int8),
    )

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if labels is None:
        axis.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.8)
    else:
        scatter = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=labels, cmap="coolwarm", alpha=0.8
        )
        colorbar = figure.colorbar(scatter, ax=axis, ticks=[0, 1])
        colorbar.ax.set_yticklabels(["correct", "hallucinated"])
        colorbar.set_label("response label")
    axis.set(
        title=f"Graph-only t-SNE ({manifest['kind']}, {matrix.shape[1]}-D descriptor)",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    figure.savefig(args.output_dir / "graph_tsne.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "graph_root": str(graph_root),
        "graph_kind": manifest["kind"],
        "sample_count": len(sample_ids),
        "descriptor_dim": int(matrix.shape[1]),
        "num_channels": num_channels,
        "tsne_input_dim": int(projected_input.shape[1]),
        "labels": str(args.labels.resolve()) if args.labels is not None else None,
        "labels_used_for_embedding": False,
        "random_state": args.random_state,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
