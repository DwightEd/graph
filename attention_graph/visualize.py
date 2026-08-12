"""Visualization of learned GNN node embeddings.

Sampling, scaling, PCA, and t-SNE are label-blind. Evaluation labels are opened
only after coordinates are fixed, solely for coloring the same points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .score import load_score_records


def visualize_embeddings(
    dataset,
    *,
    score_path,
    output_dir,
    max_nodes=10000,
    perplexity=30.0,
    seed=0,
):
    records = load_score_records(score_path)
    if len(records) < 3:
        raise ValueError("at least three token embeddings are required")
    if max_nodes is not None and len(records) > int(max_nodes):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(records), int(max_nodes), replace=False))
        records = [records[index] for index in selected]

    embeddings = np.stack([row["embedding"] for row in records]).astype(np.float32)
    scaled = StandardScaler().fit_transform(embeddings)
    pca_dim = min(50, scaled.shape[1], len(scaled) - 1)
    working = (
        PCA(n_components=pca_dim, random_state=seed).fit_transform(scaled)
        if scaled.shape[1] > pca_dim else scaled
    )
    actual_perplexity = min(float(perplexity), len(working) - 1.0)
    coordinates = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=seed,
    ).fit_transform(working)

    labels = dataset.labels()
    label_cache = {}
    y = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in label_cache:
            sample = dataset[sample_id]
            label_cache[sample_id] = labels.response_labels(sample).cpu().numpy()
            sample.release_attention()
        y.append(label_cache[sample_id][int(record["token_index"])])
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray([row["score"] for row in records], dtype=np.float32)

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    plot = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=scores, s=12, alpha=0.7, rasterized=True
    )
    figure.colorbar(plot, ax=axes[0], label="Unsupervised anomaly score")
    axes[0].set_title("Learned node embeddings — anomaly score")
    for label, name, marker in ((0, "Non-hallucination", "o"), (1, "Hallucination", "x")):
        mask = y == label
        if mask.any():
            axes[1].scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                s=14 if label == 0 else 24,
                alpha=0.45 if label == 0 else 0.9,
                marker=marker,
                label=f"{name} (n={int(mask.sum())})",
                rasterized=True,
            )
    axes[1].set_title("Same coordinates — labels opened after projection")
    axes[1].legend()
    for axis in axes:
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    figure.savefig(output / "node_embedding_tsne.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        output / "node_embedding_tsne.npz",
        coordinates=coordinates,
        embedding=embeddings,
        score=scores,
        label=y,
        sample_id=np.asarray([row["sample_id"] for row in records], dtype=str),
        token_index=np.asarray([row["token_index"] for row in records], dtype=np.int32),
    )
    return {
        "figure": str(output / "node_embedding_tsne.png"),
        "coordinates": str(output / "node_embedding_tsne.npz"),
        "selected_nodes": len(records),
        "perplexity": actual_perplexity,
        "representation": "learned_attention_gnn_node_embedding",
    }
