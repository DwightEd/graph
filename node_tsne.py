"""Dataset-level node t-SNE from interpretable graph-structural node states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_dataset import STRUCTURAL_FEATURE_NAMES, ResearchDataset


class NodeTSNEVisualizer:
    """Project response-token structural states from many sample graphs together.

    One point in the final embedding is one response token. The input vector of
    each point is the 12-D structural state returned by
    ``ResearchSample.structural_features()``; no learned encoder or GNN is used.
    """

    def __init__(
        self,
        split_root,
        *,
        device="cpu",
        verify_hashes=False,
        random_state=0,
    ):
        self.dataset = ResearchDataset(
            split_root, device=device, verify_hashes=verify_hashes
        )
        self.random_state = int(random_state)
        self.last_result = None

    def collect(self, *, max_samples=None, max_nodes=None):
        """Collect one 12-D vector per response token across the split.

        ``max_samples`` truncates dataset order before loading. ``max_nodes``
        applies deterministic uniform sampling *after* all node states are
        collected and does not inspect labels, so sampling cannot manufacture
        class separation. Set both to ``None`` to use every response token.
        """
        sample_ids = self.dataset.sample_ids
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < 1:
                raise ValueError("max_samples must be positive or None")
            sample_ids = sample_ids[:max_samples]

        features = []
        node_sample_ids = []
        response_positions = []
        task_types = []
        data_sources = []

        for sample_id in sample_ids:
            sample = self.dataset[sample_id]
            node_features = (
                sample.structural_features().cpu().numpy().astype(np.float32, copy=False)
            )
            count = len(node_features)
            if count == 0:
                continue

            features.append(node_features)
            node_sample_ids.append(np.full(count, str(sample_id), dtype=object))
            response_positions.append(np.arange(count, dtype=np.int32))
            task_types.append(np.full(count, sample.task_type, dtype=object))
            data_sources.append(np.full(count, sample.data_source, dtype=object))

        if not features:
            raise ValueError("no response-token node states were collected")

        output = {
            "features": np.concatenate(features, axis=0),
            "sample_id": np.concatenate(node_sample_ids, axis=0),
            "response_position": np.concatenate(response_positions, axis=0),
            "task_type": np.concatenate(task_types, axis=0),
            "data_source": np.concatenate(data_sources, axis=0),
            "feature_names": np.asarray(STRUCTURAL_FEATURE_NAMES),
            "sample_count": len(sample_ids),
        }

        total_nodes = len(output["features"])
        output["total_nodes_before_sampling"] = total_nodes
        if max_nodes is not None:
            max_nodes = int(max_nodes)
            if max_nodes < 3:
                raise ValueError("max_nodes must be at least 3 or None")
            if total_nodes > max_nodes:
                rng = np.random.default_rng(self.random_state)
                indices = np.sort(
                    rng.choice(total_nodes, size=max_nodes, replace=False)
                )
                for key in (
                    "features",
                    "sample_id",
                    "response_position",
                    "task_type",
                    "data_source",
                ):
                    output[key] = output[key][indices]

        output["selected_nodes"] = len(output["features"])
        return output

    def fit(self, *, max_samples=None, max_nodes=None, perplexity=30.0):
        """Fit one common t-SNE, then load labels only for its presentation."""
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        result = self.collect(max_samples=max_samples, max_nodes=max_nodes)
        matrix = result["features"]
        if len(matrix) < 3:
            raise ValueError("t-SNE needs at least three node vectors")

        scaled = StandardScaler().fit_transform(matrix)
        actual_perplexity = min(float(perplexity), len(scaled) - 1.0)
        if actual_perplexity <= 0:
            raise ValueError("perplexity must be positive")

        result["coordinates"] = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=self.random_state,
        ).fit_transform(scaled)
        result["perplexity"] = actual_perplexity
        label_store = self.dataset.labels()
        labels_by_sample = {
            sample_id: label_store.response_labels(self.dataset[sample_id])
            .cpu()
            .numpy()
            for sample_id in np.unique(result["sample_id"])
        }
        result["labels"] = np.asarray(
            [
                labels_by_sample[sample_id][position]
                for sample_id, position in zip(
                    result["sample_id"], result["response_position"]
                )
            ],
            dtype=np.int64,
        )
        self.last_result = result
        return result

    def plot(self, result=None, *, save_path=None, title=None):
        """Plot the pooled node embedding; one marker is one response token."""
        import matplotlib.pyplot as plt

        result = self.last_result if result is None else result
        if result is None or "coordinates" not in result:
            raise ValueError("call fit() first or pass a fitted result")

        coordinates = result["coordinates"]
        labels = result["labels"]
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)

        normal = labels == 0
        anomaly = labels == 1
        if normal.any():
            axis.scatter(
                coordinates[normal, 0],
                coordinates[normal, 1],
                s=10,
                alpha=0.35,
                marker="o",
                label=f"Correct token (n={int(normal.sum())})",
                rasterized=True,
            )
        if anomaly.any():
            axis.scatter(
                coordinates[anomaly, 0],
                coordinates[anomaly, 1],
                s=22,
                alpha=0.85,
                marker="x",
                label=f"Hallucination token (n={int(anomaly.sum())})",
                rasterized=True,
            )

        axis.set(
            title=title or "Node-level t-SNE from graph-structural states",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        axis.legend()
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=220, bbox_inches="tight")
        return figure

    def save(self, result=None, *, output_dir):
        """Save coordinates, original 12-D node states, labels, and provenance."""
        result = self.last_result if result is None else result
        if result is None or "coordinates" not in result:
            raise ValueError("call fit() first or pass a fitted result")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "node_tsne_coordinates.npz",
            coordinates=result["coordinates"],
            features=result["features"],
            labels=result["labels"],
            sample_id=result["sample_id"].astype(str),
            response_position=result["response_position"],
            task_type=result["task_type"].astype(str),
            data_source=result["data_source"].astype(str),
            feature_names=result["feature_names"],
        )
        metadata = {
            "split_root": str(self.dataset.root),
            "sample_count": int(result["sample_count"]),
            "total_nodes_before_sampling": int(result["total_nodes_before_sampling"]),
            "selected_nodes": int(result["selected_nodes"]),
            "correct_nodes": int((result["labels"] == 0).sum()),
            "hallucination_nodes": int((result["labels"] == 1).sum()),
            "perplexity": float(result["perplexity"]),
            "random_state": self.random_state,
            "feature_names": list(STRUCTURAL_FEATURE_NAMES),
            "node_representation": "12-D deterministic graph-structural statistics",
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return metadata

    def run(
        self,
        *,
        output_dir,
        max_samples=None,
        max_nodes=None,
        perplexity=30.0,
        title=None,
    ):
        """Collect, fit, plot, and save the dataset-level node t-SNE in one call."""
        output_dir = Path(output_dir)
        result = self.fit(
            max_samples=max_samples,
            max_nodes=max_nodes,
            perplexity=perplexity,
        )
        self.plot(
            result,
            save_path=output_dir / "node_tsne.png",
            title=title,
        )
        metadata = self.save(result, output_dir=output_dir)
        return result, metadata
