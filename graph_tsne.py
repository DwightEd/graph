"""Label-free sample-level t-SNE views of canonical attention graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from descriptors import temporal_summary
from graph_features import basic_structural_features
from research_dataset import ResearchDataset


class GraphTSNEAnalysis:
    """Project graph-topology and node temporal summaries for every response."""

    def __init__(
        self,
        split_root,
        output_dir,
        graph_root=None,
        tau=0.01,
        node_feature_mode="attention",
        seed=0,
    ):
        self.split_root = Path(split_root)
        self.output_dir = Path(output_dir)
        self.graph_root = None if graph_root is None else Path(graph_root)
        self.tau = tau
        self.node_feature_mode = node_feature_mode
        self.seed = seed

    def run(self):
        if self.node_feature_mode == "none":
            raise ValueError("node_feature_mode cannot be 'none' for graph t-SNE")
        graph_roots = None if self.graph_root is None else {"graph": self.graph_root}
        dataset = ResearchDataset(self.split_root, graph_roots=graph_roots, verify_hashes=True)
        if self.graph_root is not None:
            graph_manifest = dataset.graph_manifests["graph"]
            if graph_manifest.get("kind") != "original":
                raise ValueError("graph t-SNE topology descriptors require an original graph cache")
            if graph_manifest.get("parameters", {}).get("tau") != self.tau:
                raise ValueError("cached original graph tau does not match requested tau")
        sample_ids, response_tokens, topology_rows, node_rows = [], [], [], []

        for sample in tqdm(dataset, desc="t-SNE descriptors", unit="sample"):
            attention = sample.attention()
            graph = sample.original_graph(self.tau) if self.graph_root is None else sample.graph("graph")
            topology = basic_structural_features(attention, sample.relation_edges(graph))
            node_features = sample.node_features(self.node_feature_mode)
            sample_ids.append(sample.sample_id)
            response_tokens.append(attention.num_response_tokens)
            topology_rows.append(temporal_summary(topology).cpu().numpy())
            node_rows.append(
                temporal_summary(node_features[attention.response_idx:]).cpu().numpy()
            )

        topology = np.stack(topology_rows)
        node = np.stack(node_rows)
        combined = np.concatenate(
            (
                self._scale(topology) / np.sqrt(topology.shape[1]),
                self._scale(node) / np.sqrt(node.shape[1]),
            ),
            axis=1,
        )
        embeddings = {
            "topology": self._embed(topology),
            "node": self._embed(node),
            "combined": self._embed(combined, pre_scaled=True),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        coordinates = self.output_dir / "graph_tsne_coordinates.npz"
        np.savez_compressed(
            coordinates,
            sample_id=np.asarray(sample_ids),
            response_tokens=np.asarray(response_tokens, dtype=np.int32),
            **embeddings,
        )

        labels = dataset.labels()
        positive_response = np.asarray(
            [int(bool(labels.positive_runs(sample_id))) for sample_id in sample_ids]
        )
        figure = self.output_dir / "graph_tsne.png"
        length_figure = self.output_dir / "graph_tsne_response_length.png"
        self._plot(embeddings, positive_response, "positive response span", "coolwarm", figure)
        self._plot(embeddings, np.asarray(response_tokens), "response tokens", "viridis", length_figure)
        return {
            "samples": len(sample_ids),
            "figure": str(figure),
            "length_figure": str(length_figure),
            "coordinates": str(coordinates),
        }

    def _embed(self, matrix, pre_scaled=False):
        if len(matrix) < 2:
            raise ValueError("t-SNE needs at least two samples")
        values = matrix if pre_scaled else self._scale(matrix)
        if values.shape[1] > 50:
            values = PCA(
                n_components=min(50, values.shape[0], values.shape[1]),
                random_state=self.seed,
            ).fit_transform(values)
        return TSNE(
            n_components=2,
            perplexity=min(30, len(values) - 1),
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=self.seed,
        ).fit_transform(values)

    @staticmethod
    def _scale(matrix):
        return StandardScaler().fit_transform(matrix)

    @staticmethod
    def _plot(embeddings, colors, color_label, cmap, output):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for axis, (name, coordinates) in zip(axes, embeddings.items()):
            plot = axis.scatter(
                coordinates[:, 0], coordinates[:, 1], c=colors, cmap=cmap, alpha=0.8
            )
            axis.set(title=f"{name} t-SNE", xlabel="t-SNE 1", ylabel="t-SNE 2")
        figure.colorbar(plot, ax=axes, label=color_label)
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
