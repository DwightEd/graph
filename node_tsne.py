"""Dataset-level t-SNE for label-free graph-derived response-token states.

The default remains the original 12-D baseline. Richer modes expose the
32-D statistics used by onset analysis and the evidence-aligned mass-cover /
provenance state. Labels are loaded only after the projection is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research_dataset import ResearchDataset
from structural_discovery import REPRESENTATION_MODES, response_representation


class NodeTSNEVisualizer:
    """Project one common response-token representation from many graphs."""

    def __init__(
        self,
        split_root,
        *,
        representation="basic12",
        mass_cover=0.80,
        relay_discount=0.85,
        task_type=None,
        generator_model=None,
        device="cpu",
        verify_hashes=False,
        random_state=0,
    ):
        if representation not in REPRESENTATION_MODES:
            raise ValueError(f"representation must be one of {REPRESENTATION_MODES}")
        self.dataset = ResearchDataset(
            split_root, device=device, verify_hashes=verify_hashes
        )
        self.representation = representation
        self.mass_cover = float(mass_cover)
        self.relay_discount = float(relay_discount)
        self.task_type = task_type
        self.generator_model = generator_model
        self.random_state = int(random_state)
        self.last_result = None

    def _selected_sample_ids(self):
        output = []
        for sample_id in self.dataset.sample_ids:
            sample = self.dataset[sample_id]
            if self.task_type is not None and sample.task_type != self.task_type:
                continue
            if (
                self.generator_model is not None
                and sample.generator_model != self.generator_model
            ):
                continue
            output.append(sample_id)
        return output

    def collect(self, *, max_samples=None, max_nodes=None):
        """Collect one graph-derived vector per response token."""
        sample_ids = self._selected_sample_ids()
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
        feature_names = None

        for sample_id in sample_ids:
            sample = self.dataset[sample_id]
            node_features, names = response_representation(
                sample,
                self.representation,
                mass_cover=self.mass_cover,
                relay_discount=self.relay_discount,
            )
            count = len(node_features)
            if count == 0:
                sample.release_attention()
                continue
            features.append(node_features)
            node_sample_ids.append(np.full(count, str(sample_id), dtype=object))
            response_positions.append(np.arange(count, dtype=np.int32))
            task_types.append(np.full(count, sample.task_type, dtype=object))
            data_sources.append(np.full(count, sample.data_source, dtype=object))
            feature_names = tuple(names)
            sample.release_attention()

        if not features:
            raise ValueError("no response-token node states were collected")

        output = {
            "features": np.concatenate(features, axis=0),
            "sample_id": np.concatenate(node_sample_ids, axis=0),
            "response_position": np.concatenate(response_positions, axis=0),
            "task_type": np.concatenate(task_types, axis=0),
            "data_source": np.concatenate(data_sources, axis=0),
            "feature_names": np.asarray(feature_names),
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
                indices = np.sort(rng.choice(total_nodes, size=max_nodes, replace=False))
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
        """Fit one common t-SNE, then load labels only for presentation."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import RobustScaler

        result = self.collect(max_samples=max_samples, max_nodes=max_nodes)
        matrix = result["features"]
        if len(matrix) < 3:
            raise ValueError("t-SNE needs at least three node vectors")
        scaled = np.nan_to_num(RobustScaler().fit_transform(matrix))
        pca_dim = min(30, scaled.shape[1], max(len(scaled) - 1, 1))
        if pca_dim >= 2 and scaled.shape[1] > pca_dim:
            scaled = PCA(
                n_components=pca_dim, random_state=self.random_state
            ).fit_transform(scaled)
        actual_perplexity = min(float(perplexity), len(scaled) - 1.0)
        if actual_perplexity <= 0:
            raise ValueError("perplexity must be positive")
        result["coordinates"] = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            random_state=self.random_state,
        ).fit_transform(scaled)
        result["perplexity"] = actual_perplexity
        result["pca_dimensions"] = int(pca_dim)

        label_store = self.dataset.labels()
        labels_by_sample = {
            sample_id: label_store.response_labels(self.dataset[sample_id]).cpu().numpy()
            for sample_id in np.unique(result["sample_id"])
        }
        result["labels"] = np.asarray(
            [
                labels_by_sample[sample_id][position]
                for sample_id, position in zip(
                    result["sample_id"], result["response_position"], strict=True
                )
            ],
            dtype=np.int64,
        )
        self.last_result = result
        return result

    def plot(self, result=None, *, save_path=None, title=None):
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
                coordinates[normal, 0], coordinates[normal, 1],
                s=10, alpha=0.30, marker="o",
                label=f"Correct token (n={int(normal.sum())})", rasterized=True,
            )
        if anomaly.any():
            axis.scatter(
                coordinates[anomaly, 0], coordinates[anomaly, 1],
                s=22, alpha=0.85, marker="x",
                label=f"Hallucination token (n={int(anomaly.sum())})", rasterized=True,
            )
        axis.set(
            title=title or f"Node t-SNE: {self.representation}",
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
            "pca_dimensions": int(result["pca_dimensions"]),
            "random_state": self.random_state,
            "feature_names": result["feature_names"].tolist(),
            "node_representation": self.representation,
            "mass_cover": self.mass_cover,
            "relay_discount": self.relay_discount,
            "task_type_filter": self.task_type,
            "generator_model_filter": self.generator_model,
            "labels_used_for_fit": False,
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
        output_dir = Path(output_dir)
        result = self.fit(
            max_samples=max_samples, max_nodes=max_nodes, perplexity=perplexity
        )
        self.plot(result, save_path=output_dir / "node_tsne.png", title=title)
        metadata = self.save(result, output_dir=output_dir)
        return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--representation", choices=REPRESENTATION_MODES, default="basic12")
    parser.add_argument("--mass-cover", type=float, default=0.80)
    parser.add_argument("--relay-discount", type=float, default=0.85)
    parser.add_argument("--task-type")
    parser.add_argument("--generator-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    visualizer = NodeTSNEVisualizer(
        args.canonical_split,
        representation=args.representation,
        mass_cover=args.mass_cover,
        relay_discount=args.relay_discount,
        task_type=args.task_type,
        generator_model=args.generator_model,
        device=args.device,
        random_state=args.seed,
    )
    _, metadata = visualizer.run(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_nodes=args.max_nodes,
        perplexity=args.perplexity,
    )
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    main()
