"""Label-blind out-of-fold evaluation and fold-local embedding projection."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm

from anomaly import ConditionalStudentMixture, EmpiricalTailCalibrator
from attention_gnn import (
    RelationChannelAutoencoder,
    build_attention_graph,
    masked_view,
    reconstruction_loss,
)


class UnsupervisedGraphMethod:
    """Learn response-token graph embeddings, then fit an unlabeled density."""

    def __init__(
        self,
        *,
        num_channels,
        embedding_dim,
        message_passing_steps,
        graph_variant="full",
        epochs,
        fit_steps,
        edge_mask_rate=0.15,
        channel_mask_rate=0.15,
        seed=0,
    ):
        if num_channels < 1 or embedding_dim < 1 or message_passing_steps < 0:
            raise ValueError("model dimensions must be positive and steps non-negative")
        if graph_variant not in {"full", "rewired", "channel_mean"}:
            raise ValueError("unknown graph variant")
        if epochs < 1 or fit_steps < 1:
            raise ValueError("epochs and fit_steps must be positive")
        if not 0.0 <= edge_mask_rate <= 1.0 or not 0.0 <= channel_mask_rate <= 1.0:
            raise ValueError("mask rates must be in [0, 1]")
        self.num_channels = int(num_channels)
        self.embedding_dim = int(embedding_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.graph_variant = graph_variant
        self.epochs = int(epochs)
        self.fit_steps = int(fit_steps)
        self.edge_mask_rate = float(edge_mask_rate)
        self.channel_mask_rate = float(channel_mask_rate)
        self.seed = int(seed)
        self.model_channels = 1 if graph_variant == "channel_mean" else self.num_channels
        self.model = None
        self.density = None
        self.calibrator = None
        self.density_source_ids = ()
        self.calibration_source_ids = ()

    def fit(self, samples, *, progress=False):
        samples = list(samples)
        if not samples:
            raise ValueError("training samples must not be empty")
        first_attention = samples[0].attention()
        device = first_attention.response_values.device
        samples[0].release_attention()

        rng_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(self.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(self.seed)
            self.model = RelationChannelAutoencoder(
                num_channels=self.model_channels,
                embedding_dim=self.embedding_dim,
                message_passing_steps=self.message_passing_steps,
                dropout=0.0,
            ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        generator = torch.Generator(device=device).manual_seed(self.seed)
        self.model.train()
        epochs = tqdm(
            range(self.epochs),
            desc="self-supervised GNN",
            unit="epoch",
            leave=False,
            disable=not progress,
        )
        for epoch in epochs:
            running_loss = 0.0
            for sample in samples:
                graph = self._graph(sample, epoch=epoch)
                if graph.num_channels != self.model_channels:
                    raise ValueError("training graph channel count does not match num_channels")
                view = self._training_view(graph, generator)
                optimizer.zero_grad()
                loss = reconstruction_loss(self.model, graph, view)["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite reconstruction loss")
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach())
                sample.release_attention()
            epochs.set_postfix(loss=f"{running_loss / len(samples):.4f}")

        self.model.eval()
        self.model.requires_grad_(False)
        train_embeddings = self._embeddings(samples)
        density_indices, calibration_indices = self._density_partition(samples)
        self.density = ConditionalStudentMixture(
            fit_steps=self.fit_steps,
            seed=self.seed,
        ).fit([train_embeddings[index] for index in density_indices], progress=progress)
        calibration_scores = torch.cat(
            self.density.score([train_embeddings[index] for index in calibration_indices])
        )
        self.calibrator = EmpiricalTailCalibrator().fit(calibration_scores)
        return self

    def score(self, samples):
        if self.model is None or self.density is None or self.calibrator is None:
            raise RuntimeError("fit must be called before score")
        samples = list(samples)
        embeddings = self._embeddings(samples)
        nll = self.density.score(embeddings)
        scores = [
            -torch.log(self.calibrator.transform(values).clamp_min(1e-12))
            for values in nll
        ]
        return {
            sample.sample_id: {
                "embedding": embedding.detach().cpu().numpy(),
                "score": score.detach().cpu().numpy(),
                "nll": values.detach().cpu().numpy(),
            }
            for sample, embedding, score, values in zip(
                samples, embeddings, scores, nll, strict=True
            )
        }

    def embed(self, samples):
        if self.model is None:
            raise RuntimeError("fit must be called before embed")
        samples = list(samples)
        embeddings = self._embeddings(samples)
        return {
            sample.sample_id: embedding.detach().cpu().numpy()
            for sample, embedding in zip(samples, embeddings, strict=True)
        }

    def _training_view(self, graph, generator):
        target_relation = graph.edge_index[1] * 2 + graph.edge_type
        masked_edges = []
        for group in torch.unique(target_relation):
            members = torch.nonzero(target_relation == group, as_tuple=False).flatten()
            count = min(max(1, round(len(members) * self.edge_mask_rate)), len(members) - 1)
            if self.edge_mask_rate > 0.0 and count > 0:
                order = torch.randperm(
                    len(members), generator=generator, device=members.device
                )
                masked_edges.append(members[order[:count]])
        masked_edges = torch.cat(masked_edges) if masked_edges else None

        channel_count = min(
            round(graph.num_channels * self.channel_mask_rate), graph.num_channels - 1
        )
        masked_channels = None
        if self.channel_mask_rate > 0.0 and channel_count > 0:
            masked_channels = torch.randperm(
                graph.num_channels,
                generator=generator,
                device=graph.edge_channel.device,
            )[:channel_count]
        return masked_view(
            graph,
            masked_edges=masked_edges,
            masked_channels=masked_channels,
        )

    def _density_partition(self, samples):
        sources = sorted({sample.source_id for sample in samples})
        if len(sources) < 2:
            raise ValueError("density fitting requires at least two training source groups")
        rng = np.random.default_rng(self.seed)
        shuffled = np.asarray(sources, dtype=object)[rng.permutation(len(sources))]
        calibration_count = min(max(1, round(0.2 * len(sources))), len(sources) - 1)
        self.calibration_source_ids = tuple(shuffled[:calibration_count].tolist())
        self.density_source_ids = tuple(shuffled[calibration_count:].tolist())
        calibration = set(self.calibration_source_ids)
        density_indices = [
            index for index, sample in enumerate(samples) if sample.source_id not in calibration
        ]
        calibration_indices = [
            index for index, sample in enumerate(samples) if sample.source_id in calibration
        ]
        return density_indices, calibration_indices

    def _embeddings(self, samples):
        embeddings = []
        with torch.no_grad():
            for sample in samples:
                graph = self._graph(sample)
                if graph.num_channels != self.model_channels:
                    raise ValueError("graph channel count does not match fitted model")
                hidden = self.model.encode(graph, masked_view(graph))
                embeddings.append(hidden[graph.response_idx :].detach().float())
                sample.release_attention()
        return embeddings

    def _graph(self, sample, *, epoch=0):
        graph = build_attention_graph(sample.attention())
        if self.graph_variant == "rewired":
            graph = _rewire_sources(graph, self.seed + epoch)
        elif self.graph_variant == "channel_mean":
            mean_node = graph.node_attr.float().mean(dim=1, keepdim=True)
            edge_count = graph.edge_index.shape[1]
            graph = graph.__class__(
                **{
                    **graph.__dict__,
                    "num_channels": 1,
                    "node_attr": mean_node,
                    "edge_ptr": torch.arange(
                        edge_count + 1, device=graph.edge_ptr.device
                    ),
                    "edge_channel": torch.zeros(
                        edge_count, dtype=torch.long, device=graph.edge_channel.device
                    ),
                    "edge_value": graph.edge_weight,
                }
            )
        return graph


def _rewire_sources(graph, seed):
    """Permute sources within RP/RR groups while preserving causal validity."""
    source, target = graph.edge_index
    rewired = source.clone()
    generator = torch.Generator(device=source.device).manual_seed(seed)
    for relation in (0, 1):
        members = torch.nonzero(graph.edge_type == relation, as_tuple=False).flatten()
        for edge in members[torch.randperm(len(members), generator=generator, device=source.device)]:
            low = 0 if relation == 0 else graph.response_idx
            candidates = torch.arange(low, target[edge], device=source.device)
            occupied = rewired[(target == target[edge]) & (graph.edge_type == relation)]
            candidates = candidates[~torch.isin(candidates, occupied)]
            if len(candidates):
                rewired[edge] = candidates[
                    torch.randint(len(candidates), (), generator=generator, device=source.device)
                ]
    return graph.__class__(**{**graph.__dict__, "edge_index": torch.stack((rewired, target))})


class AllDataEvaluator:
    """Run one unsupervised fit per source-grouped fold."""

    def __init__(self, dataset, *, folds=5, seed=0):
        self.dataset = dataset
        self.folds = int(folds)
        self.seed = int(seed)

    def run(self, fit_fold):
        samples = [self.dataset[sample_id] for sample_id in self.dataset.sample_ids]
        source_ids = sorted({sample.source_id for sample in samples})
        if self.folds < 2 or self.folds > len(source_ids):
            raise ValueError("folds must be between 2 and the number of source groups")

        rng = np.random.default_rng(self.seed)
        shuffled_sources = np.asarray(source_ids, dtype=object)[rng.permutation(len(source_ids))]
        source_fold = {
            source_id: index % self.folds
            for index, source_id in enumerate(shuffled_sources.tolist())
        }

        records = []
        for fold in range(self.folds):
            heldout = [sample for sample in samples if source_fold[sample.source_id] == fold]
            train = [sample for sample in samples if source_fold[sample.source_id] != fold]
            outputs = fit_fold(train, heldout, fold)
            for sample in heldout:
                output = outputs[sample.sample_id]
                token_count = sample.attention().num_response_tokens
                sample.release_attention()
                if len(output["embedding"]) != token_count or len(output["score"]) != token_count:
                    raise ValueError("fold output must contain one embedding and score per response token")
                nll = output.get("nll")
                if nll is not None and len(nll) != token_count:
                    raise ValueError("fold output must contain one NLL per response token")
                for token_index, (embedding, score) in enumerate(
                    zip(output["embedding"], output["score"], strict=True)
                ):
                    record = {
                            "sample_id": sample.sample_id,
                            "source_id": sample.source_id,
                            "fold": fold,
                            "token_index": token_index,
                            "embedding": embedding,
                            "score": score,
                        }
                    for name in ("task_type", "data_source", "generator_model"):
                        record[name] = getattr(sample, name, None)
                    if nll is not None:
                        record["nll"] = nll[token_index]
                    records.append(record)
        return records

    def evaluate(self, records):
        labels = self.dataset.labels()
        by_sample = {}
        for sample_id in {row["sample_id"] for row in records}:
            sample = self.dataset[sample_id]
            by_sample[sample_id] = labels.response_labels(sample).cpu()
            sample.release_attention()
        return [
            {
                **row,
                "label": int(by_sample[row["sample_id"]][row["token_index"]]),
            }
            for row in records
        ]


class LearnedEmbeddingVisualizer:
    """Project held-out embeddings in a coordinate system fitted on train only."""

    def __init__(self, *, random_state=0):
        self.random_state = int(random_state)

    def project_fold(self, train_embeddings, heldout_embeddings):
        projector = PCA(n_components=2, random_state=self.random_state)
        projector.fit(np.asarray(train_embeddings))
        return projector.transform(np.asarray(heldout_embeddings))

    def plot_fold(self, train_embeddings, records, output_path):
        import matplotlib.pyplot as plt

        heldout = np.stack([row["embedding"] for row in records])
        coordinates = self.project_fold(train_embeddings, heldout)
        labels = np.asarray([row.get("label", -1) for row in records])
        scores = np.asarray([row["score"] for row in records])

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        score_plot = axes[0].scatter(
            coordinates[:, 0], coordinates[:, 1], c=scores, s=14, cmap="viridis"
        )
        figure.colorbar(score_plot, ax=axes[0], label="Unsupervised anomaly score")
        axes[0].set_title("Learned GNN node embeddings")
        for label, name, marker in ((0, "Correct", "o"), (1, "Hallucination", "x")):
            selected = labels == label
            if selected.any():
                axes[1].scatter(
                    coordinates[selected, 0], coordinates[selected, 1],
                    s=18, marker=marker, label=f"{name} (n={selected.sum()})",
                )
        axes[1].set_title("Labels opened after projection")
        axes[1].legend()
        for axis in axes:
            axis.set(xlabel="PCA 1", ylabel="PCA 2")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200)
        plt.close(figure)
        return {
            "coordinates": coordinates,
            "representation": "learned_gnn_node_embedding",
        }
