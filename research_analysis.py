"""Reusable structural analysis and visualization for canonical attention research data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from descriptors import temporal_summary
from research_dataset import (
    STRUCTURAL_FEATURE_NAMES,
    LabelStore,
    ResearchDataset,
    structural_features_from_edges,
)


GRAPH_FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES
DEFAULT_PLOT_FEATURES = (
    "prompt_mass_share",
    "history_edge_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "channel_edge_density",
)


def raw_attention_graph_features(attention, edges) -> torch.Tensor:
    """Backward-compatible wrapper around the canonical data-layer feature decoder."""
    return structural_features_from_edges(attention, edges)


def sample_graph_descriptor(sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Return response-node structural states and one mean/std/slope graph descriptor."""
    token_features = sample.structural_features()
    return token_features, temporal_summary(token_features)


def _fit_tsne(matrix, random_state=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("t-SNE needs at least three node vectors")
    scaled = StandardScaler().fit_transform(values)
    if scaled.shape[1] > 50:
        components = min(50, scaled.shape[0], scaled.shape[1])
        scaled = PCA(
            n_components=components, random_state=random_state
        ).fit_transform(scaled)
    perplexity = min(30.0, max(2.0, (len(scaled) - 1) / 3.0))
    perplexity = min(perplexity, len(scaled) - 1.0)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=random_state,
    ).fit_transform(scaled)
    return coordinates, perplexity


class SampleGraphVisualizer:
    """One-call inspection of token nodes, edges, node states, and matched controls."""

    def __init__(
        self,
        split_root,
        *,
        labels_path=None,
        device="cpu",
        verify_hashes=False,
        random_state=0,
    ):
        self.dataset = ResearchDataset(
            split_root, device=device, verify_hashes=verify_hashes
        )
        self.labels = self.dataset.label_store(labels_path)
        missing = set(self.dataset.sample_ids).difference(self.labels.rows)
        if missing:
            raise ValueError(
                f"labels are missing {len(missing)} dataset sample IDs; "
                f"first: {sorted(missing)[:3]}"
            )
        self.random_state = random_state
        self._response_length_cache = {}

    def is_hallucinated(self, sample_id) -> bool:
        return bool(self.labels.positive_runs(str(sample_id)))

    @property
    def error_sample_ids(self) -> list[str]:
        return [
            sid for sid in self.dataset.sample_ids if self.is_hallucinated(sid)
        ]

    @property
    def correct_sample_ids(self) -> list[str]:
        return [
            sid for sid in self.dataset.sample_ids if not self.is_hallucinated(sid)
        ]

    def _response_length(self, sample_id) -> int:
        sample_id = str(sample_id)
        if sample_id not in self._response_length_cache:
            self._response_length_cache[sample_id] = (
                self.dataset[sample_id].attention().num_response_tokens
            )
        return self._response_length_cache[sample_id]

    def list_errors(self, limit=20) -> list[dict]:
        rows = []
        for sample_id in self.error_sample_ids[:limit]:
            sample = self.dataset[sample_id]
            rows.append(
                {
                    **sample.metadata,
                    "response_tokens": self._response_length(sample_id),
                    "positive_runs": self.labels.positive_runs(sample_id),
                }
            )
        return rows

    def analyze(self, sample_id) -> dict:
        """Return one self-contained graph view plus its graph-level descriptor."""
        sample_id = str(sample_id)
        view = self.dataset[sample_id].graph_view(self.labels)
        view["descriptor"] = temporal_summary(view["response_features"])
        return view

    def match_correct(self, error_sample_id, *, max_candidates=256) -> str:
        """Prefer same source/task, then similar response length."""
        error_sample_id = str(error_sample_id)
        if not self.is_hallucinated(error_sample_id):
            raise ValueError(f"{error_sample_id} is not labeled hallucinated")

        correct_ids = self.correct_sample_ids
        if not correct_ids:
            raise ValueError("no fully correct sample is available for comparison")

        error_sample = self.dataset[error_sample_id]
        error_length = self._response_length(error_sample_id)
        groups = [
            [
                sid
                for sid in correct_ids
                if self.dataset[sid].source_id == error_sample.source_id
            ]
        ]
        if error_sample.task_type is not None and error_sample.data_source is not None:
            groups.append(
                [
                    sid
                    for sid in correct_ids
                    if self.dataset[sid].task_type == error_sample.task_type
                    and self.dataset[sid].data_source == error_sample.data_source
                ]
            )
        if error_sample.task_type is not None:
            groups.append(
                [
                    sid
                    for sid in correct_ids
                    if self.dataset[sid].task_type == error_sample.task_type
                ]
            )
        groups.append(correct_ids)
        candidates = next(group for group in groups if group)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        return min(
            candidates,
            key=lambda sid: (
                abs(self._response_length(sid) - error_length),
                str(sid),
            ),
        )

    def span_summary(self, sample_id, pre_window=8, post_window=8) -> list[dict]:
        result = self.analyze(sample_id)
        values = result["response_features"]
        output = []
        for run_index, (start, end) in enumerate(result["positive_runs"]):
            segments = {
                "pre": values[max(0, start - pre_window):start],
                "error": values[start:end],
                "post": values[end:min(len(values), end + post_window)],
            }
            means = {
                name: (
                    segment.mean(dim=0)
                    if len(segment)
                    else torch.full(
                        (values.shape[1],), float("nan"), dtype=values.dtype
                    )
                )
                for name, segment in segments.items()
            }
            for feature_index, feature_name in enumerate(GRAPH_FEATURE_NAMES):
                output.append(
                    {
                        "run": run_index,
                        "start": start,
                        "end": end,
                        "feature": feature_name,
                        "pre": float(means["pre"][feature_index]),
                        "error": float(means["error"][feature_index]),
                        "post": float(means["post"][feature_index]),
                        "error_minus_pre": float(
                            means["error"][feature_index]
                            - means["pre"][feature_index]
                        ),
                    }
                )
        return output

    @staticmethod
    def _feature_indices(features):
        names = tuple(features) if features is not None else DEFAULT_PLOT_FEATURES
        unknown = set(names).difference(GRAPH_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        return names, [GRAPH_FEATURE_NAMES.index(name) for name in names]

    @staticmethod
    def _select_relation_indices(
        relations,
        *,
        target_min=None,
        target_max=None,
        top_k_per_target=5,
        max_edges=300,
    ):
        target = relations["target"]
        weight = relations["weight"]
        if weight.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        mask = torch.ones(len(weight), dtype=torch.bool)
        if target_min is not None:
            mask &= target >= int(target_min)
        if target_max is not None:
            mask &= target < int(target_max)
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if not len(candidates):
            return candidates

        selected = []
        for value in torch.unique(target[candidates]):
            rows = candidates[target[candidates] == value]
            if top_k_per_target is not None and len(rows) > top_k_per_target:
                keep = torch.topk(weight[rows], top_k_per_target).indices
                rows = rows[keep]
            selected.append(rows)
        selected = torch.cat(selected) if selected else candidates[:0]
        if max_edges is not None and len(selected) > max_edges:
            keep = torch.topk(weight[selected], max_edges).indices
            selected = selected[keep]
        return selected

    def plot_token_graph(
        self,
        sample_id,
        *,
        top_k_per_target=4,
        max_edges=250,
        save_path=None,
    ):
        """Plot all token nodes on the original token axis with retained relation arcs."""
        import matplotlib.pyplot as plt

        result = self.analyze(sample_id)
        response_idx = result["response_idx"]
        num_tokens = result["num_tokens"]
        relations = result["relations"]
        selected = self._select_relation_indices(
            relations,
            top_k_per_target=top_k_per_target,
            max_edges=max_edges,
        )

        figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
        prompt_x = np.arange(response_idx)
        response_x = np.arange(response_idx, num_tokens)
        response_labels = result["response_labels"].numpy()

        if len(prompt_x):
            axis.scatter(
                prompt_x,
                np.zeros_like(prompt_x, dtype=float),
                s=18,
                label="Prompt token",
                zorder=3,
            )
        correct = response_x[response_labels == 0]
        error = response_x[response_labels == 1]
        if len(correct):
            axis.scatter(
                correct,
                np.zeros_like(correct, dtype=float),
                s=26,
                label="Correct response token",
                zorder=4,
            )
        if len(error):
            axis.scatter(
                error,
                np.zeros_like(error, dtype=float),
                s=34,
                marker="X",
                label="Hallucination token",
                zorder=5,
            )

        if len(selected):
            weights = relations["weight"][selected].numpy()
            max_weight = max(float(weights.max()), 1e-12)
            for index in selected.tolist():
                source = int(relations["source"][index])
                target = int(relations["target"][index])
                weight = float(relations["weight"][index])
                edge_type = int(relations["edge_type"][index])
                distance = target - source
                u = np.linspace(0.0, 1.0, 32)
                x = source + distance * u
                height = 0.10 + 1.15 * min(distance / max(num_tokens, 1), 0.5)
                y = height * 4.0 * u * (1.0 - u)
                axis.plot(
                    x,
                    y,
                    linewidth=0.5 + 2.0 * weight / max_weight,
                    alpha=0.15 + 0.55 * weight / max_weight,
                    linestyle="-" if edge_type == 0 else "--",
                    zorder=1,
                )

        for start, end in result["positive_runs"]:
            left = response_idx + start - 0.45
            right = response_idx + end - 0.55
            axis.axvspan(left, right, alpha=0.10, zorder=0)

        axis.axvline(
            response_idx - 0.5,
            linestyle=":",
            linewidth=1.2,
            label="Prompt / response boundary",
        )
        axis.set(
            title=f"Token graph — sample {result['sample_id']}",
            xlabel="Absolute token position",
            ylabel="Relation arc height",
            xlim=(-1, num_tokens),
        )
        axis.legend(loc="upper left", ncols=2)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure

    def plot_edge_heatmap(
        self,
        sample_id,
        *,
        center=None,
        radius=12,
        save_path=None,
    ):
        """Zoom on response targets and show their source-token relation weights."""
        import matplotlib.pyplot as plt

        result = self.analyze(sample_id)
        response_idx = result["response_idx"]
        response_count = result["num_response_tokens"]
        if center is None:
            center = (
                int(result["positive_runs"][0][0])
                if result["positive_runs"]
                else response_count // 2
            )
        center = int(center)
        start = max(0, center - radius)
        end = min(response_count, center + radius + 1)

        matrix = np.zeros((end - start, result["num_tokens"]), dtype=np.float32)
        relations = result["relations"]
        for source, target, weight in zip(
            relations["source"].tolist(),
            relations["target"].tolist(),
            relations["weight"].tolist(),
        ):
            relative_target = target - response_idx
            if start <= relative_target < end:
                matrix[relative_target - start, source] = weight

        figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(-0.5, result["num_tokens"] - 0.5, start - 0.5, end - 0.5),
        )
        axis.axvline(response_idx - 0.5, linestyle=":", linewidth=1.2)
        for error_start, error_end in result["positive_runs"]:
            overlap_start = max(start, error_start)
            overlap_end = min(end, error_end)
            if overlap_start < overlap_end:
                axis.axhspan(
                    overlap_start - 0.5,
                    overlap_end - 0.5,
                    alpha=0.12,
                )
        figure.colorbar(image, ax=axis, label="Mean-channel relation weight")
        axis.set(
            title=(
                f"Incoming relation heatmap — sample {result['sample_id']} "
                f"(response {start}:{end})"
            ),
            xlabel="Source token position (prompt left of boundary)",
            ylabel="Response target position",
        )
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure

    def plot_node_tsne(
        self,
        sample_id,
        *,
        mode="structure",
        connect_path=True,
        annotate_every=10,
        save_path=None,
    ):
        """Project token nodes; structure mode uses one 12-D response graph state per point."""
        import matplotlib.pyplot as plt

        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        result = self.analyze(sample_id)
        attention = sample.attention()

        if mode == "structure":
            matrix = result["response_features"].numpy()
            node_classes = result["response_labels"].numpy()
            node_positions = np.arange(len(matrix))
            class_specs = (
                (0, "Correct response token", "o"),
                (1, "Hallucination token", "X"),
            )
        elif mode == "attention":
            matrix = (
                sample.node_features("attention", attention=attention)
                .detach()
                .cpu()
                .numpy()
            )
            response_labels = result["response_labels"].numpy()
            node_classes = np.full(attention.num_tokens, -1, dtype=np.int64)
            node_classes[attention.response_idx:] = response_labels
            node_positions = np.arange(attention.num_tokens)
            class_specs = (
                (-1, "Prompt token", "o"),
                (0, "Correct response token", "o"),
                (1, "Hallucination token", "X"),
            )
        else:
            raise ValueError("mode must be 'structure' or 'attention'")

        coordinates, perplexity = _fit_tsne(
            matrix, random_state=self.random_state
        )
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        if connect_path:
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )
        for value, label, marker in class_specs:
            mask = node_classes == value
            if mask.any():
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    marker=marker,
                    s=34 if value != 1 else 50,
                    label=label,
                    zorder=3,
                )

        if annotate_every:
            for index in range(0, len(coordinates), int(annotate_every)):
                axis.annotate(
                    str(int(node_positions[index])),
                    coordinates[index],
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.75,
                )
        if mode == "structure":
            for start, _ in result["positive_runs"]:
                if 0 <= start < len(coordinates):
                    axis.annotate(
                        "error onset",
                        coordinates[start],
                        xytext=(8, 8),
                        textcoords="offset points",
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
                    )
        axis.set(
            title=(
                f"Node-level t-SNE ({mode}) — sample {sample_id}; "
                f"perplexity={perplexity:.2f}"
            ),
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        axis.legend()
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure, coordinates

    def plot_trajectories(self, sample_id, *, features=None, save_path=None):
        """Plot graph-state trajectories along response generation."""
        import matplotlib.pyplot as plt

        result = self.analyze(sample_id)
        names, indices = self._feature_indices(features)
        columns = 2
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 3.2 * rows),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(-1)
        x = np.arange(len(result["response_features"]))

        for axis, name, index in zip(axes, names, indices):
            axis.plot(x, result["response_features"][:, index].numpy())
            for start, end in result["positive_runs"]:
                axis.axvspan(start, max(start, end - 1), alpha=0.16)
            axis.set(title=name, xlabel="Response token position")
        for axis in axes[len(names):]:
            axis.set_visible(False)
        figure.suptitle(
            f"Graph-state trajectories — sample={result['sample_id']}"
        )
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure

    def plot(self, sample_id, *, features=None, save_path=None):
        """Backward-compatible alias for graph-state trajectory plotting."""
        return self.plot_trajectories(
            sample_id, features=features, save_path=save_path
        )

    def compare(
        self,
        error_sample_id,
        correct_sample_id=None,
        *,
        features=None,
        save_path=None,
    ):
        """Overlay a hallucinated sample and a matched fully correct control."""
        import matplotlib.pyplot as plt

        error_sample_id = str(error_sample_id)
        if not self.is_hallucinated(error_sample_id):
            raise ValueError(f"{error_sample_id} is not labeled hallucinated")
        correct_sample_id = (
            self.match_correct(error_sample_id)
            if correct_sample_id is None
            else str(correct_sample_id)
        )
        if self.is_hallucinated(correct_sample_id):
            raise ValueError(f"{correct_sample_id} is not a fully correct sample")

        error = self.analyze(error_sample_id)
        control = self.analyze(correct_sample_id)
        names, indices = self._feature_indices(features)
        columns = 2
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 3.2 * rows),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(-1)

        error_x = np.linspace(0.0, 1.0, len(error["response_features"]))
        control_x = np.linspace(0.0, 1.0, len(control["response_features"]))
        for axis, name, index in zip(axes, names, indices):
            axis.plot(
                error_x,
                error["response_features"][:, index].numpy(),
                label="Hallucinated",
            )
            axis.plot(
                control_x,
                control["response_features"][:, index].numpy(),
                label="Correct",
            )
            for start, end in error["positive_runs"]:
                denominator = max(len(error["response_features"]) - 1, 1)
                axis.axvspan(
                    start / denominator,
                    (end - 1) / denominator,
                    alpha=0.16,
                )
            axis.set(title=name, xlabel="Normalized response position")
            axis.legend()
        for axis in axes[len(names):]:
            axis.set_visible(False)
        figure.suptitle(
            f"error={error_sample_id} vs correct={correct_sample_id}"
        )
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure, correct_sample_id

    def visualize(
        self,
        sample_id,
        *,
        correct_sample_id=None,
        output_dir=None,
        local_radius=12,
        node_tsne_mode="structure",
        include_control=True,
    ):
        """Generate the full single-sample diagnostic bundle in one call."""
        sample_id = str(sample_id)
        output_dir = Path(output_dir) if output_dir is not None else None

        def path(name):
            if output_dir is None:
                return None
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{sample_id}_{name}.png"

        figures = {}
        figures["token_graph"] = self.plot_token_graph(
            sample_id, save_path=path("token_graph")
        )
        figures["edge_heatmap"] = self.plot_edge_heatmap(
            sample_id,
            radius=local_radius,
            save_path=path("edge_heatmap"),
        )
        figures["node_tsne"], node_coordinates = self.plot_node_tsne(
            sample_id,
            mode=node_tsne_mode,
            save_path=path(f"node_tsne_{node_tsne_mode}"),
        )
        figures["trajectories"] = self.plot_trajectories(
            sample_id, save_path=path("trajectories")
        )

        matched_control = None
        if include_control and self.is_hallucinated(sample_id):
            figures["control_compare"], matched_control = self.compare(
                sample_id,
                correct_sample_id=correct_sample_id,
                save_path=path("control_compare"),
            )

        return {
            "sample_id": sample_id,
            "control_sample_id": matched_control,
            "positive_runs": self.labels.positive_runs(sample_id),
            "span_summary": self.span_summary(sample_id),
            "node_tsne_coordinates": node_coordinates,
            "figures": figures,
        }


# Backward compatibility with the earlier notebook/API name.
SampleBehaviorVisualizer = SampleGraphVisualizer
