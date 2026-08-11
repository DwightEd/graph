"""Rich single-sample structural projection diagnostics.

This module keeps the projection itself label-free. Labels are loaded only after
the node representations and low-dimensional coordinates have been computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_dataset import ResearchDataset


RICH_RESPONSE_FEATURE_NAMES = (
    "incoming_mass",
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_degree",
    "prompt_degree",
    "history_degree",
    "in_density",
    "prompt_density",
    "history_density",
    "history_edge_share",
    "channel_edge_density",
    "prompt_mass",
    "history_mass",
    "prompt_entropy",
    "history_entropy",
    "top1_share",
    "top3_share",
    "hhi",
    "prompt_top1_share",
    "history_top1_share",
    "history_lag_std",
    "history_near1_share",
    "history_near4_share",
    "history_near8_share",
    "history_far16_share",
    "early_prompt_mass",
    "middle_prompt_mass",
    "late_prompt_mass",
    "early_history_mass",
    "middle_history_mass",
    "late_history_mass",
)

SOURCE_ROLE_FEATURE_NAMES = (
    "outgoing_mass",
    "out_degree",
    "out_channel_density",
    "target_position_mean",
    "target_position_std",
    "target_span",
    "outgoing_entropy",
    "top1_share",
    "hhi",
    "early_target_mass_share",
    "middle_target_mass_share",
    "late_target_mass_share",
)

PHASE_NAMES = ("far_normal", "pre_error", "hallucination", "post_error")

HEATMAP_FEATURES = (
    "prompt_mass",
    "history_mass",
    "prompt_mass_share",
    "history_edge_share",
    "prompt_density",
    "history_density",
    "normalized_entropy",
    "top1_share",
    "top3_share",
    "hhi",
    "history_lag",
    "history_lag_std",
    "history_near1_share",
    "history_near4_share",
    "history_near8_share",
    "channel_edge_density",
    "early_prompt_mass",
    "middle_prompt_mass",
    "late_prompt_mass",
    "early_history_mass",
    "middle_history_mass",
    "late_history_mass",
)


def _normalized_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[weights > 0]
    if len(weights) <= 1:
        return 0.0
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(probabilities)))


def _top_share(weights: np.ndarray, k: int) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or len(weights) == 0:
        return 0.0
    k = min(int(k), len(weights))
    return float(np.partition(weights, len(weights) - k)[-k:].sum() / total)


def _hhi(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    probabilities = weights / total
    return float(np.square(probabilities).sum())


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or len(values) == 0:
        return 0.0, 0.0
    mean = float(np.sum(values * weights) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    return mean, float(np.sqrt(max(variance, 0.0)))


def _layer_groups(num_layers: int) -> tuple[np.ndarray, np.ndarray]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    group = np.minimum(2, np.arange(num_layers, dtype=np.int64) * 3 // num_layers)
    counts = np.bincount(group, minlength=3)
    return group, counts


def rich_response_features(sample) -> np.ndarray:
    """Return a causal, incoming-only rich state for every response token.

    The 32-D representation extends the original 12 structural statistics with
    separate prompt/history mass, subset concentration, local-history distance
    bands, and early/middle/late layer-group routing. It uses only incoming
    relations available at each response position; it does not use future edges.
    """
    attention = sample.attention()
    relations = sample.relation_edges()
    raw_edges = sample.attention_edges()

    response_idx = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    num_layers = int(attention.num_layers)
    num_heads = int(attention.num_heads)

    rel_source = relations["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    rel_target = relations["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    rel_weight = relations["weight"].detach().cpu().numpy().astype(np.float64, copy=False)
    rel_channels = relations["channel_count"].detach().cpu().numpy().astype(np.float64, copy=False)

    raw_source = raw_edges["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_target = raw_edges["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_layer = raw_edges["layer"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_weight = raw_edges["weight"].detach().cpu().numpy().astype(np.float64, copy=False)

    layer_group, layer_counts = _layer_groups(num_layers)
    group_channels = layer_counts * num_heads

    matrix = np.zeros((response_count, len(RICH_RESPONSE_FEATURE_NAMES)), dtype=np.float32)
    lag_norm = float(max(response_count - 1, 1))

    for row in range(response_count):
        target_abs = response_idx + row
        mask = rel_target == target_abs
        sources = rel_source[mask]
        weights = rel_weight[mask]
        channels = rel_channels[mask]

        prompt = sources < response_idx
        history = ~prompt
        prompt_weights = weights[prompt]
        history_weights = weights[history]

        total_mass = float(weights.sum())
        prompt_mass = float(prompt_weights.sum())
        history_mass = float(history_weights.sum())

        in_degree = float(len(weights))
        prompt_degree = float(prompt.sum())
        history_degree = float(history.sum())
        absolute_target = float(max(target_abs, 1))

        prompt_share = prompt_mass / total_mass if total_mass > 0 else 0.0
        history_share = history_degree / in_degree if in_degree > 0 else 0.0

        history_lag = 0.0
        history_lag_std = 0.0
        near1 = near4 = near8 = far16 = 0.0
        if history_mass > 0:
            lags = (target_abs - sources[history]).astype(np.float64)
            history_lag, history_lag_std = _weighted_mean_std(lags / lag_norm, history_weights)
            near1 = float(history_weights[lags <= 1].sum() / history_mass)
            near4 = float(history_weights[lags <= 4].sum() / history_mass)
            near8 = float(history_weights[lags <= 8].sum() / history_mass)
            far16 = float(history_weights[lags > 16].sum() / history_mass)

        raw_mask = raw_target == target_abs
        target_raw_source = raw_source[raw_mask]
        target_raw_layer = raw_layer[raw_mask]
        target_raw_weight = raw_weight[raw_mask]
        grouped = np.zeros((3, 2), dtype=np.float64)
        for group_id in range(3):
            if group_channels[group_id] <= 0:
                continue
            group_mask = layer_group[target_raw_layer] == group_id
            if not group_mask.any():
                continue
            group_source = target_raw_source[group_mask]
            group_weight = target_raw_weight[group_mask]
            grouped[group_id, 0] = group_weight[group_source < response_idx].sum() / group_channels[group_id]
            grouped[group_id, 1] = group_weight[group_source >= response_idx].sum() / group_channels[group_id]

        channel_degree = float(channels.sum())
        channel_density = channel_degree / (float(attention.num_channels) * absolute_target)

        values = (
            total_mass,
            prompt_share,
            _normalized_entropy(weights),
            history_lag,
            in_degree,
            prompt_degree,
            history_degree,
            in_degree / absolute_target,
            prompt_degree / float(max(response_idx, 1)),
            history_degree / float(row) if row > 0 else 0.0,
            history_share,
            channel_density,
            prompt_mass,
            history_mass,
            _normalized_entropy(prompt_weights),
            _normalized_entropy(history_weights),
            _top_share(weights, 1),
            _top_share(weights, 3),
            _hhi(weights),
            _top_share(prompt_weights, 1),
            _top_share(history_weights, 1),
            history_lag_std,
            near1,
            near4,
            near8,
            far16,
            grouped[0, 0],
            grouped[1, 0],
            grouped[2, 0],
            grouped[0, 1],
            grouped[1, 1],
            grouped[2, 1],
        )
        matrix[row] = np.asarray(values, dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError("rich response features must be finite")
    return matrix


def source_role_features(sample) -> np.ndarray:
    """Return descriptive source-role graph statistics for every prompt/response token.

    These features summarize how each token is used by later response queries.
    They are intentionally descriptive and future-looking, so they are suitable
    for visualization but not for online causal detection.
    """
    attention = sample.attention()
    relations = sample.relation_edges()

    num_tokens = int(attention.num_tokens)
    response_idx = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    num_channels = int(attention.num_channels)

    source = relations["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    target = relations["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    weight = relations["weight"].detach().cpu().numpy().astype(np.float64, copy=False)
    channel_count = relations["channel_count"].detach().cpu().numpy().astype(np.float64, copy=False)

    matrix = np.zeros((num_tokens, len(SOURCE_ROLE_FEATURE_NAMES)), dtype=np.float32)
    position_norm = float(max(response_count - 1, 1))

    for node in range(num_tokens):
        mask = source == node
        weights = weight[mask]
        targets = target[mask]
        counts = channel_count[mask]
        total = float(weights.sum())
        degree = float(len(weights))

        if node < response_idx:
            possible_targets = response_count
        else:
            response_position = node - response_idx
            possible_targets = max(response_count - response_position - 1, 0)

        target_position = (targets - response_idx).astype(np.float64) / position_norm
        mean_position, std_position = _weighted_mean_std(target_position, weights)
        target_span = (
            float((target_position.max() - target_position.min()))
            if len(target_position) > 1
            else 0.0
        )
        out_channel_density = (
            float(counts.sum()) / float(num_channels * possible_targets)
            if possible_targets > 0
            else 0.0
        )

        stage_share = np.zeros(3, dtype=np.float64)
        if total > 0:
            stages = np.minimum(2, ((targets - response_idx) * 3 // max(response_count, 1)).astype(np.int64))
            for stage in range(3):
                stage_share[stage] = weights[stages == stage].sum() / total

        values = (
            total,
            degree,
            out_channel_density,
            mean_position,
            std_position,
            target_span,
            _normalized_entropy(weights),
            _top_share(weights, 1),
            _hhi(weights),
            stage_share[0],
            stage_share[1],
            stage_share[2],
        )
        matrix[node] = np.asarray(values, dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError("source-role features must be finite")
    return matrix


def response_phase_labels(response_count: int, positive_runs, pre_window=10, post_window=10) -> np.ndarray:
    """Return 0 far-normal, 1 pre-error, 2 hallucination, 3 post-error labels."""
    response_count = int(response_count)
    pre_window = int(pre_window)
    post_window = int(post_window)
    if response_count < 1:
        return np.empty(0, dtype=np.int64)
    if pre_window < 0 or post_window < 0:
        raise ValueError("pre_window and post_window must be non-negative")

    phases = np.zeros(response_count, dtype=np.int64)
    runs = [(int(start), int(end)) for start, end in positive_runs]
    for start, end in runs:
        if not (0 <= start < end <= response_count):
            raise ValueError("positive runs must be response-relative [start, end) intervals")
        phases[start:end] = 2
    for start, _ in runs:
        left = max(0, start - pre_window)
        mask = phases[left:start] == 0
        phases[left:start][mask] = 1
    for _, end in runs:
        right = min(response_count, end + post_window)
        mask = phases[end:right] == 0
        phases[end:right][mask] = 3
    return phases


def _project(matrix, *, random_state=0, perplexity=None, pca_components=20, scaler="robust"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import RobustScaler, StandardScaler

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("projection needs at least three node vectors")

    if scaler == "robust":
        scaled = RobustScaler().fit_transform(values)
    elif scaler == "standard":
        scaled = StandardScaler().fit_transform(values)
    else:
        raise ValueError("scaler must be 'robust' or 'standard'")

    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if pca_components is not None and scaled.shape[1] > int(pca_components):
        components = min(int(pca_components), scaled.shape[1], len(scaled) - 1)
        if components >= 2:
            scaled = PCA(n_components=components, random_state=random_state).fit_transform(scaled)

    if perplexity is None:
        perplexity = min(30.0, max(5.0, float(np.sqrt(len(scaled)))))
    actual_perplexity = min(float(perplexity), len(scaled) - 1.0)
    if actual_perplexity <= 0:
        raise ValueError("perplexity must be positive")

    coordinates = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=int(random_state),
    ).fit_transform(scaled)
    return coordinates, actual_perplexity, scaled


class RichSampleVisualizer:
    """Advanced single-response structural visualization without learned encoders."""

    def __init__(
        self,
        split_root,
        *,
        output_root=None,
        device="cpu",
        verify_hashes=False,
        random_state=0,
    ):
        self.dataset = ResearchDataset(
            split_root, device=device, verify_hashes=verify_hashes
        )
        self.labels = self.dataset.labels()
        self.output_root = Path(output_root) if output_root is not None else None
        self.random_state = int(random_state)

    @property
    def error_sample_ids(self):
        return [
            sample_id
            for sample_id in self.dataset.sample_ids
            if self.labels.positive_runs(sample_id)
        ]

    @property
    def correct_sample_ids(self):
        return [
            sample_id
            for sample_id in self.dataset.sample_ids
            if not self.labels.positive_runs(sample_id)
        ]

    def list_errors(self, limit=20):
        rows = []
        for sample_id in self.error_sample_ids[: int(limit)]:
            sample = self.dataset[sample_id]
            attention = sample.attention()
            rows.append(
                {
                    **sample.metadata,
                    "response_tokens": attention.num_response_tokens,
                    "positive_runs": self.labels.positive_runs(sample_id),
                }
            )
        return rows

    def _output_dir(self, sample_id, output_dir=None):
        if output_dir is not None:
            path = Path(output_dir)
        elif self.output_root is not None:
            path = self.output_root / str(sample_id)
        else:
            return None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _response_feature_state(self, sample_id):
        """Load the representation without touching hallucination labels."""
        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        features = rich_response_features(sample)
        return {
            "sample_id": sample_id,
            "sample": sample,
            "features": features,
            "feature_names": np.asarray(RICH_RESPONSE_FEATURE_NAMES),
            "response_position": np.arange(len(features), dtype=np.int32),
        }

    def response_state(self, sample_id, *, pre_window=10, post_window=10):
        """Return rich features plus evaluation-only phase annotations."""
        state = self._response_feature_state(sample_id)
        positive_runs = self.labels.positive_runs(state["sample_id"])
        state["positive_runs"] = positive_runs
        state["phases"] = response_phase_labels(
            len(state["features"]),
            positive_runs,
            pre_window=pre_window,
            post_window=post_window,
        )
        return state

    def fit_response_projection(
        self,
        sample_id,
        *,
        pre_window=10,
        post_window=10,
        perplexity=None,
        scaler="robust",
        pca_components=20,
        random_state=None,
    ):
        state = self._response_feature_state(sample_id)
        seed = self.random_state if random_state is None else int(random_state)
        coordinates, actual_perplexity, scaled = _project(
            state["features"],
            random_state=seed,
            perplexity=perplexity,
            pca_components=pca_components,
            scaler=scaler,
        )
        # Labels are deliberately loaded only after coordinates exist.
        positive_runs = self.labels.positive_runs(state["sample_id"])
        phases = response_phase_labels(
            len(state["features"]),
            positive_runs,
            pre_window=pre_window,
            post_window=post_window,
        )
        state.update(
            coordinates=coordinates,
            scaled_features=scaled,
            perplexity=actual_perplexity,
            scaler=scaler,
            positive_runs=positive_runs,
            phases=phases,
        )
        return state

    @staticmethod
    def _scatter_phases(axis, coordinates, phases):
        specs = (
            (0, "Far normal", "o", 34, 0.55),
            (1, "Pre-error", "^", 50, 0.85),
            (2, "Hallucination", "X", 75, 0.95),
            (3, "Post-error", "s", 46, 0.75),
        )
        for value, label, marker, size, alpha in specs:
            mask = phases == value
            if mask.any():
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    label=f"{label} (n={int(mask.sum())})",
                )

    def plot_response_projection(
        self,
        sample_id,
        *,
        pre_window=10,
        post_window=10,
        perplexity=None,
        scaler="robust",
        pca_components=20,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        result = self.fit_response_projection(
            sample_id,
            pre_window=pre_window,
            post_window=post_window,
            perplexity=perplexity,
            scaler=scaler,
            pca_components=pca_components,
        )
        coordinates = result["coordinates"]
        phases = result["phases"]
        features = result["features"]
        names = list(RICH_RESPONSE_FEATURE_NAMES)

        figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
        axes = axes.reshape(-1)

        self._scatter_phases(axes[0], coordinates, phases)
        axes[0].set_title("Graph-state phases")
        axes[0].legend()

        position = result["response_position"] / max(len(coordinates) - 1, 1)
        scatter = axes[1].scatter(
            coordinates[:, 0], coordinates[:, 1], c=position, s=38, alpha=0.8
        )
        figure.colorbar(scatter, ax=axes[1], label="Normalized response position")
        axes[1].set_title("Position confound check")

        prompt_share = features[:, names.index("prompt_mass_share")]
        scatter = axes[2].scatter(
            coordinates[:, 0], coordinates[:, 1], c=prompt_share, s=38, alpha=0.8
        )
        figure.colorbar(scatter, ax=axes[2], label="Prompt mass share")
        axes[2].set_title("Prompt grounding")

        hhi = features[:, names.index("hhi")]
        scatter = axes[3].scatter(
            coordinates[:, 0], coordinates[:, 1], c=hhi, s=38, alpha=0.8
        )
        figure.colorbar(scatter, ax=axes[3], label="HHI concentration")
        axes[3].set_title("Attention concentration")

        for axis in axes:
            axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")

        figure.suptitle(
            f"Rich response-node projection — sample {sample_id}; "
            f"perplexity={result['perplexity']:.2f}"
        )

        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "rich_response_tsne.png", dpi=220, bbox_inches="tight")
            np.savez_compressed(
                out / "rich_response_projection.npz",
                coordinates=coordinates,
                features=features,
                scaled_features=result["scaled_features"],
                phases=phases,
                response_position=result["response_position"],
                feature_names=result["feature_names"],
            )
        return figure, result

    def plot_projection_stability(
        self,
        sample_id,
        *,
        pre_window=10,
        post_window=10,
        perplexities=(5.0, 15.0, 30.0),
        scaler="robust",
        pca_components=20,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        base = self._response_feature_state(sample_id)
        valid = [
            float(value)
            for value in perplexities
            if 0 < float(value) < len(base["features"])
        ]
        if not valid:
            raise ValueError("no valid perplexity is smaller than the node count")

        # Phase labels are post-hoc annotations, not projection inputs.
        positive_runs = self.labels.positive_runs(str(sample_id))
        phases = response_phase_labels(
            len(base["features"]),
            positive_runs,
            pre_window=pre_window,
            post_window=post_window,
        )
        figure, axes = plt.subplots(
            1, len(valid), figsize=(5.2 * len(valid), 4.8), constrained_layout=True
        )
        axes = np.asarray(axes).reshape(-1)
        results = {}
        for axis, perplexity in zip(axes, valid):
            coordinates, actual, _ = _project(
                base["features"],
                random_state=self.random_state,
                perplexity=perplexity,
                pca_components=pca_components,
                scaler=scaler,
            )
            self._scatter_phases(axis, coordinates, phases)
            axis.set(
                title=f"perplexity={actual:g}",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
            )
            results[actual] = coordinates
        axes[0].legend()
        figure.suptitle(f"Projection stability — sample {sample_id}")

        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "projection_stability.png", dpi=220, bbox_inches="tight")
        return figure, results

    def plot_source_role_projection(
        self,
        sample_id,
        *,
        perplexity=None,
        scaler="robust",
        pca_components=10,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        attention = sample.attention()
        matrix = source_role_features(sample)
        coordinates, actual, _ = _project(
            matrix,
            random_state=self.random_state,
            perplexity=perplexity,
            pca_components=pca_components,
            scaler=scaler,
        )

        response_labels = self.labels.response_labels(sample).cpu().numpy()
        classes = np.full(attention.num_tokens, -1, dtype=np.int64)
        classes[attention.response_idx:] = response_labels
        absolute_position = np.arange(attention.num_tokens)

        figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
        specs = (
            (-1, "Prompt token", "o", 24, 0.4),
            (0, "Correct response token", "o", 36, 0.7),
            (1, "Hallucination token", "X", 65, 0.95),
        )
        for value, label, marker, size, alpha in specs:
            mask = classes == value
            if mask.any():
                axes[0].scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    label=f"{label} (n={int(mask.sum())})",
                )
        axes[0].legend()
        axes[0].set_title("Prompt / response source roles")

        position = absolute_position / max(attention.num_tokens - 1, 1)
        scatter = axes[1].scatter(
            coordinates[:, 0], coordinates[:, 1], c=position, s=28, alpha=0.7
        )
        figure.colorbar(scatter, ax=axes[1], label="Normalized absolute token position")
        axes[1].set_title("Position confound check")

        for axis in axes:
            axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        figure.suptitle(
            f"All-token source-role projection — sample {sample_id}; perplexity={actual:.2f}"
        )

        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "source_role_tsne.png", dpi=220, bbox_inches="tight")
        return figure, coordinates

    def plot_feature_heatmap(
        self,
        sample_id,
        *,
        pre_window=12,
        post_window=12,
        radius=18,
        feature_names=HEATMAP_FEATURES,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import RobustScaler

        state = self.response_state(
            sample_id, pre_window=pre_window, post_window=post_window
        )
        features = state["features"]
        names = list(RICH_RESPONSE_FEATURE_NAMES)
        selected_names = [name for name in feature_names if name in names]
        indices = [names.index(name) for name in selected_names]

        scaled = RobustScaler().fit_transform(features[:, indices]).T
        runs = state["positive_runs"]
        center = runs[0][0] if runs else len(features) // 2
        start = max(0, center - int(radius))
        end = min(len(features), center + int(radius) + 1)
        window = scaled[:, start:end]

        figure, axis = plt.subplots(
            figsize=(13, max(7, 0.34 * len(selected_names))), constrained_layout=True
        )
        image = axis.imshow(
            window,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(start - 0.5, end - 0.5, len(selected_names) - 0.5, -0.5),
        )
        axis.set_yticks(np.arange(len(selected_names)))
        axis.set_yticklabels(selected_names)
        for error_start, error_end in runs:
            left = max(start, error_start)
            right = min(end, error_end)
            if left < right:
                axis.axvspan(left - 0.5, right - 0.5, alpha=0.12)
        axis.axvline(center - 0.5, linestyle=":", linewidth=1.2)
        figure.colorbar(image, ax=axis, label="Robust-scaled feature value")
        axis.set(
            title=f"Rich graph-state heatmap around first error onset — sample {sample_id}",
            xlabel="Response token position",
            ylabel="Structural feature",
        )

        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "rich_feature_heatmap.png", dpi=220, bbox_inches="tight")
        return figure

    def feature_differences(
        self,
        sample_id,
        *,
        pre_window=10,
        post_window=10,
    ):
        from sklearn.preprocessing import StandardScaler

        state = self.response_state(
            sample_id, pre_window=pre_window, post_window=post_window
        )
        if not state["positive_runs"]:
            raise ValueError("sample has no hallucination span")
        start, end = state["positive_runs"][0]
        pre_start = max(0, start - int(pre_window))
        scaled = StandardScaler().fit_transform(state["features"])
        pre = scaled[pre_start:start]
        error = scaled[start:end]
        if len(pre) == 0 or len(error) == 0:
            raise ValueError("pre-error and hallucination segments must both be non-empty")
        difference = error.mean(axis=0) - pre.mean(axis=0)
        return {
            "feature_names": np.asarray(RICH_RESPONSE_FEATURE_NAMES),
            "difference": difference,
            "pre_range": (pre_start, start),
            "error_range": (start, end),
        }

    def plot_feature_differences(
        self,
        sample_id,
        *,
        pre_window=10,
        top_k=16,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        result = self.feature_differences(sample_id, pre_window=pre_window)
        order = np.argsort(np.abs(result["difference"]))[::-1][: int(top_k)]
        names = result["feature_names"][order]
        values = result["difference"][order]

        figure, axis = plt.subplots(
            figsize=(9, max(5, 0.38 * len(order))), constrained_layout=True
        )
        y = np.arange(len(order))
        axis.barh(y, values)
        axis.axvline(0.0, linewidth=1.0)
        axis.set_yticks(y)
        axis.set_yticklabels(names)
        axis.invert_yaxis()
        axis.set(
            title=(
                f"Hallucination minus pre-error structural change — sample {sample_id}"
            ),
            xlabel="Difference in sample-standardized feature units",
        )

        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "rich_feature_differences.png", dpi=220, bbox_inches="tight")
        return figure, result

    def separation_metrics(
        self,
        sample_id,
        *,
        pre_window=10,
        post_window=10,
    ):
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import RobustScaler

        state = self.response_state(
            sample_id, pre_window=pre_window, post_window=post_window
        )
        scaled = RobustScaler().fit_transform(state["features"])
        phases = state["phases"]
        hallucination = phases == 2
        normal = phases != 2
        pre = phases == 1

        metrics = {
            "hallucination_nodes": int(hallucination.sum()),
            "pre_error_nodes": int(pre.sum()),
            "normal_nodes": int(normal.sum()),
            "silhouette_error_vs_normal": None,
            "centroid_distance_error_vs_pre": None,
        }
        if hallucination.sum() >= 2 and normal.sum() >= 2:
            binary = hallucination.astype(np.int64)
            metrics["silhouette_error_vs_normal"] = float(
                silhouette_score(scaled, binary)
            )
        if hallucination.any() and pre.any():
            metrics["centroid_distance_error_vs_pre"] = float(
                np.linalg.norm(
                    scaled[hallucination].mean(axis=0) - scaled[pre].mean(axis=0)
                )
            )
        return metrics

    def match_correct(self, error_sample_id, *, max_candidates=256):
        error_sample_id = str(error_sample_id)
        if not self.labels.positive_runs(error_sample_id):
            raise ValueError("error_sample_id must contain a hallucination span")

        correct_ids = self.correct_sample_ids
        if not correct_ids:
            raise ValueError("no fully correct control is available")

        error = self.dataset[error_sample_id]
        error_length = error.attention().num_response_tokens
        groups = [
            [sid for sid in correct_ids if self.dataset[sid].source_id == error.source_id]
        ]
        if error.task_type is not None and error.data_source is not None:
            groups.append(
                [
                    sid
                    for sid in correct_ids
                    if self.dataset[sid].task_type == error.task_type
                    and self.dataset[sid].data_source == error.data_source
                ]
            )
        if error.task_type is not None:
            groups.append(
                [sid for sid in correct_ids if self.dataset[sid].task_type == error.task_type]
            )
        groups.append(correct_ids)

        candidates = next(group for group in groups if group)[: int(max_candidates)]
        return min(
            candidates,
            key=lambda sid: (
                abs(self.dataset[sid].attention().num_response_tokens - error_length),
                str(sid),
            ),
        )

    def plot_matched_control_projection(
        self,
        error_sample_id,
        *,
        correct_sample_id=None,
        pre_window=10,
        post_window=10,
        perplexity=None,
        scaler="robust",
        pca_components=20,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        error_sample_id = str(error_sample_id)
        if correct_sample_id is None:
            correct_sample_id = self.match_correct(error_sample_id)
        correct_sample_id = str(correct_sample_id)

        error_state = self._response_feature_state(error_sample_id)
        correct_features = rich_response_features(self.dataset[correct_sample_id])

        pooled = np.concatenate((error_state["features"], correct_features), axis=0)
        coordinates, actual, _ = _project(
            pooled,
            random_state=self.random_state,
            perplexity=perplexity,
            pca_components=pca_components,
            scaler=scaler,
        )
        split = len(error_state["features"])
        error_coordinates = coordinates[:split]
        control_coordinates = coordinates[split:]
        positive_runs = self.labels.positive_runs(error_sample_id)
        phases = response_phase_labels(
            split, positive_runs, pre_window=pre_window, post_window=post_window
        )

        figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
        self._scatter_phases(axes[0], error_coordinates, phases)
        axes[0].set_title(f"Hallucinated sample {error_sample_id}")
        axes[0].legend()
        axes[1].scatter(
            control_coordinates[:, 0],
            control_coordinates[:, 1],
            s=38,
            alpha=0.65,
            marker="o",
            label=f"Correct control (n={len(control_coordinates)})",
        )
        axes[1].set_title(f"Matched correct sample {correct_sample_id}")
        axes[1].legend()
        for axis in axes:
            axis.set(xlabel="joint t-SNE 1", ylabel="joint t-SNE 2")
        figure.suptitle(f"Joint rich-node projection; perplexity={actual:.2f}")

        out = self._output_dir(error_sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "matched_control_joint_tsne.png", dpi=220, bbox_inches="tight")
        return figure, correct_sample_id

    def visualize(
        self,
        sample_id,
        *,
        output_dir=None,
        pre_window=10,
        post_window=10,
        heatmap_radius=18,
        include_control=True,
    ):
        sample_id = str(sample_id)
        out = self._output_dir(sample_id, output_dir)

        response_figure, response_result = self.plot_response_projection(
            sample_id,
            pre_window=pre_window,
            post_window=post_window,
            output_dir=out,
        )
        stability_figure, stability = self.plot_projection_stability(
            sample_id,
            pre_window=pre_window,
            post_window=post_window,
            output_dir=out,
        )
        source_figure, source_coordinates = self.plot_source_role_projection(
            sample_id, output_dir=out
        )
        heatmap_figure = self.plot_feature_heatmap(
            sample_id,
            pre_window=pre_window,
            post_window=post_window,
            radius=heatmap_radius,
            output_dir=out,
        )
        difference_figure, differences = self.plot_feature_differences(
            sample_id, pre_window=pre_window, output_dir=out
        )
        metrics = self.separation_metrics(
            sample_id, pre_window=pre_window, post_window=post_window
        )

        control_figure = None
        control_sample_id = None
        if include_control and self.labels.positive_runs(sample_id):
            control_figure, control_sample_id = self.plot_matched_control_projection(
                sample_id,
                pre_window=pre_window,
                post_window=post_window,
                output_dir=out,
            )

        metadata = {
            "sample_id": sample_id,
            "positive_runs": self.labels.positive_runs(sample_id),
            "response_nodes": int(len(response_result["features"])),
            "rich_response_dimensions": len(RICH_RESPONSE_FEATURE_NAMES),
            "source_role_dimensions": len(SOURCE_ROLE_FEATURE_NAMES),
            "pre_window": int(pre_window),
            "post_window": int(post_window),
            "matched_control_sample_id": control_sample_id,
            "metrics": metrics,
            "representation_note": (
                "Response projection is incoming-only and causal with respect to the "
                "current response position. All-token source-role projection is "
                "descriptive and uses future response queries."
            ),
        }
        if out is not None:
            (out / "rich_visualization_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        return {
            "metadata": metadata,
            "response_projection": response_result,
            "projection_stability": stability,
            "source_role_coordinates": source_coordinates,
            "feature_differences": differences,
            "figures": {
                "response": response_figure,
                "stability": stability_figure,
                "source_role": source_figure,
                "heatmap": heatmap_figure,
                "differences": difference_figure,
                "control": control_figure,
            },
        }
