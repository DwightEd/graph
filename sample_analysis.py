"""Single-sample hallucination graph analysis and visualization.

This is the only module for the interactive/case-study workflow.  Low-level
feature construction lives in graph_features.py; canonical I/O lives in
research_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from graph_features import (
    DYNAMIC_FEATURE_NAMES,
    STATIC_BLOCK_NAMES,
    block_slices,
    collect_position_reference,
    dynamic_state,
    position_residual_blocks,
    response_graph_features,
    source_transition_features,
    static_feature_blocks,
)
from research_dataset import ResearchDataset


def _normalized_model_name(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _balanced_residual(blocks: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            blocks[name] / np.sqrt(max(blocks[name].shape[1], 1))
            for name in STATIC_BLOCK_NAMES
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _project(matrix, *, random_state=0, perplexity=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import RobustScaler

    values = RobustScaler().fit_transform(np.asarray(matrix, dtype=np.float32))
    values = np.nan_to_num(values)
    pca_dim = min(20, values.shape[1], max(len(values) - 1, 1))
    if pca_dim >= 2 and values.shape[1] > pca_dim:
        values = PCA(n_components=pca_dim, random_state=random_state).fit_transform(values)
    if perplexity is None:
        perplexity = min(15.0, max(3.0, np.sqrt(len(values))))
    perplexity = min(float(perplexity), len(values) - 1.0)
    return (
        TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            random_state=int(random_state),
        ).fit_transform(values),
        perplexity,
    )


def _pca2(matrix):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    values = np.nan_to_num(
        RobustScaler().fit_transform(np.asarray(matrix, dtype=np.float32))
    )
    if len(values) < 2:
        return np.zeros((len(values), 2), dtype=np.float32)
    dimensions = min(2, values.shape[1], len(values))
    coordinates = PCA(n_components=dimensions).fit_transform(values)
    if dimensions == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(values))))
    return coordinates.astype(np.float32)


@dataclass(frozen=True)
class RunWindow:
    run_index: int
    error_start: int
    error_end: int
    clean_pre_start: int
    clean_pre_end: int
    clean_post_start: int
    clean_post_end: int

    @property
    def pre_length(self) -> int:
        return self.clean_pre_end - self.clean_pre_start

    @property
    def error_length(self) -> int:
        return self.error_end - self.error_start

    @property
    def post_length(self) -> int:
        return self.clean_post_end - self.clean_post_start

    def as_dict(self) -> dict:
        return {
            "run_index": self.run_index,
            "error": [self.error_start, self.error_end],
            "clean_pre": [self.clean_pre_start, self.clean_pre_end],
            "clean_post": [self.clean_post_start, self.clean_post_end],
            "pre_length": self.pre_length,
            "error_length": self.error_length,
            "post_length": self.post_length,
        }


def run_windows(response_count: int, positive_runs, *, pre_window=10, post_window=10):
    """Build independent clean windows for every hallucination span."""
    response_count = int(response_count)
    runs = sorted((int(start), int(end)) for start, end in positive_runs)
    for start, end in runs:
        if not 0 <= start < end <= response_count:
            raise ValueError("positive runs must be valid response-relative intervals")
    result = []
    for index, (start, end) in enumerate(runs):
        previous_end = runs[index - 1][1] if index else 0
        next_start = runs[index + 1][0] if index + 1 < len(runs) else response_count
        result.append(
            RunWindow(
                run_index=index,
                error_start=start,
                error_end=end,
                clean_pre_start=max(previous_end, start - int(pre_window), 0),
                clean_pre_end=start,
                clean_post_start=end,
                clean_post_end=min(next_start, end + int(post_window), response_count),
            )
        )
    return result


class SampleAnalysis:
    """Analyze one hallucinated response against same-generator correct controls."""

    def __init__(
        self,
        split_root,
        *,
        output_root=None,
        generator_model=None,
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
        self.generator_model = (
            self.dataset.manifest.get("generator_model")
            if generator_model is None
            else generator_model
        )
        self._feature_cache: dict[str, np.ndarray] = {}
        self.available_generator_models = sorted(
            {
                sample.generator_model
                for sample in self.dataset
                if sample.generator_model is not None
            }
        )
        if self.generator_model is not None:
            requested = _normalized_model_name(self.generator_model)
            if not any(
                _normalized_model_name(value) == requested
                for value in self.available_generator_models
            ):
                raise ValueError(
                    f"generator_model={self.generator_model!r} is not present; "
                    f"available={self.available_generator_models}"
                )

    def _matches_generator(self, sample) -> bool:
        return self.generator_model is None or (
            _normalized_model_name(sample.generator_model)
            == _normalized_model_name(self.generator_model)
        )

    @property
    def sample_ids(self) -> list[str]:
        return [
            sample_id
            for sample_id in self.dataset.sample_ids
            if self._matches_generator(self.dataset[sample_id])
        ]

    @property
    def error_sample_ids(self) -> list[str]:
        return [
            sample_id
            for sample_id in self.sample_ids
            if self.labels.positive_runs(sample_id)
        ]

    @property
    def correct_sample_ids(self) -> list[str]:
        return [
            sample_id
            for sample_id in self.sample_ids
            if not self.labels.positive_runs(sample_id)
        ]

    def provenance(self) -> dict:
        observer = self.dataset.manifest.get("observer_model")
        selected = self.generator_model
        return {
            "split_root": str(self.dataset.root),
            "observer_model": observer,
            "manifest_generator_model": self.dataset.manifest.get("generator_model"),
            "available_generator_models": self.available_generator_models,
            "selected_generator_model": selected,
            "same_generator_and_observer": (
                _normalized_model_name(observer) == _normalized_model_name(selected)
                if observer is not None and selected is not None
                else None
            ),
            "selected_samples": len(self.sample_ids),
            "selected_hallucinated_samples": len(self.error_sample_ids),
            "selected_correct_samples": len(self.correct_sample_ids),
        }

    def _features(self, sample_id) -> np.ndarray:
        sample_id = str(sample_id)
        if sample_id not in self._feature_cache:
            self._feature_cache[sample_id] = response_graph_features(
                self.dataset[sample_id]
            )
        return self._feature_cache[sample_id]

    def control_ids(self, sample_id, *, max_controls=32) -> list[str]:
        """Rank fully correct controls, always requiring the same response generator."""
        sample = self.dataset[str(sample_id)]
        candidates = []
        for candidate_id in self.correct_sample_ids:
            candidate = self.dataset[candidate_id]
            if _normalized_model_name(candidate.generator_model) != _normalized_model_name(
                sample.generator_model
            ):
                continue
            score = (
                0
                if candidate.task_type == sample.task_type
                and candidate.data_source == sample.data_source
                else 1
                if candidate.task_type == sample.task_type
                else 2,
                abs(
                    candidate.attention().num_response_tokens
                    - sample.attention().num_response_tokens
                ),
                str(candidate_id),
            )
            candidates.append((score, candidate_id))
        candidates.sort()
        return [candidate_id for _, candidate_id in candidates[: int(max_controls)]]

    def _control_reference(self, sample_id, *, max_controls=32):
        ids = self.control_ids(sample_id, max_controls=max_controls)
        if not ids:
            raise ValueError("no same-generator fully correct controls are available")
        feature_blocks = [static_feature_blocks(self._features(control_id)) for control_id in ids]
        positions, pooled = collect_position_reference(feature_blocks)
        return ids, positions, pooled

    def states(
        self,
        sample_id,
        *,
        max_controls=32,
        position_bandwidth=0.08,
        min_position_points=48,
        rolling_window=8,
    ) -> dict:
        """Build static position-residual states and 19-D transition states."""
        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        if not self._matches_generator(sample):
            raise ValueError("sample does not match the selected generator_model")

        features = self._features(sample_id)
        blocks = static_feature_blocks(features)
        control_ids, control_positions, control_blocks = self._control_reference(
            sample_id, max_controls=max_controls
        )
        residual_blocks = position_residual_blocks(
            blocks,
            control_positions=control_positions,
            control_blocks=control_blocks,
            bandwidth=position_bandwidth,
            min_points=min_position_points,
        )
        residual_balanced = _balanced_residual(residual_blocks)
        slices = block_slices(blocks)
        changes = source_transition_features(sample, features)
        dynamics = dynamic_state(
            residual_balanced,
            slices,
            changes,
            rolling_window=rolling_window,
        )
        return {
            "sample_id": sample_id,
            "sample": sample,
            "generator_model": sample.generator_model,
            "observer_model": sample.observer_model,
            "response_features": features,
            "raw_blocks": blocks,
            "position_residual_blocks": residual_blocks,
            "position_residual_balanced": residual_balanced,
            "dynamic": dynamics,
            "dynamic_feature_names": np.asarray(DYNAMIC_FEATURE_NAMES),
            "block_slices": slices,
            "control_ids": control_ids,
            "control_positions": control_positions,
            "control_reference_blocks": control_blocks,
        }

    def runs(self, sample_id, *, pre_window=10, post_window=10):
        sample_id = str(sample_id)
        return run_windows(
            self.dataset[sample_id].attention().num_response_tokens,
            self.labels.positive_runs(sample_id),
            pre_window=pre_window,
            post_window=post_window,
        )

    def _control_state(self, control_id, state):
        control = self.dataset[control_id]
        features = self._features(control_id)
        blocks = static_feature_blocks(features)
        residual = position_residual_blocks(
            blocks,
            control_positions=state["control_positions"],
            control_blocks=state["control_reference_blocks"],
        )
        balanced = _balanced_residual(residual)
        dynamics = dynamic_state(
            balanced,
            block_slices(blocks),
            source_transition_features(control, features),
        )
        return balanced, dynamics

    def _control_null(self, state, window: RunWindow, *, max_controls=32) -> dict:
        sample = state["sample"]
        static = state["position_residual_balanced"]
        pre = static[window.clean_pre_start : window.clean_pre_end]
        error = static[window.error_start : window.error_end]
        observed = (
            float(np.linalg.norm(error.mean(0) - pre.mean(0)))
            if len(pre) and len(error)
            else None
        )
        dynamic_score = (
            float(state["dynamic"][window.error_start, 7])
            if window.error_start < len(state["dynamic"])
            else None
        )

        null_distances, null_dynamic = [], []
        relative_start = window.error_start / max(
            sample.attention().num_response_tokens - 1, 1
        )
        for control_id in state["control_ids"][: int(max_controls)]:
            control = self.dataset[control_id]
            count = control.attention().num_response_tokens
            if count < window.pre_length + window.error_length + 1:
                continue
            center = int(round(relative_start * max(count - 1, 1)))
            center = min(max(center, window.pre_length), count - window.error_length)
            balanced, dynamics = self._control_state(control_id, state)
            control_pre = balanced[center - window.pre_length : center]
            control_error = balanced[center : center + window.error_length]
            if len(control_pre) == window.pre_length and len(control_error) == window.error_length:
                null_distances.append(
                    float(np.linalg.norm(control_error.mean(0) - control_pre.mean(0)))
                )
            null_dynamic.append(float(dynamics[center, 7]))

        def percentile(value, null):
            if value is None or not null:
                return None
            array = np.asarray(null)
            return float((np.sum(array <= value) + 1) / (len(array) + 1))

        return {
            "observed_centroid_shift": observed,
            "centroid_shift_null": null_distances,
            "centroid_shift_percentile": percentile(observed, null_distances),
            "observed_rolling_transition": dynamic_score,
            "rolling_transition_null": null_dynamic,
            "rolling_transition_percentile": percentile(dynamic_score, null_dynamic),
        }

    def run_metrics(
        self,
        sample_id,
        *,
        run_index=0,
        pre_window=10,
        post_window=10,
        max_controls=32,
    ) -> dict:
        state = self.states(sample_id, max_controls=max_controls)
        windows = self.runs(sample_id, pre_window=pre_window, post_window=post_window)
        if not windows:
            raise ValueError("sample has no hallucination runs")
        if not 0 <= int(run_index) < len(windows):
            raise IndexError("run_index out of range")
        window = windows[int(run_index)]
        result = {
            "sample_id": str(sample_id),
            "run": window.as_dict(),
            "generator_model": state["generator_model"],
            "observer_model": state["observer_model"],
            "control_count": len(state["control_ids"]),
            "clean_pre_sufficient": window.pre_length >= 3,
        }
        result.update(self._control_null(state, window, max_controls=max_controls))
        return result

    def _output_dir(self, sample_id, output_dir=None):
        if output_dir is not None:
            path = Path(output_dir)
        elif self.output_root is not None:
            path = self.output_root / str(sample_id)
        else:
            return None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plot_run(
        self,
        sample_id,
        *,
        run_index=0,
        pre_window=10,
        post_window=10,
        local_radius=18,
        max_controls=32,
        output_dir=None,
    ):
        """Plot one hallucination run: static geometry plus transition magnitudes."""
        import matplotlib.pyplot as plt

        state = self.states(sample_id, max_controls=max_controls)
        windows = self.runs(sample_id, pre_window=pre_window, post_window=post_window)
        if not windows:
            raise ValueError("sample has no hallucination runs")
        window = windows[int(run_index)]
        local_start = max(0, window.error_start - int(local_radius))
        local_end = min(
            len(state["position_residual_balanced"]),
            window.error_end + int(local_radius),
        )
        indices = np.arange(local_start, local_end)
        local_static = state["position_residual_balanced"][local_start:local_end]
        local_dynamic = state["dynamic"][local_start:local_end]
        pca = _pca2(local_static)
        tsne, perplexity = _project(local_static, random_state=self.random_state)

        figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        axes = axes.reshape(-1)

        def phase_mask(kind):
            if kind == "pre":
                return (indices >= window.clean_pre_start) & (indices < window.clean_pre_end)
            if kind == "error":
                return (indices >= window.error_start) & (indices < window.error_end)
            if kind == "post":
                return (indices >= window.clean_post_start) & (indices < window.clean_post_end)
            return np.ones(len(indices), dtype=bool)

        reserved = phase_mask("pre") | phase_mask("error") | phase_mask("post")
        masks = {
            "context": ~reserved,
            "pre": phase_mask("pre"),
            "error": phase_mask("error"),
            "post": phase_mask("post"),
        }
        specs = (
            ("context", "Context", "o"),
            ("pre", "Clean pre-error", "^"),
            ("error", "Hallucination", "X"),
            ("post", "Clean post-error", "s"),
        )
        for axis, coords, title in (
            (axes[0], pca, "Position-residual static state — PCA"),
            (axes[1], tsne, f"Position-residual static state — t-SNE p={perplexity:.1f}"),
        ):
            for kind, label, marker in specs:
                mask = masks[kind]
                if mask.any():
                    axis.scatter(coords[mask, 0], coords[mask, 1], marker=marker, label=label)
            axis.set(title=title, xlabel="component 1", ylabel="component 2")
            axis.legend()

        axes[2].plot(indices, local_dynamic[:, 7])
        axes[2].axvspan(window.error_start, window.error_end - 1, alpha=0.15)
        axes[2].set(
            title="Rolling transition magnitude",
            xlabel="Response token position",
            ylabel="Transition score",
        )

        block_change = np.asarray(
            [local_dynamic[:, 8 + index] for index, _ in enumerate(STATIC_BLOCK_NAMES)]
        )
        image = axes[3].imshow(
            block_change,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(
                local_start - 0.5,
                local_end - 0.5,
                len(STATIC_BLOCK_NAMES) - 0.5,
                -0.5,
            ),
        )
        axes[3].set_yticks(np.arange(len(STATIC_BLOCK_NAMES)))
        axes[3].set_yticklabels(STATIC_BLOCK_NAMES)
        axes[3].axvspan(window.error_start - 0.5, window.error_end - 0.5, alpha=0.15)
        axes[3].set(
            title="Block-wise rolling deviation",
            xlabel="Response token position",
            ylabel="Structural block",
        )
        figure.colorbar(image, ax=axes[3], label="Block transition magnitude")
        figure.suptitle(
            f"Run-centric transition analysis — sample {sample_id}, run {run_index}"
        )
        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(
                out / f"run_{int(run_index)}_transition.png", dpi=220, bbox_inches="tight"
            )
        return figure, {
            "state": state,
            "window": window,
            "pca": pca,
            "tsne": tsne,
            "local_indices": indices,
        }

    def plot_control_null(
        self,
        sample_id,
        *,
        run_index=0,
        pre_window=10,
        post_window=10,
        max_controls=32,
        output_dir=None,
    ):
        import matplotlib.pyplot as plt

        metrics = self.run_metrics(
            sample_id,
            run_index=run_index,
            pre_window=pre_window,
            post_window=post_window,
            max_controls=max_controls,
        )
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for axis, null_key, observed_key, percentile_key, title in (
            (
                axes[0], "centroid_shift_null", "observed_centroid_shift",
                "centroid_shift_percentile", "Pre→error centroid shift",
            ),
            (
                axes[1], "rolling_transition_null", "observed_rolling_transition",
                "rolling_transition_percentile", "Onset rolling transition",
            ),
        ):
            values = metrics[null_key]
            if values:
                axis.hist(values, bins=min(12, max(4, len(values) // 2)), alpha=0.7)
            observed = metrics[observed_key]
            if observed is not None:
                axis.axvline(observed, linewidth=2.0, label="Observed hallucination run")
            percentile = metrics[percentile_key]
            suffix = "" if percentile is None else f"; percentile={100*percentile:.1f}"
            axis.set(title=title + suffix, xlabel="Transition magnitude", ylabel="Controls")
            axis.legend()
        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(
                out / f"run_{int(run_index)}_control_null.png", dpi=220, bbox_inches="tight"
            )
        return figure, metrics

    def plot_matched_controls(self, sample_id, *, max_controls=8, output_dir=None):
        """Fit one joint t-SNE for the selected response and same-generator controls."""
        import matplotlib.pyplot as plt

        state = self.states(sample_id, max_controls=max(max_controls, 16))
        sample_matrix = state["position_residual_balanced"]
        control_ids = state["control_ids"][: int(max_controls)]
        matrices = [sample_matrix]
        lengths = [len(sample_matrix)]
        for control_id in control_ids:
            matrix, _ = self._control_state(control_id, state)
            matrices.append(matrix)
            lengths.append(len(matrix))
        coordinates, perplexity = _project(
            np.concatenate(matrices, axis=0), random_state=self.random_state
        )

        figure, axes = plt.subplots(
            1, 2, figsize=(12, 5.2), constrained_layout=True, sharex=True, sharey=True
        )
        sample_coords = coordinates[: lengths[0]]
        for axis in axes:
            axis.scatter(sample_coords[:, 0], sample_coords[:, 1], label="Selected sample")
        offset = lengths[0]
        for control_id, length in zip(control_ids, lengths[1:]):
            control_coords = coordinates[offset : offset + length]
            offset += length
            axes[0].scatter(
                control_coords[:, 0], control_coords[:, 1], alpha=0.25,
                label=f"Control {control_id}",
            )
            axes[1].scatter(control_coords[:, 0], control_coords[:, 1], alpha=0.25)
        phases = np.zeros(len(sample_coords), dtype=bool)
        for start, end in self.labels.positive_runs(str(sample_id)):
            phases[start:end] = True
        if phases.any():
            axes[1].scatter(
                sample_coords[phases, 0], sample_coords[phases, 1],
                marker="X", s=70, label="Hallucination",
            )
        axes[0].set_title("Selected response vs same-generator controls")
        axes[1].set_title("Same coordinates with hallucination overlay")
        for axis in axes:
            axis.set(xlabel="joint t-SNE 1", ylabel="joint t-SNE 2")
            axis.legend(fontsize=7)
        figure.suptitle(f"Shared-axis matched controls; perplexity={perplexity:.1f}")
        out = self._output_dir(sample_id, output_dir)
        if out is not None:
            figure.savefig(out / "matched_controls_shared_axes.png", dpi=220, bbox_inches="tight")
        return figure, control_ids

    def visualize(
        self,
        sample_id,
        *,
        output_dir=None,
        pre_window=10,
        post_window=10,
        local_radius=18,
        max_controls=32,
    ) -> dict:
        """Generate the complete single-sample diagnostic bundle."""
        sample_id = str(sample_id)
        windows = self.runs(sample_id, pre_window=pre_window, post_window=post_window)
        if not windows:
            raise ValueError("sample has no hallucination runs")
        out = self._output_dir(sample_id, output_dir)
        metrics, figures = [], {}
        for window in windows:
            figure, _ = self.plot_run(
                sample_id,
                run_index=window.run_index,
                pre_window=pre_window,
                post_window=post_window,
                local_radius=local_radius,
                max_controls=max_controls,
                output_dir=out,
            )
            figures[f"run_{window.run_index}"] = figure
            null_figure, run_metric = self.plot_control_null(
                sample_id,
                run_index=window.run_index,
                pre_window=pre_window,
                post_window=post_window,
                max_controls=max_controls,
                output_dir=out,
            )
            figures[f"run_{window.run_index}_null"] = null_figure
            metrics.append(run_metric)
        controls_figure, control_ids = self.plot_matched_controls(sample_id, output_dir=out)
        figures["controls"] = controls_figure
        metadata = {
            "sample_id": sample_id,
            "provenance": self.provenance(),
            "positive_runs": self.labels.positive_runs(sample_id),
            "run_windows": [window.as_dict() for window in windows],
            "same_generator_control_ids": control_ids,
            "metrics": metrics,
            "pipeline": (
                "canonical relations -> 32D response features -> six semantic blocks -> "
                "position residualization -> 33D balanced static state -> 19D dynamics"
            ),
        }
        if out is not None:
            (out / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        return {"metadata": metadata, "figures": figures}
