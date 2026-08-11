"""Run-centric graph-state transition diagnostics for single RAGTruth samples.

This module complements ``rich_visualization.py``.  It is designed for mechanism
analysis rather than pretty clustering: it controls generator provenance and
response position, balances feature blocks, analyzes each hallucination span
independently, and compares observed transitions with matched correct controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research_dataset import ResearchDataset
from rich_visualization import RICH_RESPONSE_FEATURE_NAMES, rich_response_features


def _normalized_model_name(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


STATIC_BLOCK_NAMES = (
    "grounding",
    "self_history",
    "sparsity",
    "concentration",
    "locality",
    "layer_routing",
)

STATIC_BLOCK_FEATURES = {
    "grounding": (
        "prompt_mass_share",
        "prompt_mass",
        "prompt_degree",
        "prompt_density",
        "prompt_entropy",
        "prompt_top1_share",
    ),
    "self_history": (
        "history_mass",
        "history_degree",
        "history_density",
        "history_edge_share",
        "history_entropy",
        "history_top1_share",
    ),
    "sparsity": (
        "incoming_mass",
        "in_degree",
        "in_density",
        "channel_edge_density",
    ),
    "concentration": (
        "normalized_entropy",
        "top1_share",
        "top3_share",
        "hhi",
    ),
    "locality": (
        "history_lag",
        "history_lag_std",
        "history_bin_1",
        "history_bin_2_4",
        "history_bin_5_8",
        "history_bin_9_16",
        "history_bin_gt16",
    ),
    "layer_routing": (
        "early_prompt_mass",
        "middle_prompt_mass",
        "late_prompt_mass",
        "early_history_mass",
        "middle_history_mass",
        "late_history_mass",
    ),
}

DYNAMIC_FEATURE_NAMES = (
    "delta_total",
    "delta_grounding",
    "delta_self_history",
    "delta_sparsity",
    "delta_concentration",
    "delta_locality",
    "delta_layer_routing",
    "rolling_total",
    "rolling_grounding",
    "rolling_self_history",
    "rolling_sparsity",
    "rolling_concentration",
    "rolling_locality",
    "rolling_layer_routing",
    "source_js",
    "prompt_source_js",
    "history_source_js",
    "neighbor_change",
    "layer_routing_shift",
)


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
    """Build independent clean windows for every hallucination run."""
    response_count = int(response_count)
    runs = sorted((int(start), int(end)) for start, end in positive_runs)
    for start, end in runs:
        if not 0 <= start < end <= response_count:
            raise ValueError("positive runs must be valid response-relative intervals")
    result = []
    for index, (start, end) in enumerate(runs):
        previous_end = runs[index - 1][1] if index else 0
        next_start = runs[index + 1][0] if index + 1 < len(runs) else response_count
        pre_start = max(previous_end, start - int(pre_window), 0)
        post_end = min(next_start, end + int(post_window), response_count)
        result.append(
            RunWindow(
                run_index=index,
                error_start=start,
                error_end=end,
                clean_pre_start=pre_start,
                clean_pre_end=start,
                clean_post_start=end,
                clean_post_end=post_end,
            )
        )
    return result


def _feature_index():
    return {name: index for index, name in enumerate(RICH_RESPONSE_FEATURE_NAMES)}


def static_feature_blocks(rich_features: np.ndarray):
    """Convert 32-D rich states into six non-overlapping conceptual blocks."""
    values = np.asarray(rich_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(RICH_RESPONSE_FEATURE_NAMES):
        raise ValueError("rich_features must have shape [response_tokens, 32]")
    idx = _feature_index()

    history_mass = values[:, idx["history_mass"]]
    near1 = values[:, idx["history_near1_share"]]
    near4 = values[:, idx["history_near4_share"]]
    near8 = values[:, idx["history_near8_share"]]
    far16 = values[:, idx["history_far16_share"]]
    has_history = history_mass > 0

    locality_bins = {
        "history_bin_1": np.clip(near1, 0.0, 1.0),
        "history_bin_2_4": np.clip(near4 - near1, 0.0, 1.0),
        "history_bin_5_8": np.clip(near8 - near4, 0.0, 1.0),
        "history_bin_9_16": np.where(
            has_history,
            np.clip(1.0 - near8 - far16, 0.0, 1.0),
            0.0,
        ),
        "history_bin_gt16": np.clip(far16, 0.0, 1.0),
    }

    source = {name: values[:, index] for name, index in idx.items()}
    source.update(locality_bins)
    blocks = {}
    for block_name in STATIC_BLOCK_NAMES:
        blocks[block_name] = np.stack(
            [source[name] for name in STATIC_BLOCK_FEATURES[block_name]], axis=1
        ).astype(np.float32, copy=False)
    return blocks


def _robust_center_scale(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float32)
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = 1.4826 * mad
    q25, q75 = np.percentile(matrix, [25, 75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std = matrix.std(axis=0)
    scale = np.where(scale > 1e-6, scale, iqr_scale)
    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _block_slices(blocks):
    slices = {}
    start = 0
    for name in STATIC_BLOCK_NAMES:
        width = blocks[name].shape[1]
        slices[name] = slice(start, start + width)
        start += width
    return slices


def block_balanced_state(blocks, *, reference_blocks=None):
    """Robust-scale each block independently and give each block equal L2 budget."""
    transformed = []
    for name in STATIC_BLOCK_NAMES:
        block = np.asarray(blocks[name], dtype=np.float32)
        reference = block if reference_blocks is None else np.asarray(
            reference_blocks[name], dtype=np.float32
        )
        center, scale = _robust_center_scale(reference)
        standardized = (block - center) / scale
        standardized = np.nan_to_num(standardized)
        transformed.append(standardized / np.sqrt(max(block.shape[1], 1)))
    return np.concatenate(transformed, axis=1).astype(np.float32, copy=False)


def _collect_control_position_reference(control_feature_blocks):
    pooled_blocks = {name: [] for name in STATIC_BLOCK_NAMES}
    positions = []
    for blocks in control_feature_blocks:
        count = len(next(iter(blocks.values())))
        if count == 0:
            continue
        positions.append(np.linspace(0.0, 1.0, count, dtype=np.float32))
        for name in STATIC_BLOCK_NAMES:
            pooled_blocks[name].append(blocks[name])
    if not positions:
        raise ValueError("no correct control nodes are available for position baseline")
    return (
        np.concatenate(positions),
        {name: np.concatenate(rows, axis=0) for name, rows in pooled_blocks.items()},
    )


def position_residual_blocks(
    blocks,
    *,
    control_positions: np.ndarray,
    control_blocks,
    bandwidth=0.08,
    min_points=48,
):
    """Residualize each feature against a position-matched correct-control baseline."""
    count = len(next(iter(blocks.values())))
    target_positions = np.linspace(0.0, 1.0, count, dtype=np.float32)
    control_positions = np.asarray(control_positions, dtype=np.float32)
    output = {name: np.zeros_like(blocks[name], dtype=np.float32) for name in STATIC_BLOCK_NAMES}

    for row, position in enumerate(target_positions):
        distance = np.abs(control_positions - position)
        selected = np.flatnonzero(distance <= float(bandwidth))
        if len(selected) < int(min_points):
            take = min(int(min_points), len(control_positions))
            selected = np.argpartition(distance, take - 1)[:take]
        for name in STATIC_BLOCK_NAMES:
            reference = control_blocks[name][selected]
            center, scale = _robust_center_scale(reference)
            output[name][row] = (blocks[name][row] - center) / scale
    return output


def _probability_vector(source_ids, weights, support):
    vector = np.zeros(len(support), dtype=np.float64)
    if len(source_ids) == 0:
        return vector
    mapping = {int(value): index for index, value in enumerate(support)}
    for source, weight in zip(source_ids, weights):
        vector[mapping[int(source)]] += float(weight)
    total = vector.sum()
    if total > 0:
        vector /= total
    return vector


def _js_divergence(source_a, weight_a, source_b, weight_b):
    support = sorted(set(map(int, source_a)) | set(map(int, source_b)))
    if not support:
        return 0.0
    p = _probability_vector(source_a, weight_a, support)
    q = _probability_vector(source_b, weight_b, support)
    if p.sum() == 0 and q.sum() == 0:
        return 0.0
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def source_transition_features(sample, rich_features: np.ndarray):
    """Measure source-set and source-weight changes between consecutive targets."""
    attention = sample.attention()
    relations = sample.relation_edges()
    response_idx = int(attention.response_idx)
    count = int(attention.num_response_tokens)

    source = relations["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    target = relations["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    weight = relations["weight"].detach().cpu().numpy().astype(np.float64, copy=False)

    output = np.zeros((count, 5), dtype=np.float32)
    layer_names = (
        "early_prompt_mass",
        "middle_prompt_mass",
        "late_prompt_mass",
        "early_history_mass",
        "middle_history_mass",
        "late_history_mass",
    )
    idx = _feature_index()
    routing = rich_features[:, [idx[name] for name in layer_names]]

    previous_sources = previous_weights = None
    for row in range(count):
        absolute_target = response_idx + row
        mask = target == absolute_target
        current_sources = source[mask]
        current_weights = weight[mask]
        if row == 0:
            previous_sources, previous_weights = current_sources, current_weights
            continue

        output[row, 0] = _js_divergence(
            previous_sources, previous_weights, current_sources, current_weights
        )
        prev_prompt = previous_sources < response_idx
        curr_prompt = current_sources < response_idx
        output[row, 1] = _js_divergence(
            previous_sources[prev_prompt],
            previous_weights[prev_prompt],
            current_sources[curr_prompt],
            current_weights[curr_prompt],
        )
        prev_history = previous_sources >= response_idx
        curr_history = current_sources >= response_idx
        output[row, 2] = _js_divergence(
            previous_sources[prev_history],
            previous_weights[prev_history],
            current_sources[curr_history],
            current_weights[curr_history],
        )
        previous_set = set(map(int, previous_sources))
        current_set = set(map(int, current_sources))
        union = previous_set | current_set
        intersection = previous_set & current_set
        output[row, 3] = 1.0 - (len(intersection) / len(union) if union else 1.0)
        output[row, 4] = float(np.linalg.norm(routing[row] - routing[row - 1]))
        previous_sources, previous_weights = current_sources, current_weights
    return output


def dynamic_state(static_state: np.ndarray, block_slices, source_changes, *, rolling_window=8):
    """Return transition features from local velocity and trailing robust deviations."""
    state = np.asarray(static_state, dtype=np.float32)
    n = len(state)
    delta = np.zeros_like(state)
    if n > 1:
        delta[1:] = state[1:] - state[:-1]

    rolling_z = np.zeros_like(state)
    for row in range(1, n):
        start = max(0, row - int(rolling_window))
        history = state[start:row]
        center, scale = _robust_center_scale(history)
        rolling_z[row] = (state[row] - center) / scale

    rows = []
    for row in range(n):
        delta_norms = [float(np.linalg.norm(delta[row]))]
        rolling_norms = [float(np.linalg.norm(rolling_z[row]))]
        for name in STATIC_BLOCK_NAMES:
            sl = block_slices[name]
            delta_norms.append(float(np.linalg.norm(delta[row, sl])))
            rolling_norms.append(float(np.linalg.norm(rolling_z[row, sl])))
        rows.append(delta_norms + rolling_norms + source_changes[row].tolist())
    return np.asarray(rows, dtype=np.float32)


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
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=int(random_state),
    ).fit_transform(values)
    return coordinates, perplexity


def _pca2(matrix):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    values = RobustScaler().fit_transform(np.asarray(matrix, dtype=np.float32))
    values = np.nan_to_num(values)
    if len(values) < 2:
        return np.zeros((len(values), 2), dtype=np.float32)
    dimensions = min(2, values.shape[1], len(values))
    coordinates = PCA(n_components=dimensions).fit_transform(values)
    if dimensions == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(values))])
    return coordinates.astype(np.float32)


class TransitionSampleVisualizer:
    """Single-sample transition diagnostics with generator-aware controls."""

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
        self.requested_generator_model = generator_model
        self._feature_cache = {}

        available = sorted(
            {
                sample.generator_model
                for sample in self.dataset
                if sample.generator_model is not None
            }
        )
        self.available_generator_models = available
        self.generator_model = (
            self.dataset.manifest.get("generator_model")
            if generator_model is None
            else generator_model
        )
        if self.generator_model is not None:
            requested = _normalized_model_name(self.generator_model)
            if not any(_normalized_model_name(value) == requested for value in available):
                raise ValueError(
                    f"generator_model={self.generator_model!r} is not present; "
                    f"available={available}"
                )

    def _matches_generator(self, sample):
        if self.generator_model is None:
            return True
        return (
            _normalized_model_name(sample.generator_model)
            == _normalized_model_name(self.generator_model)
        )

    @property
    def sample_ids(self):
        return [
            sample_id
            for sample_id in self.dataset.sample_ids
            if self._matches_generator(self.dataset[sample_id])
        ]

    @property
    def error_sample_ids(self):
        return [
            sample_id
            for sample_id in self.sample_ids
            if self.labels.positive_runs(sample_id)
        ]

    @property
    def correct_sample_ids(self):
        return [
            sample_id
            for sample_id in self.sample_ids
            if not self.labels.positive_runs(sample_id)
        ]

    def provenance(self):
        observer_model = self.dataset.manifest.get("observer_model")
        selected = self.generator_model
        return {
            "split_root": str(self.dataset.root),
            "observer_model": observer_model,
            "manifest_generator_model": self.dataset.manifest.get("generator_model"),
            "available_generator_models": self.available_generator_models,
            "selected_generator_model": selected,
            "same_generator_and_observer": (
                _normalized_model_name(observer_model)
                == _normalized_model_name(selected)
                if observer_model is not None and selected is not None
                else None
            ),
            "selected_samples": len(self.sample_ids),
            "selected_hallucinated_samples": len(self.error_sample_ids),
            "selected_correct_samples": len(self.correct_sample_ids),
        }

    def _rich(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self._feature_cache:
            self._feature_cache[sample_id] = rich_response_features(
                self.dataset[sample_id]
            )
        return self._feature_cache[sample_id]

    def _control_ids(self, sample_id, *, max_controls=32):
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
        control_ids = self._control_ids(sample_id, max_controls=max_controls)
        if not control_ids:
            raise ValueError("no same-generator fully correct controls are available")
        feature_blocks = [
            static_feature_blocks(self._rich(control_id)) for control_id in control_ids
        ]
        positions, pooled = _collect_control_position_reference(feature_blocks)
        return control_ids, positions, pooled

    def states(
        self,
        sample_id,
        *,
        max_controls=32,
        position_bandwidth=0.08,
        min_position_points=48,
        rolling_window=8,
    ):
        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        if not self._matches_generator(sample):
            raise ValueError("sample does not match the selected generator_model")
        rich = self._rich(sample_id)
        blocks = static_feature_blocks(rich)
        control_ids, control_positions, control_blocks = self._control_reference(
            sample_id, max_controls=max_controls
        )

        raw_balanced = block_balanced_state(blocks, reference_blocks=control_blocks)
        residual_blocks = position_residual_blocks(
            blocks,
            control_positions=control_positions,
            control_blocks=control_blocks,
            bandwidth=position_bandwidth,
            min_points=min_position_points,
        )
        residual_balanced = np.concatenate(
            [
                residual_blocks[name] / np.sqrt(residual_blocks[name].shape[1])
                for name in STATIC_BLOCK_NAMES
            ],
            axis=1,
        ).astype(np.float32)

        slices = _block_slices(blocks)
        source_changes = source_transition_features(sample, rich)
        dynamics = dynamic_state(
            residual_balanced,
            slices,
            source_changes,
            rolling_window=rolling_window,
        )
        return {
            "sample_id": sample_id,
            "sample": sample,
            "generator_model": sample.generator_model,
            "observer_model": sample.observer_model,
            "rich_features": rich,
            "raw_blocks": blocks,
            "raw_balanced": raw_balanced,
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
        response_count = self.dataset[sample_id].attention().num_response_tokens
        return run_windows(
            response_count,
            self.labels.positive_runs(sample_id),
            pre_window=pre_window,
            post_window=post_window,
        )

    def _output_dir(self, sample_id, output_dir=None):
        if output_dir is not None:
            path = Path(output_dir)
        elif self.output_root is not None:
            path = self.output_root / str(sample_id)
        else:
            return None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _control_state(self, control_id, state):
        control = self.dataset[control_id]
        control_rich = self._rich(control_id)
        control_blocks = static_feature_blocks(control_rich)
        residual = position_residual_blocks(
            control_blocks,
            control_positions=state["control_positions"],
            control_blocks=state["control_reference_blocks"],
        )
        balanced = np.concatenate(
            [
                residual[name] / np.sqrt(residual[name].shape[1])
                for name in STATIC_BLOCK_NAMES
            ],
            axis=1,
        ).astype(np.float32)
        changes = source_transition_features(control, control_rich)
        dynamics = dynamic_state(balanced, _block_slices(control_blocks), changes)
        return balanced, dynamics

    def _control_null(self, state, window: RunWindow, *, max_controls=32):
        sample = state["sample"]
        pre = state["position_residual_balanced"][
            window.clean_pre_start : window.clean_pre_end
        ]
        error = state["position_residual_balanced"][window.error_start : window.error_end]
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

        null_distances = []
        null_dynamic = []
        relative_start = window.error_start / max(
            sample.attention().num_response_tokens - 1, 1
        )
        pre_len = window.pre_length
        error_len = window.error_length
        for control_id in state["control_ids"][: int(max_controls)]:
            control = self.dataset[control_id]
            control_count = control.attention().num_response_tokens
            if control_count < pre_len + error_len + 1:
                continue
            center = int(round(relative_start * max(control_count - 1, 1)))
            center = min(max(center, pre_len), control_count - error_len)
            balanced, dynamics = self._control_state(control_id, state)
            control_pre = balanced[center - pre_len : center]
            control_error = balanced[center : center + error_len]
            if len(control_pre) == pre_len and len(control_error) == error_len:
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
    ):
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

        transition_score = local_dynamic[:, 7]
        axes[2].plot(indices, transition_score)
        axes[2].axvspan(window.error_start, window.error_end - 1, alpha=0.15)
        axes[2].set(
            title="Rolling transition magnitude",
            xlabel="Response token position",
            ylabel="Transition score",
        )

        block_change = []
        for block_index, name in enumerate(STATIC_BLOCK_NAMES):
            block_change.append(local_dynamic[:, 8 + block_index])
        block_change = np.asarray(block_change)
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
                out / f"run_{int(run_index)}_transition.png",
                dpi=220,
                bbox_inches="tight",
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
                axes[0],
                "centroid_shift_null",
                "observed_centroid_shift",
                "centroid_shift_percentile",
                "Pre→error centroid shift",
            ),
            (
                axes[1],
                "rolling_transition_null",
                "observed_rolling_transition",
                "rolling_transition_percentile",
                "Onset rolling transition",
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
                out / f"run_{int(run_index)}_control_null.png",
                dpi=220,
                bbox_inches="tight",
            )
        return figure, metrics

    def plot_matched_controls(self, sample_id, *, max_controls=8, output_dir=None):
        """Joint overlay with shared axes; same-generator controls only."""
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
        pooled = np.concatenate(matrices, axis=0)
        coords, perplexity = _project(pooled, random_state=self.random_state)

        figure, axes = plt.subplots(
            1, 2, figsize=(12, 5.2), constrained_layout=True, sharex=True, sharey=True
        )
        offset = lengths[0]
        sample_coords = coords[: lengths[0]]
        axes[0].scatter(sample_coords[:, 0], sample_coords[:, 1], label="Selected sample")
        axes[1].scatter(sample_coords[:, 0], sample_coords[:, 1], label="Selected sample")
        for control_id, length in zip(control_ids, lengths[1:]):
            control_coords = coords[offset : offset + length]
            offset += length
            axes[0].scatter(
                control_coords[:, 0],
                control_coords[:, 1],
                alpha=0.25,
                label=f"Control {control_id}",
            )
            axes[1].scatter(control_coords[:, 0], control_coords[:, 1], alpha=0.25)
        phases = np.zeros(len(sample_coords), dtype=np.int64)
        for start, end in self.labels.positive_runs(str(sample_id)):
            phases[start:end] = 1
        if (phases == 1).any():
            axes[1].scatter(
                sample_coords[phases == 1, 0],
                sample_coords[phases == 1, 1],
                marker="X",
                s=70,
                label="Hallucination",
            )
        axes[0].set_title("Joint geometry: selected vs same-generator controls")
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
    ):
        sample_id = str(sample_id)
        windows = self.runs(sample_id, pre_window=pre_window, post_window=post_window)
        if not windows:
            raise ValueError("sample has no hallucination runs")
        out = self._output_dir(sample_id, output_dir)

        metrics = []
        figures = {}
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

        controls_figure, control_ids = self.plot_matched_controls(
            sample_id, output_dir=out
        )
        figures["controls"] = controls_figure

        metadata = {
            "sample_id": sample_id,
            "provenance": self.provenance(),
            "positive_runs": self.labels.positive_runs(sample_id),
            "run_windows": [window.as_dict() for window in windows],
            "same_generator_control_ids": control_ids,
            "metrics": metrics,
            "notes": {
                "position_residualization": (
                    "Feature-wise median/MAD baselines are estimated from fully correct "
                    "same-generator controls at nearby normalized response positions."
                ),
                "block_balance": (
                    "Six conceptual static blocks are robust-scaled and divided by "
                    "sqrt(block dimension) so blocks with more hand-crafted features do "
                    "not dominate Euclidean/PCA/t-SNE geometry."
                ),
                "locality": (
                    "Cumulative near-history shares are converted to mutually exclusive "
                    "distance bins: 1, 2-4, 5-8, 9-16, >16."
                ),
            },
        }
        if out is not None:
            (out / "transition_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        return {"metadata": metadata, "figures": figures}
