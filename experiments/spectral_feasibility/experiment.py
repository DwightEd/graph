"""Label-free robust spectral-subspace fitting, scoring, and post-hoc evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .representations import (
    SpectralConfig,
    causal_spectral_state,
    reference_positions,
    response_position_bin,
    spectral_state_dimension,
    spectral_volume,
)


REFERENCE_SCHEMA = "causal-spectral-subspace-reference-v1"
SCORE_SCHEMA = "causal-spectral-subspace-score-v1"


def _robust_location_scale(values: np.ndarray, *, epsilon=1e-6):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    deviation = np.abs(values - center)
    mad = 1.4826 * np.median(deviation, axis=0)
    std = np.std(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(std > epsilon, std, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def _position_stats(values: np.ndarray, bins: np.ndarray, count: int):
    values = np.asarray(values)
    bins = np.asarray(bins)
    if len(values) == 0:
        raise ValueError("cannot estimate position statistics from no values")
    global_center, global_scale = _robust_location_scale(values)
    centers = np.empty((count, values.shape[1]), dtype=np.float32)
    scales = np.empty_like(centers)
    for position_bin in range(count):
        selected = bins == position_bin
        if int(selected.sum()) >= 2:
            centers[position_bin], scales[position_bin] = _robust_location_scale(values[selected])
        else:
            centers[position_bin] = global_center
            scales[position_bin] = global_scale
    return centers, scales


def _scalar_position_stats(values: np.ndarray, bins: np.ndarray, count: int):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        raise ValueError("cannot estimate scalar statistics from no finite values")
    center, scale = _position_stats(values[finite, None], np.asarray(bins)[finite], count)
    return center[:, 0], scale[:, 0]


def _sample_ids(dataset, limit=None):
    sample_ids = list(dataset.sample_ids)
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("no samples selected")
    return sample_ids


def _bins_for_positions(positions, response_count, position_bins):
    return np.asarray(
        [response_position_bin(int(position), response_count, position_bins) for position in positions],
        dtype=np.int16,
    )


def _pca_residual(standardized: np.ndarray, pca: PCA) -> np.ndarray:
    scores = pca.transform(standardized)
    reconstructed = pca.inverse_transform(scores)
    return np.mean(np.square(standardized - reconstructed), axis=1).astype(np.float32)


def _fit_trimmed_pca(standardized, bins, config: SpectralConfig):
    """Fit PCA twice and drop only the largest train residual tail per bin."""
    values = np.asarray(standardized, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int16)
    if len(values) < 3:
        raise ValueError("at least three train reference tokens are required")

    probe_dim = min(config.pca_dim, 16, values.shape[0] - 1, values.shape[1])
    probe_dim = max(1, int(probe_dim))
    probe = PCA(n_components=probe_dim, svd_solver="randomized", random_state=0)
    probe.fit(values)
    probe_residual = _pca_residual(values, probe)

    keep = np.ones(len(values), dtype=bool)
    if config.trim_fraction < 1.0:
        for position_bin in range(config.position_bins):
            selected = np.flatnonzero(bins == position_bin)
            if len(selected) < 4:
                continue
            threshold = float(np.quantile(probe_residual[selected], config.trim_fraction))
            keep[selected] = probe_residual[selected] <= threshold
    if int(keep.sum()) < 3:
        keep[:] = True

    pca_dim = min(config.pca_dim, int(keep.sum()) - 1, values.shape[1])
    if pca_dim < 1:
        raise ValueError("PCA has no valid component after robust trimming")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0)
    pca.fit(values[keep])
    return pca, keep


def _transform_states(states, bins, reference, *, return_residual_vector=False):
    values = np.asarray(states, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    standardized = (values - reference["state_center"][bins]) / reference["state_scale"][bins]
    centered = standardized - reference["pca_mean"]
    scores = centered @ reference["pca_components"].T
    embedding = scores / reference["pca_whiten_scale"]
    reconstructed = scores @ reference["pca_components"] + reference["pca_mean"]
    residual_vector = standardized - reconstructed
    residual = np.mean(np.square(residual_vector), axis=1)
    output = (
        embedding.astype(np.float32, copy=False),
        residual.astype(np.float32, copy=False),
    )
    if return_residual_vector:
        return (*output, residual_vector.astype(np.float32, copy=False))
    return output


def _project_raw_innovation(current, previous, position_bin, reference):
    raw = (
        np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32)
    ) / reference["state_scale"][int(position_bin)]
    projected = raw @ reference["pca_components"].T
    return (projected / reference["pca_whiten_scale"]).astype(np.float32, copy=False)


def _knn_distances(reference_values, query_values, neighbors, *, self_query=False):
    reference_values = np.asarray(reference_values, dtype=np.float32)
    query_values = np.asarray(query_values, dtype=np.float32)
    if len(reference_values) < 2:
        return np.zeros(len(query_values), dtype=np.float32)
    extra = 1 if self_query else 0
    requested = min(int(neighbors) + extra, len(reference_values))
    model = NearestNeighbors(n_neighbors=requested, metric="euclidean")
    model.fit(reference_values)
    distances = model.kneighbors(query_values, return_distance=True)[0]
    if self_query:
        distances = distances[:, 1:]
    if distances.shape[1] == 0:
        return np.zeros(len(query_values), dtype=np.float32)
    return distances.mean(axis=1).astype(np.float32, copy=False)


def _states_for_targets(sample, targets, config):
    history = max(config.spectral_window - 1, config.dynamic_lags)
    needed = set()
    for target in map(int, targets):
        needed.update(range(max(0, target - history), target + 1))
    positions = np.asarray(sorted(needed), dtype=np.int64)
    states, prompt_volume = causal_spectral_state(sample, positions=positions, config=config)
    return positions, states, prompt_volume


def _metadata_text(value):
    return "" if value is None else str(value)


def _lag_features(embedding: np.ndarray, targets: np.ndarray, lags: int):
    rows = []
    valid_targets = []
    for target in np.asarray(targets, dtype=np.int64):
        if target < lags:
            continue
        rows.append(np.concatenate([embedding[target - lag] for lag in range(1, lags + 1)]))
        valid_targets.append(int(target))
    if not rows:
        return (
            np.empty((0, embedding.shape[1] * lags), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(valid_targets, dtype=np.int64)


def _fit_dynamic_predictor(features, targets, ridge_alpha):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if len(features) < 2:
        raise ValueError("at least two dynamic reference transitions are required")
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(features, targets)
    return model.coef_.astype(np.float32), np.asarray(model.intercept_, dtype=np.float32)


def _dynamic_residual(embedding, coef, intercept, lags):
    embedding = np.asarray(embedding, dtype=np.float32)
    result = np.full(len(embedding), np.nan, dtype=np.float32)
    features, targets = _lag_features(embedding, np.arange(len(embedding)), lags)
    if len(targets):
        prediction = features @ np.asarray(coef, dtype=np.float32).T + np.asarray(intercept, dtype=np.float32)
        result[targets] = np.mean(np.square(embedding[targets] - prediction), axis=1)
    return result


def _empirical_upper_tail(reference_values, reference_bins, values, bins, *, epsilon=1e-12):
    """Convert anomaly magnitudes to train-only -log empirical tail p-values."""
    reference_values = np.asarray(reference_values, dtype=np.float64)
    reference_bins = np.asarray(reference_bins, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    result = np.full(len(values), np.nan, dtype=np.float32)
    global_reference = np.sort(reference_values[np.isfinite(reference_values)])
    if len(global_reference) == 0:
        raise ValueError("empirical calibration has no finite train values")
    for position_bin in np.unique(bins):
        query = (bins == position_bin) & np.isfinite(values)
        if not bool(query.any()):
            continue
        reference = reference_values[(reference_bins == position_bin) & np.isfinite(reference_values)]
        reference = np.sort(reference) if len(reference) >= 2 else global_reference
        count_ge = len(reference) - np.searchsorted(reference, values[query], side="left")
        probability = (count_ge + 1.0) / (len(reference) + 1.0)
        result[query] = -np.log(np.maximum(probability, epsilon)).astype(np.float32)
    return result


def _channel_residual_attribution(
    residual_vector, *, num_channels, top_k, prompt_bins, attribution_topk
):
    residual_vector = np.asarray(residual_vector, dtype=np.float32)
    rr_width = int(num_channels) * int(top_k)
    rr = residual_vector[:, :rr_width].reshape(len(residual_vector), num_channels, top_k)
    rp = residual_vector[:, rr_width:].reshape(len(residual_vector), num_channels, prompt_bins)
    rr_energy = np.mean(np.square(rr), axis=(1, 2)).astype(np.float32)
    rp_energy = np.mean(np.square(rp), axis=(1, 2)).astype(np.float32)
    channel_energy = (
        np.square(rr).sum(axis=2) + np.square(rp).sum(axis=2)
    ) / float(top_k + prompt_bins)
    keep = min(int(attribution_topk), int(num_channels))
    order = np.argsort(channel_energy, axis=1)[:, -keep:][:, ::-1]
    energy = np.take_along_axis(channel_energy, order, axis=1)
    return rr_energy, rp_energy, order.astype(np.int32), energy.astype(np.float32)


def _query_reference_knn(reference, manifold, bins, tasks):
    result = np.zeros(len(manifold), dtype=np.float32)
    tasks = np.asarray(tasks, dtype=str)
    for task in np.unique(tasks):
        for position_bin in range(int(reference["position_bins"])):
            query = (tasks == task) & (bins == position_bin)
            if not bool(query.any()):
                continue
            source = (
                (reference["reference_task"] == task)
                & (reference["reference_bin"] == position_bin)
            )
            if source.sum() < 2:
                source = reference["reference_bin"] == position_bin
            result[query] = _knn_distances(
                reference["reference_manifold"][source],
                manifold[query],
                int(reference["neighbors"]),
                self_query=False,
            )
    return result


def fit_spectral_reference(dataset, output_path, *, config: SpectralConfig | None = None, limit=None):
    """Fit robust spectral subspace and causal dynamics without labels."""
    config = SpectralConfig() if config is None else config
    config.validate()
    sample_ids = _sample_ids(dataset, limit)

    raw_states = []
    raw_bins = []
    audit_sample = []
    audit_token = []
    geometry = None

    for sample_id in tqdm(sample_ids, desc="spectral fit pass 1"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            current_geometry = (int(attention.num_layers), int(attention.num_heads))
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError("attention geometry changes inside the split")
            targets = reference_positions(attention.num_response_tokens, config.reference_per_sample)
            states, _ = causal_spectral_state(sample, positions=targets, config=config)
            raw_states.append(states)
            raw_bins.append(
                _bins_for_positions(targets, attention.num_response_tokens, config.position_bins)
            )
            audit_sample.extend([str(sample.sample_id)] * len(targets))
            audit_token.extend(map(int, targets))
        finally:
            sample.release_attention()

    states = np.concatenate(raw_states, axis=0).astype(np.float32, copy=False)
    bins = np.concatenate(raw_bins, axis=0)
    expected_dim = spectral_state_dimension(
        geometry[0], geometry[1], config.top_k, config.prompt_bins
    )
    if states.shape[1] != expected_dim:
        raise RuntimeError("unexpected channel-preserving spectral dimension")

    state_center, state_scale = _position_stats(states, bins, config.position_bins)
    states -= state_center[bins]
    states /= state_scale[bins]
    pca, keep_mask = _fit_trimmed_pca(states, bins, config)
    whiten_scale = np.sqrt(
        np.maximum(pca.explained_variance_.astype(np.float64), config.epsilon)
    ).astype(np.float32)
    proto = {
        "state_center": state_center,
        "state_scale": state_scale,
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_whiten_scale": whiten_scale,
    }
    del states

    ref_embedding = []
    ref_innovation = []
    ref_residual = []
    ref_temporal_volume = []
    ref_prompt_volume = []
    ref_bins = []
    ref_tasks = []
    dynamic_features = []
    dynamic_targets = []
    dynamic_reference_index = []

    reference_index = 0
    for sample_id in tqdm(sample_ids, desc="spectral fit pass 2"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            targets = reference_positions(attention.num_response_tokens, config.reference_per_sample)
            positions, local_states, local_prompt_volume = _states_for_targets(sample, targets, config)
            local_bins = _bins_for_positions(
                positions, attention.num_response_tokens, config.position_bins
            )
            local_embedding, local_residual = _transform_states(local_states, local_bins, proto)
            by_position = {int(position): index for index, position in enumerate(positions)}
            task = _metadata_text(sample.task_type)
            embedding_by_token = {
                int(position): local_embedding[index] for index, position in enumerate(positions)
            }
            for target in targets.tolist():
                index = by_position[int(target)]
                position_bin = response_position_bin(
                    target, attention.num_response_tokens, config.position_bins
                )
                current = local_embedding[index]
                if target > 0:
                    innovation = _project_raw_innovation(
                        local_states[index],
                        local_states[by_position[int(target - 1)]],
                        position_bin,
                        proto,
                    )
                else:
                    innovation = np.zeros_like(current)
                start = max(0, target - config.spectral_window + 1)
                window_indices = [by_position[position] for position in range(start, target + 1)]
                temporal_volume = spectral_volume(
                    local_embedding[window_indices],
                    window=config.spectral_window,
                    alpha=config.logdet_alpha,
                    epsilon=config.epsilon,
                )[-1]

                ref_embedding.append(current)
                ref_innovation.append(innovation)
                ref_residual.append(local_residual[index])
                ref_temporal_volume.append(temporal_volume)
                ref_prompt_volume.append(local_prompt_volume[index])
                ref_bins.append(position_bin)
                ref_tasks.append(task)

                if target >= config.dynamic_lags:
                    feature = np.concatenate(
                        [embedding_by_token[target - lag] for lag in range(1, config.dynamic_lags + 1)]
                    )
                    dynamic_features.append(feature)
                    dynamic_targets.append(current)
                    dynamic_reference_index.append(reference_index)
                reference_index += 1
        finally:
            sample.release_attention()

    ref_embedding = np.asarray(ref_embedding, dtype=np.float32)
    ref_innovation = np.asarray(ref_innovation, dtype=np.float32)
    ref_residual = np.asarray(ref_residual, dtype=np.float32)
    ref_temporal_volume = np.asarray(ref_temporal_volume, dtype=np.float32)
    ref_prompt_volume = np.asarray(ref_prompt_volume, dtype=np.float32)
    ref_bins = np.asarray(ref_bins, dtype=np.int16)
    ref_tasks = np.asarray(ref_tasks, dtype=str)
    if len(ref_embedding) != len(keep_mask):
        raise RuntimeError("trim mask and second-pass reference order are misaligned")

    normal = np.asarray(keep_mask, dtype=bool)
    delta_center, delta_scale = _position_stats(
        ref_innovation[normal], ref_bins[normal], config.position_bins
    )
    innovation_standardized = (
        ref_innovation - delta_center[ref_bins]
    ) / delta_scale[ref_bins]
    reference_manifold_all = np.concatenate(
        (ref_embedding, innovation_standardized), axis=1
    ).astype(np.float32)
    reference_manifold = reference_manifold_all[normal]
    reference_bin = ref_bins[normal]
    reference_task = ref_tasks[normal]

    train_knn = np.zeros(len(reference_manifold), dtype=np.float32)
    filled = np.zeros(len(reference_manifold), dtype=bool)
    for task in np.unique(reference_task):
        for position_bin in range(config.position_bins):
            selected = (reference_task == task) & (reference_bin == position_bin)
            if selected.sum() < 2:
                continue
            train_knn[selected] = _knn_distances(
                reference_manifold[selected],
                reference_manifold[selected],
                config.neighbors,
                self_query=True,
            )
            filled[selected] = True
    for position_bin in range(config.position_bins):
        selected = (~filled) & (reference_bin == position_bin)
        if not bool(selected.any()):
            continue
        source = reference_bin == position_bin
        train_knn[selected] = _knn_distances(
            reference_manifold[source],
            reference_manifold[selected],
            config.neighbors,
            self_query=False,
        )
        filled[selected] = True

    dynamic_features = np.asarray(dynamic_features, dtype=np.float32)
    dynamic_targets = np.asarray(dynamic_targets, dtype=np.float32)
    dynamic_reference_index = np.asarray(dynamic_reference_index, dtype=np.int64)
    dynamic_keep = normal[dynamic_reference_index] if len(dynamic_reference_index) else np.empty(0, dtype=bool)
    if int(dynamic_keep.sum()) < 2:
        raise ValueError(
            "too few trimmed dynamic transitions; increase train references or lower dynamic_lags"
        )
    dynamic_coef, dynamic_intercept = _fit_dynamic_predictor(
        dynamic_features[dynamic_keep],
        dynamic_targets[dynamic_keep],
        config.dynamic_ridge,
    )
    dynamic_prediction = dynamic_features @ dynamic_coef.T + dynamic_intercept
    dynamic_residual_all = np.mean(
        np.square(dynamic_targets - dynamic_prediction), axis=1
    ).astype(np.float32)
    ref_dynamic_residual = np.full(len(ref_embedding), np.nan, dtype=np.float32)
    ref_dynamic_residual[dynamic_reference_index] = dynamic_residual_all

    temporal_center, temporal_scale = _scalar_position_stats(
        ref_temporal_volume[normal], ref_bins[normal], config.position_bins
    )
    prompt_center, prompt_scale = _scalar_position_stats(
        ref_prompt_volume[normal], ref_bins[normal], config.position_bins
    )
    temporal_absz = np.abs(
        (ref_temporal_volume - temporal_center[ref_bins]) / temporal_scale[ref_bins]
    ).astype(np.float32)
    prompt_absz = np.abs(
        (ref_prompt_volume - prompt_center[ref_bins]) / prompt_scale[ref_bins]
    ).astype(np.float32)

    calibration_bin = ref_bins[normal]
    calibration_static = ref_residual[normal]
    calibration_dynamic = ref_dynamic_residual[normal]
    calibration_temporal = temporal_absz[normal]
    calibration_prompt = prompt_absz[normal]
    calibration_knn = train_knn

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        num_layers=np.asarray(geometry[0], dtype=np.int16),
        num_heads=np.asarray(geometry[1], dtype=np.int16),
        top_k=np.asarray(config.top_k, dtype=np.int16),
        prompt_bins=np.asarray(config.prompt_bins, dtype=np.int16),
        block_rows=np.asarray(config.block_rows, dtype=np.int32),
        position_bins=np.asarray(config.position_bins, dtype=np.int16),
        pca_dim=np.asarray(pca.n_components_, dtype=np.int16),
        reference_per_sample=np.asarray(config.reference_per_sample, dtype=np.int16),
        trim_fraction=np.asarray(config.trim_fraction, dtype=np.float32),
        neighbors=np.asarray(config.neighbors, dtype=np.int16),
        spectral_window=np.asarray(config.spectral_window, dtype=np.int16),
        dynamic_lags=np.asarray(config.dynamic_lags, dtype=np.int16),
        dynamic_ridge=np.asarray(config.dynamic_ridge, dtype=np.float32),
        logdet_alpha=np.asarray(config.logdet_alpha, dtype=np.float32),
        attribution_topk=np.asarray(config.attribution_topk, dtype=np.int16),
        epsilon=np.asarray(config.epsilon, dtype=np.float32),
        state_center=state_center,
        state_scale=state_scale,
        pca_mean=proto["pca_mean"],
        pca_components=proto["pca_components"],
        pca_whiten_scale=whiten_scale,
        delta_center=delta_center,
        delta_scale=delta_scale,
        dynamic_coef=dynamic_coef,
        dynamic_intercept=dynamic_intercept,
        reference_manifold=reference_manifold,
        reference_bin=reference_bin,
        reference_task=reference_task,
        calibration_bin=calibration_bin,
        calibration_static=calibration_static,
        calibration_dynamic=calibration_dynamic,
        calibration_temporal=calibration_temporal,
        calibration_prompt=calibration_prompt,
        calibration_knn=calibration_knn,
        temporal_center=temporal_center,
        temporal_scale=temporal_scale,
        prompt_center=prompt_center,
        prompt_scale=prompt_scale,
        reference_sample_id=np.asarray(audit_sample, dtype=str),
        reference_token_index=np.asarray(audit_token, dtype=np.int32),
        reference_keep=normal.astype(np.int8),
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "reference_tokens": int(len(ref_embedding)),
        "trimmed_reference_tokens": int(normal.sum()),
        "raw_spectral_dim": int(expected_dim),
        "embedding_dim": int(pca.n_components_),
        "manifold_dim": int(reference_manifold.shape[1]),
        "dynamic_lags": int(config.dynamic_lags),
    }


def load_spectral_reference(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA:
            raise ValueError("unsupported spectral reference schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _config_from_reference(reference):
    return SpectralConfig(
        top_k=int(reference["top_k"]),
        prompt_bins=int(reference["prompt_bins"]),
        block_rows=int(reference["block_rows"]),
        position_bins=int(reference["position_bins"]),
        pca_dim=int(reference["pca_dim"]),
        reference_per_sample=int(reference["reference_per_sample"]),
        trim_fraction=float(reference["trim_fraction"]),
        neighbors=int(reference["neighbors"]),
        spectral_window=int(reference["spectral_window"]),
        dynamic_lags=int(reference["dynamic_lags"]),
        dynamic_ridge=float(reference["dynamic_ridge"]),
        logdet_alpha=float(reference["logdet_alpha"]),
        attribution_topk=int(reference["attribution_topk"]),
        epsilon=float(reference["epsilon"]),
    )


def score_spectral_dataset(dataset, reference_path, output_path, *, limit=None):
    """Score one split from a frozen train-only reference without labels."""
    reference = load_spectral_reference(reference_path)
    config = _config_from_reference(reference)
    sample_ids = _sample_ids(dataset, limit)
    columns = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "task_type",
            "data_source",
            "generator_model",
            "embedding",
            "innovation",
            "pca_residual",
            "dynamic_residual",
            "temporal_spectral_volume",
            "prompt_channel_volume",
            "knn_distance",
            "rr_residual_energy",
            "rp_residual_energy",
            "top_channel_index",
            "top_channel_energy",
            "score_static",
            "score_dynamic",
            "score_temporal_volume",
            "score_prompt_volume",
            "score_knn",
            "score",
            "score_component_count",
        )
    }

    for sample_id in tqdm(sample_ids, desc="spectral score"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            if (
                int(attention.num_layers) != int(reference["num_layers"])
                or int(attention.num_heads) != int(reference["num_heads"])
            ):
                raise ValueError("test attention geometry differs from spectral reference")
            response_count = int(attention.num_response_tokens)
            positions = np.arange(response_count, dtype=np.int64)
            states, prompt_volume = causal_spectral_state(
                sample, positions=positions, config=config
            )
            bins = _bins_for_positions(positions, response_count, config.position_bins)
            embedding, residual, residual_vector = _transform_states(
                states, bins, reference, return_residual_vector=True
            )

            innovation = np.zeros_like(embedding)
            if response_count > 1:
                raw_delta = states[1:] - states[:-1]
                scaled_delta = raw_delta / reference["state_scale"][bins[1:]]
                innovation[1:] = (
                    scaled_delta @ reference["pca_components"].T
                ) / reference["pca_whiten_scale"]
            innovation_standardized = (
                innovation - reference["delta_center"][bins]
            ) / reference["delta_scale"][bins]
            if response_count:
                innovation_standardized[0] = 0.0
            manifold = np.concatenate(
                (embedding, innovation_standardized), axis=1
            ).astype(np.float32)

            task = _metadata_text(sample.task_type)
            tasks = np.asarray([task] * response_count, dtype=str)
            knn = _query_reference_knn(reference, manifold, bins, tasks)
            dynamic_residual = _dynamic_residual(
                embedding,
                reference["dynamic_coef"],
                reference["dynamic_intercept"],
                config.dynamic_lags,
            )
            temporal_volume = spectral_volume(
                embedding,
                window=config.spectral_window,
                alpha=config.logdet_alpha,
                epsilon=config.epsilon,
            )
            temporal_absz = np.abs(
                (temporal_volume - reference["temporal_center"][bins])
                / reference["temporal_scale"][bins]
            )
            prompt_absz = np.abs(
                (prompt_volume - reference["prompt_center"][bins])
                / reference["prompt_scale"][bins]
            )

            score_static = _empirical_upper_tail(
                reference["calibration_static"],
                reference["calibration_bin"],
                residual,
                bins,
            )
            score_dynamic = _empirical_upper_tail(
                reference["calibration_dynamic"],
                reference["calibration_bin"],
                dynamic_residual,
                bins,
            )
            score_temporal = _empirical_upper_tail(
                reference["calibration_temporal"],
                reference["calibration_bin"],
                temporal_absz,
                bins,
            )
            score_prompt = _empirical_upper_tail(
                reference["calibration_prompt"],
                reference["calibration_bin"],
                prompt_absz,
                bins,
            )
            score_knn = _empirical_upper_tail(
                reference["calibration_knn"],
                reference["reference_bin"],
                knn,
                bins,
            )

            primary = np.stack(
                (score_static, score_dynamic, score_temporal, score_prompt), axis=1
            )
            valid = np.isfinite(primary)
            component_count = valid.sum(axis=1).astype(np.int8)
            score = (
                np.nansum(primary, axis=1) / np.maximum(component_count, 1)
            ).astype(np.float32)

            rr_energy, rp_energy, top_channel_index, top_channel_energy = (
                _channel_residual_attribution(
                    residual_vector,
                    num_channels=int(attention.num_channels),
                    top_k=config.top_k,
                    prompt_bins=config.prompt_bins,
                    attribution_topk=config.attribution_topk,
                )
            )

            text = lambda value: np.asarray(
                [_metadata_text(value)] * response_count, dtype=str
            )
            columns["sample_id"].append(text(sample.sample_id))
            columns["source_id"].append(text(sample.source_id))
            columns["token_index"].append(positions.astype(np.int32))
            columns["task_type"].append(tasks)
            columns["data_source"].append(text(sample.data_source))
            columns["generator_model"].append(text(sample.generator_model))
            columns["embedding"].append(embedding)
            columns["innovation"].append(innovation_standardized.astype(np.float32))
            columns["pca_residual"].append(residual)
            columns["dynamic_residual"].append(dynamic_residual)
            columns["temporal_spectral_volume"].append(temporal_volume)
            columns["prompt_channel_volume"].append(prompt_volume)
            columns["knn_distance"].append(knn)
            columns["rr_residual_energy"].append(rr_energy)
            columns["rp_residual_energy"].append(rp_energy)
            columns["top_channel_index"].append(top_channel_index)
            columns["top_channel_energy"].append(top_channel_energy)
            columns["score_static"].append(score_static)
            columns["score_dynamic"].append(score_dynamic)
            columns["score_temporal_volume"].append(score_temporal)
            columns["score_prompt_volume"].append(score_prompt)
            columns["score_knn"].append(score_knn)
            columns["score"].append(score)
            columns["score_component_count"].append(component_count)
        finally:
            sample.release_attention()

    output = {
        name: np.concatenate(values, axis=0) for name, values in columns.items()
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(Path(reference_path))),
        **output,
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["score"])),
        "embedding_dim": int(output["embedding"].shape[1]),
        "manifold_dim": int(
            output["embedding"].shape[1] + output["innovation"].shape[1]
        ),
    }


def load_score_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA:
            raise ValueError("unsupported spectral score schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _label_store_for_evaluation(dataset):
    try:
        return dataset.labels()
    except RuntimeError as error:
        if "every attention sample" not in str(error):
            raise
        for sample_id in dataset.sample_ids:
            sample = dataset[sample_id]
            sample.attention()
            sample.release_attention()
        return dataset.labels()


def _metrics(y, score):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    return {
        "tokens": int(len(y)),
        "positive_tokens": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "auprc_random_baseline": float(y.mean()),
    }


def evaluate_score_artifact(dataset, score_path, output_path):
    """Open labels only after all representations and anomaly scores are frozen."""
    artifact = load_score_artifact(score_path)
    labels = _label_store_for_evaluation(dataset)
    cache = {}
    y = np.empty(len(artifact["score"]), dtype=np.int64)
    for index, (sample_id, token_index) in enumerate(
        zip(artifact["sample_id"], artifact["token_index"], strict=True)
    ):
        sample_id = str(sample_id)
        if sample_id not in cache:
            sample = dataset[sample_id]
            cache[sample_id] = labels.response_labels(sample).cpu().numpy()
            sample.release_attention()
        y[index] = int(cache[sample_id][int(token_index)])

    metrics = _metrics(y, artifact["score"])
    if metrics is None:
        raise ValueError("evaluation requires both normal and hallucination tokens")
    component_metrics = {
        "static_subspace_residual": _metrics(y, artifact["score_static"]),
        "dynamic_prediction_residual": _metrics(y, artifact["score_dynamic"]),
        "temporal_spectral_volume": _metrics(y, artifact["score_temporal_volume"]),
        "prompt_channel_volume": _metrics(y, artifact["score_prompt_volume"]),
        "manifold_knn_diagnostic": _metrics(y, artifact["score_knn"]),
        "innovation_norm_diagnostic": _metrics(
            y, np.linalg.norm(artifact["innovation"], axis=1)
        ),
        "rr_residual_energy": _metrics(y, artifact["rr_residual_energy"]),
        "rp_residual_energy": _metrics(y, artifact["rp_residual_energy"]),
    }
    report = {
        "schema": "causal-spectral-subspace-evaluation-v1",
        "metrics": metrics,
        "components": component_metrics,
        "labels_used_during": "posthoc_evaluation_only",
        "representation": (
            "signed strongest-magnitude RR prefix LapEigvals + per-layer/head relative-prompt "
            "routing bins -> robust trimmed PCA/whitening; primary anomaly score is the train-only "
            "empirical-tail fusion of static subspace residual, causal ridge dynamics residual, "
            "temporal LogDet, and prompt-channel LogDet. kNN is diagnostic only."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
