"""Fully label-free fitting, scoring, and evaluation for causal spectral states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
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


REFERENCE_SCHEMA = "causal-dual-spectrum-reference-v1"
SCORE_SCHEMA = "causal-dual-spectrum-score-v1"


def _robust_location_scale(values: np.ndarray, *, epsilon=1e-6):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    deviation = np.abs(values - center)
    mad = 1.4826 * np.median(deviation, axis=0)
    std = np.std(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(std > epsilon, std, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def _position_stats(values: np.ndarray, bins: np.ndarray, count: int):
    global_center, global_scale = _robust_location_scale(values)
    centers = np.empty((count, values.shape[1]), dtype=np.float32)
    scales = np.empty_like(centers)
    for position_bin in range(count):
        selected = bins == position_bin
        if int(selected.sum()) >= 2:
            centers[position_bin], scales[position_bin] = _robust_location_scale(
                values[selected]
            )
        else:
            centers[position_bin] = global_center
            scales[position_bin] = global_scale
    return centers, scales


def _scalar_position_stats(values: np.ndarray, bins: np.ndarray, count: int):
    center, scale = _position_stats(
        np.asarray(values, dtype=np.float32).reshape(-1, 1), bins, count
    )
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
        [
            response_position_bin(int(position), response_count, position_bins)
            for position in positions
        ],
        dtype=np.int16,
    )


def _transform_states(states, bins, reference):
    values = np.asarray(states, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    standardized = (
        values - reference["state_center"][bins]
    ) / reference["state_scale"][bins]
    centered = standardized - reference["pca_mean"]
    scores = centered @ reference["pca_components"].T
    embedding = scores / reference["pca_whiten_scale"]
    reconstructed = scores @ reference["pca_components"] + reference["pca_mean"]
    residual = np.mean(np.square(standardized - reconstructed), axis=1)
    return (
        embedding.astype(np.float32, copy=False),
        residual.astype(np.float32, copy=False),
    )


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


def _states_for_targets(sample, targets, config, *, include_window=False):
    needed = set()
    for target in map(int, targets):
        needed.add(target)
        if target > 0:
            needed.add(target - 1)
        if include_window:
            needed.update(
                range(max(0, target - config.spectral_window + 1), target + 1)
            )
    positions = np.asarray(sorted(needed), dtype=np.int64)
    states, prompt_volume = causal_spectral_state(
        sample, positions=positions, config=config
    )
    return positions, states, prompt_volume


def _metadata_text(value):
    return "" if value is None else str(value)


def fit_spectral_reference(
    dataset,
    output_path,
    *,
    config: SpectralConfig | None = None,
    limit=None,
):
    """Fit the normal causal dual-spectrum manifold without opening labels."""
    config = SpectralConfig() if config is None else config
    config.validate()
    sample_ids = _sample_ids(dataset, limit)

    raw_states = []
    raw_bins = []
    audit_sample = []
    audit_token = []
    geometry = None

    # Pass 1: sample deterministic causal prefixes and learn robust position
    # normalization plus unsupervised layer/head combinations.
    for sample_id in tqdm(sample_ids, desc="spectral fit pass 1"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            current_geometry = (int(attention.num_layers), int(attention.num_heads))
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError("attention geometry changes inside the split")
            targets = reference_positions(
                attention.num_response_tokens, config.reference_per_sample
            )
            states, _prompt_volume = causal_spectral_state(
                sample, positions=targets, config=config
            )
            raw_states.append(states)
            raw_bins.append(
                _bins_for_positions(
                    targets, attention.num_response_tokens, config.position_bins
                )
            )
            audit_sample.extend([str(sample.sample_id)] * len(targets))
            audit_token.extend(map(int, targets))
        finally:
            sample.release_attention()

    states = np.concatenate(raw_states, axis=0)
    bins = np.concatenate(raw_bins, axis=0)
    if len(states) < 3:
        raise ValueError("at least three train reference tokens are required")
    expected_dim = spectral_state_dimension(
        geometry[0], geometry[1], config.top_k, config.prompt_sketch_dim
    )
    if states.shape[1] != expected_dim:
        raise RuntimeError("unexpected channel-preserving spectral dimension")

    state_center, state_scale = _position_stats(states, bins, config.position_bins)
    standardized = (states - state_center[bins]) / state_scale[bins]
    pca_dim = min(config.pca_dim, standardized.shape[0] - 1, standardized.shape[1])
    if pca_dim < 1:
        raise ValueError("PCA has no valid component")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0)
    pca.fit(standardized)
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
    del standardized, states

    # Pass 2: freeze coordinates and measure current state, one-step spectral
    # innovation, temporal spectral volume, and RP channel-volume.
    ref_embedding = []
    ref_delta = []
    ref_residual = []
    ref_temporal_volume = []
    ref_prompt_volume = []
    ref_bins = []
    ref_tasks = []
    for sample_id in tqdm(sample_ids, desc="spectral fit pass 2"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            targets = reference_positions(
                attention.num_response_tokens, config.reference_per_sample
            )
            positions, local_states, local_prompt_volume = _states_for_targets(
                sample, targets, config, include_window=True
            )
            local_bins = _bins_for_positions(
                positions, attention.num_response_tokens, config.position_bins
            )
            local_embedding, local_residual = _transform_states(
                local_states, local_bins, proto
            )
            by_position = {
                int(position): index for index, position in enumerate(positions)
            }
            task = _metadata_text(sample.task_type)
            for target in targets.tolist():
                index = by_position[int(target)]
                current = local_embedding[index]
                if target > 0:
                    previous = local_embedding[by_position[int(target - 1)]]
                    delta = current - previous
                else:
                    delta = np.zeros_like(current)
                start = max(0, target - config.spectral_window + 1)
                window_indices = [
                    by_position[position] for position in range(start, target + 1)
                ]
                temporal_volume = spectral_volume(
                    local_embedding[window_indices],
                    window=config.spectral_window,
                    alpha=config.logdet_alpha,
                    epsilon=config.epsilon,
                )[-1]
                ref_embedding.append(current)
                ref_delta.append(delta)
                ref_residual.append(local_residual[index])
                ref_temporal_volume.append(temporal_volume)
                ref_prompt_volume.append(local_prompt_volume[index])
                ref_bins.append(
                    response_position_bin(
                        target, attention.num_response_tokens, config.position_bins
                    )
                )
                ref_tasks.append(task)
        finally:
            sample.release_attention()

    ref_embedding = np.asarray(ref_embedding, dtype=np.float32)
    ref_delta = np.asarray(ref_delta, dtype=np.float32)
    ref_residual = np.asarray(ref_residual, dtype=np.float32)
    ref_temporal_volume = np.asarray(ref_temporal_volume, dtype=np.float32)
    ref_prompt_volume = np.asarray(ref_prompt_volume, dtype=np.float32)
    ref_bins = np.asarray(ref_bins, dtype=np.int16)
    ref_tasks = np.asarray(ref_tasks, dtype=str)

    delta_center, delta_scale = _position_stats(
        ref_delta, ref_bins, config.position_bins
    )
    delta_standardized = (
        ref_delta - delta_center[ref_bins]
    ) / delta_scale[ref_bins]
    reference_manifold = np.concatenate(
        (ref_embedding, delta_standardized), axis=1
    ).astype(np.float32, copy=False)

    train_knn = np.zeros(len(reference_manifold), dtype=np.float32)
    for task in np.unique(ref_tasks):
        for position_bin in range(config.position_bins):
            selected = (ref_tasks == task) & (ref_bins == position_bin)
            if selected.sum() < 2:
                continue
            train_knn[selected] = _knn_distances(
                reference_manifold[selected],
                reference_manifold[selected],
                config.neighbors,
                self_query=True,
            )
    # Rare empty task-position cells fall back to position-only neighbors.
    missing = train_knn == 0
    for position_bin in range(config.position_bins):
        selected = missing & (ref_bins == position_bin)
        if not bool(selected.any()):
            continue
        source = ref_bins == position_bin
        train_knn[selected] = _knn_distances(
            reference_manifold[source],
            reference_manifold[selected],
            config.neighbors,
            self_query=False,
        )

    knn_center, knn_scale = _scalar_position_stats(
        train_knn, ref_bins, config.position_bins
    )
    residual_center, residual_scale = _scalar_position_stats(
        ref_residual, ref_bins, config.position_bins
    )
    temporal_volume_center, temporal_volume_scale = _scalar_position_stats(
        ref_temporal_volume, ref_bins, config.position_bins
    )
    prompt_volume_center, prompt_volume_scale = _scalar_position_stats(
        ref_prompt_volume, ref_bins, config.position_bins
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        num_layers=np.asarray(geometry[0], dtype=np.int16),
        num_heads=np.asarray(geometry[1], dtype=np.int16),
        top_k=np.asarray(config.top_k, dtype=np.int16),
        prompt_sketch_dim=np.asarray(config.prompt_sketch_dim, dtype=np.int16),
        prompt_sketch_seed=np.asarray(config.prompt_sketch_seed, dtype=np.int64),
        block_rows=np.asarray(config.block_rows, dtype=np.int32),
        position_bins=np.asarray(config.position_bins, dtype=np.int16),
        pca_dim=np.asarray(pca_dim, dtype=np.int16),
        reference_per_sample=np.asarray(config.reference_per_sample, dtype=np.int16),
        neighbors=np.asarray(config.neighbors, dtype=np.int16),
        spectral_window=np.asarray(config.spectral_window, dtype=np.int16),
        logdet_alpha=np.asarray(config.logdet_alpha, dtype=np.float32),
        epsilon=np.asarray(config.epsilon, dtype=np.float32),
        state_center=state_center,
        state_scale=state_scale,
        pca_mean=proto["pca_mean"],
        pca_components=proto["pca_components"],
        pca_whiten_scale=whiten_scale,
        delta_center=delta_center,
        delta_scale=delta_scale,
        reference_manifold=reference_manifold,
        reference_bin=ref_bins,
        reference_task=ref_tasks,
        knn_center=knn_center,
        knn_scale=knn_scale,
        residual_center=residual_center,
        residual_scale=residual_scale,
        temporal_volume_center=temporal_volume_center,
        temporal_volume_scale=temporal_volume_scale,
        prompt_volume_center=prompt_volume_center,
        prompt_volume_scale=prompt_volume_scale,
        reference_sample_id=np.asarray(audit_sample, dtype=str),
        reference_token_index=np.asarray(audit_token, dtype=np.int32),
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "reference_tokens": int(len(reference_manifold)),
        "raw_spectral_dim": int(expected_dim),
        "embedding_dim": int(pca_dim),
        "manifold_dim": int(reference_manifold.shape[1]),
    }


def load_spectral_reference(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA:
            raise ValueError("unsupported spectral reference schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _config_from_reference(reference):
    return SpectralConfig(
        top_k=int(reference["top_k"]),
        prompt_sketch_dim=int(reference["prompt_sketch_dim"]),
        prompt_sketch_seed=int(reference["prompt_sketch_seed"]),
        block_rows=int(reference["block_rows"]),
        position_bins=int(reference["position_bins"]),
        pca_dim=int(reference["pca_dim"]),
        reference_per_sample=int(reference["reference_per_sample"]),
        neighbors=int(reference["neighbors"]),
        spectral_window=int(reference["spectral_window"]),
        logdet_alpha=float(reference["logdet_alpha"]),
        epsilon=float(reference["epsilon"]),
    )


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


def score_spectral_dataset(dataset, reference_path, output_path, *, limit=None):
    """Score one split from a frozen reference without opening labels."""
    reference = load_spectral_reference(reference_path)
    config = _config_from_reference(reference)
    sample_ids = _sample_ids(dataset, limit)
    columns = {
        name: []
        for name in (
            "sample_id", "source_id", "token_index", "task_type",
            "data_source", "generator_model", "embedding", "innovation",
            "knn_distance", "pca_residual", "temporal_spectral_volume",
            "prompt_channel_volume", "score_knn", "score_residual",
            "score_temporal_volume", "score_prompt_volume", "score",
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
                raise ValueError(
                    "test attention geometry differs from spectral reference"
                )
            response_count = int(attention.num_response_tokens)
            positions = np.arange(response_count, dtype=np.int64)
            states, prompt_volume = causal_spectral_state(
                sample, positions=positions, config=config
            )
            bins = _bins_for_positions(
                positions, response_count, config.position_bins
            )
            embedding, residual = _transform_states(states, bins, reference)
            innovation = np.zeros_like(embedding)
            if response_count > 1:
                innovation[1:] = embedding[1:] - embedding[:-1]
            innovation_standardized = (
                innovation - reference["delta_center"][bins]
            ) / reference["delta_scale"][bins]
            if response_count:
                innovation_standardized[0] = 0.0
            manifold = np.concatenate(
                (embedding, innovation_standardized), axis=1
            ).astype(np.float32, copy=False)

            task = _metadata_text(sample.task_type)
            tasks = np.asarray([task] * response_count, dtype=str)
            knn = _query_reference_knn(reference, manifold, bins, tasks)
            temporal_volume = spectral_volume(
                embedding,
                window=config.spectral_window,
                alpha=config.logdet_alpha,
                epsilon=config.epsilon,
            )

            knn_z = np.maximum(
                (knn - reference["knn_center"][bins])
                / reference["knn_scale"][bins],
                0.0,
            )
            residual_z = np.maximum(
                (residual - reference["residual_center"][bins])
                / reference["residual_scale"][bins],
                0.0,
            )
            temporal_volume_z = np.abs(
                (temporal_volume - reference["temporal_volume_center"][bins])
                / reference["temporal_volume_scale"][bins]
            )
            prompt_volume_z = np.abs(
                (prompt_volume - reference["prompt_volume_center"][bins])
                / reference["prompt_volume_scale"][bins]
            )
            score = np.sqrt(
                (
                    np.square(knn_z)
                    + np.square(residual_z)
                    + np.square(temporal_volume_z)
                    + np.square(prompt_volume_z)
                )
                / 4.0
            ).astype(np.float32)

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
            columns["innovation"].append(
                innovation_standardized.astype(np.float32)
            )
            columns["knn_distance"].append(knn)
            columns["pca_residual"].append(residual)
            columns["temporal_spectral_volume"].append(temporal_volume)
            columns["prompt_channel_volume"].append(prompt_volume)
            columns["score_knn"].append(knn_z.astype(np.float32))
            columns["score_residual"].append(residual_z.astype(np.float32))
            columns["score_temporal_volume"].append(
                temporal_volume_z.astype(np.float32)
            )
            columns["score_prompt_volume"].append(
                prompt_volume_z.astype(np.float32)
            )
            columns["score"].append(score)
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
    if np.unique(y).size < 2:
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
    """Open labels only after representation and anomaly scores are frozen."""
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
        "manifold_knn": _metrics(y, artifact["score_knn"]),
        "pca_residual": _metrics(y, artifact["score_residual"]),
        "temporal_spectral_volume": _metrics(
            y, artifact["score_temporal_volume"]
        ),
        "prompt_channel_volume": _metrics(
            y, artifact["score_prompt_volume"]
        ),
        "innovation_norm": _metrics(
            y, np.linalg.norm(artifact["innovation"], axis=1)
        ),
    }
    report = {
        "schema": "causal-dual-spectrum-evaluation-v1",
        "metrics": metrics,
        "components": component_metrics,
        "labels_used_during": "posthoc_evaluation_only",
        "representation": (
            "RR causal per-layer/head prefix LapEigvals + RP channel-preserving "
            "prompt transport sketch -> robust PCA/whitening -> current spectral "
            "state + one-step innovation manifold; LogDet volumes are calibrated "
            "as independent label-free anomaly components"
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
