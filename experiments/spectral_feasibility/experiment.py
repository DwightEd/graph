"""Fully label-free RR spectral-subspace fitting, scoring, and evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .representations import (
    SpectralConfig,
    prefix_laplacian_spectrum,
    reference_positions,
    response_position_bin,
    rr_spectral_dimension,
)


REFERENCE_SCHEMA = "rr-spectral-subspace-reference-v2"
SCORE_SCHEMA = "rr-spectral-subspace-score-v2"


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
            centers[position_bin], scales[position_bin] = _robust_location_scale(
                values[selected]
            )
        else:
            centers[position_bin] = global_center
            scales[position_bin] = global_scale
    return centers, scales


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


def _fit_pca(values: np.ndarray, requested_dim: int) -> PCA:
    values = np.asarray(values, dtype=np.float32)
    dim = min(int(requested_dim), len(values) - 1, values.shape[1])
    if dim < 1:
        raise ValueError("PCA has no valid component")
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=0)
    pca.fit(values)
    return pca


def _pca_residual(values: np.ndarray, pca: PCA) -> np.ndarray:
    scores = pca.transform(values)
    reconstructed = pca.inverse_transform(scores)
    return np.mean(np.square(values - reconstructed), axis=1).astype(np.float32)


def _fit_trimmed_pca(values, bins, config: SpectralConfig):
    """Fit an RR-only provisional PCA, trim its upper residual tail, then refit."""
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int16)
    if len(values) < 3:
        raise ValueError("at least three train reference tokens are required")

    probe_dim = min(config.pca_dim, 16, len(values) - 1, values.shape[1])
    probe = _fit_pca(values, max(1, probe_dim))
    probe_residual = _pca_residual(values, probe)

    keep = np.ones(len(values), dtype=bool)
    if config.trim_fraction < 1.0:
        for position_bin in range(config.position_bins):
            selected = np.flatnonzero(bins == position_bin)
            if len(selected) < 4:
                continue
            threshold = float(
                np.quantile(probe_residual[selected], config.trim_fraction)
            )
            keep[selected] = probe_residual[selected] <= threshold
    if int(keep.sum()) < 3:
        keep[:] = True

    pca = _fit_pca(values[keep], config.pca_dim)
    return pca, keep, probe_residual


def _standardize_rr(states, bins, reference):
    values = np.asarray(states, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    return (
        values - reference["rr_center"][bins]
    ) / reference["rr_scale"][bins]


def _transform_with_model(
    standardized,
    *,
    mean,
    components,
    whiten_scale=None,
    return_residual_vector=False,
):
    standardized = np.asarray(standardized, dtype=np.float32)
    centered = standardized - np.asarray(mean, dtype=np.float32)
    components = np.asarray(components, dtype=np.float32)
    scores = centered @ components.T
    reconstructed = scores @ components + np.asarray(mean, dtype=np.float32)
    residual_vector = standardized - reconstructed
    residual = np.mean(np.square(residual_vector), axis=1).astype(np.float32)
    if whiten_scale is None:
        embedding = scores.astype(np.float32, copy=False)
    else:
        embedding = (
            scores / np.asarray(whiten_scale, dtype=np.float32)
        ).astype(np.float32, copy=False)
    if return_residual_vector:
        return embedding, residual, residual_vector.astype(np.float32, copy=False)
    return embedding, residual


def _empirical_upper_tail(reference_values, reference_bins, values, bins, *, epsilon=1e-12):
    """Convert anomaly magnitudes to position-conditioned train-only -log p tails."""
    reference_values = np.asarray(reference_values, dtype=np.float64)
    reference_bins = np.asarray(reference_bins, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    result = np.full(len(values), np.nan, dtype=np.float32)
    finite_reference = reference_values[np.isfinite(reference_values)]
    if len(finite_reference) == 0:
        raise ValueError("empirical calibration has no finite train values")
    global_reference = np.sort(finite_reference)
    for position_bin in np.unique(bins):
        query = (bins == position_bin) & np.isfinite(values)
        if not bool(query.any()):
            continue
        reference = reference_values[
            (reference_bins == position_bin) & np.isfinite(reference_values)
        ]
        reference = np.sort(reference) if len(reference) >= 2 else global_reference
        count_ge = len(reference) - np.searchsorted(
            reference, values[query], side="left"
        )
        probability = (count_ge + 1.0) / (len(reference) + 1.0)
        result[query] = -np.log(np.maximum(probability, epsilon)).astype(np.float32)
    return result


def _channel_energy(residual_vector, *, num_channels: int, top_k: int):
    residual_vector = np.asarray(residual_vector, dtype=np.float32)
    expected = int(num_channels) * int(top_k)
    if residual_vector.ndim != 2 or residual_vector.shape[1] != expected:
        raise ValueError("RR residual vector does not match channel geometry")
    values = residual_vector.reshape(len(residual_vector), num_channels, top_k)
    return np.mean(np.square(values), axis=2).astype(np.float32)


def _localized_channel_anomaly(
    channel_energy,
    bins,
    center,
    scale,
    *,
    tail_fraction: float,
):
    """Aggregate only the strongest channel-normalized RR residual tail.

    Channel energy is first standardized against its own train distribution in
    the same response-position bin.  The fixed upper fraction is then averaged,
    so a small set of abnormal heads is not diluted by all 1024 channels.
    """
    energy = np.asarray(channel_energy, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.int64)
    normalized = np.maximum(
        (energy - np.asarray(center)[bins]) / np.asarray(scale)[bins], 0.0
    ).astype(np.float32)
    channels = normalized.shape[1]
    tail_count = min(
        channels,
        max(1, int(math.ceil(channels * float(tail_fraction)))),
    )
    start = channels - tail_count
    strongest = np.partition(normalized, start, axis=1)[:, start:]
    aggregate = strongest.mean(axis=1).astype(np.float32)
    return aggregate, normalized, tail_count


def _top_channels(channel_scores, *, count: int):
    values = np.asarray(channel_scores, dtype=np.float32)
    channels = values.shape[1]
    keep = min(max(1, int(count)), channels)
    selected = np.argpartition(values, channels - keep, axis=1)[:, -keep:]
    selected_scores = np.take_along_axis(values, selected, axis=1)
    local_order = np.argsort(selected_scores, axis=1)[:, ::-1]
    indices = np.take_along_axis(selected, local_order, axis=1)
    scores = np.take_along_axis(values, indices, axis=1)
    return indices.astype(np.int32), scores.astype(np.float32)


def _pca_artifact(prefix: str, pca: PCA, *, epsilon: float, whiten: bool):
    output = {
        f"{prefix}_mean": pca.mean_.astype(np.float32),
        f"{prefix}_components": pca.components_.astype(np.float32),
    }
    if whiten:
        output[f"{prefix}_whiten_scale"] = np.sqrt(
            np.maximum(pca.explained_variance_.astype(np.float64), epsilon)
        ).astype(np.float32)
    return output


def fit_spectral_reference(
    dataset,
    output_path,
    *,
    config: SpectralConfig | None = None,
    limit=None,
):
    """Fit RR-only robust/untrimmed subspaces without opening labels."""
    config = SpectralConfig() if config is None else config
    config.validate()
    sample_ids = _sample_ids(dataset, limit)

    raw_rr = []
    raw_bins = []
    audit_sample = []
    audit_token = []
    geometry = None

    for sample_id in tqdm(sample_ids, desc="RR spectral train references"):
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
            rr = prefix_laplacian_spectrum(
                sample, positions=targets, config=config
            )
            raw_rr.append(rr)
            raw_bins.append(
                _bins_for_positions(
                    targets,
                    attention.num_response_tokens,
                    config.position_bins,
                )
            )
            audit_sample.extend([str(sample.sample_id)] * len(targets))
            audit_token.extend(map(int, targets))
        finally:
            sample.release_attention()

    rr = np.concatenate(raw_rr, axis=0).astype(np.float32, copy=False)
    bins = np.concatenate(raw_bins, axis=0)
    expected_dim = rr_spectral_dimension(
        geometry[0], geometry[1], config.top_k
    )
    if rr.shape[1] != expected_dim:
        raise RuntimeError("unexpected RR spectral dimension")

    rr_center, rr_scale = _position_stats(rr, bins, config.position_bins)
    standardized = (rr - rr_center[bins]) / rr_scale[bins]

    trimmed_pca, keep, probe_residual = _fit_trimmed_pca(
        standardized, bins, config
    )
    untrimmed_pca = _fit_pca(standardized, config.pca_dim)

    trimmed_whiten = np.sqrt(
        np.maximum(
            trimmed_pca.explained_variance_.astype(np.float64),
            config.epsilon,
        )
    ).astype(np.float32)
    _, trimmed_residual, trimmed_vector = _transform_with_model(
        standardized,
        mean=trimmed_pca.mean_,
        components=trimmed_pca.components_,
        whiten_scale=trimmed_whiten,
        return_residual_vector=True,
    )
    _, untrimmed_residual = _transform_with_model(
        standardized,
        mean=untrimmed_pca.mean_,
        components=untrimmed_pca.components_,
    )

    channels = int(geometry[0]) * int(geometry[1])
    channel_energy = _channel_energy(
        trimmed_vector,
        num_channels=channels,
        top_k=config.top_k,
    )
    channel_center, channel_scale = _position_stats(
        channel_energy[keep],
        bins[keep],
        config.position_bins,
    )
    localized_raw, _, channel_tail_count = _localized_channel_anomaly(
        channel_energy,
        bins,
        channel_center,
        channel_scale,
        tail_fraction=config.channel_tail_fraction,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "num_layers": np.asarray(geometry[0], dtype=np.int16),
        "num_heads": np.asarray(geometry[1], dtype=np.int16),
        "top_k": np.asarray(config.top_k, dtype=np.int16),
        "block_rows": np.asarray(config.block_rows, dtype=np.int32),
        "position_bins": np.asarray(config.position_bins, dtype=np.int16),
        "pca_dim": np.asarray(trimmed_pca.n_components_, dtype=np.int16),
        "reference_per_sample": np.asarray(
            config.reference_per_sample, dtype=np.int16
        ),
        "trim_fraction": np.asarray(config.trim_fraction, dtype=np.float32),
        "channel_tail_fraction": np.asarray(
            config.channel_tail_fraction, dtype=np.float32
        ),
        "channel_tail_count": np.asarray(channel_tail_count, dtype=np.int16),
        "attribution_topk": np.asarray(config.attribution_topk, dtype=np.int16),
        "epsilon": np.asarray(config.epsilon, dtype=np.float32),
        "rr_center": rr_center,
        "rr_scale": rr_scale,
        "channel_center": channel_center,
        "channel_scale": channel_scale,
        "calibration_bin": bins[keep],
        "calibration_rr_global": trimmed_residual[keep],
        "calibration_rr_localized": localized_raw[keep],
        "untrimmed_calibration_bin": bins,
        "calibration_rr_untrimmed": untrimmed_residual,
        "reference_sample_id": np.asarray(audit_sample, dtype=str),
        "reference_token_index": np.asarray(audit_token, dtype=np.int32),
        "reference_keep": keep.astype(np.int8),
        "probe_residual": probe_residual,
    }
    artifact.update(
        _pca_artifact(
            "rr_pca",
            trimmed_pca,
            epsilon=config.epsilon,
            whiten=True,
        )
    )
    artifact.update(
        _pca_artifact(
            "rr_untrimmed_pca",
            untrimmed_pca,
            epsilon=config.epsilon,
            whiten=False,
        )
    )
    np.savez_compressed(output_path, **artifact)

    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "reference_tokens": int(len(rr)),
        "trimmed_reference_tokens": int(keep.sum()),
        "rr_spectral_dim": int(expected_dim),
        "embedding_dim": int(trimmed_pca.n_components_),
        "channel_tail_count": int(channel_tail_count),
    }


def load_spectral_reference(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA:
            raise ValueError("unsupported RR spectral reference schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _config_from_reference(reference):
    return SpectralConfig(
        top_k=int(reference["top_k"]),
        block_rows=int(reference["block_rows"]),
        position_bins=int(reference["position_bins"]),
        pca_dim=int(reference["pca_dim"]),
        reference_per_sample=int(reference["reference_per_sample"]),
        trim_fraction=float(reference["trim_fraction"]),
        channel_tail_fraction=float(reference["channel_tail_fraction"]),
        attribution_topk=int(reference["attribution_topk"]),
        epsilon=float(reference["epsilon"]),
    )


def score_spectral_dataset(dataset, reference_path, output_path, *, limit=None):
    """Score RR spectral escape from a frozen train-only reference."""
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
            "rr_embedding",
            "rr_residual_energy",
            "rr_localized_residual",
            "rr_untrimmed_residual",
            "top_channel_index",
            "top_channel_score",
            "score_rr_global",
            "score_rr_localized",
            "score_rr_untrimmed_ablation",
            "score",
        )
    }

    for sample_id in tqdm(sample_ids, desc="RR spectral score"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            if (
                int(attention.num_layers) != int(reference["num_layers"])
                or int(attention.num_heads) != int(reference["num_heads"])
            ):
                raise ValueError(
                    "test attention geometry differs from RR spectral reference"
                )
            response_count = int(attention.num_response_tokens)
            positions = np.arange(response_count, dtype=np.int64)
            rr = prefix_laplacian_spectrum(
                sample, positions=positions, config=config
            )
            bins = _bins_for_positions(
                positions, response_count, config.position_bins
            )
            standardized = _standardize_rr(rr, bins, reference)
            embedding, residual, residual_vector = _transform_with_model(
                standardized,
                mean=reference["rr_pca_mean"],
                components=reference["rr_pca_components"],
                whiten_scale=reference["rr_pca_whiten_scale"],
                return_residual_vector=True,
            )
            _, untrimmed_residual = _transform_with_model(
                standardized,
                mean=reference["rr_untrimmed_pca_mean"],
                components=reference["rr_untrimmed_pca_components"],
            )

            channel_energy = _channel_energy(
                residual_vector,
                num_channels=int(attention.num_channels),
                top_k=config.top_k,
            )
            localized_raw, channel_score, _ = _localized_channel_anomaly(
                channel_energy,
                bins,
                reference["channel_center"],
                reference["channel_scale"],
                tail_fraction=config.channel_tail_fraction,
            )
            top_channel_index, top_channel_score = _top_channels(
                channel_score,
                count=config.attribution_topk,
            )

            score_global = _empirical_upper_tail(
                reference["calibration_rr_global"],
                reference["calibration_bin"],
                residual,
                bins,
            )
            score_localized = _empirical_upper_tail(
                reference["calibration_rr_localized"],
                reference["calibration_bin"],
                localized_raw,
                bins,
            )
            score_untrimmed = _empirical_upper_tail(
                reference["calibration_rr_untrimmed"],
                reference["untrimmed_calibration_bin"],
                untrimmed_residual,
                bins,
            )

            text = lambda value: np.asarray(
                ["" if value is None else str(value)] * response_count,
                dtype=str,
            )
            columns["sample_id"].append(text(sample.sample_id))
            columns["source_id"].append(text(sample.source_id))
            columns["token_index"].append(positions.astype(np.int32))
            columns["task_type"].append(text(sample.task_type))
            columns["data_source"].append(text(sample.data_source))
            columns["generator_model"].append(text(sample.generator_model))
            columns["rr_embedding"].append(embedding)
            columns["rr_residual_energy"].append(residual)
            columns["rr_localized_residual"].append(localized_raw)
            columns["rr_untrimmed_residual"].append(untrimmed_residual)
            columns["top_channel_index"].append(top_channel_index)
            columns["top_channel_score"].append(top_channel_score)
            columns["score_rr_global"].append(score_global)
            columns["score_rr_localized"].append(score_localized)
            columns["score_rr_untrimmed_ablation"].append(score_untrimmed)
            # Primary detector is deliberately RR-global only.  Diagnostics do
            # not enter the deployed score until a separate validation supports it.
            columns["score"].append(score_global)
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
        "embedding_dim": int(output["rr_embedding"].shape[1]),
        "primary_detector": "rr_trimmed_subspace_tail",
    }


def load_score_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA:
            raise ValueError("unsupported RR spectral score schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _label_store_for_evaluation(dataset):
    try:
        return dataset.labels()
    except RuntimeError as error:
        if "every attention sample" not in str(error):
            raise
        for sample_id in tqdm(
            dataset.sample_ids,
            desc="unlock evaluation labels",
            unit="sample",
        ):
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
    """Open labels only after every RR representation and score is frozen."""
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
        "rr_trimmed_subspace": _metrics(y, artifact["score_rr_global"]),
        "rr_localized_channel_tail": _metrics(y, artifact["score_rr_localized"]),
        "rr_untrimmed_pca_ablation": _metrics(
            y, artifact["score_rr_untrimmed_ablation"]
        ),
        "rr_raw_residual_energy": _metrics(y, artifact["rr_residual_energy"]),
        "rr_localized_raw": _metrics(y, artifact["rr_localized_residual"]),
        "rr_peak_channel_score": _metrics(y, artifact["top_channel_score"][:, 0]),
    }
    report = {
        "schema": "rr-spectral-subspace-evaluation-v2",
        "metrics": metrics,
        "components": component_metrics,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_detector": "rr_trimmed_subspace_tail",
        "representation": (
            "signed strongest-magnitude per-layer/head RR causal Laplacian spectra "
            "-> position-robust normalization -> RR-only trimmed PCA. The primary "
            "score is the train-only empirical upper tail of RR reconstruction "
            "error. Untrimmed PCA and localized channel-tail scores are frozen "
            "diagnostic ablations and never enter the primary score."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
