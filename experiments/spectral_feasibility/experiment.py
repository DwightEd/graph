"""Label-free RR spectral-subspace fitting, scoring, and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .artifacts import (
    EVALUATION_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    file_sha256,
    load_score_artifact,
    load_spectral_reference,
)
from .representations import (
    SpectralConfig,
    prefix_laplacian_spectrum,
    reference_positions,
    response_position_bin,
    rr_spectral_dimension,
)
from .subspace import (
    empirical_upper_tail,
    fit_robust_pca,
    pca_artifact,
    position_location_scale,
    project_subspace,
    robust_location_scale,
    standardize_by_position,
)


def _sample_ids(dataset, limit=None):
    sample_ids = list(map(str, dataset.sample_ids))
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


def _group_key(sample) -> str:
    source_id = sample.source_id
    return (
        str(source_id)
        if source_id not in (None, "", "None", "null")
        else str(sample.sample_id)
    )


def _partition_sample_groups(dataset, sample_ids, *, fraction: float, seed: int):
    """Assign complete source groups to disjoint fit and calibration streams."""
    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        groups.setdefault(_group_key(sample), []).append(sample_id)
        sample.release_attention()
    if len(groups) < 2:
        raise ValueError(
            "RR reference fitting needs at least two independent source/sample groups"
        )

    def order_key(group: str):
        return hashlib.sha256(f"{int(seed)}:{group}".encode()).digest()

    ordered = sorted(groups, key=order_key)
    calibration_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * float(fraction)))),
    )
    calibration_groups = set(ordered[:calibration_count])
    role = {
        sample_id: ("calibration" if group in calibration_groups else "fit")
        for group, members in groups.items()
        for sample_id in members
    }
    return role, sorted(set(groups) - calibration_groups), sorted(calibration_groups)


def _channel_energy(residual_vector, *, num_channels: int, top_k: int):
    residual_vector = np.asarray(residual_vector, dtype=np.float32)
    expected = int(num_channels) * int(top_k)
    if residual_vector.ndim != 2 or residual_vector.shape[1] != expected:
        raise ValueError("RR residual vector does not match channel geometry")
    values = residual_vector.reshape(len(residual_vector), num_channels, top_k)
    return np.mean(np.square(values), axis=2).astype(np.float32)


def _localized_channel_anomaly(channel_energy, center, scale, *, tail_fraction):
    """Average the strongest standardized channel-residual tail."""
    energy = np.asarray(channel_energy, dtype=np.float32)
    normalized = np.maximum(
        (energy - np.asarray(center)) / np.asarray(scale),
        0.0,
    ).astype(np.float32)
    channels = normalized.shape[1]
    tail_count = min(
        channels,
        max(1, int(math.ceil(channels * float(tail_fraction)))),
    )
    strongest = np.partition(normalized, channels - tail_count, axis=1)[
        :, channels - tail_count :
    ]
    return strongest.mean(axis=1).astype(np.float32), normalized, tail_count


def _top_channels(channel_scores, *, count: int):
    values = np.asarray(channel_scores, dtype=np.float32)
    channels = values.shape[1]
    keep = min(max(1, int(count)), channels)
    selected = np.argpartition(values, channels - keep, axis=1)[:, -keep:]
    selected_scores = np.take_along_axis(values, selected, axis=1)
    order = np.argsort(selected_scores, axis=1)[:, ::-1]
    return (
        np.take_along_axis(selected, order, axis=1).astype(np.int32),
        np.take_along_axis(selected_scores, order, axis=1).astype(np.float32),
    )


def _collect_reference_rows(dataset, sample_ids, role, config):
    rows = {
        current: {"value": [], "bin": [], "sample": [], "token": []}
        for current in ("fit", "calibration")
    }
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
            positions = reference_positions(
                attention.num_response_tokens,
                config.reference_per_sample,
            )
            current = rows[role[sample_id]]
            current["value"].append(
                prefix_laplacian_spectrum(
                    sample,
                    positions=positions,
                    config=config,
                )
            )
            current["bin"].append(
                _bins_for_positions(
                    positions,
                    attention.num_response_tokens,
                    config.position_bins,
                )
            )
            current["sample"].extend([str(sample.sample_id)] * len(positions))
            current["token"].extend(map(int, positions))
        finally:
            sample.release_attention()

    for current in rows.values():
        current["value"] = np.concatenate(current["value"], axis=0).astype(
            np.float32,
            copy=False,
        )
        current["bin"] = np.concatenate(current["bin"], axis=0)
        if not bool(np.isfinite(current["value"]).all()):
            raise FloatingPointError("RR spectral reference contains non-finite values")
    return rows, geometry


def fit_spectral_reference(
    dataset,
    output_path,
    *,
    config: SpectralConfig | None = None,
    limit=None,
):
    """Fit on one unlabeled group stream and calibrate on a disjoint stream."""
    config = SpectralConfig() if config is None else config
    config.validate()
    sample_ids = _sample_ids(dataset, limit)
    role, fit_groups, calibration_groups = _partition_sample_groups(
        dataset,
        sample_ids,
        fraction=config.calibration_fraction,
        seed=config.split_seed,
    )
    rows, geometry = _collect_reference_rows(dataset, sample_ids, role, config)
    fit_rows = rows["fit"]
    calibration_rows = rows["calibration"]

    expected_dim = rr_spectral_dimension(*geometry, config.top_k)
    if fit_rows["value"].shape[1] != expected_dim:
        raise RuntimeError("unexpected RR spectral dimension")
    minimum_fit_rows = max(3, 4 * int(config.pca_dim))
    if len(fit_rows["value"]) < minimum_fit_rows:
        raise ValueError(
            "RR subspace fit is underdetermined: "
            f"{len(fit_rows['value'])} fit rows for dimension {config.pca_dim}; "
            f"use enough TRAIN_LIMIT samples to provide at least {minimum_fit_rows} rows"
        )
    if len(calibration_rows["value"]) < 2:
        raise ValueError("independent calibration stream has fewer than two tokens")

    rr_center, rr_scale = position_location_scale(
        fit_rows["value"],
        fit_rows["bin"],
        config.position_bins,
    )
    fit_standardized = standardize_by_position(
        fit_rows["value"], fit_rows["bin"], rr_center, rr_scale
    )
    calibration_standardized = standardize_by_position(
        calibration_rows["value"],
        calibration_rows["bin"],
        rr_center,
        rr_scale,
    )
    model, keep, provisional_residual = fit_robust_pca(
        fit_standardized,
        fit_rows["bin"],
        requested_dim=config.pca_dim,
        trim_fraction=config.trim_fraction,
        seed=config.split_seed,
        epsilon=config.epsilon,
    )
    model_artifact = pca_artifact(model, epsilon=config.epsilon)
    fit_projection = project_subspace(
        fit_standardized[keep],
        model_artifact,
    )
    calibration_projection = project_subspace(
        calibration_standardized,
        model_artifact,
    )

    fit_channel_energy = _channel_energy(
        fit_projection.residual_vector,
        num_channels=geometry[0] * geometry[1],
        top_k=config.top_k,
    )
    channel_center, channel_scale = robust_location_scale(fit_channel_energy)
    calibration_channel_energy = _channel_energy(
        calibration_projection.residual_vector,
        num_channels=geometry[0] * geometry[1],
        top_k=config.top_k,
    )
    localized, _, channel_tail_count = _localized_channel_anomaly(
        calibration_channel_energy,
        channel_center,
        channel_scale,
        tail_fraction=config.channel_tail_fraction,
    )

    artifact = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "num_layers": np.asarray(geometry[0], dtype=np.int16),
        "num_heads": np.asarray(geometry[1], dtype=np.int16),
        "top_k": np.asarray(config.top_k, dtype=np.int16),
        "block_rows": np.asarray(config.block_rows, dtype=np.int32),
        "position_bins": np.asarray(config.position_bins, dtype=np.int16),
        "subspace_dim": np.asarray(model.n_components_, dtype=np.int16),
        "reference_per_sample": np.asarray(
            config.reference_per_sample, dtype=np.int16
        ),
        "trim_fraction": np.asarray(config.trim_fraction, dtype=np.float32),
        "calibration_fraction": np.asarray(
            config.calibration_fraction, dtype=np.float32
        ),
        "split_seed": np.asarray(config.split_seed, dtype=np.int64),
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
        "calibration_rr_residual": calibration_projection.residual_energy,
        "calibration_rr_latent": calibration_projection.latent_energy,
        "calibration_rr_ppca": calibration_projection.ppca_energy,
        "calibration_rr_localized": localized,
        "fit_group_id": np.asarray(fit_groups, dtype=str),
        "calibration_group_id": np.asarray(calibration_groups, dtype=str),
        "fit_reference_sample_id": np.asarray(fit_rows["sample"], dtype=str),
        "fit_reference_token_index": np.asarray(
            fit_rows["token"], dtype=np.int32
        ),
        "fit_reference_keep": keep.astype(np.int8),
        "fit_provisional_residual": provisional_residual,
        "calibration_reference_sample_id": np.asarray(
            calibration_rows["sample"], dtype=str
        ),
        "calibration_reference_token_index": np.asarray(
            calibration_rows["token"], dtype=np.int32
        ),
        **model_artifact,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **artifact)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "fit_groups": len(fit_groups),
        "calibration_groups": len(calibration_groups),
        "fit_reference_tokens": int(len(fit_rows["value"])),
        "retained_fit_tokens": int(keep.sum()),
        "calibration_tokens": int(len(calibration_rows["value"])),
        "rr_spectral_dim": int(expected_dim),
        "embedding_dim": int(model.n_components_),
        "channel_tail_count": int(channel_tail_count),
    }


def _config_from_reference(reference):
    return SpectralConfig(
        top_k=int(reference["top_k"]),
        block_rows=int(reference["block_rows"]),
        position_bins=int(reference["position_bins"]),
        pca_dim=int(reference["subspace_dim"]),
        reference_per_sample=int(reference["reference_per_sample"]),
        trim_fraction=float(reference["trim_fraction"]),
        calibration_fraction=float(reference["calibration_fraction"]),
        split_seed=int(reference["split_seed"]),
        channel_tail_fraction=float(reference["channel_tail_fraction"]),
        attribution_topk=int(reference["attribution_topk"]),
        epsilon=float(reference["epsilon"]),
    )


def score_spectral_dataset(dataset, reference_path, output_path, *, limit=None):
    """Freeze token geometry and scores without opening labels."""
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
            "rr_latent_energy",
            "rr_ppca_energy",
            "rr_localized_residual",
            "top_channel_index",
            "top_channel_score",
            "score_rr_residual",
            "score_rr_latent",
            "score_rr_ppca",
            "score_rr_localized",
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
            bins = _bins_for_positions(
                positions,
                response_count,
                config.position_bins,
            )
            raw = prefix_laplacian_spectrum(
                sample,
                positions=positions,
                config=config,
            )
            standardized = standardize_by_position(
                raw,
                bins,
                reference["rr_center"],
                reference["rr_scale"],
            )
            projection = project_subspace(standardized, reference)
            channel_energy = _channel_energy(
                projection.residual_vector,
                num_channels=int(attention.num_channels),
                top_k=config.top_k,
            )
            localized, channel_score, _ = _localized_channel_anomaly(
                channel_energy,
                reference["channel_center"],
                reference["channel_scale"],
                tail_fraction=config.channel_tail_fraction,
            )
            top_channel_index, top_channel_score = _top_channels(
                channel_score,
                count=config.attribution_topk,
            )
            score_residual = empirical_upper_tail(
                reference["calibration_rr_residual"],
                projection.residual_energy,
            )
            score_latent = empirical_upper_tail(
                reference["calibration_rr_latent"],
                projection.latent_energy,
            )
            score_ppca = empirical_upper_tail(
                reference["calibration_rr_ppca"],
                projection.ppca_energy,
            )
            score_localized = empirical_upper_tail(
                reference["calibration_rr_localized"],
                localized,
            )

            def text(value):
                return np.asarray(
                    ["" if value is None else str(value)] * response_count,
                    dtype=str,
                )

            columns["sample_id"].append(text(sample.sample_id))
            columns["source_id"].append(text(sample.source_id))
            columns["token_index"].append(positions.astype(np.int32))
            columns["task_type"].append(text(sample.task_type))
            columns["data_source"].append(text(sample.data_source))
            columns["generator_model"].append(text(sample.generator_model))
            columns["rr_embedding"].append(projection.embedding)
            columns["rr_residual_energy"].append(projection.residual_energy)
            columns["rr_latent_energy"].append(projection.latent_energy)
            columns["rr_ppca_energy"].append(projection.ppca_energy)
            columns["rr_localized_residual"].append(localized)
            columns["top_channel_index"].append(top_channel_index)
            columns["top_channel_score"].append(top_channel_score)
            columns["score_rr_residual"].append(score_residual)
            columns["score_rr_latent"].append(score_latent)
            columns["score_rr_ppca"].append(score_ppca)
            columns["score_rr_localized"].append(score_localized)
            # One canonical score: orthogonal escape. The remaining distances
            # diagnose how the same geometry fails; they are never fused.
            columns["score"].append(score_residual)
        finally:
            sample.release_attention()

    output = {
        name: np.concatenate(values, axis=0) for name, values in columns.items()
    }
    numeric = (
        "rr_embedding",
        "rr_residual_energy",
        "rr_latent_energy",
        "rr_ppca_energy",
        "rr_localized_residual",
        "top_channel_score",
        "score_rr_residual",
        "score_rr_latent",
        "score_rr_ppca",
        "score_rr_localized",
        "score",
    )
    if any(not bool(np.isfinite(output[name]).all()) for name in numeric):
        raise FloatingPointError("RR spectral scoring produced non-finite values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(Path(reference_path))),
        reference_sha256=np.asarray(file_sha256(reference_path)),
        **output,
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["score"])),
        "embedding_dim": int(output["rr_embedding"].shape[1]),
        "primary_detector": "rr_subspace_residual_tail",
    }


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
    """Open labels only after every representation and score is frozen."""
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
    components = {
        "rr_subspace_residual_tail": _metrics(y, artifact["score_rr_residual"]),
        "rr_raw_residual_energy": _metrics(y, artifact["rr_residual_energy"]),
        "rr_in_subspace_tail": _metrics(y, artifact["score_rr_latent"]),
        "rr_ppca_tail": _metrics(y, artifact["score_rr_ppca"]),
        "rr_localized_channel_tail": _metrics(y, artifact["score_rr_localized"]),
        "rr_peak_channel_score": _metrics(y, artifact["top_channel_score"][:, 0]),
    }
    report = {
        "schema": EVALUATION_SCHEMA,
        "metrics": metrics,
        "components": components,
        "labels_used_during": "posthoc_evaluation_only",
        "primary_detector": "rr_subspace_residual_tail",
        "reference_sha256": str(np.asarray(artifact["reference_sha256"]).item()),
        "method": (
            "signed per-layer/head RR causal-Laplacian diagonal modes; "
            "fit-only position robust scaling; two-pass robust PCA; independent "
            "source-group calibration; global empirical residual tail"
        ),
        "claim_boundary": (
            "The coordinates are accumulated causal in-degree/diagonal modes, "
            "not eigenvector embeddings and not a learned message-passing model."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
