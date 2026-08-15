"""Fit, extract, and evaluate causal RR topology-dynamics features.

The representation and train reference are label-blind. Correct/hallucination
labels are opened only in :func:`evaluate_topology_artifact`, after every
feature has been frozen on disk. This experiment is a mechanism audit rather
than a new classifier: it measures route convergence, grounding, and the
layer/head/source/lag origin of RR spectral-subspace escape.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from experiments.spectral_feasibility.representations import reference_positions

from .features import (
    SCALAR_FEATURE_NAMES,
    TopologyDynamicsConfig,
    extract_sample_topology_dynamics,
    load_rr_reference,
)


REFERENCE_SCHEMA = "rr-topology-dynamics-reference-v1"
SCORE_SCHEMA = "rr-topology-dynamics-features-v1"

CONVERGENCE_FEATURES = (
    "route_effective_rank",
    "route_participation_rank",
    "route_top1_energy_share",
    "cross_head_route_consensus",
    "source_effective_number",
    "source_entropy",
    "source_top1_share",
    "channel_route_velocity",
    "source_route_velocity",
    "anchor_turnover",
    "offline_route_distance_to_final",
    "offline_source_distance_to_final",
)

GROUNDING_FEATURES = (
    "direct_prompt_share",
    "prompt_groundedness",
    "grounded_rr_relay",
    "ungrounded_rr_feedback",
    "residual_grounded_source_share",
)

RESIDUAL_FEATURES = (
    "spectral_residual_energy",
    "residual_effective_channels",
    "residual_channel_entropy",
    "residual_channel_top1_share",
    "residual_channel_top5pct_share",
    "residual_weighted_lag",
    "residual_recent_lag_share",
    "residual_mid_lag_share",
    "residual_far_lag_share",
    "residual_source_effective_number",
    "residual_source_top1_share",
)


@dataclass(frozen=True)
class TopologyAuditConfig:
    reference_per_sample: int = 16
    min_task_bin_rows: int = 8
    phase_bins: int = 10
    onset_window: int = 4
    bootstrap_replicates: int = 1000
    seed: int = 20260815

    def validate(self) -> None:
        if min(
            int(self.reference_per_sample),
            int(self.min_task_bin_rows),
            int(self.phase_bins),
            int(self.onset_window),
        ) < 1:
            raise ValueError("topology-audit integer settings must be positive")
        if int(self.bootstrap_replicates) < 0:
            raise ValueError("bootstrap_replicates must be non-negative")


def _metadata_text(value) -> str:
    return "" if value is None else str(value)


def _sample_ids(dataset, limit=None):
    values = list(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        values = values[:limit]
    if not values:
        raise ValueError("no samples selected")
    return values


def _robust_location_scale(values: np.ndarray, *, epsilon=1e-6):
    values = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(values, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    std = np.nanstd(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(std > epsilon, std, 1.0))
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > epsilon), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _fit_conditioned_reference(
    features: np.ndarray,
    position_bin: np.ndarray,
    task: np.ndarray,
    *,
    position_bins: int,
    min_task_bin_rows: int,
):
    features = np.asarray(features, dtype=np.float32)
    position_bin = np.asarray(position_bin, dtype=np.int16)
    task = np.asarray(task, dtype=str)
    feature_dim = features.shape[1]

    global_center, global_scale = _robust_location_scale(features)
    position_center = np.empty((position_bins, feature_dim), dtype=np.float32)
    position_scale = np.empty_like(position_center)
    for current_bin in range(position_bins):
        selected = position_bin == current_bin
        if int(selected.sum()) >= 2:
            position_center[current_bin], position_scale[current_bin] = (
                _robust_location_scale(features[selected])
            )
        else:
            position_center[current_bin] = global_center
            position_scale[current_bin] = global_scale

    task_names = np.asarray(sorted(set(task.tolist())), dtype=str)
    task_center = np.empty(
        (len(task_names), position_bins, feature_dim), dtype=np.float32
    )
    task_scale = np.empty_like(task_center)
    task_count = np.zeros((len(task_names), position_bins), dtype=np.int32)
    for task_index, task_name in enumerate(task_names):
        for current_bin in range(position_bins):
            selected = (task == task_name) & (position_bin == current_bin)
            count = int(selected.sum())
            task_count[task_index, current_bin] = count
            if count >= min_task_bin_rows:
                task_center[task_index, current_bin], task_scale[
                    task_index, current_bin
                ] = _robust_location_scale(features[selected])
            else:
                task_center[task_index, current_bin] = position_center[current_bin]
                task_scale[task_index, current_bin] = position_scale[current_bin]

    return {
        "global_center": global_center,
        "global_scale": global_scale,
        "position_center": position_center,
        "position_scale": position_scale,
        "task_names": task_names,
        "task_center": task_center,
        "task_scale": task_scale,
        "task_count": task_count,
    }


def _standardize_features(features, position_bin, task, reference):
    features = np.asarray(features, dtype=np.float32)
    position_bin = np.asarray(position_bin, dtype=np.int64)
    task = np.asarray(task, dtype=str)
    result = np.empty_like(features)
    task_lookup = {
        str(name): index for index, name in enumerate(reference["task_names"])
    }
    for task_name in np.unique(task):
        selected = task == task_name
        task_index = task_lookup.get(str(task_name))
        if task_index is None:
            center = reference["position_center"][position_bin[selected]]
            scale = reference["position_scale"][position_bin[selected]]
        else:
            center = reference["task_center"][task_index, position_bin[selected]]
            scale = reference["task_scale"][task_index, position_bin[selected]]
        result[selected] = (features[selected] - center) / scale
    return result.astype(np.float32, copy=False)


def fit_topology_reference(
    dataset,
    spectral_reference_path,
    output_path,
    *,
    topology_config: TopologyDynamicsConfig | None = None,
    audit_config: TopologyAuditConfig | None = None,
    limit=None,
):
    """Fit task/position robust feature scales without opening labels."""
    topology_config = (
        TopologyDynamicsConfig() if topology_config is None else topology_config
    )
    audit_config = TopologyAuditConfig() if audit_config is None else audit_config
    topology_config.validate()
    audit_config.validate()
    spectral_reference = load_rr_reference(spectral_reference_path)
    sample_ids = _sample_ids(dataset, limit)

    feature_rows = []
    position_rows = []
    task_rows = []
    sample_rows = []
    token_rows = []
    feature_names = None

    for sample_id in tqdm(
        sample_ids, desc="fit RR topology-dynamics reference", unit="sample"
    ):
        sample = dataset[sample_id]
        try:
            extracted = extract_sample_topology_dynamics(
                sample, spectral_reference, config=topology_config
            )
            names = np.asarray(extracted["feature_names"], dtype=str)
            if feature_names is None:
                feature_names = names
            elif not np.array_equal(feature_names, names):
                raise RuntimeError("topology feature order changes across samples")
            response_count = len(extracted["features"])
            selected = reference_positions(
                response_count, audit_config.reference_per_sample
            )
            feature_rows.append(extracted["features"][selected])
            position_rows.append(extracted["position_bin"][selected])
            task = _metadata_text(sample.task_type)
            task_rows.extend([task] * len(selected))
            sample_rows.extend([str(sample.sample_id)] * len(selected))
            token_rows.extend(map(int, selected.tolist()))
        finally:
            sample.release_attention()

    features = np.concatenate(feature_rows, axis=0).astype(np.float32, copy=False)
    position_bin = np.concatenate(position_rows, axis=0).astype(np.int16, copy=False)
    task = np.asarray(task_rows, dtype=str)
    fitted = _fit_conditioned_reference(
        features,
        position_bin,
        task,
        position_bins=topology_config.position_bins,
        min_task_bin_rows=audit_config.min_task_bin_rows,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        spectral_reference_path=np.asarray(str(Path(spectral_reference_path))),
        feature_names=np.asarray(feature_names, dtype=str),
        lag_bins=np.asarray(topology_config.lag_bins, dtype=np.int16),
        spectral_top_k=np.asarray(topology_config.spectral_top_k, dtype=np.int16),
        block_rows=np.asarray(topology_config.block_rows, dtype=np.int32),
        position_bins=np.asarray(topology_config.position_bins, dtype=np.int16),
        top_source_count=np.asarray(topology_config.top_source_count, dtype=np.int16),
        recent_lag_max=np.asarray(topology_config.recent_lag_max, dtype=np.int16),
        mid_lag_max=np.asarray(topology_config.mid_lag_max, dtype=np.int16),
        far_lag_fraction=np.asarray(
            topology_config.far_lag_fraction, dtype=np.float32
        ),
        epsilon=np.asarray(topology_config.epsilon, dtype=np.float32),
        reference_per_sample=np.asarray(
            audit_config.reference_per_sample, dtype=np.int16
        ),
        min_task_bin_rows=np.asarray(
            audit_config.min_task_bin_rows, dtype=np.int16
        ),
        phase_bins=np.asarray(audit_config.phase_bins, dtype=np.int16),
        onset_window=np.asarray(audit_config.onset_window, dtype=np.int16),
        bootstrap_replicates=np.asarray(
            audit_config.bootstrap_replicates, dtype=np.int32
        ),
        seed=np.asarray(audit_config.seed, dtype=np.int64),
        reference_features=features,
        reference_position_bin=position_bin,
        reference_task=task,
        reference_sample_id=np.asarray(sample_rows, dtype=str),
        reference_token_index=np.asarray(token_rows, dtype=np.int32),
        **fitted,
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "reference_tokens": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "tasks": int(len(fitted["task_names"])),
    }


def load_topology_reference(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != REFERENCE_SCHEMA:
            raise ValueError("unsupported RR topology-dynamics reference schema")
        return {name: arrays[name].copy() for name in arrays.files}


def _topology_config_from_reference(reference):
    return TopologyDynamicsConfig(
        lag_bins=int(reference["lag_bins"]),
        spectral_top_k=int(reference["spectral_top_k"]),
        block_rows=int(reference["block_rows"]),
        position_bins=int(reference["position_bins"]),
        top_source_count=int(reference["top_source_count"]),
        recent_lag_max=int(reference["recent_lag_max"]),
        mid_lag_max=int(reference["mid_lag_max"]),
        far_lag_fraction=float(reference["far_lag_fraction"]),
        epsilon=float(reference["epsilon"]),
    )


def score_topology_dataset(
    dataset,
    spectral_reference_path,
    topology_reference_path,
    output_path,
    *,
    limit=None,
):
    """Freeze full-split topology features and train-standardized coordinates."""
    spectral_reference = load_rr_reference(spectral_reference_path)
    topology_reference = load_topology_reference(topology_reference_path)
    topology_config = _topology_config_from_reference(topology_reference)
    sample_ids = _sample_ids(dataset, limit)

    columns = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "relative_position",
            "position_bin",
            "task_type",
            "data_source",
            "generator_model",
            "features_raw",
            "features_z",
            "layer_route_effective_rank",
            "layer_route_consensus",
            "layer_residual_energy",
            "spectral_rank_residual_energy",
            "rr_embedding",
        )
    }

    for sample_id in tqdm(
        sample_ids, desc="extract RR topology dynamics", unit="sample"
    ):
        sample = dataset[sample_id]
        try:
            extracted = extract_sample_topology_dynamics(
                sample, spectral_reference, config=topology_config
            )
            if not np.array_equal(
                np.asarray(extracted["feature_names"], dtype=str),
                np.asarray(topology_reference["feature_names"], dtype=str),
            ):
                raise RuntimeError("topology feature order differs from reference")
            raw = extracted["features"]
            response_count = len(raw)
            task_name = _metadata_text(sample.task_type)
            task = np.asarray([task_name] * response_count, dtype=str)
            z = _standardize_features(
                raw,
                extracted["position_bin"],
                task,
                topology_reference,
            )
            tokens = np.arange(response_count, dtype=np.int32)
            relative_position = tokens.astype(np.float32) / float(
                max(response_count - 1, 1)
            )
            text = lambda value: np.asarray(
                [_metadata_text(value)] * response_count, dtype=str
            )
            columns["sample_id"].append(text(sample.sample_id))
            columns["source_id"].append(text(sample.source_id))
            columns["token_index"].append(tokens)
            columns["relative_position"].append(relative_position)
            columns["position_bin"].append(extracted["position_bin"])
            columns["task_type"].append(task)
            columns["data_source"].append(text(sample.data_source))
            columns["generator_model"].append(text(sample.generator_model))
            columns["features_raw"].append(raw)
            columns["features_z"].append(z)
            columns["layer_route_effective_rank"].append(
                extracted["layer_route_effective_rank"]
            )
            columns["layer_route_consensus"].append(
                extracted["layer_route_consensus"]
            )
            columns["layer_residual_energy"].append(
                extracted["layer_residual_energy"]
            )
            columns["spectral_rank_residual_energy"].append(
                extracted["spectral_rank_residual_energy"]
            )
            columns["rr_embedding"].append(extracted["rr_embedding"])
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
        spectral_reference_path=np.asarray(str(Path(spectral_reference_path))),
        topology_reference_path=np.asarray(str(Path(topology_reference_path))),
        feature_names=np.asarray(topology_reference["feature_names"], dtype=str),
        **output,
    )
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["token_index"])),
        "feature_dim": int(output["features_raw"].shape[1]),
        "layers": int(output["layer_residual_energy"].shape[1]),
    }


def load_topology_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA:
            raise ValueError("unsupported RR topology-dynamics feature schema")
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


def _align_labels(dataset, artifact):
    labels = _label_store_for_evaluation(dataset)
    cache = {}
    y = np.empty(len(artifact["token_index"]), dtype=np.int64)
    for index, (sample_id, token_index) in enumerate(
        zip(artifact["sample_id"], artifact["token_index"], strict=True)
    ):
        sample_id = str(sample_id)
        if sample_id not in cache:
            sample = dataset[sample_id]
            cache[sample_id] = labels.response_labels(sample).cpu().numpy()
            sample.release_attention()
        y[index] = int(cache[sample_id][int(token_index)])
    return y


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _binary_metrics(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    auc = float(roc_auc_score(y, score))
    auc_lower = float(roc_auc_score(y, -score))
    ap_higher = float(average_precision_score(y, score))
    ap_lower = float(average_precision_score(y, -score))
    normal = score[y == 0]
    positive = score[y == 1]
    normal_median = float(np.median(normal))
    positive_median = float(np.median(positive))
    normal_mad = 1.4826 * float(np.median(np.abs(normal - normal_median)))
    robust_effect = (positive_median - normal_median) / max(normal_mad, 1e-8)
    return {
        "tokens": int(len(y)),
        "positive_tokens": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc_higher": auc,
        "auroc_lower": auc_lower,
        "orientation_free_auroc": max(auc, auc_lower),
        "direction": "higher_in_hallucination" if auc >= auc_lower else "lower_in_hallucination",
        "auprc_higher": ap_higher,
        "auprc_lower": ap_lower,
        "orientation_free_auprc": max(ap_higher, ap_lower),
        "normal_median": normal_median,
        "hallucination_median": positive_median,
        "median_difference": positive_median - normal_median,
        "robust_effect_mad": robust_effect,
    }


def _bootstrap_mean_interval(values, *, replicates: int, seed: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    result = {"groups": int(len(values)), "mean": float(values.mean())}
    if replicates < 1 or len(values) < 2:
        result.update({"ci_low": None, "ci_high": None})
        return result
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        estimates[replicate] = values[
            rng.integers(0, len(values), size=len(values))
        ].mean()
    result.update(
        {
            "ci_low": float(np.quantile(estimates, 0.025)),
            "ci_high": float(np.quantile(estimates, 0.975)),
        }
    )
    return result


def _within_sample_effect(values, y, sample_id):
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sample_id = np.asarray(sample_id, dtype=str)
    result = []
    for current in np.unique(sample_id):
        selected = sample_id == current
        finite = selected & np.isfinite(values)
        if not bool((finite & (y == 0)).any()) or not bool(
            (finite & (y == 1)).any()
        ):
            continue
        result.append(
            float(values[finite & (y == 1)].mean() - values[finite & (y == 0)].mean())
        )
    return np.asarray(result, dtype=np.float64)


def _onset_effects(values, y, sample_id, token_index, window):
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sample_id = np.asarray(sample_id, dtype=str)
    token_index = np.asarray(token_index, dtype=np.int64)
    by_sample = []
    for current in np.unique(sample_id):
        selected = np.flatnonzero(sample_id == current)
        order = selected[np.argsort(token_index[selected])]
        local_y = y[order]
        local_value = values[order]
        local_token = token_index[order]
        effects = []
        for local in range(len(order)):
            if local_y[local] != 1 or (local > 0 and local_y[local - 1] == 1):
                continue
            run_end = local
            while run_end < len(order) and local_y[run_end] == 1:
                run_end += 1
            pre_start = max(0, local - int(window))
            post_end = min(run_end, local + int(window))
            pre = local_value[pre_start:local]
            post = local_value[local:post_end]
            pre_label = local_y[pre_start:local]
            contiguous = (
                local == 0
                or local_token[local] - local_token[max(local - 1, 0)] == 1
            )
            if (
                not contiguous
                or len(pre) == 0
                or len(post) == 0
                or not bool((pre_label == 0).all())
                or not np.isfinite(pre).all()
                or not np.isfinite(post).all()
            ):
                continue
            effects.append(float(post.mean() - pre.mean()))
        if effects:
            by_sample.append(float(np.mean(effects)))
    return np.asarray(by_sample, dtype=np.float64)


def _spearman(values, residual, mask=None):
    values = np.asarray(values, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(residual)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if int(finite.sum()) < 3:
        return None
    coefficient, p_value = spearmanr(values[finite], residual[finite])
    return {
        "rho": _finite_float(coefficient),
        "p_value": _finite_float(p_value),
        "tokens": int(finite.sum()),
    }


def _feature_metric_rows(feature_names, matrix, y, *, representation):
    rows = []
    mapping = {}
    for index, name in enumerate(feature_names):
        metric = _binary_metrics(y, matrix[:, index])
        mapping[str(name)] = metric
        if metric is not None:
            rows.append(
                {"representation": representation, "feature": str(name), **metric}
            )
    return mapping, rows


def _layer_metric_rows(y, artifact):
    rows = []
    report = {}
    for family in (
        "layer_route_effective_rank",
        "layer_route_consensus",
        "layer_residual_energy",
    ):
        matrix = np.asarray(artifact[family], dtype=np.float32)
        current = []
        for layer in range(matrix.shape[1]):
            metric = _binary_metrics(y, matrix[:, layer])
            current.append(metric)
            if metric is not None:
                rows.append({"family": family, "layer": layer, **metric})
        report[family] = current
    return report, rows


def _rank_metric_rows(y, artifact):
    matrix = np.asarray(artifact["spectral_rank_residual_energy"], dtype=np.float32)
    rows = []
    report = []
    for rank in range(matrix.shape[1]):
        metric = _binary_metrics(y, matrix[:, rank])
        report.append(metric)
        if metric is not None:
            rows.append({"spectral_rank": rank, **metric})
    return report, rows


def _phase_curve_rows(feature_names, raw, y, relative_position, phase_bins):
    selected_names = tuple(
        name
        for name in (
            *CONVERGENCE_FEATURES,
            *GROUNDING_FEATURES,
            "spectral_residual_energy",
            "residual_effective_channels",
            "residual_weighted_lag",
        )
        if name in set(feature_names)
    )
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    phase = np.minimum(
        (np.asarray(relative_position) * int(phase_bins)).astype(np.int64),
        int(phase_bins) - 1,
    )
    rows = []
    for name in selected_names:
        values = raw[:, name_to_index[name]]
        for phase_bin in range(int(phase_bins)):
            for label in (0, 1):
                selected = (
                    (phase == phase_bin)
                    & (y == label)
                    & np.isfinite(values)
                )
                if not bool(selected.any()):
                    continue
                rows.append(
                    {
                        "feature": name,
                        "phase_bin": phase_bin,
                        "label": label,
                        "tokens": int(selected.sum()),
                        "mean": float(values[selected].mean()),
                        "median": float(np.median(values[selected])),
                    }
                )
    return rows


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_topology_artifact(
    dataset,
    artifact_path,
    output_dir,
    *,
    bootstrap_replicates=None,
    onset_window=None,
    phase_bins=None,
    seed=None,
):
    """Open labels post-hoc and diagnose topology differences."""
    artifact = load_topology_artifact(artifact_path)
    reference = load_topology_reference(str(np.asarray(artifact["topology_reference_path"]).item()))
    y = _align_labels(dataset, artifact)
    feature_names = np.asarray(artifact["feature_names"], dtype=str)
    raw = np.asarray(artifact["features_raw"], dtype=np.float32)
    z = np.asarray(artifact["features_z"], dtype=np.float32)
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}

    bootstrap_replicates = int(
        reference["bootstrap_replicates"]
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    onset_window = int(
        reference["onset_window"] if onset_window is None else onset_window
    )
    phase_bins = int(reference["phase_bins"] if phase_bins is None else phase_bins)
    seed = int(reference["seed"] if seed is None else seed)

    raw_metrics, feature_rows_raw = _feature_metric_rows(
        feature_names, raw, y, representation="raw"
    )
    z_metrics, feature_rows_z = _feature_metric_rows(
        feature_names, z, y, representation="train_standardized"
    )
    layer_metrics, layer_rows = _layer_metric_rows(y, artifact)
    rank_metrics, rank_rows = _rank_metric_rows(y, artifact)

    sample_effects = {}
    onset_effects = {}
    sample_rows = []
    onset_rows = []
    for index, name in enumerate(feature_names):
        within = _within_sample_effect(
            raw[:, index], y, artifact["sample_id"]
        )
        within_report = _bootstrap_mean_interval(
            within,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        sample_effects[str(name)] = within_report
        if within_report is not None:
            sample_rows.append({"feature": str(name), **within_report})

        onset = _onset_effects(
            raw[:, index],
            y,
            artifact["sample_id"],
            artifact["token_index"],
            onset_window,
        )
        onset_report = _bootstrap_mean_interval(
            onset,
            replicates=bootstrap_replicates,
            seed=seed + 10_000 + index,
        )
        onset_effects[str(name)] = onset_report
        if onset_report is not None:
            onset_rows.append({"feature": str(name), **onset_report})

    residual = raw[:, name_to_index["spectral_residual_energy"]]
    correlations = {}
    correlation_rows = []
    for index, name in enumerate(feature_names):
        current = {
            "all": _spearman(raw[:, index], residual),
            "normal": _spearman(raw[:, index], residual, y == 0),
            "hallucination": _spearman(raw[:, index], residual, y == 1),
        }
        correlations[str(name)] = current
        for population, metric in current.items():
            if metric is not None:
                correlation_rows.append(
                    {"feature": str(name), "population": population, **metric}
                )

    phase_rows = _phase_curve_rows(
        feature_names,
        raw,
        y,
        artifact["relative_position"],
        phase_bins,
    )

    overall = {
        "tokens": int(len(y)),
        "positive_tokens": int(y.sum()),
        "prevalence": float(y.mean()),
        "samples": int(len(np.unique(artifact["sample_id"]))),
    }
    report = {
        "schema": "rr-topology-dynamics-evaluation-v1",
        "overall": overall,
        "feature_metrics_raw": raw_metrics,
        "feature_metrics_train_standardized": z_metrics,
        "within_sample_effects": sample_effects,
        "hallucination_onset_effects": onset_effects,
        "layer_metrics": layer_metrics,
        "spectral_rank_metrics": rank_metrics,
        "correlation_with_spectral_residual": correlations,
        "hypotheses": {
            "route_convergence": list(CONVERGENCE_FEATURES),
            "grounding_vs_feedback": list(GROUNDING_FEATURES),
            "spectral_escape_localization": list(RESIDUAL_FEATURES),
        },
        "claim_boundaries": {
            "labels_used_during": "posthoc_evaluation_only",
            "offline_future_features": [
                "offline_route_distance_to_final",
                "offline_source_distance_to_final",
            ],
            "confidence_available": False,
            "confidence_reason": (
                "the canonical attention cache contains attention/metadata but no "
                "token logits, entropy, NLL, or calibrated confidence"
            ),
            "topology_scope": (
                "retained cache-censored causal response-query topology; prompt "
                "query rows and exact sub-floor attention values are unavailable"
            ),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "feature_metrics.csv", feature_rows_raw + feature_rows_z)
    _write_csv(output_dir / "within_sample_effects.csv", sample_rows)
    _write_csv(output_dir / "onset_effects.csv", onset_rows)
    _write_csv(output_dir / "layer_metrics.csv", layer_rows)
    _write_csv(output_dir / "spectral_rank_metrics.csv", rank_rows)
    _write_csv(output_dir / "phase_curves.csv", phase_rows)
    _write_csv(output_dir / "residual_correlations.csv", correlation_rows)
    return report
