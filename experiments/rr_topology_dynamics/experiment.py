"""Fit references and extract causal RR topology-dynamics features.

The representation and train reference are label-blind. Post-hoc evaluation
lives in :mod:`experiments.rr_topology_dynamics.evaluation` and opens labels
only after every feature has been frozen on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiment_protocol import (
    FrozenFile,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
)
from experiments.spectral_feasibility.representations import reference_positions

from .artifacts import (
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_topology_artifact,
    load_topology_reference,
)
from .features import (
    SCALAR_FEATURE_NAMES,
    TopologyDynamicsConfig,
    extract_sample_topology_dynamics,
    load_rr_reference,
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
    spectral_file = FrozenFile.capture(spectral_reference_path)
    spectral_reference = load_rr_reference(spectral_file.path)
    spectral_file.verify(spectral_file.path)
    sample_ids = _sample_ids(dataset, limit)

    feature_rows = []
    position_rows = []
    task_rows = []
    sample_rows = []
    token_rows = []
    reference_source_ids = {
        str(group_id)
        for field in ("fit_group_id", "calibration_group_id")
        for group_id in np.asarray(spectral_reference[field], dtype=str).tolist()
    }
    feature_names = None

    for sample_id in tqdm(
        sample_ids, desc="fit RR topology-dynamics reference", unit="sample"
    ):
        sample = dataset[sample_id]
        try:
            extracted = extract_sample_topology_dynamics(
                sample, spectral_reference, config=topology_config
            )
            reference_source_ids.add(canonical_source_group(sample))
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
        spectral_reference_path=np.asarray(str(spectral_file.path)),
        spectral_reference_sha256=np.asarray(spectral_file.sha256),
        reference_source_id=np.asarray(sorted(reference_source_ids), dtype=str),
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
    load_topology_reference(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "reference_tokens": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "tasks": int(len(fitted["task_names"])),
    }


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
    topology_file = FrozenFile.capture(topology_reference_path)
    topology_reference = load_topology_reference(topology_file.path)
    topology_file.verify(topology_file.path)
    spectral_path = Path(spectral_reference_path).resolve()
    bound_spectral_path = Path(
        str(np.asarray(topology_reference["spectral_reference_path"]).item())
    ).resolve()
    if spectral_path != bound_spectral_path:
        raise ValueError("spectral reference identity differs from topology reference")
    spectral_file = FrozenFile.capture(spectral_path)
    if spectral_file.sha256 != str(
        np.asarray(topology_reference["spectral_reference_sha256"]).item()
    ):
        raise ValueError("spectral reference digest differs from topology reference")
    spectral_reference = load_rr_reference(spectral_file.path)
    spectral_file.verify(spectral_file.path)
    topology_config = _topology_config_from_reference(topology_reference)
    sample_ids = _sample_ids(dataset, limit)
    source_audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=sample_ids,
        reserved_source_ids=topology_reference["reference_source_id"].tolist(),
        require_complete_split=limit is None,
    )

    columns = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
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
            sample.attention()
            source_audit.observe(sample)
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
            columns["response_length"].append(
                np.full(response_count, response_count, dtype=np.int32)
            )
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
    audit = source_audit.finish()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        spectral_reference_path=np.asarray(str(spectral_file.path)),
        spectral_reference_sha256=np.asarray(spectral_file.sha256),
        topology_reference_path=np.asarray(str(topology_file.path)),
        topology_reference_sha256=np.asarray(topology_file.sha256),
        dataset_manifest_sha256=np.asarray(dataset_manifest_sha256(dataset)),
        reference_source_id=np.asarray(
            topology_reference["reference_source_id"], dtype=str
        ),
        test_group_id=np.asarray(audit.test_source_ids, dtype=str),
        test_sample_id=np.asarray(audit.test_sample_ids, dtype=str),
        audit_scope=np.asarray(audit.test_scope),
        feature_names=np.asarray(topology_reference["feature_names"], dtype=str),
        **output,
    )
    load_topology_artifact(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["token_index"])),
        "feature_dim": int(output["features_raw"].shape[1]),
        "layers": int(output["layer_residual_energy"].shape[1]),
    }
