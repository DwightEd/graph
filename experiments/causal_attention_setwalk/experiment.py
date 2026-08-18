"""Fit, score, and evaluate causal attention SetWalk node representations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from experiment_protocol import (
    FrozenEvaluation,
    FrozenFile,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
)
from .artifacts import (
    EVALUATION_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_reference,
    load_score_artifact,
)
from .model import (
    MODEL_FIELDS,
    ReferenceConfig,
    anomaly_score,
    fit_reference_model,
    pack_model,
    reference_positions,
    response_position_bin,
    unpack_model,
)
from .representation import (
    DIAGNOSTIC_DIRECTIONS,
    DIAGNOSTIC_NAMES,
    LAYER_PROFILE_NAMES,
    VIEW_NAMES,
    SetWalkConfig,
    extract_setwalk_representations,
)


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


def _text(value):
    return "" if value is None else str(value)


def _repeat(value, count):
    return np.asarray([_text(value)] * int(count), dtype=str)


def _position_columns(response_count, bins):
    token = np.arange(response_count, dtype=np.int32)
    relative = token.astype(np.float32) / max(1, response_count - 1)
    position = np.asarray(
        [response_position_bin(int(i), response_count, bins) for i in token],
        dtype=np.int16,
    )
    return token, relative, position


def _representation_config(reference):
    return SetWalkConfig(
        fourier_features=int(reference["fourier_features"]),
        dct_components=int(reference["dct_components"]),
        recent_lag_max=int(reference["recent_lag_max"]),
        block_rows=int(reference["block_rows"]),
        seed=int(reference["seed"]),
        epsilon=float(reference["epsilon"]),
    )


def _reference_config(reference):
    return ReferenceConfig(
        reference_per_sample=int(reference["reference_per_sample"]),
        position_bins=int(reference["position_bins"]),
        min_task_bin_rows=int(reference["min_task_bin_rows"]),
        trim_fraction=float(reference["trim_fraction"]),
    )


def fit_reference(
    dataset,
    output_path,
    *,
    representation_config: SetWalkConfig | None = None,
    reference_config: ReferenceConfig | None = None,
    limit=None,
):
    """Fit every anomaly reference on unlabeled train attention only."""

    representation_config = (
        SetWalkConfig() if representation_config is None else representation_config
    )
    reference_config = (
        ReferenceConfig() if reference_config is None else reference_config
    )
    representation_config.validate()
    reference_config.validate()
    selected_ids = _sample_ids(dataset, limit)
    manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
    embeddings = {name: [] for name in VIEW_NAMES}
    sample_rows = []
    token_rows = []
    position_rows = []
    task_rows = []
    source_groups = set()

    for sample_id in tqdm(selected_ids, desc="fit causal SetWalk reference", unit="sample"):
        sample = dataset[sample_id]
        try:
            representation = extract_setwalk_representations(
                sample, representation_config
            )
            response_count = len(representation["diagnostics"])
            selected = reference_positions(
                response_count, reference_config.reference_per_sample
            )
            _, _, position = _position_columns(
                response_count, reference_config.position_bins
            )
            for view in VIEW_NAMES:
                embeddings[view].append(representation["embeddings"][view][selected])
            sample_rows.extend([str(sample.sample_id)] * len(selected))
            token_rows.extend(map(int, selected.tolist()))
            position_rows.append(position[selected])
            task_rows.extend([_text(sample.task_type)] * len(selected))
            source_groups.add(canonical_source_group(sample))
        finally:
            sample.release_attention()

    reference_embeddings = {
        view: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for view, values in embeddings.items()
    }
    position = np.concatenate(position_rows).astype(np.int16, copy=False)
    task = np.asarray(task_rows, dtype=str)
    artifact = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "train_dataset_manifest_sha256": np.asarray(manifest.sha256),
        "reference_source_id": np.asarray(sorted(source_groups), dtype=str),
        "view_names": np.asarray(VIEW_NAMES, dtype=str),
        "fourier_features": np.asarray(
            representation_config.fourier_features, dtype=np.int16
        ),
        "dct_components": np.asarray(
            representation_config.dct_components, dtype=np.int16
        ),
        "recent_lag_max": np.asarray(
            representation_config.recent_lag_max, dtype=np.int16
        ),
        "block_rows": np.asarray(representation_config.block_rows, dtype=np.int32),
        "seed": np.asarray(representation_config.seed, dtype=np.int64),
        "epsilon": np.asarray(representation_config.epsilon, dtype=np.float32),
        "reference_per_sample": np.asarray(
            reference_config.reference_per_sample, dtype=np.int16
        ),
        "position_bins": np.asarray(reference_config.position_bins, dtype=np.int16),
        "min_task_bin_rows": np.asarray(
            reference_config.min_task_bin_rows, dtype=np.int16
        ),
        "trim_fraction": np.asarray(
            reference_config.trim_fraction, dtype=np.float32
        ),
        "reference_sample_id": np.asarray(sample_rows, dtype=str),
        "reference_token_index": np.asarray(token_rows, dtype=np.int32),
        "reference_position_bin": position,
        "reference_task": task,
    }
    retained = {}
    for view in VIEW_NAMES:
        model = fit_reference_model(
            reference_embeddings[view], position, task, reference_config
        )
        artifact[f"reference_embedding_{view}"] = reference_embeddings[view]
        pack_model(artifact, view, model)
        retained[view] = int(model["retained_rows"])

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.verify(manifest.path)
    np.savez_compressed(output_path, **artifact)
    load_reference(output_path)
    return {
        "output": str(output_path),
        "schema": REFERENCE_SCHEMA,
        "labels_read": False,
        "samples": len(selected_ids),
        "reference_tokens": len(sample_rows),
        "embedding_dimensions": {
            view: int(reference_embeddings[view].shape[1]) for view in VIEW_NAMES
        },
        "retained_reference_tokens": retained,
    }


def score_dataset(dataset, reference_path, output_path, *, limit=None):
    """Freeze all token embeddings and scores before labels become available."""

    reference_file = FrozenFile.capture(reference_path)
    reference = load_reference(reference_file.path)
    representation_config = _representation_config(reference)
    reference_config = _reference_config(reference)
    selected_ids = _sample_ids(dataset, limit)
    audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=selected_ids,
        reserved_source_ids=reference["reference_source_id"],
        require_complete_split=limit is None,
    )
    manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
    embeddings = {name: [] for name in VIEW_NAMES}
    scores = {name: [] for name in VIEW_NAMES}
    diagnostics = []
    profiles = []
    rows = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "task_type",
            "data_source",
            "generator_model",
            "token_index",
            "response_length",
            "relative_position",
            "position_bin",
        )
    }
    true_order = None
    shuffled_order = None

    for sample_id in tqdm(selected_ids, desc="score causal SetWalk nodes", unit="sample"):
        sample = dataset[sample_id]
        try:
            representation = extract_setwalk_representations(
                sample, representation_config
            )
            audit.observe(sample)
            response_count = len(representation["diagnostics"])
            token, relative, position = _position_columns(
                response_count, reference_config.position_bins
            )
            task = _repeat(sample.task_type, response_count)
            for view in VIEW_NAMES:
                value = representation["embeddings"][view]
                model = unpack_model(reference, view)
                embeddings[view].append(value)
                scores[view].append(anomaly_score(value, position, task, model))
            diagnostics.append(representation["diagnostics"])
            profiles.append(representation["layer_profiles"])
            current_true = representation["true_layer_order"]
            current_shuffled = representation["shuffled_layer_order"]
            if true_order is None:
                true_order, shuffled_order = current_true, current_shuffled
            elif not (
                np.array_equal(true_order, current_true)
                and np.array_equal(shuffled_order, current_shuffled)
            ):
                raise ValueError("attention layer geometry changes across samples")
            source = canonical_source_group(sample)
            rows["sample_id"].extend([str(sample.sample_id)] * response_count)
            rows["source_id"].extend([source] * response_count)
            rows["task_type"].extend(task.tolist())
            rows["data_source"].extend(
                _repeat(sample.data_source, response_count).tolist()
            )
            rows["generator_model"].extend(
                _repeat(sample.generator_model, response_count).tolist()
            )
            rows["token_index"].extend(token.tolist())
            rows["response_length"].extend([response_count] * response_count)
            rows["relative_position"].extend(relative.tolist())
            rows["position_bin"].extend(position.tolist())
        finally:
            sample.release_attention()

    source_audit = audit.finish()
    artifact = {
        "schema": np.asarray(SCORE_SCHEMA),
        "reference_path": np.asarray(str(reference_file.path)),
        "reference_sha256": np.asarray(reference_file.sha256),
        "dataset_manifest_sha256": np.asarray(manifest.sha256),
        "reference_source_id": reference["reference_source_id"],
        "test_group_id": np.asarray(source_audit.test_source_ids, dtype=str),
        "test_sample_id": np.asarray(source_audit.test_sample_ids, dtype=str),
        "audit_scope": np.asarray(source_audit.test_scope),
        "view_names": np.asarray(VIEW_NAMES, dtype=str),
        "diagnostic_names": np.asarray(DIAGNOSTIC_NAMES, dtype=str),
        "diagnostic_directions": np.asarray(
            [DIAGNOSTIC_DIRECTIONS[name] for name in DIAGNOSTIC_NAMES], dtype=str
        ),
        "layer_profile_names": np.asarray(LAYER_PROFILE_NAMES, dtype=str),
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "data_source": np.asarray(rows["data_source"], dtype=str),
        "generator_model": np.asarray(rows["generator_model"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "relative_position": np.asarray(rows["relative_position"], dtype=np.float32),
        "position_bin": np.asarray(rows["position_bin"], dtype=np.int16),
        "diagnostics": np.concatenate(diagnostics, axis=0).astype(np.float32),
        "layer_profiles": np.concatenate(profiles, axis=0).astype(np.float32),
        "true_layer_order": np.asarray(true_order, dtype=np.int16),
        "shuffled_layer_order": np.asarray(shuffled_order, dtype=np.int16),
    }
    for view in VIEW_NAMES:
        storage_dtype = np.float32 if view == "setwalk" else np.float16
        artifact[f"embedding_{view}"] = np.concatenate(
            embeddings[view], axis=0
        ).astype(storage_dtype)
        artifact[f"score_{view}"] = np.concatenate(scores[view]).astype(np.float32)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.verify(manifest.path)
    reference_file.verify(reference_file.path)
    np.savez_compressed(output_path, **artifact)
    load_score_artifact(output_path)
    return {
        "output": str(output_path),
        "schema": SCORE_SCHEMA,
        "labels_read": False,
        "samples": len(selected_ids),
        "tokens": len(artifact["sample_id"]),
        "embedding_dimensions": {
            view: int(artifact[f"embedding_{view}"].shape[1]) for view in VIEW_NAMES
        },
    }


def _metrics(labels, score):
    labels = np.asarray(labels, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(labels.sum())
    result = {
        "tokens": int(len(labels)),
        "positive_tokens": positives,
        "prevalence": float(labels.mean()) if len(labels) else None,
        "auroc": None,
        "auprc": None,
    }
    if len(labels) and 0 < positives < len(labels):
        result["auroc"] = float(roc_auc_score(labels, score))
        result["auprc"] = float(average_precision_score(labels, score))
    return result


def _paired_cluster_difference(labels, sample_id, first, second, *, replicates, seed):
    labels = np.asarray(labels)
    samples = np.asarray(sample_id, dtype=str)
    unique = np.asarray(list(dict.fromkeys(samples.tolist())), dtype=str)
    point_first = _metrics(labels, first)
    point_second = _metrics(labels, second)
    point = {
        metric: point_first[metric] - point_second[metric]
        for metric in ("auroc", "auprc")
        if point_first[metric] is not None and point_second[metric] is not None
    }
    if int(replicates) < 1:
        return {metric: {"point": value, "ci95": None} for metric, value in point.items()}
    lookup = {sample: np.flatnonzero(samples == sample) for sample in unique}
    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in point}
    for _ in range(int(replicates)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([lookup[sample] for sample in chosen])
        first_metric = _metrics(labels[index], np.asarray(first)[index])
        second_metric = _metrics(labels[index], np.asarray(second)[index])
        for metric in draws:
            if first_metric[metric] is not None and second_metric[metric] is not None:
                draws[metric].append(first_metric[metric] - second_metric[metric])
    return {
        metric: {
            "point": float(point[metric]),
            "ci95": (
                [float(value) for value in np.quantile(draws[metric], [0.025, 0.975])]
                if draws[metric]
                else None
            ),
            "valid_replicates": len(draws[metric]),
        }
        for metric in point
    }


def evaluate(dataset, score_path, output_dir, *, bootstrap_replicates=200, seed=20260818):
    """Read labels only after the complete node artifact is frozen on disk."""

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact, evaluation = frozen.load_and_align(dataset, load_score_artifact)
    labels = evaluation.token_label.astype(np.int8, copy=False)
    view_metrics = {
        view: _metrics(labels, artifact[f"score_{view}"]) for view in VIEW_NAMES
    }
    task_metrics = {}
    task = np.asarray(artifact["task_type"], dtype=str)
    for task_name in sorted(set(task.tolist())):
        selected = task == task_name
        task_metrics[task_name] = {
            view: _metrics(labels[selected], artifact[f"score_{view}"][selected])
            for view in VIEW_NAMES
        }

    diagnostic_metrics = {}
    diagnostic_values = np.asarray(artifact["diagnostics"], dtype=np.float32)
    for index, name in enumerate(DIAGNOSTIC_NAMES):
        raw = diagnostic_values[:, index]
        direction = DIAGNOSTIC_DIRECTIONS[name]
        oriented = raw if direction == "higher" else -raw
        diagnostic_metrics[name] = {
            "predeclared_anomaly_direction": direction,
            "raw": _metrics(labels, raw),
            "oriented": _metrics(labels, oriented),
        }

    comparisons = {}
    for ablation in ("no_walk", "pairwise_walk", "layer_shuffled"):
        comparisons[f"setwalk_vs_{ablation}"] = _paired_cluster_difference(
            labels,
            artifact["sample_id"],
            artifact["score_setwalk"],
            artifact[f"score_{ablation}"],
            replicates=bootstrap_replicates,
            seed=seed,
        )

    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "primary_method": "setwalk",
        "method_claim": (
            "fixed characteristic-set hyperedge encoding plus exact two-hop "
            "layer-causal SetWalk expectation; no gradient training"
        ),
        "view_metrics": view_metrics,
        "by_task_type": task_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "structural_comparisons": comparisons,
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "score_artifact": str(frozen.artifact.path),
        "score_artifact_sha256": frozen.artifact.sha256,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "task_type", "method", "auroc", "auprc", "prevalence"),
        )
        writer.writeheader()
        for view, metric in view_metrics.items():
            writer.writerow(
                {
                    "scope": "overall",
                    "task_type": "ALL",
                    "method": view,
                    "auroc": metric["auroc"],
                    "auprc": metric["auprc"],
                    "prevalence": metric["prevalence"],
                }
            )
        for task_name, methods in task_metrics.items():
            for view, metric in methods.items():
                writer.writerow(
                    {
                        "scope": "task",
                        "task_type": task_name,
                        "method": view,
                        "auroc": metric["auroc"],
                        "auprc": metric["auprc"],
                        "prevalence": metric["prevalence"],
                    }
                )
    return {"output": str(output), **report}
