"""Label-free PR/RR signal extraction, scoring, and mechanism audit."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from attention_graph.aligned_reservoir import AlignedReservoir
from experiment_protocol import (
    FrozenEvaluation,
    FrozenFile,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
    partition_source_groups,
)

from .artifacts import (
    EVALUATION_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_reference,
    load_score_artifact,
    score_temporal_scope,
    verify_score_provenance,
)
from .collapse import (
    collapse_reference,
    collapse_reference_fields,
    collapse_scores,
)
from .components import (
    COLLAPSE_DIRECTIONS,
    COLLAPSE_FEATURE_NAMES,
    EVIDENCE_DIRECTIONS,
    EVIDENCE_FEATURE_NAMES,
    EVIDENCE_REGISTRY,
    SIGNAL_BLOCKS,
    RRSignalConfig,
    extract_rr_signal_features,
    features_per_channel,
)
from .geometry import (
    CONDITION_MODES,
    SCORE_KINDS,
    RRGeometryConfig,
    calibrated_scores,
    calibration_fields,
    cluster_gap_summary,
    condition_keys,
    fit_geometry,
    flatten_model,
    ppca_nll,
    project_geometry,
    relative_position_bins,
    shuffle_channel_blocks,
    unflatten_model,
)


_METADATA_BLOCKS = (
    "__collapse_global",
    "__evidence_global",
    "__task_code",
    "__causal_bin",
    "__group_low",
    "__group_high",
)
_GROUP_BASE = 1024


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _text(value) -> str:
    return "" if value is None else str(value)


def _sample_ids(dataset, limit=None) -> tuple[str, ...]:
    values = tuple(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        values = values[:limit]
    if not values:
        raise ValueError("no samples selected")
    return values


def _metadata_maps(dataset, sample_ids, split):
    tasks = sorted(
        {
            _text(dataset[sample_id].task_type)
            for sample_id in sample_ids
        }
    )
    task_code = {name: index for index, name in enumerate(tasks)}
    groups = sorted(
        set(split["fit_group_ids"]) | set(split["calibration_group_ids"])
    )
    group_code = {name: index for index, name in enumerate(groups)}
    return np.asarray(tasks, dtype=str), task_code, np.asarray(groups, dtype=str), group_code


def _encode_group(code: int, rows: int):
    code = int(code)
    return (
        np.full((rows, 1), code % _GROUP_BASE, dtype=np.float32),
        np.full((rows, 1), code // _GROUP_BASE, dtype=np.float32),
    )


def _decode_integer_column(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("metadata reservoir block must have width one")
    rounded = np.rint(values[:, 0]).astype(np.int64)
    if not np.allclose(values[:, 0], rounded, atol=0.01):
        raise ValueError("metadata reservoir values lost integer identity")
    return rounded


def _decode_groups(blocks, group_names):
    low = _decode_integer_column(blocks["__group_low"])
    high = _decode_integer_column(blocks["__group_high"])
    code = low + _GROUP_BASE * high
    if bool((code < 0).any()) or bool((code >= len(group_names)).any()):
        raise ValueError("reservoir group code is outside the source inventory")
    return np.asarray(group_names, dtype=str)[code]


def _decode_tasks(blocks, task_names):
    code = _decode_integer_column(blocks["__task_code"])
    if bool((code < 0).any()) or bool((code >= len(task_names)).any()):
        raise ValueError("reservoir task code is outside the task inventory")
    return np.asarray(task_names, dtype=str)[code]


def _conditions(task, relative_bin, causal_bin):
    return {
        "relative": condition_keys(task, relative_bin),
        "causal": condition_keys(task, causal_bin),
    }


def score_contract() -> tuple[str, ...]:
    names: list[str] = []
    for block in SIGNAL_BLOCKS:
        for mode in CONDITION_MODES:
            for kind in SCORE_KINDS:
                names.append(f"{block}.{mode}.{kind}_tail")
    for mode in CONDITION_MODES:
        for feature, direction in zip(
            COLLAPSE_FEATURE_NAMES,
            COLLAPSE_DIRECTIONS,
            strict=True,
        ):
            if int(direction) != 0:
                tail = "upper" if int(direction) > 0 else "lower"
                names.append(f"collapse.{mode}.{feature}.{tail}_tail")
        for feature in COLLAPSE_FEATURE_NAMES:
            names.append(f"collapse.{mode}.{feature}.two_sided")
        names.append(f"collapse.{mode}.composite")
    return tuple(names)


def _reservoir_blocks(features, *, task_code: int, group_code: int):
    rows = features.response_count
    group_low, group_high = _encode_group(group_code, rows)
    return {
        **features.blocks,
        "__collapse_global": features.collapse_global,
        "__evidence_global": features.evidence_global,
        "__task_code": np.full((rows, 1), task_code, dtype=np.float32),
        "__causal_bin": features.causal_position_bucket[:, None].astype(np.float32),
        "__group_low": group_low,
        "__group_high": group_high,
    }


def fit_rr_signal_audit(
    dataset,
    output_path,
    *,
    signal_config: RRSignalConfig | None = None,
    geometry_config: RRGeometryConfig | None = None,
    limit=None,
):
    """Fit decomposition references and coordination controls without labels."""

    signal_config = RRSignalConfig() if signal_config is None else signal_config
    geometry_config = (
        RRGeometryConfig() if geometry_config is None else geometry_config
    )
    signal_config.validate()
    geometry_config.validate()
    train_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
    sample_ids = _sample_ids(dataset, limit)
    split = partition_source_groups(
        dataset,
        sample_ids,
        calibration_fraction=geometry_config.calibration_fraction,
        seed=geometry_config.seed,
    )
    task_names, task_code, group_names, group_code = _metadata_maps(
        dataset,
        sample_ids,
        split,
    )
    sample_role = {
        sample_id: role
        for role, field in (
            ("fit", "fit_sample_ids"),
            ("cal", "calibration_sample_ids"),
        )
        for sample_id in split[field]
    }

    reservoir = AlignedReservoir(
        position_bins=geometry_config.relative_position_bins,
        size=geometry_config.reservoir_rows,
        seed=geometry_config.seed,
    )
    geometry: tuple[int, int] | None = None
    for sample_id in tqdm(
        sample_ids,
        desc="extract attention signal audit train rows",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            features = extract_rr_signal_features(
                sample,
                config=signal_config,
            )
            current_geometry = (features.num_layers, features.num_heads)
            if geometry is None:
                geometry = current_geometry
            elif geometry != current_geometry:
                raise ValueError("attention geometry changes inside the train split")
            reservoir.add(
                sample_role[sample_id],
                _reservoir_blocks(
                    features,
                    task_code=task_code[_text(sample.task_type)],
                    group_code=group_code[canonical_source_group(sample)],
                ),
                features.relative_position,
            )
        finally:
            sample.release_attention()

    if geometry is None:
        raise RuntimeError("RR signal audit extracted no train features")
    fit_relative_bin = reservoir.bins("fit")
    calibration_relative_bin = reservoir.bins("cal")
    fit_metadata = {
        name: reservoir.block("fit", name)
        for name in _METADATA_BLOCKS
    }
    calibration_metadata = {
        name: reservoir.block("cal", name)
        for name in _METADATA_BLOCKS
    }
    fit_task = _decode_tasks(fit_metadata, task_names)
    calibration_task = _decode_tasks(calibration_metadata, task_names)
    fit_causal_bin = _decode_integer_column(fit_metadata["__causal_bin"])
    calibration_causal_bin = _decode_integer_column(
        calibration_metadata["__causal_bin"]
    )
    fit_conditions = _conditions(
        fit_task,
        fit_relative_bin,
        fit_causal_bin,
    )
    calibration_conditions = _conditions(
        calibration_task,
        calibration_relative_bin,
        calibration_causal_bin,
    )
    calibration_group = _decode_groups(calibration_metadata, group_names)

    artifact: dict[str, np.ndarray] = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "train_dataset_manifest_sha256": np.asarray(train_manifest.sha256),
        "signal_config_json": np.asarray(_json(asdict(signal_config))),
        "geometry_config_json": np.asarray(_json(asdict(geometry_config))),
        "fit_group_id": np.asarray(split["fit_group_ids"], dtype=str),
        "calibration_group_id": np.asarray(
            split["calibration_group_ids"],
            dtype=str,
        ),
        "num_layers": np.asarray(geometry[0], dtype=np.int16),
        "num_heads": np.asarray(geometry[1], dtype=np.int16),
        "task_names": task_names,
        "source_group_names": group_names,
        "block_names": np.asarray(SIGNAL_BLOCKS, dtype=str),
        "evidence_feature_names": np.asarray(EVIDENCE_FEATURE_NAMES, dtype=str),
        "evidence_directions": np.asarray(EVIDENCE_DIRECTIONS, dtype=np.int8),
        "evidence_registry_json": np.asarray(_json(EVIDENCE_REGISTRY)),
        "score_names": np.asarray(score_contract(), dtype=str),
        "fit_reservoir_rows": np.asarray(
            len(fit_relative_bin),
            dtype=np.int32,
        ),
        "calibration_reservoir_rows": np.asarray(
            len(calibration_relative_bin),
            dtype=np.int32,
        ),
    }

    coordination_summary: dict[str, dict[str, object]] = {}
    num_channels = geometry[0] * geometry[1]
    for block_index, block in enumerate(SIGNAL_BLOCKS):
        fit_values = reservoir.block("fit", block)
        calibration_values = reservoir.block("cal", block)
        for mode_index, mode in enumerate(CONDITION_MODES):
            prefix = f"{block}__{mode}"
            model = fit_geometry(
                fit_values,
                fit_conditions[mode],
                config=geometry_config,
                seed_offset=1000 * block_index + 100 * mode_index,
            )
            calibration_projection = project_geometry(
                calibration_values,
                calibration_conditions[mode],
                model,
            )
            artifact.update(flatten_model(prefix, model))
            artifact.update(
                calibration_fields(prefix, calibration_projection)
            )
            shuffled = shuffle_channel_blocks(
                calibration_projection["standardized"],
                num_channels=num_channels,
                features_per_channel=features_per_channel(
                    block,
                    signal_config,
                ),
                conditions=calibration_conditions[mode],
                seed=(
                    geometry_config.seed
                    + 100_000
                    + 1000 * block_index
                    + 100 * mode_index
                ),
            )
            shuffled_nll = ppca_nll(shuffled, model)
            gate = cluster_gap_summary(
                shuffled_nll - calibration_projection["ppca_nll"],
                calibration_group,
                bootstrap_replicates=geometry_config.bootstrap_replicates,
                seed=(
                    geometry_config.seed
                    + 200_000
                    + 1000 * block_index
                    + 100 * mode_index
                ),
            )
            coordination_summary[prefix] = gate
            for key, value in gate.items():
                artifact[f"{prefix}__coordination_{key}"] = np.asarray(
                    np.nan if value is None else value
                )
        del fit_values, calibration_values

    calibration_collapse = np.asarray(
        calibration_metadata["__collapse_global"],
        dtype=np.float32,
    )
    artifact.update(
        collapse_reference_fields(
            calibration_collapse,
            calibration_conditions["relative"],
            calibration_conditions["causal"],
        )
    )
    artifact["coordination_summary_json"] = np.asarray(
        _json(coordination_summary)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_manifest.verify(train_manifest.path)
    np.savez_compressed(output_path, **artifact)
    load_reference(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "fit_samples": len(split["fit_sample_ids"]),
        "calibration_samples": len(split["calibration_sample_ids"]),
        "fit_reservoir_rows": int(len(fit_relative_bin)),
        "calibration_reservoir_rows": int(len(calibration_relative_bin)),
        "num_layers": geometry[0],
        "num_heads": geometry[1],
        "block_dims": {
            block: int(reservoir.block_widths[block])
            for block in SIGNAL_BLOCKS
        },
        "coordination": coordination_summary,
    }


def _configs_from_reference(reference):
    return (
        RRSignalConfig(
            **json.loads(str(reference["signal_config_json"].item()))
        ),
        RRGeometryConfig(
            **json.loads(str(reference["geometry_config_json"].item()))
        ),
    )


def _collapse_score_map(
    reference,
    collapse_values,
    conditions,
    *,
    mode: str,
    geometry_config: RRGeometryConfig,
):
    calibration, relative_reference, causal_reference = collapse_reference(reference)
    reference_conditions = (
        relative_reference if mode == "relative" else causal_reference
    )
    scores, _, _ = collapse_scores(
        calibration,
        reference_conditions,
        collapse_values,
        conditions,
        minimum_condition_rows=geometry_config.min_condition_rows,
    )
    result = {}
    for name, values in scores.items():
        suffix = name.removeprefix("collapse.")
        result[f"collapse.{mode}.{suffix}"] = values
    return result


def score_rr_signal_audit(
    dataset,
    reference_path,
    output_path,
    *,
    limit=None,
):
    """Freeze all preregistered attention scores without opening labels."""

    reference_file = FrozenFile.capture(reference_path)
    reference = load_reference(reference_file.path)
    signal_config, geometry_config = _configs_from_reference(reference)
    sample_ids = _sample_ids(dataset, limit)
    audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=sample_ids,
        reserved_source_ids=np.concatenate(
            (
                reference["fit_group_id"],
                reference["calibration_group_id"],
            )
        ),
        require_complete_split=limit is None,
    )
    expected_geometry = (
        int(reference["num_layers"]),
        int(reference["num_heads"]),
    )
    score_names = tuple(
        map(str, np.asarray(reference["score_names"], dtype=str).tolist())
    )
    rows: dict[str, list] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
            "relative_position",
            "causal_position_bucket",
            "task_type",
            "data_source",
            "generator_model",
            "scores",
            "collapse_raw",
            "evidence_raw",
        )
    }

    for sample_id in tqdm(
        sample_ids,
        desc="score attention signal audit",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            audit.observe(sample)
            features = extract_rr_signal_features(
                sample,
                config=signal_config,
            )
            if (features.num_layers, features.num_heads) != expected_geometry:
                raise ValueError("test attention geometry differs from reference")
            relative_bin = relative_position_bins(
                features.relative_position,
                geometry_config.relative_position_bins,
            )
            task = np.asarray(
                [_text(sample.task_type)] * features.response_count,
                dtype=str,
            )
            conditions = _conditions(
                task,
                relative_bin,
                features.causal_position_bucket,
            )
            score_map: dict[str, np.ndarray] = {}
            for block in SIGNAL_BLOCKS:
                values = features.blocks[block]
                for mode in CONDITION_MODES:
                    prefix = f"{block}__{mode}"
                    model = unflatten_model(reference, prefix)
                    projected = project_geometry(
                        values,
                        conditions[mode],
                        model,
                    )
                    score_map.update(
                        calibrated_scores(prefix, projected, reference)
                    )
            for mode in CONDITION_MODES:
                score_map.update(
                    _collapse_score_map(
                        reference,
                        features.collapse_global,
                        conditions[mode],
                        mode=mode,
                        geometry_config=geometry_config,
                    )
                )
            missing = set(score_names).difference(score_map)
            extra = set(score_map).difference(score_names)
            if missing or extra:
                raise RuntimeError(
                    f"score contract mismatch missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )
            score_matrix = np.column_stack(
                [score_map[name] for name in score_names]
            ).astype(np.float32)
            count = features.response_count
            rows["sample_id"].extend([str(sample.sample_id)] * count)
            rows["source_id"].extend(
                [canonical_source_group(sample)] * count
            )
            rows["token_index"].append(features.token_index)
            rows["response_length"].append(
                np.full(count, count, dtype=np.int32)
            )
            rows["relative_position"].append(features.relative_position)
            rows["causal_position_bucket"].append(
                features.causal_position_bucket
            )
            rows["task_type"].extend([_text(sample.task_type)] * count)
            rows["data_source"].extend([_text(sample.data_source)] * count)
            rows["generator_model"].extend(
                [_text(sample.generator_model)] * count
            )
            rows["scores"].append(score_matrix)
            rows["collapse_raw"].append(features.collapse_global)
            rows["evidence_raw"].append(features.evidence_global)
        finally:
            sample.release_attention()

    audit_result = audit.finish()
    output = {
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "token_index": np.concatenate(rows["token_index"]).astype(np.int32),
        "response_length": np.concatenate(rows["response_length"]).astype(np.int32),
        "relative_position": np.concatenate(rows["relative_position"]).astype(
            np.float32
        ),
        "causal_position_bucket": np.concatenate(
            rows["causal_position_bucket"]
        ).astype(np.int16),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "data_source": np.asarray(rows["data_source"], dtype=str),
        "generator_model": np.asarray(rows["generator_model"], dtype=str),
        "score_names": np.asarray(score_names, dtype=str),
        "scores": np.concatenate(rows["scores"]).astype(np.float32),
        "collapse_feature_names": np.asarray(
            COLLAPSE_FEATURE_NAMES,
            dtype=str,
        ),
        "collapse_raw": np.concatenate(rows["collapse_raw"]).astype(np.float32),
        "evidence_feature_names": np.asarray(
            EVIDENCE_FEATURE_NAMES,
            dtype=str,
        ),
        "evidence_directions": np.asarray(EVIDENCE_DIRECTIONS, dtype=np.int8),
        "evidence_raw": np.concatenate(rows["evidence_raw"]).astype(np.float32),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_file.verify(reference_file.path)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(reference_file.path)),
        reference_sha256=np.asarray(reference_file.sha256),
        dataset_manifest_sha256=np.asarray(dataset_manifest_sha256(dataset)),
        fit_group_id=np.asarray(reference["fit_group_id"], dtype=str),
        calibration_group_id=np.asarray(
            reference["calibration_group_id"],
            dtype=str,
        ),
        test_group_id=np.asarray(audit_result.test_source_ids, dtype=str),
        test_sample_id=np.asarray(audit_result.test_sample_ids, dtype=str),
        audit_scope=np.asarray(audit_result.test_scope),
        **output,
    )
    load_score_artifact(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["sample_id"])),
        "scores": len(score_names),
        "purpose": "mechanism_audit_not_model_selection",
    }


def _metrics(labels, values):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(labels) == 0 or np.unique(labels).size < 2:
        return None
    auc = float(roc_auc_score(labels, values))
    return {
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": auc,
        "orientation_free_auroc": max(auc, 1.0 - auc),
        "auprc": float(average_precision_score(labels, values)),
        "correct_median": float(np.median(values[labels == 0])),
        "hallucination_median": float(np.median(values[labels == 1])),
    }


def _bootstrap_mean(values, *, replicates: int, seed: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    result = {
        "samples": int(len(values)),
        "mean_effect": float(values.mean()),
        "median_effect": float(np.median(values)),
    }
    if int(replicates) > 0 and len(values) >= 2:
        rng = np.random.default_rng(int(seed))
        estimates = np.empty(int(replicates), dtype=np.float64)
        for index in range(int(replicates)):
            selected = rng.integers(0, len(values), size=len(values))
            estimates[index] = float(values[selected].mean())
        result["ci_low"] = float(np.quantile(estimates, 0.025))
        result["ci_high"] = float(np.quantile(estimates, 0.975))
    else:
        result["ci_low"] = None
        result["ci_high"] = None
    return result


def _onset_effects(
    values,
    labels,
    sample_id,
    token_index,
    *,
    window: int,
):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    sample_id = np.asarray(sample_id, dtype=str)
    token_index = np.asarray(token_index, dtype=np.int64)
    effects = []
    for current in np.unique(sample_id):
        selected = np.flatnonzero(sample_id == current)
        order = selected[np.argsort(token_index[selected])]
        y = labels[order]
        x = values[order]
        token = token_index[order]
        sample_effects = []
        for index in range(len(order)):
            if y[index] != 1 or (index > 0 and y[index - 1] == 1):
                continue
            run_end = index
            while run_end < len(order) and y[run_end] == 1:
                run_end += 1
            pre_start = max(0, index - int(window))
            post_end = min(run_end, index + int(window))
            if index <= pre_start or post_end <= index:
                continue
            if index > 0 and token[index] - token[index - 1] != 1:
                continue
            if not bool((y[pre_start:index] == 0).all()):
                continue
            pre = x[pre_start:index]
            post = x[index:post_end]
            if not np.isfinite(pre).all() or not np.isfinite(post).all():
                continue
            sample_effects.append(float(post.mean() - pre.mean()))
        if sample_effects:
            effects.append(float(np.mean(sample_effects)))
    return np.asarray(effects, dtype=np.float64)


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_rr_signal_audit(
    dataset,
    score_path,
    output_dir,
    *,
    onset_window: int = 4,
    bootstrap_replicates: int = 1000,
    seed: int = 20260818,
):
    """Open labels only after all attention scores have been frozen."""

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact = load_score_artifact(frozen.artifact.path)
    reference = verify_score_provenance(artifact)
    aligned = frozen.align_loaded(dataset, artifact)
    labels = aligned.token_label
    score_names = np.asarray(artifact["score_names"], dtype=str)
    scores = np.asarray(artifact["scores"], dtype=np.float32)
    collapse_names = np.asarray(
        artifact["collapse_feature_names"],
        dtype=str,
    )
    collapse_raw = np.asarray(artifact["collapse_raw"], dtype=np.float32)
    evidence_names = np.asarray(
        artifact["evidence_feature_names"], dtype=str
    )
    evidence_directions = np.asarray(
        artifact["evidence_directions"], dtype=np.int8
    )
    evidence_raw = np.asarray(artifact["evidence_raw"], dtype=np.float32)

    score_metrics = {}
    metric_rows = []
    for index, name in enumerate(score_names):
        current = _metrics(labels, scores[:, index])
        score_metrics[str(name)] = current
        if current is not None:
            metric_rows.append(
                {"family": "frozen_score", "name": str(name), **current}
            )

    collapse_metrics = {}
    onset = {}
    onset_rows = []
    for index, (name, direction) in enumerate(
        zip(collapse_names, COLLAPSE_DIRECTIONS, strict=True)
    ):
        raw = collapse_raw[:, index]
        oriented = raw if int(direction) >= 0 else -raw
        current = _metrics(labels, oriented)
        collapse_metrics[str(name)] = {
            "predeclared_direction": (
                "higher"
                if int(direction) > 0
                else "lower"
                if int(direction) < 0
                else "none_diagnostic_orientation_only"
            ),
            "oriented_metrics": current,
        }
        if current is not None:
            metric_rows.append(
                {
                    "family": "raw_collapse_feature",
                    "name": str(name),
                    "predeclared_direction": collapse_metrics[str(name)][
                        "predeclared_direction"
                    ],
                    **current,
                }
            )
        effects = _onset_effects(
            raw,
            labels,
            artifact["sample_id"],
            artifact["token_index"],
            window=onset_window,
        )
        summary = _bootstrap_mean(
            effects,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        onset[str(name)] = summary
        if summary is not None:
            onset_rows.append({"feature": str(name), **summary})

    evidence_metrics = {}
    for index, (name, direction) in enumerate(
        zip(evidence_names, evidence_directions, strict=True)
    ):
        raw = evidence_raw[:, index]
        oriented = raw if int(direction) > 0 else -raw
        raw_metrics = _metrics(labels, raw)
        oriented_metrics = _metrics(labels, oriented)
        evidence_metrics[str(name)] = {
            "historical_registry": EVIDENCE_REGISTRY[str(name)],
            "raw_metrics": raw_metrics,
            "fixed_oriented_metrics": oriented_metrics,
        }
        if oriented_metrics is not None:
            metric_rows.append(
                {
                    "family": "historically_frozen_scalar_baseline",
                    "name": str(name),
                    "predeclared_direction": (
                        "higher" if int(direction) > 0 else "lower"
                    ),
                    **oriented_metrics,
                }
            )

    task_metrics = {}
    task = np.asarray(artifact["task_type"], dtype=str)
    for task_name in sorted(set(task.tolist())):
        selected = task == task_name
        task_metrics[task_name] = {
            "frozen_scores": {
                str(name): _metrics(labels[selected], scores[selected, index])
                for index, name in enumerate(score_names)
            },
            "evidence_baselines": {
                str(name): _metrics(
                    labels[selected],
                    evidence_raw[selected, index]
                    if int(evidence_directions[index]) > 0
                    else -evidence_raw[selected, index],
                )
                for index, name in enumerate(evidence_names)
            },
        }

    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "purpose": (
            "PR/RR signal decomposition and hypothesis audit; score ranking is not "
            "a preregistered final detector selection"
        ),
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "score_metrics": score_metrics,
        "evidence_baselines": evidence_metrics,
        "raw_collapse_metrics": collapse_metrics,
        "first_hallucination_onset_effects": onset,
        "task_metrics": task_metrics,
        "coordination": json.loads(
            str(reference["coordination_summary_json"].item())
        ),
        "temporal_scope": score_temporal_scope().as_dict(),
        "reference_sha256": str(artifact["reference_sha256"].item()),
        "score_artifact_sha256": frozen.artifact.sha256,
        "claim_boundary": (
            "The audit preserves prompt/history route fields and separates "
            "diagonal, future received support, persistence ratio, current-row "
            "collapse, and joint-vs-independent geometry. Historically oriented "
            "scalars are exploratory baselines, not independently held-out feature "
            "selection. The audit does not infer correctness structure from a "
            "passed channel-shuffle gate alone."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "score_metrics.csv", metric_rows)
    _write_csv(output_dir / "onset_effects.csv", onset_rows)
    return report
