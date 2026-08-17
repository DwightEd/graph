"""Label-free fit, scoring and post-hoc evaluation for CITG."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from attention_graph.causal_events import (
    MultiplexEventConfig,
    extract_causal_multiplex_events,
)
from attention_graph.topology_controls import (
    rewire_causal_sources,
    token_rewire_mask,
)
from experiment_protocol import (
    FrozenEvaluation,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
    file_sha256,
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
from .geometry import (
    GeometryConfig,
    VARIANTS,
    calibrate_energies,
    condition_keys,
    energy_by_variant,
    fit_all_geometries,
    reference_positions,
    topology_gate_summary,
)
from .signatures import SignatureConfig, extract_trajectory_features


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _text(value) -> str:
    return "" if value is None else str(value)


def _selected_sample_ids(dataset, limit=None) -> tuple[str, ...]:
    sample_ids = tuple(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("no samples selected")
    return sample_ids


def _extract(
    sample,
    *,
    event_config: MultiplexEventConfig,
    signature_config: SignatureConfig,
):
    events = extract_causal_multiplex_events(
        sample,
        config=event_config,
    )
    return events, extract_trajectory_features(
        events,
        config=signature_config,
    )


def _sample_conditions(sample, feature_set) -> np.ndarray:
    task = np.asarray(
        [_text(sample.task_type)] * feature_set.response_count,
        dtype=str,
    )
    return condition_keys(task, feature_set.position_bucket)


def _feature_names(feature_set) -> dict[str, np.ndarray]:
    return feature_set.names()


def fit_citg(
    dataset,
    output_dir,
    *,
    event_config: MultiplexEventConfig | None = None,
    signature_config: SignatureConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    limit=None,
):
    """Fit conditioned PPCA geometry and calibration without labels."""
    event_config = (
        MultiplexEventConfig() if event_config is None else event_config
    )
    signature_config = (
        SignatureConfig() if signature_config is None else signature_config
    )
    geometry_config = (
        GeometryConfig() if geometry_config is None else geometry_config
    )
    event_config.validate()
    signature_config.validate()
    geometry_config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = _selected_sample_ids(dataset, limit)
    split = partition_source_groups(
        dataset,
        selected_ids,
        calibration_fraction=geometry_config.calibration_fraction,
        seed=geometry_config.seed,
    )
    if len(split["fit_group_ids"]) < 2:
        raise ValueError(
            "CITG needs at least two fit source groups; increase TRAIN_LIMIT"
        )

    fit_rows = {variant: [] for variant in VARIANTS}
    condition_rows: list[np.ndarray] = []
    names_by_variant: dict[str, np.ndarray] | None = None
    for sample_id in tqdm(
        split["fit_sample_ids"],
        desc="fit CITG trajectory reference",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            _, features = _extract(
                sample,
                event_config=event_config,
                signature_config=signature_config,
            )
            selected = reference_positions(
                features.response_count,
                geometry_config.reference_per_sample,
            )
            current_names = _feature_names(features)
            if names_by_variant is None:
                names_by_variant = current_names
            else:
                for variant in VARIANTS:
                    if not np.array_equal(
                        names_by_variant[variant],
                        current_names[variant],
                    ):
                        raise RuntimeError(
                            f"{variant} feature names changed across samples"
                        )
            for variant, values in features.variants().items():
                fit_rows[variant].append(values[selected])
            condition_rows.append(
                _sample_conditions(sample, features)[selected]
            )
        finally:
            sample.release_attention()

    if names_by_variant is None:
        raise RuntimeError("CITG fit stream produced no features")
    fit_values = {
        variant: np.concatenate(rows, axis=0).astype(
            np.float32, copy=False
        )
        for variant, rows in fit_rows.items()
    }
    fit_conditions = np.concatenate(condition_rows).astype(str)
    geometry = fit_all_geometries(
        fit_values,
        fit_conditions,
        names_by_variant,
        config=geometry_config,
    )

    calibration_energy_rows = {
        variant: [] for variant in VARIANTS
    }
    true_full_rows: list[np.ndarray] = []
    rewired_full_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    calibration_token_count = 0
    for sample_id in tqdm(
        split["calibration_sample_ids"],
        desc="calibrate CITG trajectory geometry",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            events, features = _extract(
                sample,
                event_config=event_config,
                signature_config=signature_config,
            )
            conditions = _sample_conditions(sample, features)
            true_energy = energy_by_variant(
                features.variants(),
                conditions,
                geometry,
            )
            rewired_events, changed = rewire_causal_sources(
                events,
                seed=geometry_config.seed,
            )
            rewired_features = extract_trajectory_features(
                rewired_events,
                config=signature_config,
            )
            rewired_energy = energy_by_variant(
                rewired_features.variants(),
                conditions,
                geometry,
            )
            for variant in VARIANTS:
                calibration_energy_rows[variant].append(
                    true_energy[variant]
                )
            true_full_rows.append(true_energy["full"])
            rewired_full_rows.append(rewired_energy["full"])
            valid_rows.append(
                token_rewire_mask(events, changed)
                .detach()
                .cpu()
                .numpy()
            )
            source_rows.append(
                np.asarray(
                    [canonical_source_group(sample)]
                    * features.response_count,
                    dtype=str,
                )
            )
            calibration_token_count += features.response_count
        finally:
            sample.release_attention()

    calibration_energy = {
        variant: np.concatenate(rows).astype(np.float32, copy=False)
        for variant, rows in calibration_energy_rows.items()
    }
    gate = topology_gate_summary(
        np.concatenate(true_full_rows),
        np.concatenate(rewired_full_rows),
        np.concatenate(valid_rows),
        np.concatenate(source_rows),
        config=geometry_config,
    )

    reference_path = output_dir / "reference.npz"
    gate_value = lambda name: (
        np.nan if gate[name] is None else gate[name]
    )
    np.savez_compressed(
        reference_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        train_dataset_manifest_sha256=np.asarray(
            dataset_manifest_sha256(dataset)
        ),
        event_config_json=np.asarray(_json(asdict(event_config))),
        signature_config_json=np.asarray(
            _json(asdict(signature_config))
        ),
        geometry_config_json=np.asarray(_json(asdict(geometry_config))),
        fit_group_id=np.asarray(split["fit_group_ids"], dtype=str),
        calibration_group_id=np.asarray(
            split["calibration_group_ids"], dtype=str
        ),
        fit_samples=np.asarray(
            len(split["fit_sample_ids"]), dtype=np.int32
        ),
        calibration_samples=np.asarray(
            len(split["calibration_sample_ids"]), dtype=np.int32
        ),
        fit_reference_tokens=np.asarray(
            len(fit_conditions), dtype=np.int32
        ),
        calibration_tokens=np.asarray(
            calibration_token_count, dtype=np.int32
        ),
        topology_gate_token_count=np.asarray(
            gate["token_count"], dtype=np.int32
        ),
        topology_gate_evaluated_tokens=np.asarray(
            gate["evaluated_tokens"], dtype=np.int32
        ),
        topology_gate_coverage=np.asarray(
            gate["coverage"], dtype=np.float32
        ),
        topology_gate_source_groups=np.asarray(
            gate["source_groups"], dtype=np.int32
        ),
        topology_gate_mean_gap=np.asarray(
            gate_value("mean_gap"), dtype=np.float32
        ),
        topology_gate_median_gap=np.asarray(
            gate_value("median_gap"), dtype=np.float32
        ),
        topology_gate_positive_group_fraction=np.asarray(
            gate_value("positive_group_fraction"),
            dtype=np.float32,
        ),
        topology_gate_ci_low=np.asarray(
            gate_value("ci_low"), dtype=np.float32
        ),
        topology_gate_ci_high=np.asarray(
            gate_value("ci_high"), dtype=np.float32
        ),
        topology_gate_pass=np.asarray(gate["pass"]),
        **{
            f"calibration_energy_{variant}": values
            for variant, values in calibration_energy.items()
        },
        **geometry,
    )
    load_reference(reference_path)
    return {
        "output": str(reference_path),
        "labels_read": False,
        "fit_samples": len(split["fit_sample_ids"]),
        "calibration_samples": len(
            split["calibration_sample_ids"]
        ),
        "fit_reference_tokens": int(len(fit_conditions)),
        "calibration_tokens": int(calibration_token_count),
        "feature_dims": {
            variant: int(fit_values[variant].shape[1])
            for variant in VARIANTS
        },
        "topology_gate": gate,
    }


def _configs_from_reference(reference):
    return (
        MultiplexEventConfig(
            **json.loads(
                str(reference["event_config_json"].item())
            )
        ),
        SignatureConfig(
            **json.loads(
                str(reference["signature_config_json"].item())
            )
        ),
        GeometryConfig(
            **json.loads(
                str(reference["geometry_config_json"].item())
            )
        ),
    )


def score_citg(
    dataset,
    reference_path,
    output_path,
    *,
    limit=None,
):
    """Freeze causal trajectory scores without opening labels."""
    reference_path = Path(reference_path).resolve()
    reference = load_reference(reference_path)
    event_config, signature_config, geometry_config = (
        _configs_from_reference(reference)
    )
    sample_ids = _selected_sample_ids(dataset, limit)
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
    rows: dict[str, list] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
            "task_type",
            "data_source",
            "generator_model",
            "score",
            "score_static",
            "score_topology",
            "score_mass",
            "energy_full",
            "energy_static",
            "energy_topology",
            "energy_mass",
            "rewired_energy_full",
            "rewire_energy_gap",
            "rewire_valid",
        )
    }

    for sample_id in tqdm(
        sample_ids,
        desc="score CITG test trajectories",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            audit.observe(sample)
            events, features = _extract(
                sample,
                event_config=event_config,
                signature_config=signature_config,
            )
            conditions = _sample_conditions(sample, features)
            energies = energy_by_variant(
                features.variants(),
                conditions,
                reference,
            )
            scores = calibrate_energies(energies, reference)
            rewired_events, changed = rewire_causal_sources(
                events,
                seed=geometry_config.seed,
            )
            rewired_features = extract_trajectory_features(
                rewired_events,
                config=signature_config,
            )
            rewired_energy = energy_by_variant(
                rewired_features.variants(),
                conditions,
                reference,
            )["full"]
            valid = (
                token_rewire_mask(events, changed)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            count = features.response_count
            source_id = canonical_source_group(sample)
            rows["sample_id"].extend([str(sample.sample_id)] * count)
            rows["source_id"].extend([source_id] * count)
            rows["token_index"].extend(range(count))
            rows["response_length"].extend([count] * count)
            rows["task_type"].extend([_text(sample.task_type)] * count)
            rows["data_source"].extend(
                [_text(sample.data_source)] * count
            )
            rows["generator_model"].extend(
                [_text(sample.generator_model)] * count
            )
            rows["score"].append(scores["full"])
            rows["score_static"].append(scores["static"])
            rows["score_topology"].append(scores["topology"])
            rows["score_mass"].append(scores["mass"])
            for variant in VARIANTS:
                rows[f"energy_{variant}"].append(energies[variant])
            rewired_diagnostic = rewired_energy.astype(
                np.float32, copy=True
            )
            gap_diagnostic = (
                rewired_energy - energies["full"]
            ).astype(np.float32, copy=True)
            rewired_diagnostic[~valid] = np.nan
            gap_diagnostic[~valid] = np.nan
            rows["rewired_energy_full"].append(rewired_diagnostic)
            rows["rewire_energy_gap"].append(gap_diagnostic)
            rows["rewire_valid"].append(valid)
        finally:
            sample.release_attention()

    audit_result = audit.finish()
    output = {
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(
            rows["response_length"], dtype=np.int32
        ),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "data_source": np.asarray(rows["data_source"], dtype=str),
        "generator_model": np.asarray(
            rows["generator_model"], dtype=str
        ),
    }
    for name in (
        "score",
        "score_static",
        "score_topology",
        "score_mass",
        "energy_full",
        "energy_static",
        "energy_topology",
        "energy_mass",
        "rewired_energy_full",
        "rewire_energy_gap",
    ):
        output[name] = np.concatenate(rows[name]).astype(
            np.float32, copy=False
        )
    output["rewire_valid"] = np.concatenate(
        rows["rewire_valid"]
    ).astype(bool, copy=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(reference_path)),
        reference_sha256=np.asarray(file_sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(
            dataset_manifest_sha256(dataset)
        ),
        fit_group_id=np.asarray(
            reference["fit_group_id"], dtype=str
        ),
        calibration_group_id=np.asarray(
            reference["calibration_group_id"], dtype=str
        ),
        test_group_id=np.asarray(
            audit_result.test_source_ids, dtype=str
        ),
        test_sample_id=np.asarray(
            audit_result.test_sample_ids, dtype=str
        ),
        audit_scope=np.asarray(audit_result.test_scope),
        **output,
    )
    load_score_artifact(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(output["score"])),
        "primary_detector": "conditioned_causal_trajectory_ppca",
        "topology_gate_pass": bool(
            reference["topology_gate_pass"]
        ),
    }


def _binary_metrics(labels, values):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(labels) == 0 or np.unique(labels).size < 2:
        return None
    prevalence = float(labels.mean())
    return {
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": prevalence,
        "auprc_random_baseline": prevalence,
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "correct_median": float(np.median(values[labels == 0])),
        "hallucination_median": float(
            np.median(values[labels == 1])
        ),
    }


def evaluate_citg(dataset, score_path, output_path):
    """Open labels only after the CITG score artifact is frozen."""
    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact = load_score_artifact(frozen.artifact.path)
    reference = verify_score_provenance(artifact)
    labels = frozen.align_loaded(dataset, artifact).token_label
    components = {
        "conditioned_causal_trajectory_ppca": artifact["score"],
        "static_state_ablation": artifact["score_static"],
        "topology_trajectory_ablation": artifact["score_topology"],
        "mass_trajectory_ablation": artifact["score_mass"],
        "raw_full_ppca_energy": artifact["energy_full"],
        "rewire_energy_gap": artifact["rewire_energy_gap"],
    }
    component_metrics = {
        name: _binary_metrics(labels, values)
        for name, values in components.items()
    }
    primary = component_metrics[
        "conditioned_causal_trajectory_ppca"
    ]
    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "primary_detector": "conditioned_causal_trajectory_ppca",
        "metrics": primary,
        "components": component_metrics,
        "topology_gate": {
            "token_count": int(
                reference["topology_gate_token_count"]
            ),
            "evaluated_tokens": int(
                reference["topology_gate_evaluated_tokens"]
            ),
            "coverage": float(reference["topology_gate_coverage"]),
            "source_groups": int(
                reference["topology_gate_source_groups"]
            ),
            "mean_gap": float(
                reference["topology_gate_mean_gap"]
            ),
            "median_gap": float(
                reference["topology_gate_median_gap"]
            ),
            "positive_group_fraction": float(
                reference[
                    "topology_gate_positive_group_fraction"
                ]
            ),
            "ci_low": float(reference["topology_gate_ci_low"]),
            "ci_high": float(reference["topology_gate_ci_high"]),
            "pass": bool(reference["topology_gate_pass"]),
        },
        "temporal_scope": score_temporal_scope().as_dict(),
        "reference_sha256": str(
            artifact["reference_sha256"].item()
        ),
        "score_artifact_sha256": frozen.artifact.sha256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
