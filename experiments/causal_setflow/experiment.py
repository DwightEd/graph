"""Label-free fit, frozen scoring, and post-hoc evaluation for CASF."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import (
    FrozenEvaluation,
    FrozenFile,
    HeldOutSourceAudit,
    dataset_manifest_sha256,
    file_sha256,
    partition_source_groups,
)

from .artifacts import (
    EVALUATION_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_checkpoint,
    load_reference,
    load_score_artifact,
    save_checkpoint,
)
from .calibration import (
    COMPONENT_NAMES,
    CalibrationConfig,
    calibrate_component_matrix,
    causal_condition,
    component_matrix,
    fit_latent_reference,
    latent_mahalanobis,
)
from .config import SetFlowModelConfig, SourceSetConfig, TrainingConfig
from .data import extract_causal_source_set_graph
from .model import CausalSetFlowModel
from .trainer import extract_frozen_rows, select_reference_rows, train_label_free


def fit_setflow(
    dataset,
    output_dir,
    *,
    source_config: SourceSetConfig | None = None,
    model_config: SetFlowModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    calibration_config: CalibrationConfig | None = None,
    device: str = "cuda",
    limit=None,
):
    """Train CASF and fit disjoint unlabeled latent/calibration references."""

    source_config = SourceSetConfig() if source_config is None else source_config
    model_config = SetFlowModelConfig() if model_config is None else model_config
    training_config = TrainingConfig() if training_config is None else training_config
    calibration_config = (
        CalibrationConfig(latent_trim_fraction=training_config.latent_trim_fraction)
        if calibration_config is None
        else calibration_config
    )
    source_config.validate()
    model_config.validate()
    training_config.validate()
    calibration_config.validate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    reference_path = output_dir / "reference.npz"
    if model_path.exists() or reference_path.exists():
        raise FileExistsError("CASF fit refuses to overwrite frozen artifacts")

    sample_ids = _sample_ids(dataset, limit)
    split = partition_source_groups(
        dataset,
        sample_ids,
        calibration_fraction=training_config.calibration_fraction,
        seed=training_config.seed,
    )
    first_sample = dataset[split["fit_sample_ids"][0]]
    try:
        graph = extract_causal_source_set_graph(first_sample, source_config)
        num_layers, num_heads = graph.num_layers, graph.num_heads
    finally:
        first_sample.release_attention()
    model = CausalSetFlowModel(
        num_layers,
        num_heads,
        source_config=source_config,
        model_config=model_config,
    )
    history = train_label_free(
        model,
        dataset,
        split["fit_sample_ids"],
        source_config=source_config,
        training_config=training_config,
        device=device,
    )
    save_checkpoint(
        model_path,
        {
            "state_dict": model.state_dict(),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "source_config": asdict(source_config),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
        },
    )

    fit_rows = extract_frozen_rows(
        model,
        dataset,
        split["fit_sample_ids"],
        source_config=source_config,
        deterministic_masks=training_config.deterministic_masks,
        seed=training_config.seed + 300_000,
        device=device,
        precision=training_config.precision,
    )
    selected = select_reference_rows(
        fit_rows, training_config.reference_per_sample
    )
    latent_reference = fit_latent_reference(
        fit_rows["embedding"][selected],
        trim_fraction=calibration_config.latent_trim_fraction,
        epsilon=calibration_config.epsilon,
    )

    calibration_rows = extract_frozen_rows(
        model,
        dataset,
        split["calibration_sample_ids"],
        source_config=source_config,
        deterministic_masks=training_config.deterministic_masks,
        seed=training_config.seed + 600_000,
        device=device,
        precision=training_config.precision,
    )
    calibration_matrix = component_matrix(
        _raw_components(calibration_rows, latent_reference)
    )
    calibration_conditions = causal_condition(
        calibration_rows["task_type"], calibration_rows["token_index"]
    )
    train_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
    artifact = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "model_path": np.asarray(str(model_path.resolve())),
        "model_sha256": np.asarray(file_sha256(model_path)),
        "train_dataset_manifest_sha256": np.asarray(train_manifest.sha256),
        "source_config_json": np.asarray(_json(asdict(source_config))),
        "model_config_json": np.asarray(_json(asdict(model_config))),
        "training_config_json": np.asarray(_json(asdict(training_config))),
        "calibration_config_json": np.asarray(_json(asdict(calibration_config))),
        "fit_group_id": np.asarray(split["fit_group_ids"], dtype=str),
        "calibration_group_id": np.asarray(
            split["calibration_group_ids"], dtype=str
        ),
        "component_names": np.asarray(COMPONENT_NAMES, dtype=str),
        "calibration_components": calibration_matrix,
        "calibration_conditions": calibration_conditions,
        "calibration_sample_id": calibration_rows["sample_id"],
        "calibration_token_index": calibration_rows["token_index"],
        "training_history_json": np.asarray(_json(list(history.rows))),
        "fit_samples": np.asarray(len(split["fit_sample_ids"]), dtype=np.int32),
        "calibration_samples": np.asarray(
            len(split["calibration_sample_ids"]), dtype=np.int32
        ),
        "fit_reference_tokens": np.asarray(len(selected), dtype=np.int32),
        "calibration_tokens": np.asarray(
            len(calibration_rows["sample_id"]), dtype=np.int32
        ),
        **latent_reference,
    }
    train_manifest.verify(train_manifest.path)
    np.savez_compressed(reference_path, **artifact)
    load_reference(reference_path)
    return {
        "output": str(reference_path),
        "model": str(model_path),
        "labels_read": False,
        "fit_samples": len(split["fit_sample_ids"]),
        "calibration_samples": len(split["calibration_sample_ids"]),
        "fit_reference_tokens": int(len(selected)),
        "calibration_tokens": int(len(calibration_rows["sample_id"])),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": model_config.hidden_dim,
        "precision": training_config.precision,
        "activation_checkpointing": model_config.activation_checkpointing,
    }


def score_setflow(
    dataset,
    reference_path,
    output_path,
    *,
    device: str = "cuda",
    limit=None,
):
    """Freeze held-out token embeddings and anomaly scores without labels."""

    reference_file = FrozenFile.capture(reference_path)
    reference = load_reference(reference_file.path)
    model_path = Path(str(reference["model_path"].item()))
    if file_sha256(model_path) != str(reference["model_sha256"].item()):
        raise ValueError("CASF model digest differs from frozen reference")
    checkpoint = load_checkpoint(model_path)
    source_config = SourceSetConfig(**checkpoint["source_config"])
    model_config = SetFlowModelConfig(**checkpoint["model_config"])
    training_config = TrainingConfig(**checkpoint["training_config"])
    calibration_config = CalibrationConfig(
        **json.loads(str(reference["calibration_config_json"].item()))
    )
    model = CausalSetFlowModel(
        int(checkpoint["num_layers"]),
        int(checkpoint["num_heads"]),
        source_config=source_config,
        model_config=model_config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    sample_ids = _sample_ids(dataset, limit)
    audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=sample_ids,
        reserved_source_ids=np.concatenate(
            (reference["fit_group_id"], reference["calibration_group_id"])
        ),
        require_complete_split=limit is None,
    )
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            audit.observe(sample)
        finally:
            sample.release_attention()
    rows = extract_frozen_rows(
        model,
        dataset,
        sample_ids,
        source_config=source_config,
        deterministic_masks=training_config.deterministic_masks,
        seed=training_config.seed + 900_000,
        device=device,
        precision=training_config.precision,
    )
    audit_result = audit.finish()
    raw = component_matrix(_raw_components(rows, reference))
    conditions = causal_condition(rows["task_type"], rows["token_index"])
    tail, score = calibrate_component_matrix(
        reference["calibration_components"],
        reference["calibration_conditions"],
        raw,
        conditions,
        min_condition_rows=calibration_config.min_condition_rows,
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("CASF score refuses to overwrite frozen artifact")
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(reference_file.path)),
        reference_sha256=np.asarray(reference_file.sha256),
        model_path=np.asarray(str(model_path.resolve())),
        model_sha256=np.asarray(file_sha256(model_path)),
        dataset_manifest_sha256=np.asarray(dataset_manifest_sha256(dataset)),
        fit_group_id=reference["fit_group_id"],
        calibration_group_id=reference["calibration_group_id"],
        test_group_id=np.asarray(audit_result.test_source_ids, dtype=str),
        test_sample_id=np.asarray(audit_result.test_sample_ids, dtype=str),
        audit_scope=np.asarray(audit_result.test_scope),
        component_names=np.asarray(COMPONENT_NAMES, dtype=str),
        embedding=rows["embedding"],
        components_raw=raw,
        components_tail=tail,
        score=score,
        **{
            name: rows[name]
            for name in (
                "sample_id",
                "source_id",
                "task_type",
                "data_source",
                "generator_model",
                "token_index",
                "response_length",
            )
        },
    )
    load_score_artifact(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": int(len(score)),
        "primary_detector": "empirically_recalibrated_fisher_setflow",
        "precision": training_config.precision,
    }


def evaluate_setflow(dataset, score_path, output_path):
    """Open labels only after the complete score artifact is frozen."""

    evaluation = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact = load_score_artifact(evaluation.artifact.path)
    labels = evaluation.align_loaded(dataset, artifact).token_label
    components = {
        "primary": artifact["score"],
        **{
            name: artifact["components_tail"][:, index]
            for index, name in enumerate(COMPONENT_NAMES)
        },
    }
    metrics = {name: _metrics(labels, value) for name, value in components.items()}
    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "primary_detector": "empirically_recalibrated_fisher_setflow",
        "metrics": metrics["primary"],
        "components": metrics,
        "reference_sha256": str(artifact["reference_sha256"].item()),
        "score_artifact_sha256": evaluation.artifact.sha256,
        "claim_boundary": (
            "CASF is trained only on attention-derived masked reconstruction. "
            "The primary score is a fixed empirical combination of reconstruction "
            "and latent-density components; no test-label direction or weight is fitted."
        ),
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _raw_components(rows, latent_reference):
    result = {
        name: np.asarray(rows[name], dtype=np.float32)
        for name in COMPONENT_NAMES
        if name != "latent_mahalanobis"
    }
    result["latent_mahalanobis"] = latent_mahalanobis(
        rows["embedding"], latent_reference
    )
    return result


def _metrics(labels, score):
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    labels, score = labels[finite], score[finite]
    if not len(labels) or np.unique(labels).size < 2:
        return None
    return {
        "tokens": int(len(labels)),
        "positive_tokens": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auprc_random_baseline": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
        "correct_median": float(np.median(score[labels == 0])),
        "hallucination_median": float(np.median(score[labels == 1])),
    }


def _sample_ids(dataset, limit=None):
    values = tuple(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        values = values[:limit]
    if not values:
        raise ValueError("no samples selected")
    return values


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))