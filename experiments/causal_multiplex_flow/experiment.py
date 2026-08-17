"""Label-free fit, calibration, scoring, and post-hoc evaluation for CMRP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from experiment_protocol import (
    FrozenEvaluation,
    FrozenFile,
    HeldOutSourceAudit,
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
from .calibration import (
    empirical_upper_tail,
    finite_reference,
    split_source_groups,
    topology_gate_summary,
)
from .events import EventConfig, extract_causal_events
from .model import CausalMultiplexRouter, ModelConfig

MODEL_SCHEMA = "cmrp-model-v1"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    calibration_fraction: float = 0.25
    seed: int = 20260817

    def validate(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError("epochs must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive and finite")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metadata_text(value) -> str:
    return "" if value is None else str(value)


def _selected_sample_ids(dataset, limit=None):
    sample_ids = list(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("no samples selected")
    return sample_ids


def _save_model(
    model: CausalMultiplexRouter,
    path: Path,
    *,
    event_config: EventConfig,
    train_config: TrainConfig,
) -> None:
    payload = {
        "schema": MODEL_SCHEMA,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "model_config": model.config_dict,
        "event_config": asdict(event_config),
        "train_config": asdict(train_config),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def _load_model(reference_file: FrozenFile, *, device):
    reference_file.verify(reference_file.path)
    reference = load_reference(reference_file.path)
    reference_file.verify(reference_file.path)
    model_path = reference_file.path.parent / str(
        np.asarray(reference["model_file"]).item()
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_file = FrozenFile.capture(model_path)
    if model_file.sha256 != str(np.asarray(reference["model_sha256"]).item()):
        raise ValueError("CMRP model digest differs from the reference")
    payload = torch.load(model_file.path, map_location=device, weights_only=False)
    model_file.verify(model_file.path)
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported CMRP model schema")
    model = CausalMultiplexRouter(
        num_layers=int(payload["num_layers"]),
        num_heads=int(payload["num_heads"]),
        config=ModelConfig(**payload["model_config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    event_config = EventConfig(**payload["event_config"])
    return model, event_config, reference, model_file


def _score_samples(
    model,
    dataset,
    sample_ids,
    *,
    event_config,
    description,
    source_audit=None,
):
    rows = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
            "task_type",
            "data_source",
            "generator_model",
            "raw_route_surprise",
            "presence_nll",
            "source_nll",
            "weight_error",
            "rewired_source_nll",
            "rewire_gap",
            "selected_rr_edges",
        )
    }
    edge_gaps = []
    with torch.no_grad():
        for sample_id in tqdm(sample_ids, desc=description, unit="sample"):
            sample = dataset[sample_id]
            try:
                sample.attention()
                if source_audit is not None:
                    source_audit.observe(sample)
                events = extract_causal_events(sample, config=event_config)
                output = model(events)
                values = output.detached_numpy()
                edge_gaps.append(values["rewire_edge_gap"])
                count = events.response_count
                rows["sample_id"].extend([str(sample.sample_id)] * count)
                rows["source_id"].extend([_metadata_text(sample.source_id)] * count)
                rows["token_index"].extend(range(count))
                rows["response_length"].extend([count] * count)
                rows["task_type"].extend([_metadata_text(sample.task_type)] * count)
                rows["data_source"].extend([_metadata_text(sample.data_source)] * count)
                rows["generator_model"].extend(
                    [_metadata_text(sample.generator_model)] * count
                )
                for name in (
                    "raw_route_surprise",
                    "presence_nll",
                    "source_nll",
                    "weight_error",
                    "rewired_source_nll",
                    "rewire_gap",
                    "selected_rr_edges",
                ):
                    rows[name].append(values[name])
            finally:
                sample.release_attention()
    output = {
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "data_source": np.asarray(rows["data_source"], dtype=str),
        "generator_model": np.asarray(rows["generator_model"], dtype=str),
    }
    for name in (
        "raw_route_surprise",
        "presence_nll",
        "source_nll",
        "weight_error",
        "rewired_source_nll",
        "rewire_gap",
    ):
        output[name] = np.concatenate(rows[name]).astype(np.float32, copy=False)
    output["selected_rr_edges"] = np.concatenate(rows["selected_rr_edges"]).astype(
        np.int32, copy=False
    )
    return output, np.concatenate(edge_gaps).astype(np.float32, copy=False)


def fit_cmrp(
    dataset,
    output_dir,
    *,
    event_config: EventConfig | None = None,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    limit=None,
):
    """Train on fit groups and calibrate on disjoint unlabeled source groups."""
    event_config = EventConfig() if event_config is None else event_config
    model_config = ModelConfig() if model_config is None else model_config
    train_config = TrainConfig() if train_config is None else train_config
    event_config.validate()
    model_config.validate()
    train_config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")

    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    split = split_source_groups(
        dataset,
        calibration_fraction=train_config.calibration_fraction,
        seed=train_config.seed,
        limit=limit,
    )
    if len(split["fit_group_ids"]) < 2:
        raise ValueError(
            "CMRP needs at least two fit source groups; increase TRAIN_LIMIT"
        )
    geometry = (
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
    )
    device = getattr(dataset, "device", "cpu")
    model = CausalMultiplexRouter(
        num_layers=geometry[0],
        num_heads=geometry[1],
        config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    rng = np.random.default_rng(train_config.seed)
    epoch_losses = []
    fit_ids = list(split["fit_sample_ids"])
    for epoch in range(train_config.epochs):
        model.train()
        order = rng.permutation(len(fit_ids))
        losses = []
        progress = tqdm(order, desc=f"fit CMRP epoch {epoch + 1}", unit="sample")
        for index in progress:
            sample = dataset[fit_ids[int(index)]]
            try:
                events = extract_causal_events(sample, config=event_config)
                optimizer.zero_grad(set_to_none=True)
                result = model(events)
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError("CMRP training loss is non-finite")
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.gradient_clip
                )
                optimizer.step()
                losses.append(float(result.loss.detach().cpu()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}")
            finally:
                sample.release_attention()
        epoch_losses.append(float(np.mean(losses)))

    model_path = output_dir / "model.pt"
    _save_model(
        model,
        model_path,
        event_config=event_config,
        train_config=train_config,
    )
    model_file = FrozenFile.capture(model_path)
    model.eval()
    calibration_rows, calibration_edge_gaps = _score_samples(
        model,
        dataset,
        split["calibration_sample_ids"],
        event_config=event_config,
        description="calibrate CMRP",
    )
    calibration_raw = finite_reference(
        calibration_rows["raw_route_surprise"], minimum=2
    ).astype(np.float32)
    gate = topology_gate_summary(
        calibration_edge_gaps,
        selected_edge_count=int(calibration_rows["selected_rr_edges"].sum()),
    )
    train_manifest.verify(train_manifest.path)
    model_file.verify(model_file.path)
    reference_path = output_dir / "reference.npz"
    np.savez_compressed(
        reference_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        model_file=np.asarray(model_path.name),
        model_sha256=np.asarray(model_file.sha256),
        train_dataset_manifest_sha256=np.asarray(train_manifest.sha256),
        num_layers=np.asarray(geometry[0], dtype=np.int16),
        num_heads=np.asarray(geometry[1], dtype=np.int16),
        event_config_json=np.asarray(_json(asdict(event_config))),
        model_config_json=np.asarray(_json(asdict(model_config))),
        train_config_json=np.asarray(_json(asdict(train_config))),
        fit_group_id=np.asarray(split["fit_group_ids"], dtype=str),
        calibration_group_id=np.asarray(split["calibration_group_ids"], dtype=str),
        calibration_raw_route_surprise=calibration_raw,
        topology_gate_mean_gap=np.asarray(
            np.nan if gate["mean_gap"] is None else gate["mean_gap"], dtype=np.float32
        ),
        topology_gate_median_gap=np.asarray(
            np.nan if gate["median_gap"] is None else gate["median_gap"], dtype=np.float32
        ),
        topology_gate_evaluated_edge_count=np.asarray(
            gate["evaluated_edge_count"], dtype=np.int32
        ),
        topology_gate_selected_edge_count=np.asarray(
            gate["selected_edge_count"], dtype=np.int32
        ),
        topology_gate_coverage=np.asarray(gate["coverage"], dtype=np.float32),
        topology_gate_positive_fraction=np.asarray(
            np.nan
            if gate["positive_fraction"] is None
            else gate["positive_fraction"],
            dtype=np.float32,
        ),
        topology_gate_pass=np.asarray(gate["pass"]),
        epoch_loss=np.asarray(epoch_losses, dtype=np.float32),
        fit_samples=np.asarray(len(split["fit_sample_ids"]), dtype=np.int32),
        calibration_samples=np.asarray(
            len(split["calibration_sample_ids"]), dtype=np.int32
        ),
        calibration_tokens=np.asarray(len(calibration_raw), dtype=np.int32),
    )
    # Validate the completed reference before returning it to the runner.
    load_reference(reference_path)
    return {
        "output_dir": str(output_dir),
        "model": str(model_path),
        "reference": str(reference_path),
        "labels_read": False,
        "fit_samples": len(split["fit_sample_ids"]),
        "calibration_samples": len(split["calibration_sample_ids"]),
        "calibration_tokens": len(calibration_raw),
        "epoch_loss": epoch_losses,
        "topology_gate": gate,
    }


def score_cmrp(dataset, reference_path, output_path, *, limit=None):
    """Freeze calibrated CMRP test scores without opening token labels."""
    device = getattr(dataset, "device", "cpu")
    reference_file = FrozenFile.capture(reference_path)
    model, event_config, reference, model_file = _load_model(
        reference_file, device=device
    )
    dataset_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
    if (
        int(dataset.manifest["num_layers"]) != model.num_layers
        or int(dataset.manifest["num_heads"]) != model.num_heads
    ):
        raise ValueError("test attention geometry differs from the CMRP model")
    sample_ids = _selected_sample_ids(dataset, limit)
    source_audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=sample_ids,
        reserved_source_ids=(
            reference["fit_group_id"].tolist()
            + reference["calibration_group_id"].tolist()
        ),
        require_complete_split=limit is None,
    )
    rows, _ = _score_samples(
        model,
        dataset,
        sample_ids,
        event_config=event_config,
        description="score CMRP test",
        source_audit=source_audit,
    )
    audit = source_audit.finish()
    score = empirical_upper_tail(
        reference["calibration_raw_route_surprise"],
        rows["raw_route_surprise"],
    ).astype(np.float32)
    reference_file.verify(reference_file.path)
    model_file.verify(model_file.path)
    dataset_manifest.verify(dataset_manifest.path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        reference_path=np.asarray(str(reference_file.path)),
        reference_sha256=np.asarray(reference_file.sha256),
        model_path=np.asarray(str(model_file.path)),
        model_sha256=np.asarray(model_file.sha256),
        dataset_manifest_sha256=np.asarray(dataset_manifest.sha256),
        fit_group_id=np.asarray(reference["fit_group_id"], dtype=str),
        calibration_group_id=np.asarray(
            reference["calibration_group_id"], dtype=str
        ),
        test_group_id=np.asarray(audit.test_source_ids, dtype=str),
        test_sample_id=np.asarray(audit.test_sample_ids, dtype=str),
        audit_scope=np.asarray(audit.test_scope),
        score=score,
        **rows,
    )
    load_score_artifact(output_path)
    return {
        "output": str(output_path),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": len(score),
        "primary_detector": "calibrated_causal_route_surprise",
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
        "tokens": len(labels),
        "positive_tokens": int(labels.sum()),
        "prevalence": prevalence,
        "auprc_random_baseline": prevalence,
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "correct_median": float(np.median(values[labels == 0])),
        "hallucination_median": float(np.median(values[labels == 1])),
    }


def evaluate_cmrp(dataset, score_path, output_path):
    """Open labels only after the CMRP score artifact is frozen."""
    evaluation = FrozenEvaluation.capture(score_path, expected_split="test")
    artifact = load_score_artifact(evaluation.artifact.path)
    verify_score_provenance(artifact)
    aligned = evaluation.align_loaded(dataset, artifact)
    labels = aligned.token_label
    components = {
        "calibrated_causal_route_surprise": artifact["score"],
        "raw_route_surprise": artifact["raw_route_surprise"],
        "presence_nll": artifact["presence_nll"],
        "source_nll": artifact["source_nll"],
        "weight_error": artifact["weight_error"],
        "rewired_source_nll": artifact["rewired_source_nll"],
        "rewire_gap": artifact["rewire_gap"],
    }
    metrics = {
        name: _binary_metrics(labels, values) for name, values in components.items()
    }
    primary = metrics["calibrated_causal_route_surprise"]
    report = {
        "schema": EVALUATION_SCHEMA,
        "labels_read": True,
        "primary_detector": "calibrated_causal_route_surprise",
        "metrics": primary,
        "components": metrics,
        "reference_sha256": str(np.asarray(artifact["reference_sha256"]).item()),
        "model_sha256": str(np.asarray(artifact["model_sha256"]).item()),
        "score_artifact": str(Path(score_path)),
        **score_temporal_scope().as_dict(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
