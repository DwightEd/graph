"""Fit the copied DBGNN encoder and export one embedding per token node."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import FrozenEvaluation, scalar_text, validate_source_audit
from research_dataset import open_research_dataset
from experiments.grounded_route.artifacts import (
    load_embedding_index,
    load_scores,
    save_embedding_index,
    save_encoded_graph,
    save_scores,
    sha256,
)
from experiments.grounded_route.detection import (
    PCAKNNConfig,
    fit as fit_detector,
    save_reference,
)
from experiments.grounded_route.evaluate import metrics, source_bootstrap
from experiments.grounded_route.graph_effectiveness.data import load_bundle
from experiments.grounded_route.pipeline import (
    concatenate_embedding_chunks,
    embedding_chunk,
    validate_calibration_provenance,
)

from .config import DBGNNConfig
from .graph import build_dbgnn_graph
from .learning import self_supervised_loss
from .upstream import LinkPredictionModel, OfficialNodeEncoder, UPSTREAM_COMMIT


CHECKPOINT_SCHEMA = "dbgnn-original-code-reference-checkpoint"
CHECKPOINT_VERSION = 1


def implementation_sha256() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in (
        "config.py",
        "graph.py",
        "learning.py",
        "pipeline.py",
        "upstream.py",
        "vendor/dbgnn.py",
        "vendor/gcn.py",
    ):
        digest.update(relative.encode())
        digest.update((package / relative).read_bytes())
    return digest.hexdigest()


def build_model(config: DBGNNConfig) -> LinkPredictionModel:
    encoder = OfficialNodeEncoder(
        encoder=config.encoder,
        first_order_dim=4,
        higher_order_dim=10,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
    return LinkPredictionModel(encoder, config.embedding_dim)


def training_protocol_sha256(config: DBGNNConfig) -> str:
    payload = config.as_dict()
    payload.pop("encoder")
    payload.pop("higher_order_mode")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit(
    train_index_path,
    checkpoint_path,
    *,
    config: DBGNNConfig | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Train with masked endpoint prediction; no hallucination labels are read."""

    config = DBGNNConfig() if config is None else config
    bundle = load_bundle(train_index_path)
    if scalar_text(bundle.metadata, "split") != "train":
        raise ValueError("DBGNN fitting requires graph data from the train split")
    fit_records, validation_records, calibration_records = source_split(
        bundle.records,
        config.validation_fraction,
        config.detector_fraction,
        config.seed,
    )

    set_seed(config.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(config.epochs):
        model.train()
        order = list(fit_records)
        random.Random(config.seed + epoch).shuffle(order)
        training = []
        route = []
        variance = []
        positives = 0
        eligible = 0
        positive_graphs = 0
        for record in tqdm(order, desc=f"fit {config.encoder} epoch {epoch + 1}", unit="graph"):
            graph = record.load()
            generator = sample_generator(config.seed, record.sample_id, epoch)
            optimizer.zero_grad(set_to_none=True)
            output = self_supervised_loss(model, graph, config, generator)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training.append(float(output.loss.detach().item()))
            route.append(float(output.route_loss.detach().item()))
            variance.append(float(output.variance_loss.detach().item()))
            positives += output.positive_count
            eligible += output.eligible_count
            positive_graphs += int(output.positive_count > 0)

        validation = validation_epoch(
            model,
            validation_records,
            config,
            config.seed,
        )
        if positives == 0 or validation["positive_pairs"] == 0:
            raise RuntimeError("endpoint objective found no train/validation RR pairs")
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(training)),
            "train_route_loss": float(np.mean(route)),
            "train_variance_loss": float(np.mean(variance)),
            "positive_pairs": positives,
            "eligible_pairs": eligible,
            "graphs_with_positive_pairs": positive_graphs,
            "validation_loss": validation["loss"],
            "validation_route_loss": validation["route_loss"],
            "validation_variance_loss": validation["variance_loss"],
            "validation_positive_pairs": validation["positive_pairs"],
            "validation_eligible_pairs": validation["eligible_pairs"],
            "validation_graphs_with_positive_pairs": validation[
                "graphs_with_positive_pairs"
            ],
        }
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": CHECKPOINT_VERSION,
        "labels_included": False,
        "config": config.as_dict(),
        "state_dict": best_state,
        "history": history,
        "best_validation_loss": best_loss,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "implementation_sha256": implementation_sha256(),
        "upstream_commit": UPSTREAM_COMMIT,
        "train_index_sha256": bundle.index_sha256,
        "dataset_manifest_sha256": scalar_text(
            bundle.metadata, "dataset_manifest_sha256"
        ),
        "fit_sample_ids": tuple(record.sample_id for record in fit_records),
        "validation_sample_ids": tuple(
            record.sample_id for record in validation_records
        ),
        "calibration_sample_ids": tuple(
            record.sample_id for record in calibration_records
        ),
        "fit_source_ids": unique_sources(fit_records),
        "validation_source_ids": unique_sources(validation_records),
        "calibration_source_ids": unique_sources(calibration_records),
        "train_source_ids": unique_sources(bundle.records),
    }
    save_checkpoint(checkpoint_path, payload)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "encoder": config.encoder,
        "higher_order_mode": config.higher_order_mode,
        "samples": len(fit_records),
        "calibration_samples": len(calibration_records),
        "best_validation_loss": best_loss,
        "parameter_count": payload["parameter_count"],
        "positive_pairs": history[-1]["positive_pairs"],
        "eligible_pairs": history[-1]["eligible_pairs"],
        "labels_read": False,
    }


def encode(
    source_index_path,
    checkpoint_path,
    output_dir,
    *,
    scope: str,
    device: str = "cpu",
) -> dict[str, object]:
    """Run the full clean graph and persist the pre-classifier node tensor."""

    bundle = load_bundle(source_index_path)
    checkpoint = load_checkpoint(checkpoint_path)
    config = DBGNNConfig(**checkpoint["config"])
    if scope == "calibration":
        if scalar_text(bundle.metadata, "split") != "train":
            raise ValueError("calibration embeddings must come from train graphs")
        if bundle.index_sha256 != checkpoint["train_index_sha256"]:
            raise ValueError("calibration graphs differ from encoder training input")
        selected = set(checkpoint["calibration_sample_ids"])
        records = tuple(
            record for record in bundle.records if record.sample_id in selected
        )
    elif scope == "all":
        if scalar_text(bundle.metadata, "split") != "test":
            raise ValueError("all-scope DBGNN encoding expects test graphs")
        overlap = set(checkpoint["train_source_ids"]).intersection(
            unique_sources(bundle.records)
        )
        if overlap:
            raise ValueError("DBGNN train and test source groups overlap")
        records = bundle.records
    else:
        raise ValueError("scope must be 'calibration' or 'all'")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    paths = []
    ids = []
    hashes = []
    first_edges = 0
    higher_edges = 0
    with torch.no_grad():
        for number, record in enumerate(
            tqdm(records, desc=f"encode {config.encoder} {scope}", unit="graph")
        ):
            source = record.load()
            lifted = build_dbgnn_graph(
                source,
                delta_layers=config.delta_layers,
                higher_order_mode=config.higher_order_mode,
            ).to(device)
            embedding = model.encode(lifted).detach().cpu()
            encoded = replace(source, node_embedding=embedding)
            relative = Path("graphs") / f"{number:08d}.pt"
            path = output_dir / relative
            save_encoded_graph(path, encoded)
            chunks.append(embedding_chunk(encoded))
            paths.append(relative.as_posix())
            ids.append(encoded.sample_id)
            hashes.append(sha256(path))
            first_edges += lifted.edge_index_fo.shape[1]
            higher_edges += lifted.edge_index.shape[1]

    index = concatenate_embedding_chunks(chunks)
    checkpoint_hash = sha256(checkpoint_path)
    metadata = {
        "dataset_manifest_sha256": scalar_text(
            bundle.metadata, "dataset_manifest_sha256"
        ),
        "graph_spec_sha256": scalar_text(bundle.metadata, "graph_spec_sha256"),
        "source_index_sha256": bundle.index_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "implementation_sha256": checkpoint["implementation_sha256"],
        "split": scalar_text(bundle.metadata, "split"),
        "scope": scope,
        "encoded_graph_sample_ids": ids,
        "encoded_graph_paths": paths,
        "encoded_graph_sha256": hashes,
        "encoder_family": f"official_lisiq_{config.encoder}",
        "parameter_count": checkpoint["parameter_count"],
        "embedding_dim": config.embedding_dim,
        "training_seed": config.seed,
        "training_protocol_sha256": training_protocol_sha256(config),
        "upstream_commit": UPSTREAM_COMMIT,
        "debruijn_order": 2 if config.encoder == "dbgnn" else 1,
        "delta_layers": config.delta_layers,
        "higher_order_mode": config.higher_order_mode,
        "variant": "real",
        "message_mode": "neighbor",
        "changed_fraction": 0.0,
        "lineage_used_by_encoder": False,
        "lineage_used_by_detector": False,
    }
    if scope == "calibration":
        metadata.update(
            calibration_sample_ids=tuple(ids),
            calibration_source_ids=checkpoint["calibration_source_ids"],
            encoder_source_ids=tuple(
                sorted(
                    set(checkpoint["fit_source_ids"])
                    | set(checkpoint["validation_source_ids"])
                )
            ),
        )
    else:
        metadata.update(
            audit_scope=scalar_text(bundle.metadata, "audit_scope"),
            reserved_source_ids=checkpoint["train_source_ids"],
            test_source_ids=unique_sources(bundle.records),
            test_sample_ids=tuple(ids),
        )
    index_path = output_dir / "index.npz"
    save_embedding_index(index_path, index, **metadata)
    return {
        "embeddings": str(index_path.resolve()),
        "encoder": config.encoder,
        "higher_order_mode": config.higher_order_mode,
        "samples": len(ids),
        "nodes": len(index.sample_id),
        "first_order_edges": first_edges,
        "higher_order_edges": higher_edges,
        "labels_read": False,
    }


def detect(
    calibration_path,
    test_path,
    reference_path,
    score_path,
    *,
    detector_config: PCAKNNConfig | None = None,
) -> dict[str, object]:
    calibration, calibration_meta = load_embedding_index(calibration_path)
    test, test_meta = load_embedding_index(test_path)
    if (
        scalar_text(calibration_meta, "split") != "train"
        or scalar_text(calibration_meta, "scope") != "calibration"
        or scalar_text(test_meta, "split") != "test"
        or scalar_text(test_meta, "scope") != "all"
    ):
        raise ValueError("DBGNN detection needs train calibration and held-out test embeddings")
    checkpoint = scalar_text(calibration_meta, "checkpoint_sha256")
    if checkpoint != scalar_text(test_meta, "checkpoint_sha256"):
        raise ValueError("calibration and test embeddings use different DBGNN encoders")
    validate_calibration_provenance(calibration, calibration_meta)
    for field in (
        "encoder_family",
        "higher_order_mode",
        "upstream_commit",
        "training_protocol_sha256",
    ):
        if scalar_text(calibration_meta, field) != scalar_text(test_meta, field):
            raise ValueError(f"calibration and test embeddings differ in {field}")
    validate_source_audit(
        reserved_source_ids=test_meta["reserved_source_ids"],
        test_source_ids=test_meta["test_source_ids"],
        test_sample_ids=test_meta["test_sample_ids"],
        row_sample_ids=test.sample_id,
        row_source_ids=test.source_id,
        audit_scope=scalar_text(test_meta, "audit_scope"),
    )
    reference = fit_detector(calibration.embedding, detector_config)
    save_reference(
        reference_path,
        reference,
        checkpoint_sha256=checkpoint,
        calibration_embedding_sha256=sha256(calibration_path),
        encoder_family=scalar_text(calibration_meta, "encoder_family"),
    )
    score = reference.score(test.embedding)
    save_scores(
        score_path,
        test,
        score,
        model_type=scalar_text(test_meta, "encoder_family"),
        higher_order_mode=scalar_text(test_meta, "higher_order_mode"),
        variant="real",
        changed_fraction=0.0,
        calibration_changed_fraction=0.0,
        dataset_manifest_sha256=scalar_text(
            test_meta, "dataset_manifest_sha256"
        ),
        checkpoint_sha256=checkpoint,
        detector_sha256=sha256(reference_path),
        test_embedding_sha256=sha256(test_path),
        audit_scope=scalar_text(test_meta, "audit_scope"),
        reserved_source_ids=test_meta["reserved_source_ids"],
        test_source_ids=test_meta["test_source_ids"],
        test_sample_ids=test_meta["test_sample_ids"],
    )
    return {
        "reference": str(Path(reference_path).resolve()),
        "scores": str(Path(score_path).resolve()),
        "samples": len(set(test.sample_id.tolist())),
        "nodes": len(score),
        "labels_read": False,
    }


def evaluate(
    test_root,
    score_path,
    output_path,
    *,
    bootstrap_replicates: int = 500,
    seed: int = 20260826,
) -> dict[str, object]:
    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    scores = load_scores(score_path)
    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    labels = frozen.align_loaded(dataset, scores)
    label = labels.token_label.astype(np.int8)
    result = {
        "schema": "dbgnn-reference-evaluation",
        "version": 1,
        "model_type": scalar_text(scores, "model_type"),
        "higher_order_mode": scalar_text(scores, "higher_order_mode"),
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "samples": len(set(scores["sample_id"].astype(str).tolist())),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        **metrics(label, scores["score"]),
        "source_bootstrap": source_bootstrap(
            label,
            scores["score"],
            labels.source_id,
            bootstrap_replicates,
            seed,
        ),
        "score_artifact": str(frozen.artifact.path),
        "score_sha256": frozen.artifact.sha256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**result, "evaluation": str(output_path.resolve())}


def validation_epoch(model, records, config, seed: int) -> dict[str, float | int]:
    model.eval()
    loss = []
    route = []
    variance = []
    positives = 0
    eligible = 0
    positive_graphs = 0
    with torch.no_grad():
        for record in records:
            output = self_supervised_loss(
                model,
                record.load(),
                config,
                sample_generator(seed, record.sample_id, "validation"),
            )
            loss.append(float(output.loss.item()))
            route.append(float(output.route_loss.item()))
            variance.append(float(output.variance_loss.item()))
            positives += output.positive_count
            eligible += output.eligible_count
            positive_graphs += int(output.positive_count > 0)
    return {
        "loss": float(np.mean(loss)),
        "route_loss": float(np.mean(route)),
        "variance_loss": float(np.mean(variance)),
        "positive_pairs": positives,
        "eligible_pairs": eligible,
        "graphs_with_positive_pairs": positive_graphs,
    }


def source_split(
    records,
    validation_fraction: float,
    detector_fraction: float,
    seed: int,
):
    groups = {}
    for record in records:
        groups.setdefault(record.source_id, []).append(record)
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )
    if len(ordered) < 3:
        raise ValueError("DBGNN fit needs three source groups for fit/validation/calibration")
    detector_count = min(
        len(ordered) - 2,
        max(1, round(len(ordered) * detector_fraction)),
    )
    calibration = set(ordered[:detector_count])
    encoder_groups = ordered[detector_count:]
    validation_count = min(
        len(encoder_groups) - 1,
        max(
            1,
            round(
                len(encoder_groups)
                * validation_fraction
                / (1.0 - detector_fraction)
            ),
        ),
    )
    validation = set(encoder_groups[:validation_count])
    return (
        tuple(
            record
            for record in records
            if record.source_id not in validation
            and record.source_id not in calibration
        ),
        tuple(record for record in records if record.source_id in validation),
        tuple(record for record in records if record.source_id in calibration),
    )


def unique_sources(records) -> tuple[str, ...]:
    return tuple(sorted({record.source_id for record in records}))


def sample_generator(seed: int, sample_id: str, stream) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}\0{sample_id}\0{stream}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as file:
        temporary = Path(file.name)
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or int(payload.get("version", -1)) != CHECKPOINT_VERSION
        or bool(payload.get("labels_included", True))
        or payload.get("upstream_commit") != UPSTREAM_COMMIT
        or payload.get("implementation_sha256") != implementation_sha256()
    ):
        raise ValueError("unsupported or stale DBGNN reference checkpoint")
    return payload
