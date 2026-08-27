"""Dataset-level orchestration for GroundedRoute.

The five public stages are deliberately explicit: build a lightweight graph
specification, fit the encoder, encode node graphs, fit/score one detector, and
evaluate a frozen score artifact.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import (
    HeldOutSourceAudit,
    dataset_manifest_sha256,
    partition_source_groups,
    scalar_text,
    sha256_text,
    validate_source_audit,
)
from research_dataset import open_research_dataset

from .artifacts import (
    EmbeddingIndex,
    EncodedTokenGraph,
    GraphSpec,
    load_checkpoint,
    load_embedding_index,
    load_graph_spec,
    save_checkpoint,
    save_embedding_index,
    save_encoded_graph,
    save_graph_spec,
    save_scores,
    sha256,
)
from .config import (
    MESSAGE_MODES,
    GraphConfig,
    GroundedRouteConfig,
    InterventionConfig,
    LearningConfig,
    ModelConfig,
    TrainConfig,
)
from .controls import apply_variant
from .detection import PCAKNNConfig, save_reference
from .detection import fit as fit_detector
from .graph import build_graph
from .learning import self_supervised_loss
from .model import GroundedRouteEncoder

ENCODER_IMPLEMENTATION_FILES = (
    "artifacts.py",
    "config.py",
    "controls.py",
    "graph.py",
    "learning.py",
    "model.py",
    "pipeline.py",
)


def implementation_sha256() -> str:
    """Bind a checkpoint to the exact encoder implementation used to fit it."""

    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for name in ENCODER_IMPLEMENTATION_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build(
    data_root,
    output_path,
    *,
    task: str = "QA",
    limit: int | None = None,
    graph_config: GraphConfig | None = None,
) -> dict[str, object]:
    """Freeze a label-free sample selection without materializing its graphs."""

    graph_config = GraphConfig() if graph_config is None else graph_config
    dataset = open_research_dataset(data_root, device="cpu")
    sample_ids = select_samples(dataset, task, limit)
    spec = GraphSpec(
        dataset_root=str(Path(data_root).resolve()),
        dataset_manifest_sha256=dataset_manifest_sha256(dataset),
        split=str(dataset.manifest["split"]),
        task=task,
        sample_ids=sample_ids,
        layer_count=int(dataset.manifest["num_layers"]),
        head_count=int(dataset.manifest["num_heads"]),
        graph_config=asdict(graph_config),
    )
    save_graph_spec(output_path, spec)
    return {
        "spec": str(Path(output_path).resolve()),
        "samples": len(sample_ids),
        "labels_read": False,
    }


def fit(
    spec_path,
    checkpoint_path,
    *,
    device="cpu",
    train_config: TrainConfig | None = None,
    variant: str = "real",
    minimum_changed_fraction: float = 0.01,
    message_mode: str = "neighbor",
) -> dict[str, object]:
    """Fit the causal endpoint encoder on source-disjoint unlabeled streams."""

    train_config = TrainConfig() if train_config is None else train_config
    spec = load_graph_spec(spec_path)
    if spec.split != "train":
        raise ValueError("encoder fitting requires a train graph spec")
    dataset = open_spec_dataset(spec, "cpu")
    split = source_split(dataset, spec.sample_ids, train_config)
    config = GroundedRouteConfig(
        graph=GraphConfig(**spec.graph_config),
        model=ModelConfig(message_mode=message_mode),
        train=train_config,
        intervention=InterventionConfig(
            variant=variant,
            minimum_changed_fraction=minimum_changed_fraction,
        ),
    )

    set_random_seed(train_config.seed)
    model = GroundedRouteEncoder(
        spec.layer_count,
        spec.head_count,
        config.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    changed_edges = 0
    variant_edges = 0

    for epoch in range(train_config.epochs):
        model.train()
        order = list(split["fit_sample_ids"])
        random.Random(train_config.seed + epoch).shuffle(order)
        training_loss: list[float] = []
        for sample_id in tqdm(order, desc=f"fit epoch {epoch + 1}", unit="sample"):
            sample = dataset[sample_id]
            try:
                graph = build_graph(sample, config.graph)
            finally:
                sample.release_attention()
            graph = graph.to(device)
            graph, changed, total = controlled_graph(graph, config)
            if epoch == 0:
                changed_edges += changed
                variant_edges += total
            generator = graph_generator(
                "cpu",
                train_config.seed,
                sample_id,
                epoch,
            )
            optimizer.zero_grad(set_to_none=True)
            output = self_supervised_loss(
                model,
                graph,
                config.learning,
                generator,
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.gradient_clip,
            )
            optimizer.step()
            training_loss.append(float(output.loss.detach().item()))
            del graph, output

        validation, validation_changed, validation_edges = validation_epoch(
            model,
            dataset,
            split["validation_sample_ids"],
            config,
            train_config.seed,
        )
        if epoch == 0:
            changed_edges += validation_changed
            variant_edges += validation_edges
            require_effective_control(config.intervention, changed_edges, variant_edges)
        current_training = float(np.mean(training_loss))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": current_training,
                "validation_loss": validation,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("encoder fitting produced no checkpoint")

    train_sources = tuple(
        sorted(
            set(split["fit_source_ids"])
            | set(split["validation_source_ids"])
            | set(split["calibration_source_ids"])
        )
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    actual_changed_fraction = change_fraction(changed_edges, variant_edges)
    save_checkpoint(
        checkpoint_path,
        {
            "config": config.as_dict(),
            "layer_count": spec.layer_count,
            "head_count": spec.head_count,
            "state_dict": best_state,
            "history": history,
            "best_validation_loss": best_validation,
            "parameter_count": parameter_count,
            "implementation_sha256": implementation_sha256(),
            "graph_spec_sha256": sha256(spec_path),
            "fit_sample_ids": split["fit_sample_ids"],
            "validation_sample_ids": split["validation_sample_ids"],
            "calibration_sample_ids": split["calibration_sample_ids"],
            "fit_source_ids": split["fit_source_ids"],
            "validation_source_ids": split["validation_source_ids"],
            "calibration_source_ids": split["calibration_source_ids"],
            "train_source_ids": train_sources,
            "variant": config.intervention.variant,
            "message_mode": config.model.message_mode,
            "changed_fraction": actual_changed_fraction,
            "changed_edges": changed_edges,
            "variant_edges": variant_edges,
        },
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "samples": len(split["fit_sample_ids"]),
        "train_loss": history[-1]["train_loss"],
        "best_validation_loss": best_validation,
        "parameter_count": parameter_count,
        "implementation_sha256": implementation_sha256(),
        "variant": config.intervention.variant,
        "message_mode": config.model.message_mode,
        "changed_fraction": actual_changed_fraction,
        "labels_read": False,
    }


def encode(
    spec_path,
    checkpoint_path,
    output_dir,
    *,
    scope: str,
    device="cpu",
    variant: str | None = None,
    message_mode: str | None = None,
) -> dict[str, object]:
    """Save one self-contained encoded token graph per selected sample."""

    spec = load_graph_spec(spec_path)
    dataset = open_spec_dataset(spec, "cpu")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_geometry(spec, checkpoint)
    validate_checkpoint_implementation(checkpoint)
    model, config = restore_model(checkpoint, device)
    if config.graph != GraphConfig(**spec.graph_config):
        raise ValueError("checkpoint and graph spec use different graph construction")
    if variant is not None and variant != config.intervention.variant:
        raise ValueError("requested variant differs from the checkpoint variant")
    if message_mode is not None and message_mode != config.model.message_mode:
        raise ValueError("requested message mode differs from the checkpoint")

    if scope == "calibration":
        if spec.split != "train" or sha256(spec_path) != checkpoint["graph_spec_sha256"]:
            raise ValueError("calibration encoding requires the fitted train graph spec")
        sample_ids = tuple(map(str, checkpoint["calibration_sample_ids"]))
        audit = None
    elif scope == "all":
        sample_ids = spec.sample_ids
        audit = None
        if spec.split == "test":
            audit = HeldOutSourceAudit(
                dataset,
                selected_sample_ids=sample_ids,
                reserved_source_ids=checkpoint["train_source_ids"],
                require_complete_split=set(sample_ids) == set(map(str, dataset.sample_ids)),
            )
    else:
        raise ValueError("encode scope must be 'all' or 'calibration'")

    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, np.ndarray]] = []
    graph_paths: list[str] = []
    graph_sample_ids: list[str] = []
    graph_sha256: list[str] = []
    edge_count = 0
    changed_edges = 0
    variant_edges = 0

    model.eval()
    with torch.no_grad():
        for number, sample_id in enumerate(
            tqdm(sample_ids, desc=f"encode {scope}", unit="sample")
        ):
            sample = dataset[sample_id]
            try:
                if audit is not None:
                    audit.observe(sample)
                graph = build_graph(sample, config.graph)
            finally:
                sample.release_attention()
            graph = graph.to(device)
            graph, changed, total = controlled_graph(graph, config)
            changed_edges += changed
            variant_edges += total
            output = model.encode(graph)
            encoded = EncodedTokenGraph.from_output(graph, output)
            relative_path = Path("graphs") / f"{number:08d}.pt"
            graph_path = output_dir / relative_path
            save_encoded_graph(graph_path, encoded)
            chunks.append(embedding_chunk(encoded))
            graph_paths.append(relative_path.as_posix())
            graph_sample_ids.append(encoded.sample_id)
            graph_sha256.append(sha256(graph_path))
            edge_count += graph.edge_count
            del graph, output, encoded

    require_effective_control(config.intervention, changed_edges, variant_edges)
    actual_changed_fraction = change_fraction(changed_edges, variant_edges)
    index = concatenate_embedding_chunks(chunks)
    metadata: dict[str, object] = {
        "dataset_manifest_sha256": spec.dataset_manifest_sha256,
        "checkpoint_sha256": sha256(checkpoint_path),
        "implementation_sha256": checkpoint["implementation_sha256"],
        "graph_spec_sha256": sha256(spec_path),
        "split": spec.split,
        "scope": scope,
        "encoded_graph_sample_ids": graph_sample_ids,
        "encoded_graph_paths": graph_paths,
        "encoded_graph_sha256": graph_sha256,
        "variant": config.intervention.variant,
        "message_mode": config.model.message_mode,
        "changed_fraction": actual_changed_fraction,
    }
    if audit is not None:
        source_audit = audit.finish()
        metadata.update(
            audit_scope=source_audit.test_scope,
            reserved_source_ids=checkpoint["train_source_ids"],
            test_source_ids=source_audit.test_source_ids,
            test_sample_ids=source_audit.test_sample_ids,
        )
    else:
        metadata.update(
            calibration_sample_ids=checkpoint["calibration_sample_ids"],
            calibration_source_ids=checkpoint["calibration_source_ids"],
            encoder_source_ids=tuple(
                sorted(
                    set(checkpoint["fit_source_ids"])
                    | set(checkpoint["validation_source_ids"])
                )
            ),
        )

    index_path = output_dir / "index.npz"
    save_embedding_index(index_path, index, **metadata)
    return {
        "embeddings": str(index_path.resolve()),
        "samples": len(sample_ids),
        "nodes": len(index.sample_id),
        "edges": edge_count,
        "variant": config.intervention.variant,
        "message_mode": config.model.message_mode,
        "changed_fraction": actual_changed_fraction,
        "labels_read": False,
    }


def detect(
    calibration_index_path,
    test_index_path,
    reference_path,
    score_path,
    *,
    config: PCAKNNConfig | None = None,
) -> dict[str, object]:
    """Fit PCA-kNN on calibration embeddings and score test embeddings."""

    calibration, calibration_meta = load_embedding_index(calibration_index_path)
    test, test_meta = load_embedding_index(test_index_path)
    if scalar_text(calibration_meta, "scope") != "calibration":
        raise ValueError("detector reference must use calibration embeddings")
    validate_calibration_provenance(calibration, calibration_meta)
    checkpoint_hash = sha256_text(calibration_meta, "checkpoint_sha256")
    if checkpoint_hash != sha256_text(test_meta, "checkpoint_sha256"):
        raise ValueError("calibration and test embeddings use different encoders")
    variant = scalar_text(calibration_meta, "variant")
    if variant != scalar_text(test_meta, "variant"):
        raise ValueError("calibration and test embeddings use different variants")
    message_mode = artifact_message_mode(calibration_meta)
    if message_mode != artifact_message_mode(test_meta):
        raise ValueError("calibration and test embeddings use different message modes")
    calibration_changed_fraction = scalar_number(
        calibration_meta,
        "changed_fraction",
    )
    test_changed_fraction = scalar_number(test_meta, "changed_fraction")
    validate_source_audit(
        reserved_source_ids=test_meta["reserved_source_ids"],
        test_source_ids=test_meta["test_source_ids"],
        test_sample_ids=test_meta["test_sample_ids"],
        row_sample_ids=test.sample_id,
        row_source_ids=test.source_id,
        audit_scope=scalar_text(test_meta, "audit_scope"),
    )

    reference = fit_detector(calibration.embedding, config)
    save_reference(
        reference_path,
        reference,
        checkpoint_sha256=checkpoint_hash,
        calibration_embedding_sha256=sha256(calibration_index_path),
        variant=variant,
        message_mode=message_mode,
        changed_fraction=calibration_changed_fraction,
    )
    score = reference.score(test.embedding)
    save_scores(
        score_path,
        test,
        score,
        model_type="grounded_route",
        variant=variant,
        message_mode=message_mode,
        changed_fraction=test_changed_fraction,
        calibration_changed_fraction=calibration_changed_fraction,
        dataset_manifest_sha256=sha256_text(
            test_meta,
            "dataset_manifest_sha256",
        ),
        checkpoint_sha256=checkpoint_hash,
        detector_sha256=sha256(reference_path),
        test_embedding_sha256=sha256(test_index_path),
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
        "variant": variant,
        "message_mode": message_mode,
        "changed_fraction": test_changed_fraction,
        "labels_read": False,
    }


def select_samples(dataset, task: str, limit: int | None) -> tuple[str, ...]:
    selected: list[str] = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        try:
            if task.casefold() == "all" or str(sample.task_type).casefold() == task.casefold():
                selected.append(str(sample_id))
        finally:
            sample.release_attention()
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no samples match the requested task")
    return tuple(selected)


def open_spec_dataset(spec: GraphSpec, device):
    dataset = open_research_dataset(spec.dataset_root, device=device)
    if dataset_manifest_sha256(dataset) != spec.dataset_manifest_sha256:
        raise ValueError("dataset manifest differs from the graph spec")
    if str(dataset.manifest["split"]) != spec.split:
        raise ValueError("dataset split differs from the graph spec")
    if (
        int(dataset.manifest["num_layers"]) != spec.layer_count
        or int(dataset.manifest["num_heads"]) != spec.head_count
    ):
        raise ValueError("dataset geometry differs from the graph spec")
    return dataset


def source_split(dataset, sample_ids, config: TrainConfig) -> dict[str, tuple[str, ...]]:
    outer = partition_source_groups(
        dataset,
        sample_ids,
        calibration_fraction=config.detector_fraction,
        seed=config.seed,
    )
    inner = partition_source_groups(
        dataset,
        outer["fit_sample_ids"],
        calibration_fraction=(
            config.validation_fraction / (1.0 - config.detector_fraction)
        ),
        seed=config.seed + 1,
    )
    return {
        "fit_sample_ids": inner["fit_sample_ids"],
        "validation_sample_ids": inner["calibration_sample_ids"],
        "calibration_sample_ids": outer["calibration_sample_ids"],
        "fit_source_ids": inner["fit_group_ids"],
        "validation_source_ids": inner["calibration_group_ids"],
        "calibration_source_ids": outer["calibration_group_ids"],
    }


def validation_epoch(
    model,
    dataset,
    sample_ids,
    config,
    seed: int,
) -> tuple[float, int, int]:
    model.eval()
    values: list[float] = []
    changed_edges = 0
    variant_edges = 0
    with torch.no_grad():
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                graph = build_graph(sample, config.graph)
            finally:
                sample.release_attention()
            graph = graph.to(next(model.parameters()).device)
            graph, changed, total = controlled_graph(graph, config)
            changed_edges += changed
            variant_edges += total
            generator = graph_generator("cpu", seed, sample_id, 10_000)
            output = self_supervised_loss(
                model,
                graph,
                config.learning,
                generator,
            )
            values.append(float(output.loss.item()))
            del graph, output
    return float(np.mean(values)), changed_edges, variant_edges


def restore_model(checkpoint, device):
    payload = checkpoint["config"]
    config = GroundedRouteConfig(
        graph=GraphConfig(**payload["graph"]),
        model=ModelConfig(**payload["model"]),
        learning=LearningConfig(**payload["learning"]),
        train=TrainConfig(**payload["train"]),
        intervention=InterventionConfig(**payload["intervention"]),
    )
    if str(checkpoint["variant"]) != config.intervention.variant:
        raise ValueError("checkpoint variant differs from its frozen configuration")
    if str(checkpoint.get("message_mode", "neighbor")) != config.model.message_mode:
        raise ValueError("checkpoint message mode differs from its frozen configuration")
    model = GroundedRouteEncoder(
        int(checkpoint["layer_count"]),
        int(checkpoint["head_count"]),
        config.model,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config


def validate_checkpoint_implementation(checkpoint) -> None:
    expected = checkpoint.get("implementation_sha256")
    if expected is None:
        raise ValueError("checkpoint does not record its encoder implementation")
    if str(expected) != implementation_sha256():
        raise ValueError("checkpoint encoder implementation differs from current code")


def graph_generator(device, seed: int, sample_id: str, stream):
    digest = hashlib.sha256(f"{seed}\0{stream}\0{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    return torch.Generator(device=device).manual_seed(value)


def controlled_graph(graph, config: GroundedRouteConfig):
    variant = config.intervention.variant
    generator = graph_generator(
        "cpu",
        config.train.seed,
        graph.sample_id,
        f"variant:{variant}",
    )
    controlled = apply_variant(
        graph,
        variant,
        generator,
        endpoint_rewire_passes=config.intervention.endpoint_rewire_passes,
    )
    if variant == "weight_shuffle":
        # The audit observes float16 sidecars, so count only persisted changes.
        changed = int(
            (
                controlled.edges.weight.to(torch.float16)
                != graph.edges.weight.to(torch.float16)
            ).sum().item()
        )
    elif variant == "endpoint_rewire":
        changed = int((controlled.edges.source != graph.edges.source).sum().item())
    elif variant == "no_message":
        changed = graph.edge_count
    else:
        changed = 0
    return controlled, changed, graph.edge_count


def change_fraction(changed_edges: int, variant_edges: int) -> float:
    return float(changed_edges / variant_edges) if variant_edges else 0.0


def require_effective_control(
    config: InterventionConfig,
    changed_edges: int,
    variant_edges: int,
) -> None:
    if (
        config.variant != "real"
        and change_fraction(changed_edges, variant_edges)
        < config.minimum_changed_fraction
    ):
        raise RuntimeError(
            f"{config.variant} changed fewer than "
            f"{config.minimum_changed_fraction:.1%} of retained edges"
        )


def validate_checkpoint_geometry(spec: GraphSpec, checkpoint) -> None:
    if (
        int(checkpoint["layer_count"]) != spec.layer_count
        or int(checkpoint["head_count"]) != spec.head_count
    ):
        raise ValueError("checkpoint and graph spec use different layer/head geometry")


def validate_calibration_provenance(
    calibration: EmbeddingIndex,
    metadata,
) -> None:
    validate_source_audit(
        reserved_source_ids=metadata["encoder_source_ids"],
        test_source_ids=metadata["calibration_source_ids"],
        test_sample_ids=metadata["calibration_sample_ids"],
        row_sample_ids=calibration.sample_id,
        row_source_ids=calibration.source_id,
        audit_scope="selected_samples",
    )


def scalar_number(mapping, name: str) -> float:
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"artifact field {name!r} must be a scalar number")
    return float(value.item())


def artifact_message_mode(metadata) -> str:
    mode = (
        scalar_text(metadata, "message_mode")
        if "message_mode" in metadata
        else "neighbor"
    )
    if mode not in MESSAGE_MODES:
        raise ValueError(f"artifact message_mode must be one of {MESSAGE_MODES}")
    return mode


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embedding_chunk(graph: EncodedTokenGraph) -> dict[str, np.ndarray]:
    count = graph.response_count
    return {
        "sample_id": np.repeat(graph.sample_id, count),
        "source_id": np.repeat(graph.source_id, count),
        "task_type": np.repeat(graph.task_type, count),
        "token_index": np.arange(count, dtype=np.int32),
        "response_length": np.full(count, count, dtype=np.int32),
        "response_token_id": graph.token_ids[graph.response_start :]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64),
        "embedding": graph.response_embedding.detach().cpu().numpy().astype(np.float32),
    }


def concatenate_embedding_chunks(chunks: list[dict[str, np.ndarray]]) -> EmbeddingIndex:
    if not chunks:
        raise RuntimeError("encoding produced no token graphs")
    return EmbeddingIndex(
        **{
            name: np.concatenate([chunk[name] for chunk in chunks])
            for name in chunks[0]
        }
    )
