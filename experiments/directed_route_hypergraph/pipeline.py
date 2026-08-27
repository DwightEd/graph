"""Train, export and evaluate directed route-hypergraph node embeddings."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import canonical_source_group, scalar_text, sha256_text
from experiments.grounded_route.artifacts import (
    EncodedTokenGraph,
    load_embedding_index,
    load_scores,
    save_embedding_index,
    save_encoded_graph,
    save_scores,
    sha256,
)
from experiments.grounded_route.config import GraphConfig
from experiments.grounded_route.controls import apply_variant
from experiments.grounded_route.detection import PCAKNNConfig, save_reference
from experiments.grounded_route.detection import fit as fit_detector
from experiments.grounded_route.evaluate import metrics, source_bootstrap
from experiments.grounded_route.evaluation.data import EmbeddingTable
from experiments.grounded_route.graph import build_graph
from experiments.grounded_route.pipeline import (
    concatenate_embedding_chunks,
    embedding_chunk,
    graph_generator,
    select_samples,
    source_split,
)
from research_dataset import open_research_dataset

from .config import LearningConfig, ModelConfig, TrainConfig
from .learning import self_supervised_loss
from .model import DirectedRouteHypergraphEncoder

METHOD = "directed_route_hypergraph_flow_dae"


def fit(
    data_root,
    checkpoint_path,
    *,
    task: str = "QA",
    limit: int | None = None,
    device: str = "cpu",
    variant: str = "real",
    graph_config: GraphConfig | None = None,
    model_config: ModelConfig | None = None,
    learning_config: LearningConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, object]:
    """Fit the hypergraph encoder without hallucination labels."""

    graph_config = GraphConfig() if graph_config is None else graph_config
    model_config = ModelConfig() if model_config is None else model_config
    learning_config = LearningConfig() if learning_config is None else learning_config
    train_config = TrainConfig() if train_config is None else train_config
    dataset = open_research_dataset(data_root, device="cpu")
    train_split = str(dataset.manifest["split"])
    if train_split != "train":
        raise ValueError("encoder fitting requires the train split")
    sample_ids = select_samples(dataset, task, limit)
    split = source_split(dataset, sample_ids, train_config)

    set_seed(train_config.seed)
    model = DirectedRouteHypergraphEncoder(
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
        model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    best_state = None
    best_validation = float("inf")
    history = []

    for epoch in range(train_config.epochs):
        model.train()
        order = list(split["fit_sample_ids"])
        random.Random(train_config.seed + epoch).shuffle(order)
        losses = []
        row_losses = []
        flow_losses = []
        variance_losses = []
        for sample_id in tqdm(order, desc=f"fit epoch {epoch + 1}", unit="sample"):
            graph = load_graph(
                dataset,
                sample_id,
                variant,
                train_config.seed,
                graph_config,
                device,
            )
            generator = graph_generator("cpu", train_config.seed, sample_id, epoch)
            optimizer.zero_grad(set_to_none=True)
            output = self_supervised_loss(
                model,
                graph,
                learning_config,
                generator,
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip
            )
            optimizer.step()
            losses.append(float(output.loss.detach().item()))
            row_losses.append(float(output.row.detach().item()))
            flow_losses.append(float(output.flow.detach().item()))
            variance_losses.append(float(output.variance.detach().item()))

        validation = validation_epoch(
            model,
            dataset,
            split["validation_sample_ids"],
            variant,
            graph_config,
            learning_config,
            train_config,
            device,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_row_loss": float(np.mean(row_losses)),
            "train_flow_loss": float(np.mean(flow_losses)),
            "train_variance_loss": float(np.mean(variance_losses)),
            "validation_loss": validation["loss"],
            "validation_row_loss": validation["row"],
            "validation_flow_loss": validation["flow"],
            "validation_variance_loss": validation["variance"],
        }
        history.append(row)
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    payload = {
        "method": METHOD,
        "labels_included": False,
        "model_config": asdict(model_config),
        "learning_config": asdict(learning_config),
        "train_config": asdict(train_config),
        "graph_config": asdict(graph_config),
        "task": task,
        "train_split": train_split,
        "layer_count": int(dataset.manifest["num_layers"]),
        "head_count": int(dataset.manifest["num_heads"]),
        "state_dict": best_state,
        "history": history,
        "best_validation_loss": best_validation,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "variant": variant,
        "train_source_ids": tuple(
            sorted(
                set(split["fit_source_ids"])
                | set(split["validation_source_ids"])
                | set(split["calibration_source_ids"])
            )
        ),
        **split,
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "samples": len(split["fit_sample_ids"]),
        "calibration_samples": len(split["calibration_sample_ids"]),
        "best_validation_loss": best_validation,
        "parameter_count": payload["parameter_count"],
        "labels_read": False,
    }


def encode(
    data_root,
    checkpoint_path,
    output_dir,
    *,
    scope: str,
    task: str = "QA",
    limit: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Save one typed graph and one learned vector per token node."""

    dataset = open_research_dataset(data_root, device="cpu")
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint, model = restore_model(checkpoint_path, device)
    graph_config = GraphConfig(**checkpoint["graph_config"])
    if str(checkpoint["method"]) != METHOD:
        raise ValueError("checkpoint belongs to a different method")
    if task.casefold() != str(checkpoint["task"]).casefold():
        raise ValueError("encoding task differs from the checkpoint task")
    geometry = (
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
    )
    checkpoint_geometry = (
        int(checkpoint["layer_count"]),
        int(checkpoint["head_count"]),
    )
    if geometry != checkpoint_geometry:
        raise ValueError("encoding dataset geometry differs from the checkpoint")

    dataset_split = str(dataset.manifest["split"])
    if scope == "calibration":
        if dataset_split != str(checkpoint["train_split"]):
            raise ValueError("calibration encoding requires the fitted train split")
        sample_ids = tuple(map(str, checkpoint["calibration_sample_ids"]))
    elif scope == "all":
        if dataset_split != "test":
            raise ValueError("test encoding requires the test split")
        sample_ids = select_samples(dataset, task, limit)
    else:
        raise ValueError("scope must be 'calibration' or 'all'")

    source_ids = set()
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            source_ids.add(canonical_source_group(sample))
        finally:
            sample.release_attention()
    if scope == "all" and source_ids.intersection(checkpoint["train_source_ids"]):
        raise ValueError("train and test source groups overlap")

    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    edge_count = 0

    model.eval()
    with torch.no_grad():
        for number, sample_id in enumerate(
            tqdm(sample_ids, desc=f"encode {scope}", unit="sample")
        ):
            graph = load_graph(
                dataset,
                sample_id,
                str(checkpoint["variant"]),
                int(checkpoint["train_config"]["seed"]),
                graph_config,
                device,
            )
            output = model.encode(graph)
            encoded = EncodedTokenGraph.from_output(graph, output)
            save_encoded_graph(graph_dir / f"{number:08d}.pt", encoded)
            chunks.append(embedding_chunk(encoded))
            edge_count += graph.edge_count

    index = concatenate_embedding_chunks(chunks)
    index_path = output_dir / "index.npz"
    save_embedding_index(
        index_path,
        index,
        method=METHOD,
        scope=scope,
        variant=str(checkpoint["variant"]),
        checkpoint_sha256=checkpoint_hash,
        source_ids=tuple(sorted(source_ids)),
    )
    return {
        "embeddings": str(index_path.resolve()),
        "samples": len(sample_ids),
        "nodes": len(index.sample_id),
        "edges": edge_count,
        "labels_read": False,
    }


def detect(
    calibration_path,
    test_path,
    reference_path,
    score_path,
    *,
    config: PCAKNNConfig | None = None,
) -> dict[str, object]:
    """Fit a node-only one-class detector and freeze test scores."""

    calibration, calibration_meta = load_embedding_index(calibration_path)
    test, test_meta = load_embedding_index(test_path)
    if (
        scalar_text(calibration_meta, "method") != METHOD
        or scalar_text(test_meta, "method") != METHOD
    ):
        raise ValueError("embedding index belongs to a different method")
    if scalar_text(calibration_meta, "scope") != "calibration":
        raise ValueError("detector reference must use calibration embeddings")
    if scalar_text(test_meta, "scope") != "all":
        raise ValueError("detector test input must use all-scope embeddings")
    variant = scalar_text(calibration_meta, "variant")
    if variant != scalar_text(test_meta, "variant"):
        raise ValueError("calibration and test embeddings use different variants")
    checkpoint_hash = sha256_text(calibration_meta, "checkpoint_sha256")
    if checkpoint_hash != sha256_text(test_meta, "checkpoint_sha256"):
        raise ValueError("calibration and test embeddings use different checkpoints")

    reference = fit_detector(calibration.embedding, config)
    save_reference(
        reference_path,
        reference,
        method=METHOD,
        variant=variant,
        checkpoint_sha256=checkpoint_hash,
    )
    score = reference.score(test.embedding)
    save_scores(
        score_path,
        test,
        score,
        method=METHOD,
        variant=variant,
        checkpoint_sha256=checkpoint_hash,
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
    seed: int = 20260827,
) -> dict[str, object]:
    """Open token labels only after the node-only scores have been saved."""

    arrays = load_scores(score_path)
    if scalar_text(arrays, "method") != METHOD:
        raise ValueError("score artifact belongs to a different method")
    sha256_text(arrays, "checkpoint_sha256")
    table = EmbeddingTable(
        sample_id=arrays["sample_id"].astype(str),
        source_id=arrays["source_id"].astype(str),
        token_index=arrays["token_index"].astype(np.int32),
        response_length=arrays["response_length"].astype(np.int32),
        response_token_id=arrays["response_token_id"].astype(np.int64),
        embedding=np.empty((len(arrays["sample_id"]), 0), dtype=np.float32),
    )
    label = load_validated_labels(table, test_root)
    score = arrays["score"].astype(np.float64)
    result = {
        "method": METHOD,
        "labels_used_during": "posthoc_evaluation_only",
        "samples": len(set(table.sample_id.tolist())),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        **metrics(label, score),
        "source_bootstrap": source_bootstrap(
            label,
            score,
            table.source_id,
            bootstrap_replicates,
            seed,
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**result, "evaluation": str(output_path.resolve())}


def load_validated_labels(table: EmbeddingTable, test_root) -> np.ndarray:
    """Bind score rows to the exact test tokens before opening their labels."""

    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    if str(dataset.manifest["split"]) != "test":
        raise ValueError("evaluation labels must come from the test split")

    sample_ids = list(dict.fromkeys(table.sample_id.tolist()))
    label_store = dataset.prepare_evaluation_labels(sample_ids)
    labels = np.empty(len(table.sample_id), dtype=np.int8)
    for sample_id in sample_ids:
        rows = np.flatnonzero(table.sample_id == sample_id)
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            response_token_ids = (
                attention.token_ids[int(attention.response_idx) :]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            if not np.all(table.source_id[rows] == canonical_source_group(sample)):
                raise ValueError("evaluation source group differs from score rows")
            if not np.all(table.response_length[rows] == len(response_token_ids)):
                raise ValueError("evaluation response length differs from score rows")
            if not np.array_equal(
                table.response_token_id[rows],
                response_token_ids[table.token_index[rows]],
            ):
                raise ValueError("evaluation token IDs differ from score rows")
            current = label_store.response_labels(sample).cpu().numpy().astype(np.int8)
            labels[rows] = current[table.token_index[rows]]
        finally:
            sample.release_attention()
    return labels


def load_graph(
    dataset,
    sample_id,
    variant: str,
    seed: int,
    graph_config: GraphConfig,
    device: str,
):
    sample = dataset[sample_id]
    try:
        graph = build_graph(sample, graph_config)
    finally:
        sample.release_attention()
    generator = graph_generator("cpu", seed, str(sample_id), f"variant:{variant}")
    return apply_variant(graph, variant, generator).to(device)


def validation_epoch(
    model,
    dataset,
    sample_ids,
    variant,
    graph_config,
    learning_config,
    train_config,
    device,
) -> dict[str, float]:
    model.eval()
    values = {name: [] for name in ("loss", "row", "flow", "variance")}
    with torch.no_grad():
        for sample_id in sample_ids:
            graph = load_graph(
                dataset,
                sample_id,
                variant,
                train_config.seed,
                graph_config,
                device,
            )
            generator = graph_generator("cpu", train_config.seed, sample_id, "validation")
            output = self_supervised_loss(
                model,
                graph,
                learning_config,
                generator,
            )
            values["loss"].append(float(output.loss.item()))
            values["row"].append(float(output.row.item()))
            values["flow"].append(float(output.flow.item()))
            values["variance"].append(float(output.variance.item()))
    return {name: float(np.mean(current)) for name, current in values.items()}


def restore_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DirectedRouteHypergraphEncoder(
        int(checkpoint["layer_count"]),
        int(checkpoint["head_count"]),
        ModelConfig(**checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
