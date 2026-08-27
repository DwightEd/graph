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

METHOD = "directed_route_hypergraph_endpoint_recovery"
ARCHITECTURE_VERSION = 2


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
    if variant == "no_message":
        raise ValueError(
            "no_message requires a clean endpoint teacher and a separate "
            "student-view ablation"
        )
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

    if learning_config.kl_warmup_epochs < 0:
        raise ValueError("kl_warmup_epochs must be non-negative")
    for epoch in range(train_config.epochs):
        model.train()
        order = list(split["fit_sample_ids"])
        random.Random(train_config.seed + epoch).shuffle(order)
        values = {
            name: []
            for name in (
                "loss",
                "endpoint",
                "flow",
                "layout",
                "layout_sink",
                "layout_self",
                "layout_external",
                "layout_self_coverage",
                "layout_external_coverage",
                "variance",
                "kl",
                "raw_kl",
                "effective_kl_weight",
                "masked_row_fraction",
                "native_unresolved_mass_mean",
                "masked_mass_mean",
                "active_latent_dimensions",
                "posterior_logvar_mean",
            )
        }
        endpoint_pair_counts = []
        heldout_edge_counts = []
        masked_edge_counts = []
        masked_mass_totals = []
        warmup_fraction = (
            1.0
            if learning_config.kl_warmup_epochs == 0
            else min(
                (epoch + 1) / learning_config.kl_warmup_epochs,
                1.0,
            )
        )
        epoch_kl_weight = learning_config.kl_weight * warmup_fraction
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
                kl_weight=epoch_kl_weight,
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip
            )
            optimizer.step()
            for name in (
                "loss",
                "endpoint",
                "flow",
                "layout",
                "layout_sink",
                "layout_self",
                "layout_external",
                "variance",
                "kl",
                "raw_kl",
            ):
                values[name].append(float(getattr(output, name).detach().item()))
            denominator = max(graph.response_count, 1)
            values["layout_self_coverage"].append(
                output.layout_self_row_count / denominator
            )
            values["layout_external_coverage"].append(
                output.layout_external_row_count / denominator
            )
            values["effective_kl_weight"].append(output.effective_kl_weight)
            values["masked_row_fraction"].append(output.masked_row_fraction)
            values["native_unresolved_mass_mean"].append(
                output.native_unresolved_mass_mean
            )
            values["masked_mass_mean"].append(output.masked_mass_mean)
            values["active_latent_dimensions"].append(
                output.active_latent_dimensions
            )
            values["posterior_logvar_mean"].append(
                output.posterior_logvar_mean
            )
            endpoint_pair_counts.append(output.pair_count)
            heldout_edge_counts.append(output.heldout_edge_count)
            masked_edge_counts.append(output.masked_edge_count)
            masked_mass_totals.append(output.masked_mass_total)

        if not any(endpoint_pair_counts):
            raise RuntimeError(
                "no role/lag-matched endpoint pairs were available in the fit epoch"
            )
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
        plain_metrics = {
            "loss",
            "effective_kl_weight",
            "masked_row_fraction",
            "layout_self_coverage",
            "layout_external_coverage",
            "native_unresolved_mass_mean",
            "masked_mass_mean",
            "active_latent_dimensions",
            "posterior_logvar_mean",
        }
        row = {
            "epoch": epoch + 1,
            "train_endpoint_pair_count": int(sum(endpoint_pair_counts)),
            "train_heldout_edge_count": int(sum(heldout_edge_counts)),
            "train_graphs_with_pairs": int(
                sum(count > 0 for count in endpoint_pair_counts)
            ),
            "train_masked_edge_count": int(sum(masked_edge_counts)),
            "train_masked_mass_total": float(sum(masked_mass_totals)),
        }
        for name, current in values.items():
            suffix = "" if name in plain_metrics else "_loss"
            row[f"train_{name}{suffix}"] = float(np.mean(current))
        validation_plain = plain_metrics | {
            "endpoint_pair_count",
            "heldout_edge_count",
            "graphs_with_pairs",
            "masked_edge_count",
            "masked_mass_total",
        }
        for name, value in validation.items():
            suffix = "" if name in validation_plain else "_loss"
            row[f"validation_{name}{suffix}"] = value
        history.append(row)
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    payload = {
        "method": METHOD,
        "architecture_version": ARCHITECTURE_VERSION,
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
    values = {
        name: []
        for name in (
            "loss",
            "endpoint",
            "flow",
            "layout",
            "layout_sink",
            "layout_self",
            "layout_external",
            "layout_self_coverage",
            "layout_external_coverage",
            "variance",
            "kl",
            "raw_kl",
            "effective_kl_weight",
            "masked_row_fraction",
            "native_unresolved_mass_mean",
            "masked_mass_mean",
            "active_latent_dimensions",
            "posterior_logvar_mean",
        )
    }
    endpoint_pair_count = 0
    heldout_edge_count = 0
    graphs_with_pairs = 0
    masked_edge_count = 0
    masked_mass_total = 0.0
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
            for name in (
                "loss",
                "endpoint",
                "flow",
                "layout",
                "layout_sink",
                "layout_self",
                "layout_external",
                "variance",
                "kl",
                "raw_kl",
            ):
                values[name].append(float(getattr(output, name).item()))
            denominator = max(graph.response_count, 1)
            values["layout_self_coverage"].append(
                output.layout_self_row_count / denominator
            )
            values["layout_external_coverage"].append(
                output.layout_external_row_count / denominator
            )
            values["effective_kl_weight"].append(output.effective_kl_weight)
            values["masked_row_fraction"].append(output.masked_row_fraction)
            values["native_unresolved_mass_mean"].append(
                output.native_unresolved_mass_mean
            )
            values["masked_mass_mean"].append(output.masked_mass_mean)
            values["active_latent_dimensions"].append(
                output.active_latent_dimensions
            )
            values["posterior_logvar_mean"].append(
                output.posterior_logvar_mean
            )
            endpoint_pair_count += output.pair_count
            heldout_edge_count += output.heldout_edge_count
            graphs_with_pairs += int(output.pair_count > 0)
            masked_edge_count += output.masked_edge_count
            masked_mass_total += output.masked_mass_total
    result = {name: float(np.mean(current)) for name, current in values.items()}
    return {
        **result,
        "endpoint_pair_count": int(endpoint_pair_count),
        "heldout_edge_count": int(heldout_edge_count),
        "graphs_with_pairs": int(graphs_with_pairs),
        "masked_edge_count": int(masked_edge_count),
        "masked_mass_total": float(masked_mass_total),
    }


def restore_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if str(checkpoint.get("method", "")) != METHOD:
        raise ValueError("checkpoint belongs to a different method")
    if int(checkpoint.get("architecture_version", -1)) != ARCHITECTURE_VERSION:
        raise ValueError("checkpoint has an incompatible architecture version")
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
