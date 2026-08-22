"""Label-free training and frozen structural recovery scores."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.source_reuse_contrast.data import collect_source_reuse_graph, select_sample_ids

from .artifacts import CHECKPOINT_SCHEMA, GRAPH_SCHEMA, SCORE_SCHEMA, save_npz, write_json
from .config import RecoveryConfig
from .controls import collapse_channels, rewire_endpoints, shuffle_heads, shuffle_layers
from .graph_data import MultiplexGraph, build_multiplex_graph
from .masking import mask_graph
from .model import LayeredGraphRecovery


def _open_dataset(split_root, *, device: str):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device=device)


def _stable_seed(sample_id: str, base: int, offset: int = 0) -> int:
    digest = hashlib.sha256(str(sample_id).encode()).digest()
    return base + offset + int.from_bytes(digest[:4], "little")


def _source_split(dataset, sample_ids, fraction: float, seed: int):
    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        groups.setdefault(str(dataset[sample_id].source_id), []).append(sample_id)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    count = max(1, min(len(names) - 1, round(len(names) * fraction)))
    validation = set(names[:count])
    fit = [item for name in names if name not in validation for item in groups[name]]
    heldout = [item for name in names if name in validation for item in groups[name]]
    return fit, heldout


def _geometry(dataset, sample_id: str, config: RecoveryConfig):
    sample = dataset[sample_id]
    raw = collect_source_reuse_graph(sample, block_rows=config.block_rows)
    sample.release_attention()
    return raw.num_layers, raw.num_heads


def _run_epoch(
    model,
    dataset,
    sample_ids,
    *,
    config: RecoveryConfig,
    optimizer,
    epoch: int,
    description: str,
):
    training = optimizer is not None
    model.train(training)
    values: list[float] = []
    tokens = 0
    valid_tokens = 0
    iterator = tqdm(
        sample_ids,
        desc=description,
        unit="sample",
        leave=False,
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for sample_id in iterator:
        sample = dataset[sample_id]
        raw = collect_source_reuse_graph(sample, block_rows=config.block_rows)
        graph = build_multiplex_graph(raw)
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(_stable_seed(graph.sample_id, config.random_seed, epoch * 100003))
        masked = collapse_channels(mask_graph(graph, config, generator=generator), config.representation)
        output = model(graph, masked)
        sample.release_attention()

        if training:
            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        values.append(float(output.loss.detach().cpu()))
        tokens += graph.num_response_tokens
        valid_tokens += int(output.valid.sum())
        iterator.set_postfix(
            loss=f"{np.mean(values[-20:]):.4f}",
            coverage=f"{valid_tokens / max(tokens, 1):.3f}",
        )
    return {
        "loss": float(np.mean(values)),
        "samples": len(values),
        "tokens": tokens,
        "valid_tokens": valid_tokens,
        "coverage": valid_tokens / max(tokens, 1),
    }


def train_recovery_model(
    *,
    train_split,
    output_dir,
    device="cpu",
    config: RecoveryConfig | None = None,
    task_type=None,
    limit=None,
):
    config = RecoveryConfig() if config is None else config
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    dataset = _open_dataset(train_split, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    fit_ids, validation_ids = _source_split(
        dataset, sample_ids, config.validation_fraction, config.random_seed
    )
    num_layers, num_heads = _geometry(dataset, fit_ids[0], config)
    model = LayeredGraphRecovery(
        num_layers=num_layers,
        num_heads=num_heads,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    history = []
    best = float("inf")
    patience = 0
    for epoch in range(config.epochs):
        order = np.asarray(fit_ids, dtype=str)
        np.random.default_rng(config.random_seed + epoch).shuffle(order)
        train = _run_epoch(
            model,
            dataset,
            order.tolist(),
            config=config,
            optimizer=optimizer,
            epoch=epoch,
            description=f"recovery train {epoch + 1}/{config.epochs}",
        )
        validation = _run_epoch(
            model,
            dataset,
            validation_ids,
            config=config,
            optimizer=None,
            epoch=0,
            description=f"recovery val {epoch + 1}/{config.epochs}",
        )
        history.append({"epoch": epoch + 1, "train": train, "validation": validation})
        if validation["loss"] < best:
            best = validation["loss"]
            patience = 0
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "validation_loss": best,
                    "labels_read": False,
                },
                checkpoint,
            )
        else:
            patience += 1
            if patience >= config.patience:
                break

    write_json(
        output_dir / "training.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "labels_read": False,
            "train_split": str(Path(train_split).resolve()),
            "task_type": task_type,
            "fit_samples": len(fit_ids),
            "validation_samples": len(validation_ids),
            "config": asdict(config),
            "best_validation_loss": best,
            "history": history,
            "checkpoint": "model.pt",
        },
    )
    return checkpoint


def load_recovery_model(checkpoint_path, *, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = RecoveryConfig(**checkpoint["config"])
    model = LayeredGraphRecovery(
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _score_round(model, graph: MultiplexGraph, config: RecoveryConfig, seed: int):
    generator = torch.Generator(device=graph.device)
    generator.manual_seed(seed)
    base = mask_graph(graph, config, generator=generator)
    full_input = collapse_channels(base, config.representation)
    full = model(graph, full_input)
    no_message = model(graph, full_input, message_passing=False)
    layer = model(graph, shuffle_layers(full_input, generator=generator))
    head = model(graph, shuffle_heads(full_input, generator=generator))
    endpoint_graph = rewire_endpoints(graph, generator=generator)
    endpoint = model(endpoint_graph, full_input)
    layer_mean = model(graph, collapse_channels(base, "layer_mean"))
    global_mean = model(graph, collapse_channels(base, "global_mean"))
    return {
        "recovery": full.token_loss,
        "edge_recovery": full.edge_loss,
        "diagonal_recovery": full.diagonal_loss,
        "message_gain": no_message.token_loss - full.token_loss,
        "layer_order_gain": layer.token_loss - full.token_loss,
        "head_identity_gain": head.token_loss - full.token_loss,
        "endpoint_gain": endpoint.token_loss - full.token_loss,
        "layer_head_gain": layer_mean.token_loss - full.token_loss,
        "full_channel_gain": global_mean.token_loss - full.token_loss,
        "valid": full.valid,
        "embedding": full.embedding,
    }


def _graph_file_name(sample_id: str) -> str:
    return f"sample_{hashlib.sha256(sample_id.encode()).hexdigest()[:20]}.npz"


def score_recovery_split(
    *,
    split_root,
    checkpoint_path,
    output_dir,
    device="cpu",
    task_type=None,
    limit=None,
    save_graphs=True,
):
    model, checkpoint = load_recovery_model(checkpoint_path, device=device)
    model.eval()
    config = model.config
    dataset = _open_dataset(split_root, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    score_names = (
        "recovery",
        "edge_recovery",
        "diagonal_recovery",
        "message_gain",
        "layer_order_gain",
        "head_identity_gain",
        "endpoint_gain",
        "layer_head_gain",
        "full_channel_gain",
    )
    rows = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "token_index": [],
        "response_length": [],
        "incoming_pairs": [],
        "active_channels": [],
        "retained_mass": [],
        "valid_rounds": [],
        **{name: [] for name in score_names},
    }
    embeddings: list[np.ndarray] = []
    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    if save_graphs:
        graph_dir.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(
        sample_ids,
        desc="recovery score",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    with torch.no_grad():
        for sample_id in iterator:
            sample = dataset[sample_id]
            raw = collect_source_reuse_graph(sample, block_rows=config.block_rows)
            graph = build_multiplex_graph(raw)
            rounds = [
                _score_round(
                    model,
                    graph,
                    config,
                    _stable_seed(graph.sample_id, config.random_seed, (index + 1) * 1000003),
                )
                for index in range(config.score_rounds)
            ]
            sample.release_attention()

            count = graph.num_response_tokens
            rows["sample_id"].extend([graph.sample_id] * count)
            rows["source_id"].extend([graph.source_id] * count)
            rows["task_type"].extend([graph.task_type] * count)
            rows["token_index"].extend(range(count))
            rows["response_length"].extend([count] * count)
            pair_count = (graph.target_ptr[1:] - graph.target_ptr[:-1]).cpu().tolist()
            channel_count = []
            retained_mass = []
            for token in range(count):
                current = graph.incoming(token)
                channel_count.append(int(graph.edge_observed[current].sum()))
                retained_mass.append(float(graph.edge_attr[current].sum() / (graph.num_layers * graph.num_heads)))
            rows["incoming_pairs"].extend(pair_count)
            rows["active_channels"].extend(channel_count)
            rows["retained_mass"].extend(retained_mass)
            rows["valid_rounds"].extend(
                torch.stack([item["valid"].float() for item in rounds]).sum(0).cpu().tolist()
            )
            for name in score_names:
                value = torch.stack([item[name] for item in rounds]).mean(0)
                rows[name].extend(value.cpu().tolist())
            embeddings.append(rounds[0]["embedding"].cpu().half().numpy())

            if save_graphs:
                save_npz(
                    graph_dir / _graph_file_name(graph.sample_id),
                    schema=np.asarray(GRAPH_SCHEMA),
                    labels_included=np.asarray(False),
                    sample_id=np.asarray(graph.sample_id),
                    source_id=np.asarray(graph.source_id),
                    task_type=np.asarray(graph.task_type),
                    response_idx=np.asarray(graph.response_idx, dtype=np.int32),
                    num_layers=np.asarray(graph.num_layers, dtype=np.int16),
                    num_heads=np.asarray(graph.num_heads, dtype=np.int16),
                    **graph.numpy_dict(),
                )

    artifact = {
        "schema": np.asarray(SCORE_SCHEMA),
        "labels_included": np.asarray(False),
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "incoming_pairs": np.asarray(rows["incoming_pairs"], dtype=np.int16),
        "active_channels": np.asarray(rows["active_channels"], dtype=np.int32),
        "retained_mass": np.asarray(rows["retained_mass"], dtype=np.float32),
        "valid_rounds": np.asarray(rows["valid_rounds"], dtype=np.int16),
        **{name: np.asarray(rows[name], dtype=np.float32) for name in score_names},
        "embedding": np.concatenate(embeddings, axis=0),
    }
    score_path = output_dir / "scores.npz"
    save_npz(score_path, **artifact)
    write_json(
        output_dir / "manifest.json",
        {
            "schema": SCORE_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "config": config.to_dict(),
            "samples": len(sample_ids),
            "tokens": len(rows["sample_id"]),
            "graphs_saved": save_graphs,
            "score_file": "scores.npz",
        },
    )
    return score_path
