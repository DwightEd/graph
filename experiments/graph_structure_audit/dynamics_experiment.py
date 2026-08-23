"""Label-free training and frozen cross-layer routing dynamics scores."""

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.source_reuse_contrast.data import select_sample_ids

from .artifacts import save_npz, write_json
from .controls import rewire_endpoints
from .dynamics_config import DynamicsConfig
from .dynamics_model import CrossOriginRoutingDynamics
from .graph_data import MultiplexGraph, load_multiplex_graph


CHECKPOINT_SCHEMA = "cross-origin-routing-dynamics-checkpoint-v1"
SCORE_SCHEMA = "cross-origin-routing-dynamics-scores-v1"


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


def _shuffle_layers(graph: MultiplexGraph, generator: torch.Generator) -> MultiplexGraph:
    order = torch.randperm(graph.num_layers, generator=generator, device=graph.device)
    return replace(
        graph,
        edge_attr=graph.edge_attr[:, order],
        edge_observed=graph.edge_observed[:, order],
        diagonal=graph.diagonal[:, order],
        diagonal_observed=graph.diagonal_observed[:, order],
    )


def _shuffle_heads(graph: MultiplexGraph, generator: torch.Generator) -> MultiplexGraph:
    order = torch.randperm(graph.num_heads, generator=generator, device=graph.device)
    return replace(
        graph,
        edge_attr=graph.edge_attr[:, :, order],
        edge_observed=graph.edge_observed[:, :, order],
        diagonal=graph.diagonal[:, :, order],
        diagonal_observed=graph.diagonal_observed[:, :, order],
    )


def _run_epoch(
    model,
    dataset,
    sample_ids,
    *,
    config: DynamicsConfig,
    optimizer,
    description: str,
):
    training = optimizer is not None
    model.train(training)
    losses = []
    iterator = tqdm(
        sample_ids,
        desc=description,
        unit="sample",
        leave=False,
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for sample_id in iterator:
        graph = load_multiplex_graph(dataset[sample_id], block_rows=config.block_rows)
        with torch.set_grad_enabled(training):
            output = model(graph, input_dropout=training)
            if training:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(output.loss.detach().cpu()))
        iterator.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        del output, graph
    return {"loss": float(np.mean(losses)), "samples": len(losses)}


def train_dynamics_model(
    *,
    train_split,
    output_dir,
    device="cpu",
    config: DynamicsConfig | None = None,
    task_type=None,
    limit=None,
):
    config = DynamicsConfig() if config is None else config
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    dataset = _open_dataset(train_split, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    fit_ids, validation_ids = _source_split(
        dataset, sample_ids, config.validation_fraction, config.random_seed
    )
    first = load_multiplex_graph(dataset[fit_ids[0]], block_rows=config.block_rows)
    model = CrossOriginRoutingDynamics(
        num_layers=first.num_layers,
        num_heads=first.num_heads,
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
            description=f"dynamics train {epoch + 1}/{config.epochs}",
        )
        validation = _run_epoch(
            model,
            dataset,
            validation_ids,
            config=config,
            optimizer=None,
            description=f"dynamics val {epoch + 1}/{config.epochs}",
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
                    "num_layers": first.num_layers,
                    "num_heads": first.num_heads,
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


def load_dynamics_model(checkpoint_path, *, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DynamicsConfig(**checkpoint["config"])
    model = CrossOriginRoutingDynamics(
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _score_round(model, graph: MultiplexGraph, seed: int):
    generator = torch.Generator(device=graph.device)
    generator.manual_seed(seed)
    full = model(graph, message_mode="full", input_dropout=False)
    none = model(graph, message_mode="none", input_dropout=False)
    prompt_only = model(graph, message_mode="prompt", input_dropout=False)
    response_only = model(graph, message_mode="response", input_dropout=False)
    layer = model(_shuffle_layers(graph, generator), input_dropout=False)
    head = model(_shuffle_heads(graph, generator), input_dropout=False)
    endpoint = model(rewire_endpoints(graph, generator=generator), input_dropout=False)
    prompt_gain = response_only.token_loss - full.token_loss
    response_gain = prompt_only.token_loss - full.token_loss
    return {
        "transition_recovery": full.token_loss,
        "edge_transition": full.edge_loss,
        "prompt_edge_transition": full.prompt_edge_loss,
        "response_edge_transition": full.response_edge_loss,
        "diagonal_transition": full.diagonal_loss,
        "support_transition": full.support_loss,
        "edge_state_gap": full.edge_loss - full.diagonal_loss,
        "origin_gap": full.response_edge_loss - full.prompt_edge_loss,
        "message_gain": none.token_loss - full.token_loss,
        "prompt_gain": prompt_gain,
        "response_gain": response_gain,
        "closure": response_gain - prompt_gain,
        "layer_order_gain": layer.token_loss - full.token_loss,
        "head_identity_gain": head.token_loss - full.token_loss,
        "endpoint_gain": endpoint.token_loss - full.token_loss,
        "edge_error_map": full.edge_error_map,
        "prompt_edge_error_map": full.prompt_edge_error_map,
        "response_edge_error_map": full.response_edge_error_map,
        "diagonal_error_map": full.diagonal_error_map,
        "support_error_map": full.support_error_map,
        "self_gate": full.self_gate,
        "prompt_gate": full.prompt_gate,
        "response_gate": full.response_gate,
        "edge_count_map": full.edge_count_map,
        "prompt_edge_count_map": full.prompt_edge_count_map,
        "response_edge_count_map": full.response_edge_count_map,
        "embedding": full.embedding,
        "valid": full.valid,
    }


def _robust_group_z(value, task, token_index):
    result = np.zeros_like(value, dtype=np.float32)
    bucket = np.floor(np.log2(token_index + 1)).astype(np.int16)
    for task_name in np.unique(task):
        for position in np.unique(bucket[task == task_name]):
            selected = (task == task_name) & (bucket == position)
            current = value[selected].astype(np.float64)
            median = np.median(current)
            mad = np.median(np.abs(current - median))
            scale = max(1.4826 * mad, 1e-6)
            result[selected] = ((current - median) / scale).astype(np.float32)
    return result


def score_dynamics_split(
    *,
    split_root,
    checkpoint_path,
    output_dir,
    device="cpu",
    task_type=None,
    limit=None,
):
    model, checkpoint = load_dynamics_model(checkpoint_path, device=device)
    model.eval()
    config = model.config
    dataset = _open_dataset(split_root, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    scalar_names = (
        "transition_recovery",
        "edge_transition",
        "prompt_edge_transition",
        "response_edge_transition",
        "diagonal_transition",
        "support_transition",
        "edge_state_gap",
        "origin_gap",
        "message_gain",
        "prompt_gain",
        "response_gain",
        "closure",
        "layer_order_gain",
        "head_identity_gain",
        "endpoint_gain",
    )
    map_names = (
        "edge_error_map",
        "prompt_edge_error_map",
        "response_edge_error_map",
        "diagonal_error_map",
        "support_error_map",
        "self_gate",
        "prompt_gate",
        "response_gate",
        "edge_count_map",
        "prompt_edge_count_map",
        "response_edge_count_map",
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
        **{name: [] for name in scalar_names},
    }
    maps = {name: [] for name in map_names}
    embeddings = []

    iterator = tqdm(
        sample_ids,
        desc="routing dynamics score",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    with torch.no_grad():
        for sample_id in iterator:
            graph = load_multiplex_graph(dataset[sample_id], block_rows=config.block_rows)
            rounds = [
                _score_round(
                    model,
                    graph,
                    _stable_seed(graph.sample_id, config.random_seed, (index + 1) * 1000003),
                )
                for index in range(config.score_rounds)
            ]
            count = graph.num_response_tokens
            rows["sample_id"].extend([graph.sample_id] * count)
            rows["source_id"].extend([graph.source_id] * count)
            rows["task_type"].extend([graph.task_type] * count)
            rows["token_index"].extend(range(count))
            rows["response_length"].extend([count] * count)
            pair_count = (graph.target_ptr[1:] - graph.target_ptr[:-1]).cpu().tolist()
            rows["incoming_pairs"].extend(pair_count)
            for token in range(count):
                current = graph.incoming(token)
                rows["active_channels"].append(int(graph.edge_observed[current].sum()))
                rows["retained_mass"].append(
                    float(graph.edge_attr[current].sum() / (graph.num_layers * graph.num_heads))
                )
            rows["valid_rounds"].extend(
                torch.stack([item["valid"].float() for item in rounds]).sum(0).cpu().tolist()
            )
            for name in scalar_names:
                value = torch.stack([item[name] for item in rounds]).mean(0)
                rows[name].extend(value.cpu().tolist())
            for name in map_names:
                value = torch.stack([item[name] for item in rounds]).mean(0)
                maps[name].append(value.cpu().half().numpy())
            embeddings.append(rounds[0]["embedding"].cpu().half().numpy())

    task = np.asarray(rows["task_type"], dtype=str)
    token_index = np.asarray(rows["token_index"], dtype=np.int32)
    edge_z = _robust_group_z(np.asarray(rows["edge_transition"]), task, token_index)
    diagonal_z = _robust_group_z(
        np.asarray(rows["diagonal_transition"]), task, token_index
    )
    prompt_z = _robust_group_z(
        np.asarray(rows["prompt_edge_transition"]), task, token_index
    )
    response_z = _robust_group_z(
        np.asarray(rows["response_edge_transition"]), task, token_index
    )
    rows["edge_state_decoupling"] = (edge_z - diagonal_z).tolist()
    rows["origin_fracture"] = (response_z - prompt_z).tolist()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": np.asarray(SCORE_SCHEMA),
        "labels_included": np.asarray(False),
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "task_type": task,
        "token_index": token_index,
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "incoming_pairs": np.asarray(rows["incoming_pairs"], dtype=np.int16),
        "active_channels": np.asarray(rows["active_channels"], dtype=np.int32),
        "retained_mass": np.asarray(rows["retained_mass"], dtype=np.float32),
        "valid_rounds": np.asarray(rows["valid_rounds"], dtype=np.int16),
        **{
            name: np.asarray(rows[name], dtype=np.float32)
            for name in (*scalar_names, "edge_state_decoupling", "origin_fracture")
        },
        **{name: np.concatenate(values, axis=0) for name, values in maps.items()},
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
            "score_file": "scores.npz",
        },
    )
    return score_path
