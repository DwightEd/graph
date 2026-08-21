"""Label-free training and scoring for grounding-sensitive graph refinement."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .artifacts import (
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_SCORE_SCHEMA,
    save_npz,
    sha256_file,
    write_json,
)
from .data import collect_source_reuse_graph, select_sample_ids
from .grounding_config import GroundingGraphConfig
from .grounding_model import GroundingSensitiveGraphModel


def _stable_seed(sample_id: str, base_seed: int, offset: int = 0) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return base_seed + offset + int.from_bytes(digest[:4], "little")


def _open_dataset(split_root, *, device: str):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device=device)


def _geometry(dataset, sample_ids: list[str], config: GroundingGraphConfig) -> tuple[int, int]:
    sample = dataset[sample_ids[0]]
    graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
    sample.release_attention()
    return graph.num_layers, graph.num_heads


def _source_disjoint_split(
    dataset,
    sample_ids: list[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        source_id = str(dataset[sample_id].source_id)
        groups.setdefault(source_id, []).append(sample_id)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    count = max(1, round(len(names) * validation_fraction))
    count = min(count, max(len(names) - 1, 1))
    validation_sources = set(names[:count])
    fit = [sample_id for name in names if name not in validation_sources for sample_id in groups[name]]
    validation = [sample_id for name in names if name in validation_sources for sample_id in groups[name]]
    if not fit or not validation:
        raise ValueError("source-disjoint split requires at least two source groups")
    return fit, validation


def _run_epoch(
    model: GroundingSensitiveGraphModel,
    dataset,
    sample_ids: list[str],
    *,
    config: GroundingGraphConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    valid_tokens = 0
    total_tokens = 0
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
        graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
        output = model(
            graph,
            seed=_stable_seed(graph.sample_id, config.random_seed, epoch * 100003),
        )
        sample.release_attention()
        total_tokens += graph.num_response_tokens
        valid_tokens += int(output.valid.sum())
        if not bool(output.valid.any()):
            continue
        if training:
            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        losses.append(float(output.loss.detach().cpu()))
        if losses:
            iterator.set_postfix(
                loss=f"{np.mean(losses[-20:]):.4f}",
                coverage=f"{valid_tokens / max(total_tokens, 1):.3f}",
            )
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "samples": float(len(losses)),
        "valid_tokens": float(valid_tokens),
        "tokens": float(total_tokens),
        "coverage": float(valid_tokens / max(total_tokens, 1)),
    }


def train_grounding_model(
    *,
    train_split,
    output_dir,
    device: str = "cpu",
    config: GroundingGraphConfig | None = None,
    task_type: str | None = None,
    limit: int | None = None,
) -> Path:
    config = GroundingGraphConfig() if config is None else config
    config.validate()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    dataset = _open_dataset(train_split, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    fit_ids, validation_ids = _source_disjoint_split(
        dataset,
        sample_ids,
        validation_fraction=config.validation_fraction,
        seed=config.random_seed,
    )
    num_layers, num_heads = _geometry(dataset, fit_ids, config)
    model = GroundingSensitiveGraphModel(
        num_layers=num_layers,
        num_heads=num_heads,
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    patience = 0

    for epoch in range(config.epochs):
        rng = np.random.default_rng(config.random_seed + epoch)
        order = np.asarray(fit_ids, dtype=str)
        rng.shuffle(order)
        train_metrics = _run_epoch(
            model,
            dataset,
            order.tolist(),
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            description=f"grounding train {epoch + 1}/{config.epochs}",
        )
        validation_metrics = _run_epoch(
            model,
            dataset,
            validation_ids,
            config=config,
            epoch=0,
            optimizer=None,
            description=f"grounding val {epoch + 1}/{config.epochs}",
        )
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        validation_loss = float(validation_metrics["loss"])
        if np.isfinite(validation_loss) and validation_loss < best_validation:
            best_validation = validation_loss
            patience = 0
            torch.save(
                {
                    "schema": GROUNDING_CHECKPOINT_SCHEMA,
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                    "labels_read": False,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("no finite validation checkpoint was produced")
    write_json(
        output_dir / "training.json",
        {
            "schema": GROUNDING_CHECKPOINT_SCHEMA,
            "labels_read": False,
            "train_split": str(Path(train_split).resolve()),
            "task_type": task_type,
            "fit_samples": len(fit_ids),
            "validation_samples": len(validation_ids),
            "fit_source_ids": sorted({str(dataset[item].source_id) for item in fit_ids}),
            "validation_source_ids": sorted({str(dataset[item].source_id) for item in validation_ids}),
            "config": asdict(config),
            "best_validation_loss": best_validation,
            "history": history,
            "checkpoint": "model.pt",
        },
    )
    return checkpoint_path


def load_grounding_model(
    checkpoint_path,
    *,
    device: str = "cpu",
) -> tuple[GroundingSensitiveGraphModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GroundingGraphConfig(**checkpoint["config"])
    model = GroundingSensitiveGraphModel(
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def score_grounding_split(
    *,
    split_root,
    checkpoint_path,
    output_dir,
    device: str = "cpu",
    task_type: str | None = None,
    limit: int | None = None,
    save_embeddings: bool = True,
) -> Path:
    model, checkpoint = load_grounding_model(checkpoint_path, device=device)
    model.eval()
    config = model.config
    dataset = _open_dataset(split_root, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    score_names = (
        "reconstruction",
        "raw_reconstruction",
        "prompt_gain",
        "response_gain",
        "closure",
        "fragility",
        "refinement_gain",
        "state_gain",
        "memory_specificity",
        "endpoint_specificity",
        "rewire_changed_fraction",
        "sensitivity_mean",
        "gate_mean",
        "prompt_gate_mean",
        "response_gate_mean",
    )
    rows: dict[str, list] = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "token_index": [],
        "response_length": [],
        "valid_rounds": [],
        **{name: [] for name in score_names},
    }
    embeddings: list[np.ndarray] = []

    iterator = tqdm(
        sample_ids,
        desc="grounding score",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for sample_id in iterator:
        sample = dataset[sample_id]
        graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
        if graph.num_layers != checkpoint["num_layers"] or graph.num_heads != checkpoint["num_heads"]:
            raise ValueError("score split attention geometry differs from checkpoint")
        round_values: dict[str, list[torch.Tensor]] = {name: [] for name in score_names}
        round_valid: list[torch.Tensor] = []
        first_embedding: torch.Tensor | None = None
        for round_index in range(config.score_rounds):
            output = model(
                graph,
                seed=_stable_seed(
                    graph.sample_id,
                    config.random_seed,
                    (round_index + 1) * 1000003,
                ),
            )
            first_embedding = output.embedding if first_embedding is None else first_embedding
            for name in score_names:
                round_values[name].append(getattr(output, name).detach())
            round_valid.append(output.valid.float())
        sample.release_attention()
        response_count = graph.num_response_tokens
        rows["sample_id"].extend([graph.sample_id] * response_count)
        rows["source_id"].extend([graph.source_id] * response_count)
        rows["task_type"].extend([graph.task_type] * response_count)
        rows["token_index"].extend(range(response_count))
        rows["response_length"].extend([response_count] * response_count)
        rows["valid_rounds"].extend(torch.stack(round_valid).sum(0).cpu().tolist())
        for name in score_names:
            value = torch.stack(round_values[name]).mean(0)
            rows[name].extend(value.cpu().tolist())
        if save_embeddings and first_embedding is not None:
            embeddings.append(first_embedding.detach().cpu().half().numpy())

    artifact = {
        "schema": np.asarray(GROUNDING_SCORE_SCHEMA),
        "labels_included": np.asarray(False),
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "valid_rounds": np.asarray(rows["valid_rounds"], dtype=np.int16),
        **{name: np.asarray(rows[name], dtype=np.float32) for name in score_names},
    }
    if save_embeddings:
        artifact["embedding"] = np.concatenate(embeddings, axis=0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.npz"
    save_npz(score_path, **artifact)
    write_json(
        output_dir / "manifest.json",
        {
            "schema": GROUNDING_SCORE_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "task_type": task_type,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": config.to_dict(),
            "samples": len(sample_ids),
            "tokens": len(rows["sample_id"]),
            "score_file": "scores.npz",
            "embeddings_saved": save_embddings,
        },
    )
    return score_path
