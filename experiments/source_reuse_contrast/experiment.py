"""Label-free training, validation, and scoring for source predictability."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm, trange

from .artifacts import CHECKPOINT_SCHEMA, SCORE_SCHEMA, save_npz, sha256_file, write_json
from .config import SourceReuseConfig
from .data import collect_source_reuse_graph, select_sample_ids
from .model import PredictabilityScores, SourceReusePredictor


def _stable_seed(sample_id: str, base_seed: int, offset: int = 0) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return base_seed + offset + int.from_bytes(digest[:4], "little")


def _open_dataset(split_root, *, device: str):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device=device)


def _geometry(
    dataset,
    config: SourceReuseConfig,
    sample_id: str,
) -> tuple[int, int]:
    sample = dataset[sample_id]
    graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
    sample.release_attention()
    return graph.num_layers, graph.num_heads


def source_disjoint_split(
    dataset,
    sample_ids: list[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split by source ID so validation cannot share source documents with fit."""

    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[str(sample_id)]
        groups.setdefault(str(sample.source_id), []).append(str(sample_id))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    validation_count = max(1, int(round(len(names) * validation_fraction)))
    validation_names = set(names[:validation_count])
    fit = [
        sample_id
        for name in names
        if name not in validation_names
        for sample_id in groups[name]
    ]
    validation = [
        sample_id
        for name in names
        if name in validation_names
        for sample_id in groups[name]
    ]
    if not fit or not validation:
        raise ValueError("source-disjoint split requires at least two source groups")
    return fit, validation


def _summary(values: list[np.ndarray]) -> dict[str, float]:
    empty = {
        "nll": float("nan"),
        "shuffled_nll": float("nan"),
        "shuffle_gap": float("nan"),
        "accuracy": float("nan"),
        "margin": float("nan"),
        "coverage": 0.0,
        "candidate_count": float("nan"),
        "positive_logit_mean": float("nan"),
        "positive_logit_std": float("nan"),
        "hardest_negative_logit_mean": float("nan"),
        "hardest_negative_logit_std": float("nan"),
    }
    if not values:
        return empty
    array = np.concatenate(values, axis=0)
    valid = array[:, 0] > 0
    if not valid.any():
        return empty
    selected = array[valid]
    return {
        "nll": float(selected[:, 1].mean()),
        "shuffled_nll": float(selected[:, 2].mean()),
        "shuffle_gap": float((selected[:, 2] - selected[:, 1]).mean()),
        "accuracy": float(selected[:, 3].mean()),
        "margin": float(selected[:, 4].mean()),
        "coverage": float(valid.mean()),
        "candidate_count": float(selected[:, 5].mean()),
        "positive_logit_mean": float(selected[:, 6].mean()),
        "positive_logit_std": float(selected[:, 6].std()),
        "hardest_negative_logit_mean": float(selected[:, 7].mean()),
        "hardest_negative_logit_std": float(selected[:, 7].std()),
    }


def _output_rows(output: PredictabilityScores) -> np.ndarray:
    return np.stack(
        (
            output.valid_pairs.detach().cpu().numpy(),
            output.endpoint_nll.detach().cpu().numpy(),
            output.shuffled_nll.detach().cpu().numpy(),
            output.accuracy.detach().cpu().numpy(),
            output.margin.detach().cpu().numpy(),
            output.candidate_count.detach().cpu().numpy(),
            output.positive_logit.detach().cpu().numpy(),
            output.hardest_negative_logit.detach().cpu().numpy(),
        ),
        axis=1,
    )


def validate_model(
    model: SourceReusePredictor,
    dataset,
    sample_ids: list[str],
    *,
    config: SourceReuseConfig,
    description: str,
) -> dict[str, float]:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        iterator = tqdm(
            sample_ids,
            desc=description,
            unit="sample",
            leave=False,
            dynamic_ncols=True,
            disable=not config.show_progress,
        )
        for sample_id in iterator:
            sample = dataset[str(sample_id)]
            graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
            output = model(
                graph,
                seed=_stable_seed(graph.sample_id, config.random_seed, 9000001),
            )
            values.append(_output_rows(output))
            sample.release_attention()
    return _summary(values)


def train_model(
    *,
    train_split,
    output_dir,
    device: str = "cpu",
    config: SourceReuseConfig | None = None,
    limit: int | None = None,
    task_type: str | None = None,
) -> Path:
    """Train masked exact-source prediction without hallucination labels."""

    config = SourceReuseConfig() if config is None else config
    config.validate()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    dataset = _open_dataset(train_split, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    if len(sample_ids) < 2:
        raise ValueError("training requires at least two samples")
    fit_ids, validation_ids = source_disjoint_split(
        dataset,
        sample_ids,
        validation_fraction=config.validation_fraction,
        seed=config.random_seed,
    )

    num_layers, num_heads = _geometry(dataset, config, sample_ids[0])
    model = SourceReusePredictor(
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
    history = []
    best_validation_nll = float("inf")
    stale_epochs = 0
    rng = np.random.default_rng(config.random_seed)

    epoch_bar = trange(
        config.epochs,
        desc=f"CaSH-{config.memory_mode}",
        unit="epoch",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for epoch in epoch_bar:
        model.train()
        order = np.asarray(fit_ids, dtype=str).copy()
        rng.shuffle(order)
        losses: list[float] = []
        valid_tokens = 0
        total_tokens = 0
        sample_bar = tqdm(
            order,
            desc=f"epoch {epoch + 1}/{config.epochs}",
            unit="sample",
            leave=False,
            dynamic_ncols=True,
            disable=not config.show_progress,
        )
        for step, sample_id in enumerate(sample_bar, start=1):
            sample = dataset[str(sample_id)]
            graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
            output = model(
                graph,
                seed=_stable_seed(
                    graph.sample_id, config.random_seed, epoch * 100003
                ),
            )
            sample.release_attention()
            total_tokens += graph.num_response_tokens
            valid_tokens += int(output.valid.sum())
            if not bool(output.valid.any()):
                continue

            loss = model.loss(output)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 10 == 0:
                sample_bar.set_postfix(
                    loss=f"{np.mean(losses[-20:]):.4f}" if losses else "nan",
                    coverage=f"{valid_tokens / max(total_tokens, 1):.3f}",
                )

        train_loss = float(np.mean(losses)) if losses else float("nan")
        validation = validate_model(
            model,
            dataset,
            validation_ids,
            config=config,
            description="validation",
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_samples": len(losses),
            "train_valid_tokens": valid_tokens,
            "train_total_tokens": total_tokens,
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(record)
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{validation['nll']:.4f}",
            coverage=f"{validation['coverage']:.3f}",
        )

        validation_nll = validation["nll"]
        if np.isfinite(validation_nll) and validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            stale_epochs = 0
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "epoch": epoch + 1,
                    "validation": validation,
                    "fit_source_ids": sorted(
                        {str(dataset[sample_id].source_id) for sample_id in fit_ids}
                    ),
                    "validation_source_ids": sorted(
                        {
                            str(dataset[sample_id].source_id)
                            for sample_id in validation_ids
                        }
                    ),
                    "labels_read": False,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("validation did not admit any matched endpoint prediction")

    write_json(
        output_dir / "training.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "labels_read": False,
            "train_split": str(Path(train_split).resolve()),
            "task_type_filter": task_type,
            "config": asdict(config),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "fit_samples": len(fit_ids),
            "validation_samples": len(validation_ids),
            "best_validation_nll": best_validation_nll,
            "history": history,
            "checkpoint": "model.pt",
        },
    )
    return checkpoint_path


def load_model(
    checkpoint_path,
    *,
    device: str = "cpu",
) -> tuple[SourceReusePredictor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = SourceReuseConfig(**checkpoint["config"])
    model = SourceReusePredictor(
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def score_split(
    *,
    split_root,
    checkpoint_path,
    output_dir,
    device: str = "cpu",
    limit: int | None = None,
    task_type: str | None = None,
    save_embeddings: bool = True,
) -> Path:
    """Freeze raw predictive scores before hallucination labels are opened."""

    model, checkpoint = load_model(checkpoint_path, device=device)
    model.eval()
    config = model.config
    dataset = _open_dataset(split_root, device=device)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)

    scalar_names = (
        "endpoint_nll",
        "shuffled_nll",
        "margin",
        "accuracy",
        "valid_pairs",
        "candidate_count",
        "positive_logit",
        "hardest_negative_logit",
        "mean_match_distance",
    )
    rows: dict[str, list] = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "token_index": [],
        "response_length": [],
        "valid_rounds": [],
        **{name: [] for name in scalar_names},
    }
    query_embedding = []
    source_embedding = []

    iterator = tqdm(
        sample_ids,
        desc=f"score-{config.memory_mode}",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    with torch.no_grad():
        for sample_id in iterator:
            sample = dataset[str(sample_id)]
            graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
            round_outputs = []
            for round_index in range(config.score_rounds):
                round_outputs.append(
                    model(
                        graph,
                        seed=_stable_seed(
                            graph.sample_id,
                            config.random_seed,
                            (round_index + 1) * 1000003,
                        ),
                    )
                )

            response_count = graph.num_response_tokens
            rows["sample_id"].extend([graph.sample_id] * response_count)
            rows["source_id"].extend([graph.source_id] * response_count)
            rows["task_type"].extend([graph.task_type] * response_count)
            rows["token_index"].extend(range(response_count))
            rows["response_length"].extend([response_count] * response_count)
            valid_rounds = torch.stack(
                [output.valid.float() for output in round_outputs]
            ).sum(dim=0)
            rows["valid_rounds"].extend(valid_rounds.cpu().tolist())
            for name in scalar_names:
                value = torch.stack(
                    [getattr(output, name) for output in round_outputs]
                ).mean(dim=0)
                rows[name].extend(value.cpu().tolist())
            if save_embeddings:
                query_embedding.append(
                    round_outputs[0].query_embedding.cpu().half().numpy()
                )
                source_embedding.append(
                    round_outputs[0].source_embedding.cpu().half().numpy()
                )
            sample.release_attention()

    artifact = {
        "schema": np.asarray(SCORE_SCHEMA),
        "labels_included": np.asarray(False),
        "memory_mode": np.asarray(config.memory_mode),
        "sample_id": np.asarray(rows["sample_id"], dtype=str),
        "source_id": np.asarray(rows["source_id"], dtype=str),
        "task_type": np.asarray(rows["task_type"], dtype=str),
        "token_index": np.asarray(rows["token_index"], dtype=np.int32),
        "response_length": np.asarray(rows["response_length"], dtype=np.int32),
        "valid_rounds": np.asarray(rows["valid_rounds"], dtype=np.int16),
        **{name: np.asarray(rows[name], dtype=np.float32) for name in scalar_names},
    }
    if save_embeddings:
        artifact["query_embedding"] = np.concatenate(query_embedding, axis=0)
        artifact["source_embedding"] = np.concatenate(source_embedding, axis=0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.npz"
    save_npz(score_path, **artifact)
    valid = artifact["valid_rounds"] > 0
    diagnostics = {
        "valid_token_fraction": float(valid.mean()),
        "endpoint_nll_mean": float(artifact["endpoint_nll"][valid].mean())
        if valid.any()
        else float("nan"),
        "endpoint_nll_std": float(artifact["endpoint_nll"][valid].std())
        if valid.any()
        else float("nan"),
        "shuffled_nll_mean": float(artifact["shuffled_nll"][valid].mean())
        if valid.any()
        else float("nan"),
        "memory_shuffle_gap": float(
            (artifact["shuffled_nll"][valid] - artifact["endpoint_nll"][valid]).mean()
        )
        if valid.any()
        else float("nan"),
        "margin_mean": float(artifact["margin"][valid].mean())
        if valid.any()
        else float("nan"),
        "margin_std": float(artifact["margin"][valid].std())
        if valid.any()
        else float("nan"),
        "positive_logit_mean": float(artifact["positive_logit"][valid].mean())
        if valid.any()
        else float("nan"),
        "positive_logit_std": float(artifact["positive_logit"][valid].std())
        if valid.any()
        else float("nan"),
        "hardest_negative_logit_mean": float(
            artifact["hardest_negative_logit"][valid].mean()
        )
        if valid.any()
        else float("nan"),
        "hardest_negative_logit_std": float(
            artifact["hardest_negative_logit"][valid].std()
        )
        if valid.any()
        else float("nan"),
        "unique_endpoint_nll_1e6": int(
            np.unique(np.round(artifact["endpoint_nll"][valid], 6)).size
        )
        if valid.any()
        else 0,
        "candidate_count_mean": float(artifact["candidate_count"][valid].mean())
        if valid.any()
        else float("nan"),
    }
    write_json(
        output_dir / "manifest.json",
        {
            "schema": SCORE_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "task_type_filter": task_type,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": config.to_dict(),
            "samples": len(sample_ids),
            "tokens": len(rows["endpoint_nll"]),
            "diagnostics": diagnostics,
            "score_file": "scores.npz",
            "embeddings_saved": save_embeddings,
        },
    )
    return score_path
