"""Label-blind training, validation, and train-only anomaly calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .graph import GraphBuildConfig, build_attention_graph
from .model import MaskedAttentionAutoencoder, random_target_view, reconstruction_losses
from .score import RobustResidualCalibrator, score_graph_raw


@dataclass(frozen=True)
class TrainingConfig:
    embedding_dim: int = 64
    message_steps: int = 2
    dropout: float = 0.1
    epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    target_mask_rate: float = 0.20
    channel_drop_rate: float = 0.10
    support_weight: float = 1.0
    attention_weight: float = 1.0
    distribution_weight: float = 1.0
    node_weight: float = 0.25
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    target_block_size: int = 1
    seed: int = 0

    def validate(self):
        if min(self.embedding_dim, self.epochs, self.patience, self.target_block_size) < 1:
            raise ValueError("model/training dimensions must be positive")
        if self.message_steps < 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid message_steps/dropout")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if not 0.0 < self.target_mask_rate <= 1.0:
            raise ValueError("target_mask_rate must be in (0,1]")
        if not 0.0 <= self.channel_drop_rate < 1.0:
            raise ValueError("channel_drop_rate must be in [0,1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0,1)")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if self.validation_fraction + self.calibration_fraction >= 1.0:
            raise ValueError("validation + calibration fractions must be < 1")
        if min(
            self.support_weight,
            self.attention_weight,
            self.distribution_weight,
            self.node_weight,
        ) < 0:
            raise ValueError("loss weights cannot be negative")


def _source_split(dataset, config: TrainingConfig):
    sources = np.asarray(sorted(dataset.source_ids), dtype=object)
    if len(sources) < 3:
        raise ValueError("at least three source groups are required")
    rng = np.random.default_rng(config.seed)
    sources = sources[rng.permutation(len(sources))]
    val_count = max(1, round(len(sources) * config.validation_fraction))
    cal_count = max(1, round(len(sources) * config.calibration_fraction))
    if val_count + cal_count >= len(sources):
        val_count = cal_count = 1
    validation = set(sources[:val_count].tolist())
    calibration = set(sources[val_count : val_count + cal_count].tolist())
    train = set(sources[val_count + cal_count :].tolist())
    if not train:
        raise ValueError("source split left no training groups")
    groups = {"train": train, "validation": validation, "calibration": calibration}
    return {
        name: [
            dataset[sample_id]
            for sample_id in dataset.sample_ids
            if dataset[sample_id].source_id in source_ids
        ]
        for name, source_ids in groups.items()
    }, {name: sorted(values) for name, values in groups.items()}


def _one_epoch(model, samples, graph_config, config, *, epoch, optimizer=None):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    order = np.arange(len(samples))
    if training:
        order = np.random.default_rng(config.seed + epoch).permutation(order)
    totals = {name: 0.0 for name in ("support", "weight", "distribution", "node", "total")}
    for position, index in enumerate(order.tolist()):
        sample = samples[index]
        graph = build_attention_graph(sample.attention(), graph_config).to(device)
        generator = torch.Generator(device=device).manual_seed(
            config.seed + epoch * 1_000_003 + position * 1009
        )
        view = random_target_view(
            graph,
            target_mask_rate=config.target_mask_rate,
            channel_drop_rate=config.channel_drop_rate if training else 0.0,
            generator=generator,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = reconstruction_losses(
                model,
                graph,
                view,
                support_weight=config.support_weight,
                attention_weight=config.attention_weight,
                distribution_weight=config.distribution_weight,
                node_weight=config.node_weight,
                generator=generator,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError("non-finite reconstruction loss")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for name in totals:
            totals[name] += float(getattr(losses, name).detach().cpu())
        sample.release_attention()
    return {name: value / len(samples) for name, value in totals.items()}


def train_unsupervised(
    dataset,
    *,
    output_dir,
    graph_config: GraphBuildConfig | None = None,
    config: TrainingConfig | None = None,
):
    """Train only on the supplied canonical training split; never read labels."""
    config = TrainingConfig() if config is None else config
    config.validate()
    graph_config = GraphBuildConfig() if graph_config is None else graph_config
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_name = str(dataset.manifest.get("split", "")).casefold()
    if split_name and split_name != "train":
        raise ValueError("unsupervised training requires a canonical train split")
    groups, source_groups = _source_split(dataset, config)
    if any(not values for values in groups.values()):
        raise ValueError("train/validation/calibration sample groups must be non-empty")

    num_channels = int(dataset.manifest["num_layers"]) * int(dataset.manifest["num_heads"])
    device = torch.device(dataset.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    model = MaskedAttentionAutoencoder(
        num_channels=num_channels,
        embedding_dim=config.embedding_dim,
        message_steps=config.message_steps,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _one_epoch(
            model, groups["train"], graph_config, config, epoch=epoch, optimizer=optimizer
        )
        validation_loss = _one_epoch(
            model, groups["validation"], graph_config, config, epoch=epoch, optimizer=None
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_loss.items()},
            **{f"validation_{key}": value for key, value in validation_loss.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        current = validation_loss["total"]
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)

    calibration_rows = []
    for sample_index, sample in enumerate(groups["calibration"]):
        graph = build_attention_graph(sample.attention(), graph_config).to(device)
        _embedding, residuals = score_graph_raw(
            model,
            graph,
            target_block_size=config.target_block_size,
            seed=config.seed + sample_index * 1009,
        )
        calibration_rows.append(residuals)
        sample.release_attention()
    calibrator = RobustResidualCalibrator.fit(np.concatenate(calibration_rows, axis=0))

    checkpoint = {
        "schema": "attention-graph-unsupervised-v1",
        "model_config": {
            "num_channels": num_channels,
            "embedding_dim": config.embedding_dim,
            "message_steps": config.message_steps,
            "dropout": config.dropout,
        },
        "graph_config": asdict(graph_config),
        "training_config": asdict(config),
        "source_groups": source_groups,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "calibrator": calibrator.to_dict(),
        "state_dict": best_state,
        "observer_model": dataset.manifest.get("observer_model"),
        "generator_model": dataset.manifest.get("generator_model"),
    }
    checkpoint_path = output / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    (output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_split.json").write_text(
        json.dumps(source_groups, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_samples": len(groups["train"]),
        "validation_samples": len(groups["validation"]),
        "calibration_samples": len(groups["calibration"]),
    }
