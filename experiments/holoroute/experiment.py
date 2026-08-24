"""Label-free HoloRoute training, conditional calibration and held-out scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import canonical_source_group
from experiments.attention_holonomy_audit.graph import build_attention_event_graph

from .artifacts import CHECKPOINT_SCHEMA, DENSITY_SCHEMA, SCORE_SCHEMA, load_npz, save_npz, sha256
from .config import (
    DensityConfig,
    HoloRouteConfig,
    LossConfig,
    MaskConfig,
    ModelConfig,
    TrainConfig,
)
from .density import AlignedReservoir, ConditionalDensity
from .model import HoloRouteEncoder
from .objectives import SCORE_FEATURES, score_graph, self_supervised_loss

NUISANCE_NAMES = (
    "log_absolute_position",
    "relative_position",
    "relative_position_squared",
    "relative_position_cubed",
    "log_response_length",
    "log_event_count",
    "log_relay_count",
    "log_diamond_count",
    "retained_mass",
    "observed_head_fraction",
    "unresolved_mass",
)


@dataclass(frozen=True)
class SourceSplit:
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    fit_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]


def _require_split(dataset, expected: str) -> None:
    actual = str(dataset.manifest.get("split"))
    if actual != expected:
        raise ValueError(f"expected dataset split {expected!r}, got {actual!r}")


def _selected_ids(dataset, task_type: str, limit: int | None) -> tuple[str, ...]:
    selected: list[str] = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        try:
            if task_type.casefold() == "all" or str(sample.task_type).casefold() == task_type.casefold():
                selected.append(str(sample_id))
        finally:
            sample.release_attention()
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no samples match the requested scope")
    return tuple(selected)


def _group_order(name: str, seed: int) -> bytes:
    return hashlib.sha256(f"holoroute-v1\0{seed}\0{name}".encode()).digest()


def _split_sources(
    dataset,
    sample_ids: tuple[str, ...],
    config: TrainConfig,
) -> SourceSplit:
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            grouped.setdefault(canonical_source_group(sample), []).append(sample_id)
        finally:
            sample.release_attention()
    if len(grouped) < 3:
        raise ValueError("HoloRoute needs at least three source groups")

    names = sorted(grouped, key=lambda name: _group_order(name, config.seed))
    validation_count = max(1, round(len(names) * config.validation_fraction))
    calibration_count = max(1, round(len(names) * config.calibration_fraction))
    while validation_count + calibration_count >= len(names):
        if calibration_count >= validation_count and calibration_count > 1:
            calibration_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("source groups cannot form fit/validation/calibration streams")

    validation_groups = set(names[:validation_count])
    calibration_groups = set(names[validation_count : validation_count + calibration_count])
    fit_groups = set(names) - validation_groups - calibration_groups

    def samples(groups: set[str]) -> tuple[str, ...]:
        return tuple(
            sample_id
            for group, ids in grouped.items()
            if group in groups
            for sample_id in ids
        )

    return SourceSplit(
        fit_ids=samples(fit_groups),
        validation_ids=samples(validation_groups),
        calibration_ids=samples(calibration_groups),
        fit_groups=tuple(sorted(fit_groups)),
        validation_groups=tuple(sorted(validation_groups)),
        calibration_groups=tuple(sorted(calibration_groups)),
    )


def _seed_for_sample(seed: int, sample_id: str, epoch: int = 0) -> int:
    payload = f"{seed}\0{epoch}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _config_from_dict(payload: dict) -> HoloRouteConfig:
    return HoloRouteConfig(
        model=ModelConfig(**payload["model"]),
        masking=MaskConfig(**payload["masking"]),
        loss=LossConfig(**payload["loss"]),
        train=TrainConfig(**payload["train"]),
        density=DensityConfig(**payload["density"]),
    )


def _save_checkpoint(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(checkpoint_path, *, device="cpu") -> tuple[HoloRouteEncoder, HoloRouteConfig, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported HoloRoute checkpoint")
    config = _config_from_dict(payload["config"])
    model = HoloRouteEncoder(
        int(payload["num_layers"]),
        int(payload["num_heads"]),
        config.model,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config, payload


def graph_nuisance(graph) -> np.ndarray:
    tokens = graph.num_response_tokens
    token = np.arange(tokens, dtype=np.float32)
    relative = token / max(tokens - 1, 1)
    event_count = np.bincount(
        graph.event_query.detach().cpu().numpy(),
        minlength=tokens,
    ).astype(np.float32)

    relay_count = np.zeros(tokens, dtype=np.float32)
    if graph.relay_edge_index.shape[1]:
        relay_token = graph.event_query[graph.relay_edge_index[1]].detach().cpu().numpy()
        relay_count += np.bincount(relay_token, minlength=tokens).astype(np.float32)
    diamond_count = np.zeros(tokens, dtype=np.float32)
    if graph.diamond_index.shape[1]:
        diamond_token = graph.event_query[graph.diamond_index[3]].detach().cpu().numpy()
        diamond_count += np.bincount(diamond_token, minlength=tokens).astype(np.float32)

    event_mass = graph.event_mass.detach().cpu().numpy().astype(np.float32)
    observed = graph.event_head_observed.float().mean(dim=-1).detach().cpu().numpy().astype(np.float32)
    retained_sum = np.bincount(
        graph.event_query.detach().cpu().numpy(),
        weights=event_mass,
        minlength=tokens,
    ).astype(np.float32)
    observed_sum = np.bincount(
        graph.event_query.detach().cpu().numpy(),
        weights=observed,
        minlength=tokens,
    ).astype(np.float32)
    denominator = np.maximum(event_count, 1.0)
    retained_mass = retained_sum / denominator
    observed_fraction = observed_sum / denominator
    unresolved = graph.unresolved_mass.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)

    return np.column_stack(
        (
            np.log1p(token),
            relative,
            relative**2,
            relative**3,
            np.full(tokens, np.log1p(tokens), dtype=np.float32),
            np.log1p(event_count),
            np.log1p(relay_count),
            np.log1p(diamond_count),
            retained_mass,
            observed_fraction,
            unresolved,
        )
    ).astype(np.float32)


def _mean_loss(model, dataset, sample_ids, config, *, epoch: int) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                graph = build_attention_event_graph(sample)
                if graph.num_events == 0:
                    continue
                generator = torch.Generator(device=graph.device)
                generator.manual_seed(_seed_for_sample(config.train.seed, sample_id, epoch))
                loss = self_supervised_loss(model, graph, config, generator=generator)
                values.append(float(loss.total.item()))
            finally:
                sample.release_attention()
    return float(np.mean(values)) if values else float("inf")


def train_reference(
    dataset,
    checkpoint_path,
    density_path,
    *,
    config: HoloRouteConfig | None = None,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Train the neural graph encoder and fit a label-free conditional density."""

    config = HoloRouteConfig() if config is None else config
    _require_split(dataset, "train")
    sample_ids = _selected_ids(dataset, task_type, limit)
    split = _split_sources(dataset, sample_ids, config.train)
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    device = torch.device(str(getattr(dataset, "device", "cpu")))

    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    model = HoloRouteEncoder(layers, heads, config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    best_state = None
    best_validation = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(config.train.epochs):
        model.train()
        order = list(split.fit_ids)
        random.Random(config.train.seed + epoch).shuffle(order)
        epoch_loss: list[float] = []
        for sample_id in tqdm(order, desc=f"HoloRoute epoch {epoch + 1}", unit="sample"):
            sample = dataset[sample_id]
            try:
                graph = build_attention_event_graph(sample)
                if graph.num_events == 0:
                    continue
                generator = torch.Generator(device=graph.device)
                generator.manual_seed(_seed_for_sample(config.train.seed, sample_id, epoch))
                optimizer.zero_grad(set_to_none=True)
                loss = self_supervised_loss(model, graph, config, generator=generator)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                epoch_loss.append(float(loss.total.item()))
            finally:
                sample.release_attention()
        validation = _mean_loss(
            model,
            dataset,
            split.validation_ids,
            config,
            epoch=epoch,
        )
        train_mean = float(np.mean(epoch_loss)) if epoch_loss else float("inf")
        history.append({"epoch": epoch + 1, "train_loss": train_mean, "validation_loss": validation})
        if validation < best_validation:
            best_validation = validation
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise ValueError("HoloRoute training produced no valid graph batches")
    model.load_state_dict(best_state)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "num_layers": layers,
        "num_heads": heads,
        "state_dict": best_state,
        "fit_group_id": split.fit_groups,
        "validation_group_id": split.validation_groups,
        "calibration_group_id": split.calibration_groups,
        "history": history,
    }
    _save_checkpoint(checkpoint_path, checkpoint_payload)

    reservoir = AlignedReservoir(config.density.reservoir_rows, config.train.seed + 73)
    for sample_id in tqdm(split.calibration_ids, desc="calibrate HoloRoute density", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample)
            if graph.num_events == 0:
                continue
            feature, coverage = score_graph(
                model,
                graph,
                config,
                seed=_seed_for_sample(config.train.seed + 991, sample_id),
            )
            reservoir.add(
                feature=feature,
                nuisance=graph_nuisance(graph),
                task=np.repeat(str(sample.task_type or ""), graph.num_response_tokens),
                coverage=coverage,
            )
        finally:
            sample.release_attention()
    calibration = reservoir.values()
    density = ConditionalDensity.fit(
        calibration["feature"],
        calibration["nuisance"],
        calibration["task"],
        ridge_alpha=config.density.ridge_alpha,
        covariance_shrinkage=config.density.covariance_shrinkage,
        scale_floor=config.density.scale_floor,
    )
    save_npz(
        density_path,
        schema=np.asarray(DENSITY_SCHEMA),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        train_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        score_feature_names=np.asarray(SCORE_FEATURES),
        nuisance_names=np.asarray(NUISANCE_NAMES),
        calibration_coverage_mean=np.nanmean(calibration["coverage"], axis=0).astype(np.float32),
        **density.arrays(),
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "density": str(Path(density_path).resolve()),
        "labels_read": False,
        "best_validation_loss": best_validation,
        "fit_groups": len(split.fit_groups),
        "validation_groups": len(split.validation_groups),
        "calibration_groups": len(split.calibration_groups),
    }


def load_density(path) -> tuple[ConditionalDensity, dict[str, np.ndarray]]:
    arrays = load_npz(path)
    if str(arrays["schema"].item()) != DENSITY_SCHEMA:
        raise ValueError("unsupported HoloRoute density reference")
    return ConditionalDensity.from_arrays(arrays), arrays


def score_split(
    dataset,
    checkpoint_path,
    density_path,
    score_path,
    *,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Freeze token-level neural graph anomaly scores without labels."""

    _require_split(dataset, "test")
    model, config, checkpoint = load_model(
        checkpoint_path,
        device=getattr(dataset, "device", "cpu"),
    )
    density, density_arrays = load_density(density_path)
    if sha256(checkpoint_path) != str(density_arrays["checkpoint_sha256"].item()):
        raise ValueError("density reference does not match the checkpoint")
    if int(checkpoint["num_layers"]) != int(dataset.manifest["num_layers"]):
        raise ValueError("checkpoint and test split use different layers")
    if int(checkpoint["num_heads"]) != int(dataset.manifest["num_heads"]):
        raise ValueError("checkpoint and test split use different heads")

    sample_ids = _selected_ids(dataset, task_type, limit)
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "task_type",
            "token_index",
            "response_length",
            "response_token_id",
            "score",
            "mechanism_feature",
            "standardized_feature",
            "coverage",
            "nuisance",
        )
    }
    model.eval()
    for sample_id in tqdm(sample_ids, desc="score HoloRoute", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample)
            if graph.num_events == 0:
                continue
            feature, coverage = score_graph(
                model,
                graph,
                config,
                seed=_seed_for_sample(config.train.seed + 1997, sample_id),
            )
            nuisance = graph_nuisance(graph)
            task = np.repeat(str(sample.task_type or ""), graph.num_response_tokens)
            score, standardized = density.score(feature, nuisance, task)
            tokens = graph.num_response_tokens
            rows["sample_id"].append(np.repeat(str(sample.sample_id), tokens))
            rows["source_id"].append(np.repeat(str(sample.source_id), tokens))
            rows["task_type"].append(task)
            rows["token_index"].append(np.arange(tokens, dtype=np.int32))
            rows["response_length"].append(np.full(tokens, tokens, dtype=np.int32))
            rows["response_token_id"].append(
                graph.response_token_ids.detach().cpu().numpy().astype(np.int64)
            )
            rows["score"].append(score)
            rows["mechanism_feature"].append(feature)
            rows["standardized_feature"].append(standardized)
            rows["coverage"].append(coverage)
            rows["nuisance"].append(nuisance)
        finally:
            sample.release_attention()
    arrays = {name: np.concatenate(value) for name, value in rows.items()}
    save_npz(
        score_path,
        schema=np.asarray(SCORE_SCHEMA),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        density_path=np.asarray(str(Path(density_path).resolve())),
        density_sha256=np.asarray(sha256(density_path)),
        dataset_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        primary_score=np.asarray("score"),
        score_feature_names=np.asarray(SCORE_FEATURES),
        nuisance_names=np.asarray(NUISANCE_NAMES),
        **arrays,
    )
    return {
        "scores": str(Path(score_path).resolve()),
        "labels_read": False,
        "samples": len(set(arrays["sample_id"].astype(str).tolist())),
        "tokens": len(arrays["score"]),
    }
