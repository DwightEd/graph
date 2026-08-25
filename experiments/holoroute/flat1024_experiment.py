"""Label-free training and scoring for the flat all-layer HoloRoute baseline."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.attention_holonomy_audit.graph import build_attention_event_graph

from .artifacts import DENSITY_SCHEMA, SCORE_SCHEMA, load_npz, save_npz, sha256
from .config import DensityConfig, TrainConfig
from .density import AlignedReservoir, ConditionalDensity
from .experiment import (
    NUISANCE_NAMES,
    _require_split,
    _seed_for_sample,
    _selected_ids,
    _split_sources,
    graph_nuisance,
)
from .flat1024 import (
    FLAT_CHECKPOINT_SCHEMA,
    FLAT_MODEL_TYPE,
    FLAT_SCORE_FEATURES,
    Flat1024Config,
    Flat1024MaskConfig,
    Flat1024Model,
    Flat1024ModelConfig,
    build_flat_pair_view,
    flat1024_loss,
    score_flat1024,
)


def _config_from_dict(payload: dict) -> Flat1024Config:
    return Flat1024Config(
        model=Flat1024ModelConfig(**payload["model"]),
        masking=Flat1024MaskConfig(**payload["masking"]),
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


def load_flat_model(checkpoint_path, *, device="cpu"):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["schema"] != FLAT_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported flat-1024 checkpoint")
    config = _config_from_dict(payload["config"])
    model = Flat1024Model(
        int(payload["num_layers"]),
        int(payload["num_heads"]),
        config.model,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config, payload


def load_flat_density(path):
    arrays = load_npz(path)
    if str(arrays["schema"].item()) != DENSITY_SCHEMA:
        raise ValueError("unsupported flat-1024 density reference")
    if str(arrays["model_type"].item()) != FLAT_MODEL_TYPE:
        raise ValueError("density reference is not a flat-1024 baseline")
    return ConditionalDensity.from_arrays(arrays), arrays


def _validation_loss(model, dataset, sample_ids, config) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                graph = build_attention_event_graph(sample)
                view = build_flat_pair_view(graph)
                if view.num_pairs == 0:
                    continue
                generator = torch.Generator(device=view.device)
                generator.manual_seed(_seed_for_sample(config.train.seed, sample_id, 0))
                values.append(float(flat1024_loss(model, view, config, generator=generator).item()))
            finally:
                sample.release_attention()
    return float(np.mean(values)) if values else float("inf")


def train_flat_reference(
    dataset,
    checkpoint_path,
    density_path,
    *,
    config: Flat1024Config | None = None,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Train the no-adjacency all-layer baseline and fit its conditional density."""

    config = Flat1024Config() if config is None else config
    _require_split(dataset, "train")
    sample_ids = _selected_ids(dataset, task_type, limit)
    split = _split_sources(dataset, sample_ids, config.train)
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    device = torch.device(str(getattr(dataset, "device", "cpu")))

    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    model = Flat1024Model(layers, heads, config.model).to(device)
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
        np.random.default_rng(config.train.seed + epoch).shuffle(order)
        epoch_loss: list[float] = []
        for sample_id in tqdm(order, desc=f"flat-1024 epoch {epoch + 1}", unit="sample"):
            sample = dataset[sample_id]
            try:
                graph = build_attention_event_graph(sample)
                view = build_flat_pair_view(graph)
                if view.num_pairs == 0:
                    continue
                generator = torch.Generator(device=view.device)
                generator.manual_seed(_seed_for_sample(config.train.seed, sample_id, epoch))
                optimizer.zero_grad(set_to_none=True)
                loss = flat1024_loss(model, view, config, generator=generator)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                epoch_loss.append(float(loss.item()))
            finally:
                sample.release_attention()
        validation = _validation_loss(model, dataset, split.validation_ids, config)
        train_mean = float(np.mean(epoch_loss)) if epoch_loss else float("inf")
        history.append({"epoch": epoch + 1, "train_loss": train_mean, "validation_loss": validation})
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise ValueError("flat-1024 training produced no valid pair tensors")
    model.load_state_dict(best_state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _save_checkpoint(
        checkpoint_path,
        {
            "schema": FLAT_CHECKPOINT_SCHEMA,
            "model_type": FLAT_MODEL_TYPE,
            "config": config.to_dict(),
            "num_layers": layers,
            "num_heads": heads,
            "state_dict": best_state,
            "fit_group_id": split.fit_groups,
            "validation_group_id": split.validation_groups,
            "calibration_group_id": split.calibration_groups,
            "history": history,
            "parameter_count": parameter_count,
        },
    )

    reservoir = AlignedReservoir(config.density.reservoir_rows, config.train.seed + 73)
    for sample_id in tqdm(split.calibration_ids, desc="calibrate flat-1024 density", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample)
            view = build_flat_pair_view(graph)
            if view.num_pairs == 0:
                continue
            feature, coverage = score_flat1024(
                model,
                view,
                rounds=config.masking.score_rounds,
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
        model_type=np.asarray(FLAT_MODEL_TYPE),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        train_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        score_feature_names=np.asarray(FLAT_SCORE_FEATURES),
        nuisance_names=np.asarray(NUISANCE_NAMES),
        calibration_coverage_mean=np.nanmean(calibration["coverage"], axis=0).astype(np.float32),
        **density.arrays(),
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "density": str(Path(density_path).resolve()),
        "model_type": FLAT_MODEL_TYPE,
        "parameter_count": parameter_count,
        "raw_feature_dim": layers * heads,
        "best_validation_loss": best_validation,
        "labels_read": False,
        "fit_groups": len(split.fit_groups),
        "validation_groups": len(split.validation_groups),
        "calibration_groups": len(split.calibration_groups),
    }


def score_flat_split(
    dataset,
    checkpoint_path,
    density_path,
    score_path,
    *,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Freeze flat-1024 token scores using the same nuisance model as HoloRoute."""

    _require_split(dataset, "test")
    model, config, checkpoint = load_flat_model(
        checkpoint_path, device=getattr(dataset, "device", "cpu")
    )
    density, density_arrays = load_flat_density(density_path)
    if sha256(checkpoint_path) != str(density_arrays["checkpoint_sha256"].item()):
        raise ValueError("flat density does not match the checkpoint")
    if int(checkpoint["num_layers"]) != int(dataset.manifest["num_layers"]):
        raise ValueError("flat checkpoint and test split use different layers")
    if int(checkpoint["num_heads"]) != int(dataset.manifest["num_heads"]):
        raise ValueError("flat checkpoint and test split use different heads")

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
    for sample_id in tqdm(sample_ids, desc="score flat-1024", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample)
            view = build_flat_pair_view(graph)
            if view.num_pairs == 0:
                continue
            feature, coverage = score_flat1024(
                model,
                view,
                rounds=config.masking.score_rounds,
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
            rows["response_token_id"].append(view.response_token_ids.detach().cpu().numpy().astype(np.int64))
            rows["score"].append(score)
            rows["mechanism_feature"].append(feature)
            rows["standardized_feature"].append(standardized)
            rows["coverage"].append(coverage)
            rows["nuisance"].append(nuisance)
        finally:
            sample.release_attention()
    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    save_npz(
        score_path,
        schema=np.asarray(SCORE_SCHEMA),
        model_type=np.asarray(FLAT_MODEL_TYPE),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        density_path=np.asarray(str(Path(density_path).resolve())),
        density_sha256=np.asarray(sha256(density_path)),
        dataset_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        primary_score=np.asarray("score"),
        score_feature_names=np.asarray(FLAT_SCORE_FEATURES),
        nuisance_names=np.asarray(NUISANCE_NAMES),
        **arrays,
    )
    return {
        "scores": str(Path(score_path).resolve()),
        "model_type": FLAT_MODEL_TYPE,
        "samples": len(set(arrays["sample_id"].astype(str).tolist())),
        "tokens": len(arrays["score"]),
        "labels_read": False,
    }
