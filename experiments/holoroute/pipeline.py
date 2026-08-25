"""Dataset-level training, calibration and scoring for HoloRoute."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import canonical_source_group

from .artifacts import (
    CHECKPOINT_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_npz,
    save_checkpoint,
    save_npz,
    sha256,
)
from .baseline import (
    FLAT_CHECKPOINT_SCHEMA,
    FLAT_MODEL_TYPE,
    FLAT_RESIDUAL_NAMES,
    Flat1024,
    build_pairs,
    flat_loss,
    score_flat as score_flat_graph,
)
from .config import DetectionConfig, GraphConfig, HoloRouteConfig, LossConfig, ModelConfig, TrainConfig
from .detection import (
    CONDITION_NAMES,
    RESIDUAL_NAMES,
    ConditionalReference,
    ResidualReservoir,
    TokenResiduals,
    score_graph,
    token_conditions,
)
from .graph import build_graph
from .learning import sample_seed, train_model
from .model import HoloRoute

HOLOROUTE_MODEL_TYPE = "holoroute"


@dataclass(frozen=True)
class SourceSplit:
    fit: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]
    fit_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]


@dataclass(frozen=True)
class ScoreRows:
    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    response_token_id: np.ndarray
    score: np.ndarray
    residual: np.ndarray
    standardized: np.ndarray
    coverage: np.ndarray
    condition: np.ndarray


def require_split(dataset, expected: str) -> None:
    actual = str(dataset.manifest.get("split"))
    if actual != expected:
        raise ValueError(f"expected {expected!r} data, found {actual!r}")


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


def source_order(group: str, seed: int) -> bytes:
    return hashlib.sha256(f"holoroute\0{seed}\0{group}".encode()).digest()


def split_sources(dataset, sample_ids: tuple[str, ...], config: TrainConfig) -> SourceSplit:
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            grouped.setdefault(canonical_source_group(sample), []).append(sample_id)
        finally:
            sample.release_attention()
    if len(grouped) < 3:
        raise ValueError("training needs at least three independent source groups")

    ordered = sorted(grouped, key=lambda group: source_order(group, config.seed))
    validation_count = max(1, round(len(ordered) * config.validation_fraction))
    calibration_count = max(1, round(len(ordered) * config.calibration_fraction))
    while validation_count + calibration_count >= len(ordered):
        if calibration_count > 1:
            calibration_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("source groups cannot form three disjoint streams")

    validation_groups = set(ordered[:validation_count])
    calibration_groups = set(ordered[validation_count : validation_count + calibration_count])
    fit_groups = set(ordered) - validation_groups - calibration_groups

    def samples(groups: set[str]) -> tuple[str, ...]:
        return tuple(
            sample_id
            for group, identifiers in grouped.items()
            if group in groups
            for sample_id in identifiers
        )

    return SourceSplit(
        fit=samples(fit_groups),
        validation=samples(validation_groups),
        calibration=samples(calibration_groups),
        fit_groups=tuple(sorted(fit_groups)),
        validation_groups=tuple(sorted(validation_groups)),
        calibration_groups=tuple(sorted(calibration_groups)),
    )


def config_from_dict(payload: dict[str, object]) -> HoloRouteConfig:
    return HoloRouteConfig(
        graph=GraphConfig(**payload["graph"]),
        model=ModelConfig(**payload["model"]),
        train=TrainConfig(**payload["train"]),
        loss=LossConfig(**payload["loss"]),
        detection=DetectionConfig(**payload["detection"]),
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_reference(dataset, sample_ids, config, score_sample, description):
    reservoir = ResidualReservoir(config.detection.reservoir_rows, config.train.seed + 73)
    for sample_id in tqdm(sample_ids, desc=description, unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            if not graph.event_count:
                continue
            residuals = score_sample(graph, sample_id)
            reservoir.add(
                residual=residuals.value,
                condition=token_conditions(graph),
                task=np.repeat(str(sample.task_type or ""), graph.response_count),
                coverage=residuals.coverage,
            )
        finally:
            sample.release_attention()
    calibration = reservoir.values()
    reference = ConditionalReference.fit(
        calibration["residual"],
        calibration["condition"],
        calibration["task"],
        config.detection,
    )
    coverage = np.nanmean(calibration["coverage"], axis=0).astype(np.float32)
    return reference, coverage


def save_reference(path, reference, checkpoint_path, dataset, model_type, residual_names, coverage):
    save_npz(
        path,
        schema=np.asarray(REFERENCE_SCHEMA),
        model_type=np.asarray(model_type),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        train_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        residual_names=np.asarray(residual_names),
        condition_names=np.asarray(CONDITION_NAMES),
        calibration_coverage_mean=coverage,
        **reference.arrays(),
    )


def load_reference(path, model_type: str):
    arrays = load_npz(path)
    if str(arrays["schema"].item()) != REFERENCE_SCHEMA:
        raise ValueError("unsupported HoloRoute reference")
    if str(arrays["model_type"].item()) != model_type:
        raise ValueError("reference and model type differ")
    return ConditionalReference.from_arrays(arrays), arrays


def load_holoroute(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["model_type"] != HOLOROUTE_MODEL_TYPE:
        raise ValueError("unsupported HoloRoute checkpoint")
    config = config_from_dict(payload["config"])
    model = HoloRoute(payload["layer_count"], payload["head_count"], config.model).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config, payload


def load_flat(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["schema"] != FLAT_CHECKPOINT_SCHEMA or payload["model_type"] != FLAT_MODEL_TYPE:
        raise ValueError("unsupported Flat-1024 checkpoint")
    config = config_from_dict(payload["config"])
    model = Flat1024(
        payload["layer_count"],
        payload["head_count"],
        hidden=payload["hidden"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config, payload


def flat_validation_loss(model, dataset, sample_ids, config) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                pairs = build_pairs(build_graph(sample, config.graph))
                if not pairs.count:
                    continue
                repeated = []
                for mask_index in range(config.train.validation_masks):
                    generator = torch.Generator(device=pairs.device)
                    generator.manual_seed(sample_seed(config.train.seed, sample_id, 20_000 + mask_index))
                    repeated.append(float(flat_loss(model, pairs, config, generator).item()))
                losses.append(float(np.mean(repeated)))
            finally:
                sample.release_attention()
    return float(np.mean(losses)) if losses else float("inf")


def train_flat_model(model, dataset, split: SourceSplit, config: HoloRouteConfig):
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
        order = list(split.fit)
        random.Random(config.train.seed + epoch).shuffle(order)
        losses: list[float] = []
        for sample_id in tqdm(order, desc=f"Flat-1024 epoch {epoch + 1}", unit="sample"):
            sample = dataset[sample_id]
            try:
                pairs = build_pairs(build_graph(sample, config.graph))
                if not pairs.count:
                    continue
                generator = torch.Generator(device=pairs.device)
                generator.manual_seed(sample_seed(config.train.seed, sample_id, epoch))
                optimizer.zero_grad(set_to_none=True)
                loss = flat_loss(model, pairs, config, generator)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                losses.append(float(loss.item()))
            finally:
                sample.release_attention()

        validation = flat_validation_loss(model, dataset, split.validation, config)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)) if losses else float("inf"),
                "validation_loss": validation,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("flat training produced no pair tensors")
    return best_state, history, best_validation


def train_holoroute(dataset, checkpoint_path, reference_path, config, task="QA", limit=None):
    require_split(dataset, "train")
    split = split_sources(dataset, select_samples(dataset, task, limit), config.train)
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    device = torch.device(str(getattr(dataset, "device", "cpu")))
    set_random_seed(config.train.seed)

    model = HoloRoute(layers, heads, config.model).to(device)
    training = train_model(model, dataset, split.fit, split.validation, config)
    model.load_state_dict(training.state_dict)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_checkpoint(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_type": HOLOROUTE_MODEL_TYPE,
            "config": config.as_dict(),
            "layer_count": layers,
            "head_count": heads,
            "state_dict": training.state_dict,
            "history": training.history,
            "fit_groups": split.fit_groups,
            "validation_groups": split.validation_groups,
            "calibration_groups": split.calibration_groups,
            "parameter_count": parameter_count,
        },
    )

    reference, coverage = collect_reference(
        dataset,
        split.calibration,
        config,
        lambda graph, sample_id: score_graph(
            model,
            graph,
            config,
            sample_seed(config.train.seed + 991, sample_id),
        ),
        "calibrate HoloRoute",
    )
    save_reference(
        reference_path,
        reference,
        checkpoint_path,
        dataset,
        HOLOROUTE_MODEL_TYPE,
        RESIDUAL_NAMES,
        coverage,
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "best_validation_loss": training.best_validation_loss,
        "parameter_count": parameter_count,
        "labels_read": False,
    }


def train_flat(dataset, checkpoint_path, reference_path, config, task="QA", limit=None, hidden=96):
    require_split(dataset, "train")
    split = split_sources(dataset, select_samples(dataset, task, limit), config.train)
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    device = torch.device(str(getattr(dataset, "device", "cpu")))
    set_random_seed(config.train.seed)

    model = Flat1024(layers, heads, hidden=hidden).to(device)
    state, history, validation = train_flat_model(model, dataset, split, config)
    model.load_state_dict(state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_checkpoint(
        checkpoint_path,
        {
            "schema": FLAT_CHECKPOINT_SCHEMA,
            "model_type": FLAT_MODEL_TYPE,
            "config": config.as_dict(),
            "layer_count": layers,
            "head_count": heads,
            "hidden": hidden,
            "state_dict": state,
            "history": history,
            "fit_groups": split.fit_groups,
            "validation_groups": split.validation_groups,
            "calibration_groups": split.calibration_groups,
            "parameter_count": parameter_count,
        },
    )

    reference, coverage = collect_reference(
        dataset,
        split.calibration,
        config,
        lambda graph, sample_id: score_flat_graph(
            model,
            build_pairs(graph),
            config,
            sample_seed(config.train.seed + 991, sample_id),
        ),
        "calibrate Flat-1024",
    )
    save_reference(
        reference_path,
        reference,
        checkpoint_path,
        dataset,
        FLAT_MODEL_TYPE,
        FLAT_RESIDUAL_NAMES,
        coverage,
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "best_validation_loss": validation,
        "parameter_count": parameter_count,
        "raw_feature_dim": layers * heads,
        "labels_read": False,
    }


def make_score_rows(graph, residuals: TokenResiduals, reference: ConditionalReference) -> ScoreRows:
    condition = token_conditions(graph)
    task = np.repeat(graph.task_type, graph.response_count)
    score, standardized = reference.transform(residuals.value, condition, task)
    tokens = graph.response_count
    return ScoreRows(
        sample_id=np.repeat(graph.sample_id, tokens),
        source_id=np.repeat(graph.source_id, tokens),
        task_type=task,
        token_index=np.arange(tokens, dtype=np.int32),
        response_length=np.full(tokens, tokens, dtype=np.int32),
        response_token_id=graph.response_token_ids.detach().cpu().numpy().astype(np.int64),
        score=score,
        residual=residuals.value,
        standardized=standardized,
        coverage=residuals.coverage,
        condition=condition,
    )


def merge_score_rows(rows: list[ScoreRows]) -> dict[str, np.ndarray]:
    if not rows:
        raise RuntimeError("scoring produced no token rows")
    return {
        field: np.concatenate([getattr(row, field) for row in rows])
        for field in ScoreRows.__dataclass_fields__
    }


def save_scores(path, arrays, dataset, checkpoint_path, reference_path, model_type, residual_names):
    save_npz(
        path,
        schema=np.asarray(SCORE_SCHEMA),
        model_type=np.asarray(model_type),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        residual_names=np.asarray(residual_names),
        condition_names=np.asarray(CONDITION_NAMES),
        **arrays,
    )


def score_dataset(dataset, config, reference, task, limit, description, score_sample):
    rows: list[ScoreRows] = []
    for sample_id in tqdm(select_samples(dataset, task, limit), desc=description, unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            if not graph.event_count:
                continue
            rows.append(make_score_rows(graph, score_sample(graph, sample_id), reference))
        finally:
            sample.release_attention()
    return merge_score_rows(rows)


def score_holoroute(dataset, checkpoint_path, reference_path, output_path, task="QA", limit=None):
    require_split(dataset, "test")
    model, config, checkpoint = load_holoroute(checkpoint_path, getattr(dataset, "device", "cpu"))
    reference, reference_arrays = load_reference(reference_path, HOLOROUTE_MODEL_TYPE)
    if sha256(checkpoint_path) != str(reference_arrays["checkpoint_sha256"].item()):
        raise ValueError("reference was fitted for a different checkpoint")
    if checkpoint["layer_count"] != int(dataset.manifest["num_layers"]):
        raise ValueError("checkpoint and test data have different layers")
    if checkpoint["head_count"] != int(dataset.manifest["num_heads"]):
        raise ValueError("checkpoint and test data have different heads")

    arrays = score_dataset(
        dataset,
        config,
        reference,
        task,
        limit,
        "score HoloRoute",
        lambda graph, sample_id: score_graph(
            model,
            graph,
            config,
            sample_seed(config.train.seed + 1997, sample_id),
        ),
    )
    save_scores(
        output_path,
        arrays,
        dataset,
        checkpoint_path,
        reference_path,
        HOLOROUTE_MODEL_TYPE,
        RESIDUAL_NAMES,
    )
    return {
        "scores": str(Path(output_path).resolve()),
        "samples": len(set(arrays["sample_id"].astype(str).tolist())),
        "tokens": len(arrays["score"]),
        "labels_read": False,
    }


def score_flat(dataset, checkpoint_path, reference_path, output_path, task="QA", limit=None):
    require_split(dataset, "test")
    model, config, checkpoint = load_flat(checkpoint_path, getattr(dataset, "device", "cpu"))
    reference, reference_arrays = load_reference(reference_path, FLAT_MODEL_TYPE)
    if sha256(checkpoint_path) != str(reference_arrays["checkpoint_sha256"].item()):
        raise ValueError("reference was fitted for a different checkpoint")
    if checkpoint["layer_count"] != int(dataset.manifest["num_layers"]):
        raise ValueError("checkpoint and test data have different layers")
    if checkpoint["head_count"] != int(dataset.manifest["num_heads"]):
        raise ValueError("checkpoint and test data have different heads")

    arrays = score_dataset(
        dataset,
        config,
        reference,
        task,
        limit,
        "score Flat-1024",
        lambda graph, sample_id: score_flat_graph(
            model,
            build_pairs(graph),
            config,
            sample_seed(config.train.seed + 1997, sample_id),
        ),
    )
    save_scores(
        output_path,
        arrays,
        dataset,
        checkpoint_path,
        reference_path,
        FLAT_MODEL_TYPE,
        FLAT_RESIDUAL_NAMES,
    )
    return {
        "scores": str(Path(output_path).resolve()),
        "samples": len(set(arrays["sample_id"].astype(str).tolist())),
        "tokens": len(arrays["score"]),
        "labels_read": False,
    }
