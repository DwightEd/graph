"""Label-free optimization and frozen token-row extraction."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import SourceSetConfig, TrainingConfig
from .data import extract_causal_source_set_graph
from .model import CausalSetFlowModel


@dataclass(frozen=True)
class TrainingHistory:
    rows: tuple[dict[str, float | int], ...]


def train_label_free(
    model: CausalSetFlowModel,
    dataset,
    sample_ids,
    *,
    source_config: SourceSetConfig,
    training_config: TrainingConfig,
    device: str | torch.device,
) -> TrainingHistory:
    training_config.validate()
    device = torch.device(device)
    model.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.learning_rate),
        weight_decay=float(training_config.weight_decay),
    )
    sample_ids = list(map(str, sample_ids))
    if not sample_ids:
        raise ValueError("training sample list is empty")
    history = []

    for epoch in range(1, int(training_config.epochs) + 1):
        order = sample_ids.copy()
        random.Random(int(training_config.seed) + epoch).shuffle(order)
        totals = {
            name: 0.0
            for name in (
                "loss",
                "route",
                "memory",
                "head",
                "layer",
                "temporal",
                "variance",
            )
        }
        pending = 0
        optimiser.zero_grad(set_to_none=True)
        model.train()
        for sample_index, sample_id in enumerate(
            tqdm(order, desc=f"train causal Set-Flow epoch {epoch}", unit="sample")
        ):
            sample = dataset[sample_id]
            try:
                graph = extract_causal_source_set_graph(sample, source_config)
                output = model(
                    graph,
                    mask_seed=(
                        int(training_config.seed)
                        + 1_000_003 * epoch
                        + sample_index
                    ),
                    apply_masks=True,
                    device=device,
                )
                (
                    output.loss.total / int(training_config.gradient_accumulation)
                ).backward()
                pending += 1
                if pending >= int(training_config.gradient_accumulation):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(training_config.gradient_clip_norm),
                    )
                    optimiser.step()
                    optimiser.zero_grad(set_to_none=True)
                    pending = 0
                losses = output.loss
                totals["loss"] += float(losses.total.detach().cpu())
                totals["route"] += float(losses.route_element.detach().cpu())
                totals["memory"] += float(losses.memory_element.detach().cpu())
                totals["head"] += float(losses.head.detach().cpu())
                totals["layer"] += float(losses.layer.detach().cpu())
                totals["temporal"] += float(losses.temporal.detach().cpu())
                totals["variance"] += float(losses.variance.detach().cpu())
            finally:
                sample.release_attention()
        if pending:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config.gradient_clip_norm)
            )
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        denominator = float(len(order))
        row = {
            "epoch": epoch,
            **{name: value / denominator for name, value in totals.items()},
        }
        history.append(row)
        print(row, flush=True)
    return TrainingHistory(rows=tuple(history))


@torch.no_grad()
def extract_frozen_rows(
    model: CausalSetFlowModel,
    dataset,
    sample_ids,
    *,
    source_config: SourceSetConfig,
    deterministic_masks: int,
    seed: int,
    device: str | torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    model.to(device)
    columns: dict[str, list] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "task_type",
            "data_source",
            "generator_model",
            "token_index",
            "response_length",
            "embedding",
            "route_element",
            "memory_element",
            "head_reconstruction",
            "layer_reconstruction",
            "temporal_prediction",
        )
    }
    for sample_index, sample_id in enumerate(
        tqdm(list(map(str, sample_ids)), desc="freeze causal Set-Flow rows", unit="sample")
    ):
        sample = dataset[sample_id]
        try:
            graph = extract_causal_source_set_graph(sample, source_config)
            values = model.deterministic_scores(
                graph,
                masks=int(deterministic_masks),
                seed=int(seed) + 10_007 * sample_index,
            )
            count = graph.response_count
            source_id = (
                str(sample.source_id)
                if sample.source_id is not None
                else str(sample.sample_id)
            )
            columns["sample_id"].extend([str(sample.sample_id)] * count)
            columns["source_id"].extend([source_id] * count)
            columns["task_type"].extend([_text(sample.task_type)] * count)
            columns["data_source"].extend([_text(sample.data_source)] * count)
            columns["generator_model"].extend(
                [_text(sample.generator_model)] * count
            )
            columns["token_index"].append(np.arange(count, dtype=np.int32))
            columns["response_length"].append(
                np.full(count, count, dtype=np.int32)
            )
            for name in (
                "embedding",
                "route_element",
                "memory_element",
                "head_reconstruction",
                "layer_reconstruction",
                "temporal_prediction",
            ):
                columns[name].append(values[name].detach().cpu().numpy())
        finally:
            sample.release_attention()

    output = {
        "sample_id": np.asarray(columns["sample_id"], dtype=str),
        "source_id": np.asarray(columns["source_id"], dtype=str),
        "task_type": np.asarray(columns["task_type"], dtype=str),
        "data_source": np.asarray(columns["data_source"], dtype=str),
        "generator_model": np.asarray(columns["generator_model"], dtype=str),
        "token_index": np.concatenate(columns["token_index"]).astype(np.int32),
        "response_length": np.concatenate(columns["response_length"]).astype(
            np.int32
        ),
    }
    for name in (
        "embedding",
        "route_element",
        "memory_element",
        "head_reconstruction",
        "layer_reconstruction",
        "temporal_prediction",
    ):
        output[name] = np.concatenate(columns[name], axis=0).astype(np.float32)
    return output


def select_reference_rows(
    rows: dict[str, np.ndarray], count_per_sample: int
) -> np.ndarray:
    sample_id = np.asarray(rows["sample_id"], dtype=str)
    token_index = np.asarray(rows["token_index"], dtype=np.int64)
    selected = []
    for sample in dict.fromkeys(sample_id.tolist()):
        rows_for_sample = np.flatnonzero(sample_id == sample)
        order = rows_for_sample[np.argsort(token_index[rows_for_sample])]
        keep = min(len(order), int(count_per_sample))
        position = np.unique(
            np.rint(np.linspace(0, len(order) - 1, keep)).astype(np.int64)
        )
        selected.extend(order[position].tolist())
    return np.asarray(selected, dtype=np.int64)


def _text(value) -> str:
    return "" if value is None else str(value)
