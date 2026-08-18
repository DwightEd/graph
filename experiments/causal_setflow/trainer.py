"""Label-free optimization and frozen token-row extraction.

The trainer keeps the full CASF architecture and source-set bounds.  Peak CUDA
memory is reduced by whole-model activation checkpointing and automatic mixed
precision; neither changes the self-supervised targets or graph semantics.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from .config import SourceSetConfig, TrainingConfig
from .data import extract_causal_source_set_graph
from .model import CausalSetFlowModel


@dataclass(frozen=True)
class TrainingHistory:
    rows: tuple[dict[str, float | int | str], ...]


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
    amp_dtype = _resolve_amp_dtype(device, training_config.precision)
    scaler = _make_grad_scaler(device, amp_dtype)
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
        if device.type == "cuda" and training_config.profile_cuda_memory:
            torch.cuda.reset_peak_memory_stats(device)

        for sample_index, sample_id in enumerate(
            tqdm(order, desc=f"train causal Set-Flow epoch {epoch}", unit="sample")
        ):
            sample = dataset[sample_id]
            graph = None
            losses = None
            try:
                graph = extract_causal_source_set_graph(sample, source_config)
                mask_seed = (
                    int(training_config.seed)
                    + 1_000_003 * epoch
                    + sample_index
                )
                if model.config.activation_checkpointing:
                    losses = _checkpointed_loss_tuple(
                        model,
                        graph,
                        mask_seed=mask_seed,
                        device=device,
                        amp_dtype=amp_dtype,
                    )
                else:
                    with _autocast_context(device, amp_dtype):
                        output = model(
                            graph,
                            mask_seed=mask_seed,
                            apply_masks=True,
                            device=device,
                        )
                        losses = _loss_tuple(output)

                scaled_loss = losses[0] / int(
                    training_config.gradient_accumulation
                )
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                pending += 1

                if pending >= int(training_config.gradient_accumulation):
                    _optimizer_step(
                        optimiser,
                        model,
                        scaler,
                        clip_norm=float(training_config.gradient_clip_norm),
                    )
                    pending = 0

                for name, value in zip(
                    (
                        "loss",
                        "route",
                        "memory",
                        "head",
                        "layer",
                        "temporal",
                        "variance",
                    ),
                    losses,
                    strict=True,
                ):
                    totals[name] += float(value.detach().float().cpu())
            except torch.OutOfMemoryError as error:
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 2**30
                    reserved = torch.cuda.memory_reserved(device) / 2**30
                    peak = torch.cuda.max_memory_allocated(device) / 2**30
                    raise RuntimeError(
                        "CASF CUDA OOM after exact sparse materialization; "
                        f"sample_id={sample_id} allocated_gib={allocated:.3f} "
                        f"reserved_gib={reserved:.3f} peak_gib={peak:.3f}. "
                        "Do not reduce source-set fidelity; lower only execution "
                        "chunk sizes if necessary."
                    ) from error
                raise
            finally:
                del losses, graph
                sample.release_attention()

        if pending:
            _optimizer_step(
                optimiser,
                model,
                scaler,
                clip_norm=float(training_config.gradient_clip_norm),
            )

        denominator = float(len(order))
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "precision": _precision_name(amp_dtype),
            "activation_checkpointing": int(
                bool(model.config.activation_checkpointing)
            ),
            **{name: value / denominator for name, value in totals.items()},
        }
        if device.type == "cuda" and training_config.profile_cuda_memory:
            row["peak_cuda_allocated_gib"] = float(
                torch.cuda.max_memory_allocated(device) / 2**30
            )
            row["peak_cuda_reserved_gib"] = float(
                torch.cuda.max_memory_reserved(device) / 2**30
            )
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
    precision: str = "auto",
) -> dict[str, np.ndarray]:
    device = torch.device(device)
    amp_dtype = _resolve_amp_dtype(device, precision)
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
        tqdm(
            list(map(str, sample_ids)),
            desc="freeze causal Set-Flow rows",
            unit="sample",
        )
    ):
        sample = dataset[sample_id]
        graph = None
        values = None
        try:
            graph = extract_causal_source_set_graph(sample, source_config)
            with _autocast_context(device, amp_dtype):
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
                columns[name].append(values[name].float().cpu().numpy())
        finally:
            del values, graph
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


def _checkpointed_loss_tuple(
    model: CausalSetFlowModel,
    graph,
    *,
    mask_seed: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, ...]:
    anchor = torch.zeros((), device=device, requires_grad=True)

    def run(_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with _autocast_context(device, amp_dtype):
            output = model(
                graph,
                mask_seed=mask_seed,
                apply_masks=True,
                device=device,
            )
            return _loss_tuple(output)

    return checkpoint(
        run,
        anchor,
        use_reentrant=False,
        preserve_rng_state=True,
        determinism_check="default",
    )


def _loss_tuple(output) -> tuple[torch.Tensor, ...]:
    loss = output.loss
    return (
        loss.total,
        loss.route_element,
        loss.memory_element,
        loss.head,
        loss.layer,
        loss.temporal,
        loss.variance,
    )


def _optimizer_step(
    optimiser,
    model,
    scaler,
    *,
    clip_norm: float,
) -> None:
    if scaler is None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        optimiser.step()
    else:
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        scaler.step(optimiser)
        scaler.update()
    optimiser.zero_grad(set_to_none=True)


def _resolve_amp_dtype(
    device: torch.device, precision: str
) -> torch.dtype | None:
    precision = str(precision)
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this CUDA device lacks support")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    raise ValueError("precision must be auto, bf16, fp16, or fp32")


def _autocast_context(
    device: torch.device, dtype: torch.dtype | None
):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(
    device: torch.device, dtype: torch.dtype | None
):
    if device.type != "cuda" or dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _precision_name(dtype: torch.dtype | None) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return "fp32"


def _text(value) -> str:
    return "" if value is None else str(value)