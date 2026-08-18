"""Label-free MG-CASF optimization and frozen energy extraction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    CORRUPTION_NAMES,
    CorruptionConfig,
    SourceSetConfig,
    TrainingConfig,
)
from .corruptions import sample_corruption_plan
from .data import extract_causal_source_set_graph
from .losses import (
    LossBreakdown,
    combine_breakdown,
    corrupted_energy_loss,
    corruption_type_loss,
    cosine_recovery_loss,
    pairwise_ranking_loss,
    robust_clean_energy_loss,
    variance_covariance_loss,
)
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
    corruption_config: CorruptionConfig,
    training_config: TrainingConfig,
    device: str | torch.device,
) -> TrainingHistory:
    """Optimize paired clean/corrupted Set-Flow energy without labels."""

    source_config.validate()
    corruption_config.validate()
    training_config.validate()
    device = torch.device(device)
    model.to(device)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.learning_rate),
        weight_decay=float(training_config.weight_decay),
    )
    amp_dtype = _resolve_amp_dtype(device, training_config.precision)
    scaler = _make_grad_scaler(device, amp_dtype)
    sample_ids = list(map(str, sample_ids))
    if not sample_ids:
        raise ValueError("training sample list is empty")
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, int(training_config.epochs) + 1):
        order = sample_ids.copy()
        random.Random(int(training_config.seed) + epoch).shuffle(order)
        totals = {
            name: 0.0
            for name in (
                "loss",
                "clean_energy",
                "corrupt_energy",
                "ranking",
                "type_token",
                "type_channel",
                "clean_recovery",
                "context_recovery",
                "variance",
                "covariance",
                "energy_gap",
                "embedding_std",
            )
        }
        pending = 0
        optimiser.zero_grad(set_to_none=True)
        model.train()
        if device.type == "cuda" and training_config.profile_cuda_memory:
            torch.cuda.reset_peak_memory_stats(device)

        for sample_index, sample_id in enumerate(
            tqdm(order, desc=f"train mechanism-guided Set-Flow epoch {epoch}", unit="sample")
        ):
            sample = dataset[sample_id]
            graph = None
            teacher = clean = corrupted = None
            clean_energy = corrupted_energy = None
            clean_projected = corrupted_projected = None
            breakdown = None
            try:
                graph = extract_causal_source_set_graph(sample, source_config)
                seed = (
                    int(training_config.seed)
                    + 1_000_003 * epoch
                    + sample_index
                )
                generator = torch.Generator(device=device.type)
                generator.manual_seed(seed)
                forced_type = (
                    (epoch - 1) * len(order) + sample_index
                ) % len(CORRUPTION_NAMES)
                plan = sample_corruption_plan(
                    graph.response_count,
                    graph.num_layers,
                    graph.num_heads,
                    corruption_config,
                    device=device,
                    generator=generator,
                    forced_type=forced_type,
                )

                with _autocast_context(device, amp_dtype):
                    teacher = model.encode_teacher(graph, device=device)
                    clean = model.encode_online(graph, device=device)
                    clean_energy = model.energy(clean)
                    clean_projected = model.project(clean)

                    clean_token_loss = robust_clean_energy_loss(
                        clean_energy.general,
                        corruption_config.clean_keep_fraction,
                    )
                    clean_channel_values = clean_energy.channel_general[
                        clean.channel_active
                    ]
                    clean_channel_loss = robust_clean_energy_loss(
                        clean_channel_values,
                        corruption_config.clean_keep_fraction,
                    )
                    clean_energy_loss = 0.5 * (
                        clean_token_loss + clean_channel_loss
                    )
                    clean_recovery = 0.5 * (
                        cosine_recovery_loss(
                            clean_projected.token,
                            teacher.token_embedding,
                            epsilon=model.config.epsilon,
                        )
                        + cosine_recovery_loss(
                            clean_projected.channel,
                            teacher.channel_state,
                            teacher.channel_active,
                            epsilon=model.config.epsilon,
                        )
                    )
                    variance, covariance = variance_covariance_loss(
                        clean.token_embedding
                    )
                    clean_total = (
                        float(training_config.clean_energy_weight)
                        * clean_energy_loss
                        + float(training_config.clean_recovery_weight)
                        * clean_recovery
                        + float(training_config.variance_weight) * variance
                        + float(training_config.covariance_weight) * covariance
                    )

                _backward(
                    clean_total
                    / float(training_config.gradient_accumulation),
                    scaler,
                )
                clean_general_detached = clean_energy.general.detach()
                clean_channel_detached = clean_energy.channel_general.detach()

                with _autocast_context(device, amp_dtype):
                    corrupted = model.encode_online(
                        graph,
                        corruption_plan=plan,
                        corruption_config=corruption_config,
                        device=device,
                    )
                    corrupted_energy = model.energy(corrupted)
                    corrupted_projected = model.project(corrupted)
                    channel_mask = corrupted.channel_corruption_mask
                    token_mask = channel_mask.any(dim=(1, 2))

                    corrupt_energy_loss = 0.5 * (
                        corrupted_energy_loss(
                            corrupted_energy.general, token_mask
                        )
                        + corrupted_energy_loss(
                            corrupted_energy.channel_general, channel_mask
                        )
                    )
                    ranking = 0.5 * (
                        pairwise_ranking_loss(
                            clean_general_detached,
                            corrupted_energy.general,
                            token_mask,
                            margin=corruption_config.margin,
                        )
                        + pairwise_ranking_loss(
                            clean_channel_detached,
                            corrupted_energy.channel_general,
                            channel_mask,
                            margin=corruption_config.margin,
                        )
                    )
                    type_token = corruption_type_loss(
                        corrupted_energy.type_energy,
                        plan.type_index,
                        token_mask,
                    )
                    type_channel = corruption_type_loss(
                        corrupted_energy.channel_type,
                        plan.type_index,
                        channel_mask,
                    )
                    context_recovery = 0.5 * (
                        cosine_recovery_loss(
                            corrupted_projected.token,
                            teacher.token_embedding,
                            ~token_mask,
                            epsilon=model.config.epsilon,
                        )
                        + cosine_recovery_loss(
                            corrupted_projected.channel,
                            teacher.channel_state,
                            teacher.channel_active & ~channel_mask,
                            epsilon=model.config.epsilon,
                        )
                    )
                    corrupt_total = (
                        float(training_config.corrupt_energy_weight)
                        * corrupt_energy_loss
                        + float(training_config.ranking_weight) * ranking
                        + float(training_config.type_weight)
                        * (type_token + type_channel)
                        + float(training_config.context_recovery_weight)
                        * context_recovery
                    )

                _backward(
                    corrupt_total
                    / float(training_config.gradient_accumulation),
                    scaler,
                )
                pending += 1
                if pending >= int(training_config.gradient_accumulation):
                    _optimizer_step(
                        optimiser,
                        model,
                        scaler,
                        clip_norm=float(training_config.gradient_clip_norm),
                        ema_momentum=float(training_config.ema_momentum),
                    )
                    pending = 0

                breakdown = combine_breakdown(
                    clean_energy=clean_energy_loss,
                    corrupt_energy=corrupt_energy_loss,
                    ranking=ranking,
                    type_token=type_token,
                    type_channel=type_channel,
                    clean_recovery=clean_recovery,
                    context_recovery=context_recovery,
                    variance=variance,
                    covariance=covariance,
                    weights=training_config,
                )
                gap = (
                    corrupted_energy.general[token_mask].mean()
                    - clean_general_detached[token_mask].mean()
                    if bool(token_mask.any())
                    else torch.zeros((), device=device)
                )
                embedding_std = clean.token_embedding.float().std(
                    dim=0, unbiased=False
                ).mean()
                for name, value in (
                    ("loss", breakdown.total),
                    ("clean_energy", breakdown.clean_energy),
                    ("corrupt_energy", breakdown.corrupt_energy),
                    ("ranking", breakdown.ranking),
                    ("type_token", breakdown.type_token),
                    ("type_channel", breakdown.type_channel),
                    ("clean_recovery", breakdown.clean_recovery),
                    ("context_recovery", breakdown.context_recovery),
                    ("variance", breakdown.variance),
                    ("covariance", breakdown.covariance),
                    ("energy_gap", gap),
                    ("embedding_std", embedding_std),
                ):
                    totals[name] += float(value.detach().float().cpu())
            except torch.OutOfMemoryError as error:
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 2**30
                    reserved = torch.cuda.memory_reserved(device) / 2**30
                    peak = torch.cuda.max_memory_allocated(device) / 2**30
                    raise RuntimeError(
                        "MG-CASF CUDA OOM after exact chunked materialization; "
                        f"sample_id={sample_id} allocated_gib={allocated:.3f} "
                        f"reserved_gib={reserved:.3f} peak_gib={peak:.3f}. "
                        "Lower only execution chunk sizes; do not reduce source-set "
                        "or encoder fidelity."
                    ) from error
                raise
            finally:
                del (
                    breakdown,
                    corrupted_projected,
                    clean_projected,
                    corrupted_energy,
                    clean_energy,
                    corrupted,
                    clean,
                    teacher,
                    graph,
                )
                sample.release_attention()

        if pending:
            _optimizer_step(
                optimiser,
                model,
                scaler,
                clip_norm=float(training_config.gradient_clip_norm),
                ema_momentum=float(training_config.ema_momentum),
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


@torch.inference_mode()
def extract_frozen_rows(
    model: CausalSetFlowModel,
    dataset,
    sample_ids,
    *,
    source_config: SourceSetConfig,
    device: str | torch.device,
    precision: str = "auto",
) -> dict[str, np.ndarray]:
    """Freeze clean-graph learned energies before labels are available."""

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
            "general_energy",
            "token_energy",
            "channel_energy",
            "channel_energy_max",
            "type_energy",
        )
    }
    for sample_id in tqdm(
        list(map(str, sample_ids)),
        desc="freeze mechanism-guided Set-Flow rows",
        unit="sample",
    ):
        sample = dataset[sample_id]
        graph = values = None
        try:
            graph = extract_causal_source_set_graph(sample, source_config)
            with _autocast_context(device, amp_dtype):
                values = model.score_graph(graph, device=device)
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
                "general_energy",
                "token_energy",
                "channel_energy",
                "channel_energy_max",
                "type_energy",
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
        "general_energy",
        "token_energy",
        "channel_energy",
        "channel_energy_max",
        "type_energy",
    ):
        output[name] = np.concatenate(columns[name], axis=0).astype(np.float32)
    return output


def _backward(loss: torch.Tensor, scaler) -> None:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def _optimizer_step(
    optimiser,
    model: CausalSetFlowModel,
    scaler,
    *,
    clip_norm: float,
    ema_momentum: float,
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
    model.update_teacher(float(ema_momentum))


def _resolve_amp_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    precision = str(precision)
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but unsupported by this CUDA device")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError("precision must be auto, bf16, fp16, or fp32")


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(device: torch.device, dtype: torch.dtype | None):
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