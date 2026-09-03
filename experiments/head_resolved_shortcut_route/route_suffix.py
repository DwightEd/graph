"""Reverse the observed same-position suffix of a bias-free Llama."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObservedSuffix:
    """Native gates needed after a strict ``source -> predictor`` arrival."""

    query_position: Tensor  # [event]
    attention: Sequence[Tensor]  # layer: [event, query_head, source]
    attention_rms_multiplier: Sequence[Tensor]  # layer: [event, hidden]
    mlp_rms_multiplier: Sequence[Tensor]  # layer: [event, hidden]
    native_gate: Sequence[Tensor]  # layer: [event, intermediate]
    native_up: Sequence[Tensor]  # layer: [event, intermediate]
    value_weight: Sequence[Tensor]  # layer: [kv_head * head_dim, hidden]
    output_weight: Sequence[Tensor]  # layer: [hidden, query_head * head_dim]
    gate_weight: Sequence[Tensor]  # layer: [intermediate, hidden]
    up_weight: Sequence[Tensor]  # layer: [intermediate, hidden]
    down_weight: Sequence[Tensor]  # layer: [hidden, intermediate]
    final_rms_multiplier: Tensor  # [event, hidden]
    readout_direction: Tensor  # [event, hidden]


@dataclass(frozen=True)
class SuffixAdjoints:
    """Adjoints at every attention-write boundary and at predictor input."""

    attention_write: tuple[Tensor, ...]  # layer: [event, hidden]
    layer_output: tuple[Tensor, ...]  # layer: [event, hidden]
    input: Tensor  # [event, hidden]


def symmetric_swiglu_root_write(
    normalized_roots: Tensor,
    native_gate: Tensor,
    native_up: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Allocate one native SwiGLU write across normalized observed roots.

    The native gate and up activations stay fixed.  Together with the shared
    RMS multiplier, this is the forward operator whose transpose is
    implemented by :func:`symmetric_swiglu_adjoint`.
    """

    normalized = normalized_roots.float()
    gate_roots = normalized @ gate_weight.float().T
    up_roots = normalized @ up_weight.float().T
    gate_ratio = torch.sigmoid(native_gate.float()).unsqueeze(1)
    allocated = (
        0.5
        * gate_ratio
        * (
            gate_roots * native_up.float().unsqueeze(1)
            + native_gate.float().unsqueeze(1) * up_roots
        )
    )
    return allocated @ down_weight.float().T


def reverse_observed_suffix(observed: ObservedSuffix) -> SuffixAdjoints:
    """Pull the fixed target readout through MLP, self-attention, and residual."""

    validate_suffix(observed)
    layer_count = len(observed.attention)
    lam = observed.final_rms_multiplier.float() * observed.readout_direction.float()
    arrival: list[Tensor | None] = [None] * layer_count
    layer_output: list[Tensor | None] = [None] * layer_count
    for layer in range(layer_count - 1, -1, -1):
        layer_output[layer] = lam
        eta = symmetric_swiglu_adjoint(
            lam,
            observed.mlp_rms_multiplier[layer],
            observed.native_gate[layer],
            observed.native_up[layer],
            observed.gate_weight[layer],
            observed.up_weight[layer],
            observed.down_weight[layer],
        )
        arrival[layer] = eta
        lam = self_attention_adjoint(
            eta,
            observed.attention[layer],
            observed.query_position,
            observed.attention_rms_multiplier[layer],
            observed.value_weight[layer],
            observed.output_weight[layer],
        )
    return SuffixAdjoints(
        tuple(arrival),  # type: ignore[arg-type]
        tuple(layer_output),  # type: ignore[arg-type]
        lam,
    )


def symmetric_swiglu_adjoint(
    downstream: Tensor,
    multiplier: Tensor,
    native_gate: Tensor,
    native_up: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Transpose residual plus the observed symmetric SwiGLU allocation."""

    lam = downstream.float()
    intermediate = lam @ down_weight.float()
    sigmoid = torch.sigmoid(native_gate.float())
    gate_branch = (sigmoid * native_up.float() * intermediate) @ gate_weight.float()
    up_branch = (sigmoid * native_gate.float() * intermediate) @ up_weight.float()
    return lam + 0.5 * multiplier.float() * (gate_branch + up_branch)


def self_attention_adjoint(
    downstream: Tensor,
    attention: Tensor,
    query_position: Tensor,
    multiplier: Tensor,
    value_weight: Tensor,
    output_weight: Tensor,
) -> Tensor:
    """Transpose residual plus the observed predictor-self AVWO message."""

    events, heads, _ = attention.shape
    head_dim = output_weight.shape[1] // heads
    kv_heads = value_weight.shape[0] // head_dim
    repeats = heads // kv_heads
    blocks = output_weight.float().reshape(-1, heads, head_dim).permute(1, 0, 2)
    head_adjoint = torch.einsum("ed,hdk->ehk", downstream.float(), blocks)
    query = query_position.to(device=attention.device, dtype=torch.long)
    self_weight = attention[
        torch.arange(events, device=attention.device)[:, None],
        torch.arange(heads, device=attention.device)[None],
        query[:, None],
    ]
    grouped = (self_weight[..., None] * head_adjoint).reshape(
        events, kv_heads, repeats, head_dim
    )
    value_adjoint = grouped.sum(dim=2).flatten(1) @ value_weight.float()
    return downstream.float() + multiplier.float() * value_adjoint


def injection_contribution(input_roots: Tensor, input_adjoint: Tensor) -> Tensor:
    """Project the predictor input roots through the complete local suffix."""

    if input_roots.ndim != 3 or input_roots.shape[0] != len(input_adjoint):
        raise ValueError("input roots must be [event, root, hidden]")
    if input_roots.shape[-1] != input_adjoint.shape[-1]:
        raise ValueError("input roots and suffix adjoint must share hidden size")
    return torch.einsum("erd,ed->er", input_roots.float(), input_adjoint.float())


def validate_suffix(observed: ObservedSuffix) -> None:
    """Validate only shape identities that could silently broadcast."""

    layer_count = len(observed.attention)
    sequences = (
        observed.attention_rms_multiplier,
        observed.mlp_rms_multiplier,
        observed.native_gate,
        observed.native_up,
        observed.value_weight,
        observed.output_weight,
        observed.gate_weight,
        observed.up_weight,
        observed.down_weight,
    )
    if not layer_count or any(len(sequence) != layer_count for sequence in sequences):
        raise ValueError("observed suffix has inconsistent layer sequences")
    events = len(observed.query_position)
    if observed.query_position.ndim != 1:
        raise ValueError("query_position must be one vector")
    if observed.final_rms_multiplier.shape != observed.readout_direction.shape:
        raise ValueError("final RMS multiplier and readout direction must align")
    if observed.final_rms_multiplier.shape[0] != events:
        raise ValueError("final suffix tensors must contain one row per event")
    hidden = observed.final_rms_multiplier.shape[1]
    for layer in range(layer_count):
        attention = observed.attention[layer]
        if attention.ndim != 3 or attention.shape[0] != events:
            raise ValueError("attention must be [event, head, source]")
        heads, sources = attention.shape[1:]
        query = observed.query_position.to(attention.device)
        if (query < 0).any() or (query >= sources).any():
            raise ValueError("predictor self must exist in every attention row")
        output_weight = observed.output_weight[layer]
        value_weight = observed.value_weight[layer]
        if output_weight.shape[0] != hidden or output_weight.shape[1] % heads:
            raise ValueError("output projection does not match hidden/query heads")
        head_dim = output_weight.shape[1] // heads
        if value_weight.shape[1] != hidden or value_weight.shape[0] % head_dim:
            raise ValueError("value projection does not match GQA geometry")
        kv_heads = value_weight.shape[0] // head_dim
        if heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        intermediate = observed.native_gate[layer].shape[1]
        shapes = (
            (observed.attention_rms_multiplier[layer], (events, hidden)),
            (observed.mlp_rms_multiplier[layer], (events, hidden)),
            (observed.native_gate[layer], (events, intermediate)),
            (observed.native_up[layer], (events, intermediate)),
            (observed.gate_weight[layer], (intermediate, hidden)),
            (observed.up_weight[layer], (intermediate, hidden)),
            (observed.down_weight[layer], (hidden, intermediate)),
        )
        if any(value.shape != shape for value, shape in shapes):
            raise ValueError("observed suffix layer tensors do not align")
