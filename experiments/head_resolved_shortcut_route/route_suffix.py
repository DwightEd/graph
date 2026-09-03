"""Reverse the observed same-position suffix of a bias-free Llama."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

_SWIGLU_TOKEN_BLOCK = 128
_SWIGLU_INTERMEDIATE_BLOCK = 1024


def _validate_block_sizes(token_block_size: int, intermediate_block_size: int) -> None:
    if token_block_size <= 0 or intermediate_block_size <= 0:
        raise ValueError("SwiGLU block sizes must be positive")


def _float_block(value: Tensor, device: torch.device) -> Tensor:
    """Move only one contiguous block to the compute device in FP32."""

    return value.contiguous().to(device=device, dtype=torch.float32)


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
    *,
    token_block_size: int = _SWIGLU_TOKEN_BLOCK,
    intermediate_block_size: int = _SWIGLU_INTERMEDIATE_BLOCK,
) -> Tensor:
    """Allocate one native SwiGLU write across normalized observed roots.

    The native gate and up activations stay fixed.  Together with the shared
    RMS multiplier, this is the forward operator whose transpose is
    implemented by :func:`symmetric_swiglu_adjoint`.
    """

    _validate_block_sizes(token_block_size, intermediate_block_size)
    tokens, roots, hidden = normalized_roots.shape
    intermediate = native_gate.shape[1]
    device = down_weight.device
    token_ranges = tuple(
        (start, min(start + token_block_size, tokens))
        for start in range(0, tokens, token_block_size)
    )
    allocated = [
        torch.zeros(
            stop - start,
            roots,
            hidden,
            dtype=torch.float32,
            device=device,
        )
        for start, stop in token_ranges
    ]
    for start in range(0, intermediate, intermediate_block_size):
        stop = min(start + intermediate_block_size, intermediate)
        gate_block = _float_block(gate_weight[start:stop], device)
        up_block = _float_block(up_weight[start:stop], device)
        down_block = _float_block(down_weight[:, start:stop], device)
        for block, (token_start, token_stop) in enumerate(token_ranges):
            normalized = _float_block(normalized_roots[token_start:token_stop], device)
            native_gate_block = _float_block(
                native_gate[token_start:token_stop, start:stop], device
            )
            native_up_block = _float_block(
                native_up[token_start:token_stop, start:stop], device
            )
            gate_ratio = torch.sigmoid(native_gate_block).unsqueeze(1)
            gate_roots = torch.nn.functional.linear(normalized, gate_block)
            up_roots = torch.nn.functional.linear(normalized, up_block)
            interaction = (0.5 * gate_ratio) * (
                gate_roots * native_up_block.unsqueeze(1)
                + native_gate_block.unsqueeze(1) * up_roots
            )
            contribution = torch.nn.functional.linear(interaction, down_block)
            allocated[block] = allocated[block] + contribution
    return torch.cat(allocated, dim=0)


def reverse_observed_suffix(
    observed: ObservedSuffix,
    *,
    token_block_size: int = _SWIGLU_TOKEN_BLOCK,
    intermediate_block_size: int = _SWIGLU_INTERMEDIATE_BLOCK,
) -> SuffixAdjoints:
    """Pull the fixed target readout through MLP, self-attention, and residual."""

    validate_suffix(observed)
    _validate_block_sizes(token_block_size, intermediate_block_size)
    layer_count = len(observed.attention)
    device = observed.down_weight[-1].device
    lam = _float_block(observed.final_rms_multiplier, device) * _float_block(
        observed.readout_direction, device
    )
    arrival: list[Tensor | None] = [None] * layer_count
    layer_output: list[Tensor | None] = [None] * layer_count
    for layer in range(layer_count - 1, -1, -1):
        layer_output[layer] = lam.detach().cpu()
        eta = symmetric_swiglu_adjoint(
            lam,
            observed.mlp_rms_multiplier[layer],
            observed.native_gate[layer],
            observed.native_up[layer],
            observed.gate_weight[layer],
            observed.up_weight[layer],
            observed.down_weight[layer],
            token_block_size=token_block_size,
            intermediate_block_size=intermediate_block_size,
        )
        arrival[layer] = eta.detach().cpu()
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
        lam.detach().cpu(),
    )


def symmetric_swiglu_adjoint(
    downstream: Tensor,
    multiplier: Tensor,
    native_gate: Tensor,
    native_up: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    token_block_size: int = _SWIGLU_TOKEN_BLOCK,
    intermediate_block_size: int = _SWIGLU_INTERMEDIATE_BLOCK,
) -> Tensor:
    """Transpose residual plus the observed symmetric SwiGLU allocation."""

    _validate_block_sizes(token_block_size, intermediate_block_size)
    events, hidden = downstream.shape
    intermediate = native_gate.shape[1]
    device = down_weight.device
    lam = _float_block(downstream, device)
    token_ranges = tuple(
        (start, min(start + token_block_size, events))
        for start in range(0, events, token_block_size)
    )
    pulled_back = [
        torch.zeros(stop - start, hidden, dtype=torch.float32, device=device)
        for start, stop in token_ranges
    ]
    for start in range(0, intermediate, intermediate_block_size):
        stop = min(start + intermediate_block_size, intermediate)
        gate_block = _float_block(gate_weight[start:stop], device)
        up_block = _float_block(up_weight[start:stop], device)
        down_block = _float_block(down_weight[:, start:stop], device)
        for block, (token_start, token_stop) in enumerate(token_ranges):
            native_gate_block = _float_block(
                native_gate[token_start:token_stop, start:stop], device
            )
            native_up_block = _float_block(
                native_up[token_start:token_stop, start:stop], device
            )
            intermediate_adjoint = lam[token_start:token_stop] @ down_block
            sigmoid = torch.sigmoid(native_gate_block)
            gate_branch = (
                sigmoid * native_up_block * intermediate_adjoint
            ) @ gate_block
            up_branch = (sigmoid * native_gate_block * intermediate_adjoint) @ up_block
            pulled_back[block] = pulled_back[block] + gate_branch + up_branch
    scale = _float_block(multiplier, device)
    return lam + 0.5 * scale * torch.cat(pulled_back, dim=0)


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
    device = output_weight.device
    head_dim = output_weight.shape[1] // heads
    kv_heads = value_weight.shape[0] // head_dim
    repeats = heads // kv_heads
    blocks = (
        _float_block(output_weight, device)
        .reshape(-1, heads, head_dim)
        .permute(1, 0, 2)
    )
    lam = _float_block(downstream, device)
    head_adjoint = torch.einsum("ed,hdk->ehk", lam, blocks)
    query = query_position.to(device=attention.device, dtype=torch.long)
    self_weight = attention[
        torch.arange(events, device=attention.device)[:, None],
        torch.arange(heads, device=attention.device)[None],
        query[:, None],
    ].to(device=device, dtype=torch.float32)
    grouped = (self_weight[..., None] * head_adjoint).reshape(
        events, kv_heads, repeats, head_dim
    )
    value_adjoint = grouped.sum(dim=2).flatten(1) @ _float_block(value_weight, device)
    return lam + _float_block(multiplier, device) * value_adjoint


def injection_contribution(input_roots: Tensor, input_adjoint: Tensor) -> Tensor:
    """Project the predictor input roots through the complete local suffix."""

    if input_roots.ndim != 3 or input_roots.shape[0] != len(input_adjoint):
        raise ValueError("input roots must be [event, root, hidden]")
    if input_roots.shape[-1] != input_adjoint.shape[-1]:
        raise ValueError("input roots and suffix adjoint must share hidden size")
    device = input_adjoint.device
    return torch.einsum(
        "erd,ed->er",
        _float_block(input_roots, device),
        _float_block(input_adjoint, device),
    )


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
