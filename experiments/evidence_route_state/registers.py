"""Origin registers carried through one frozen Llama layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .messages import gqa_head_index, output_projection_blocks

ORIGIN_NAMES = ("evidence", "prompt", "response", "endogenous")
EVIDENCE, PROMPT, RESPONSE, ENDOGENOUS = range(len(ORIGIN_NAMES))


def initialize_registers(
    hidden: torch.Tensor,
    position: torch.Tensor,
    evidence_mask: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Assign each observed input state to exactly one information origin."""

    position = position.to(device=hidden.device, dtype=torch.long)
    prompt_position = position < response_start
    evidence = torch.zeros_like(prompt_position)
    evidence[prompt_position] = evidence_mask.to(hidden.device)[
        position[prompt_position]
    ]
    origin = torch.full_like(position, PROMPT)
    origin[evidence] = EVIDENCE
    origin[position >= response_start] = RESPONSE

    registers = torch.zeros(
        len(position),
        len(ORIGIN_NAMES),
        hidden.shape[-1],
        dtype=torch.float32,
        device=hidden.device,
    )
    registers[torch.arange(len(position), device=hidden.device), origin] = (
        hidden.float()
    )
    return registers


def rmsnorm_registers(
    registers: torch.Tensor,
    total: torch.Tensor,
    module: torch.nn.Module,
) -> torch.Tensor:
    """Apply the observed RMSNorm diagonal to every origin register."""

    state = total.float()
    scale = torch.rsqrt(
        state.square().mean(dim=-1, keepdim=True) + module.variance_epsilon
    )
    ideal_multiplier = scale * module.weight.float()
    native = module.weight * (state * scale).to(total.dtype)
    observed_multiplier = torch.where(
        state != 0,
        native.float() / state,
        ideal_multiplier,
    )
    return registers.float() * observed_multiplier[:, None]


def project_register_values(
    normalized: torch.Tensor,
    v_proj: torch.nn.Linear,
    kv_heads: int,
    head_dim: int,
    native_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project origins and assign native low-precision closure to endogenous."""

    projected = F.linear(normalized.float(), v_proj.weight.float())
    projected = projected.reshape(
        normalized.shape[0], len(ORIGIN_NAMES), kv_heads, head_dim
    )
    if native_value is not None:
        native = native_value.float().reshape(normalized.shape[0], kv_heads, head_dim)
        projected[:, ENDOGENOUS] += native - projected.sum(1)
    return projected


def route_register_values(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_weight: torch.Tensor,
    native_head_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route origin values through every GQA head and its matching ``W_O`` block."""

    heads, _queries, sources = attention.shape
    kv_heads, head_dim = values.shape[2:]
    head_to_kv = gqa_head_index(heads, kv_heads, values.device)
    values_by_head = values[:sources, :, head_to_kv]
    head_context = torch.einsum(
        "hqs,schd->qhcd",
        attention[:, :, :sources].float(),
        values_by_head.float(),
    )
    if native_head_context is not None:
        head_context[:, :, ENDOGENOUS] += (
            native_head_context.float() - head_context.sum(2)
        )
    output_blocks = output_projection_blocks(output_weight, heads, head_dim)
    register_write = torch.einsum("qhcd,hdk->qck", head_context, output_blocks)
    return register_write, head_context


def add_attention(
    registers: torch.Tensor,
    writes: torch.Tensor,
    native_mid: torch.Tensor,
) -> torch.Tensor:
    """Add routed writes and put numerical closure in the endogenous register."""

    result = registers.float() + writes.float()
    result[:, ENDOGENOUS] += native_mid.float() - result.sum(dim=1)
    return result


def add_mlp(
    registers: torch.Tensor,
    native_mlp: torch.Tensor,
    native_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add the same-token MLP write and optionally close to the native layer output."""

    result = registers.float().clone()
    result[:, ENDOGENOUS] += native_mlp.float()
    if native_output is not None:
        result[:, ENDOGENOUS] += native_output.float() - result.sum(dim=1)
    return result


def final_readout_contributions(
    normalized: torch.Tensor,
    lm_weight: torch.Tensor,
    target: torch.Tensor,
    competitor: torch.Tensor,
) -> torch.Tensor:
    """Decompose the target-versus-competitor logit margin by origin."""

    target = target.to(device=lm_weight.device, dtype=torch.long)
    competitor = competitor.to(device=lm_weight.device, dtype=torch.long)
    direction = lm_weight[target].float() - lm_weight[competitor].float()
    return torch.einsum("qcd,qd->qc", normalized.float(), direction)
