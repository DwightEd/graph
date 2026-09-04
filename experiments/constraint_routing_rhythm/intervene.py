"""Exact Value-message deletion for Llama relay motifs.

The custom attention backend applies binary gates after softmax and before the
Value sum.  Deleted probability mass is not redistributed.  A rerun starts at
the first affected decoder layer and recomputes every later attention, residual
update, and MLP while keeping the target-versus-runner readout fixed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AttentionInterface, AttentionMaskInterface

ATTENTION_BACKEND = "constraint_routing_gate"
READOUT_CHUNK = 64
VALIDATION_TOKENS = 8
VALIDATED_ATTRIBUTE = "_constraint_routing_backend_validated"


@dataclass(frozen=True)
class ForwardCache:
    """CPU state needed for an exact suffix rerun at any decoder layer."""

    layer_input: dict[int, Tensor]  # checkpoint layer -> [source, hidden]
    layer_count: int
    query: Tensor  # [response]
    target: Tensor  # [response]
    runner: Tensor  # [response]
    readout_direction: Tensor  # [response, hidden]
    full_margin: Tensor  # [response]
    baseline_target_logprob: Tensor  # FP32 [response]
    baseline_entropy: Tensor  # FP32 [response], nats


@dataclass(frozen=True)
class RelayGate:
    """Token-message sets deleted on either side of a layer split.

    Edge matrices are indexed ``[target, source]``.  Upstream edges are gated
    in layers ``< split_layer`` and downstream edges in layers
    ``>= split_layer``.  Evidence sources, when enabled, are gated in every
    layer and for every target and query head.
    """

    upstream_edges: Tensor  # bool [source, source]
    downstream_edges: Tensor  # bool [source, source]
    split_layer: int
    cut_evidence: bool
    cut_upstream: bool
    cut_downstream: bool
    evidence_mask: Tensor  # bool [source]
    evidence_targets: Tensor | None = None  # optional bool [target]


AttentionObserver = Callable[[int, Tensor, Tensor, Tensor], None]


def repeat_kv(states: Tensor, groups: int) -> Tensor:
    """Map KV heads to their query heads for grouped-query attention."""

    batch, kv_heads, tokens, head_dim = states.shape
    if groups == 1:
        return states
    return (
        states[:, :, None]
        .expand(batch, kv_heads, groups, tokens, head_dim)
        .reshape(batch, kv_heads * groups, tokens, head_dim)
    )


def gated_eager_attention(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    scaling: float,
    dropout: float = 0.0,
    relay_gate: RelayGate | None = None,
    relay_observer: AttentionObserver | None = None,
    **_: Any,
) -> tuple[Tensor, None]:
    """Llama eager attention with an unnormalized post-softmax binary gate."""

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    probability = torch.matmul(query, key.transpose(2, 3)) * scaling
    probability = probability + attention_mask[..., : key.shape[-2]]
    probability = probability.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
    probability = F.dropout(probability, p=dropout, training=module.training)

    # The observer consumes the native probabilities online.  Returning None
    # prevents Transformers from retaining a full attention matrix per layer.
    if relay_observer is not None:
        relay_observer(module.layer_idx, probability, value, module.o_proj.weight)

    if relay_gate is not None:
        deleted = torch.zeros(
            probability.shape[-2:], dtype=torch.bool, device=probability.device
        )
        if relay_gate.cut_evidence:
            evidence_edges = relay_gate.evidence_mask[None, :]
            if relay_gate.evidence_targets is not None:
                evidence_edges = relay_gate.evidence_targets[:, None] & evidence_edges
            deleted |= evidence_edges
        if relay_gate.cut_upstream and module.layer_idx < relay_gate.split_layer:
            deleted |= relay_gate.upstream_edges
        if relay_gate.cut_downstream and module.layer_idx >= relay_gate.split_layer:
            deleted |= relay_gate.downstream_edges
        probability.masked_fill_(deleted[None, None], 0)

    output = torch.matmul(probability, value)
    return output.transpose(1, 2).contiguous(), None


def install_attention_backend(model: Any) -> None:
    """Use the same functional attention implementation in all model runs."""

    AttentionInterface.register(ATTENTION_BACKEND, gated_eager_attention)
    AttentionMaskInterface.register(
        ATTENTION_BACKEND,
        AttentionMaskInterface()["eager"],
    )
    model.set_attn_implementation(ATTENTION_BACKEND)


def causal_mask(tokens: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Build the explicit mask required by a custom Transformers backend."""

    mask = torch.full(
        (tokens, tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.triu_(diagonal=1)[None, None]


def position_embeddings(model: Any, hidden: Tensor) -> tuple[Tensor, Tensor]:
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
    return model.model.rotary_emb(hidden[:1], positions)


def forward_layers(
    model: Any,
    hidden: Tensor,
    start_layer: int,
    *,
    gate: RelayGate | None = None,
    observer: AttentionObserver | None = None,
    save_inputs: dict[int, Tensor] | None = None,
    save_layers: set[int] | None = None,
) -> Tensor:
    """Run a decoder suffix, optionally filling requested CPU checkpoints."""

    mask = causal_mask(hidden.shape[1], hidden.dtype, hidden.device)
    rotary = position_embeddings(model, hidden)
    for layer_index, layer in enumerate(
        model.model.layers[start_layer:], start=start_layer
    ):
        if save_inputs is not None and (
            save_layers is None or layer_index in save_layers
        ):
            save_inputs[layer_index] = hidden[0].detach().cpu()
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_embeddings=rotary,
            use_cache=False,
            relay_gate=gate,
            relay_observer=observer,
        )
    return model.model.norm(hidden)


def validate_attention_backend(
    model: Any,
    token_ids: Tensor | Sequence[int],
) -> float:
    """Check native eager against an executed all-one custom gate once.

    This check uses an unpadded full sequence without a KV cache, matching the
    experiment's scope.  At most eight tokens are used, so it does not
    reproduce the sample's quadratic peak.
    """

    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(token_ids, dtype=torch.long).flatten()
    ids = ids[:VALIDATION_TOKENS].to(device)[None]
    if ids.shape[1] < 2:
        raise ValueError("attention validation requires at least two tokens")

    empty = torch.zeros(ids.shape[1], ids.shape[1], dtype=torch.bool, device=device)
    all_one = RelayGate(
        upstream_edges=empty,
        downstream_edges=empty,
        split_layer=len(model.model.layers) // 2,
        cut_evidence=True,
        cut_upstream=True,
        cut_downstream=True,
        evidence_mask=torch.zeros(ids.shape[1], dtype=torch.bool, device=device),
    )

    model.eval()
    model.set_attn_implementation("eager")
    with torch.inference_mode():
        native = model.model(input_ids=ids, use_cache=False).last_hidden_state

    install_attention_backend(model)
    with torch.inference_mode():
        custom = model.model(
            input_ids=ids,
            use_cache=False,
            relay_gate=all_one,
        ).last_hidden_state

    tolerance = {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }
    rtol, atol = tolerance[model.dtype]
    if not torch.allclose(native, custom, rtol=rtol, atol=atol):
        error = float((native.float() - custom.float()).abs().max())
        raise RuntimeError(
            f"custom attention does not reproduce native eager; max_abs={error:.6g}"
        )

    error = float((native.float() - custom.float()).abs().max())
    setattr(model, VALIDATED_ATTRIBUTE, True)
    return error


def baseline_forward(
    model: Any,
    full_token_ids: Tensor | Sequence[int],
    response_start: int,
    observer: AttentionObserver | None = None,
    checkpoint_layers: Sequence[int] = (0,),
) -> ForwardCache:
    """Run a teacher-forced baseline with a fixed, FP32 margin readout.

    The runner is selected once from native-dtype baseline logits.  Target and
    runner weights are then converted separately to FP32 before subtraction;
    every intervention reuses that same readout direction.  Target log-probability
    and entropy are computed in FP32 from those same chunked baseline logits.
    """

    if getattr(model.config, "attention_bias", False):
        raise ValueError(
            "the functional message decomposition requires bias-free attention"
        )
    if getattr(model.lm_head, "bias", None) is not None:
        raise ValueError("the fixed margin readout requires a bias-free lm_head")
    if not getattr(model, VALIDATED_ATTRIBUTE, False):
        validate_attention_backend(model, full_token_ids)
    install_attention_backend(model)
    model.eval()
    device = model.get_input_embeddings().weight.device
    full_ids = torch.as_tensor(full_token_ids, dtype=torch.long)
    source_ids = full_ids[:-1].to(device)[None]
    query = torch.arange(response_start - 1, len(full_ids) - 1)
    target = full_ids[response_start:].to(device)

    layer_count = len(model.model.layers)
    checkpoints = {int(layer) for layer in checkpoint_layers}
    if not checkpoints <= set(range(layer_count)):
        raise ValueError("checkpoint layer is outside the decoder")
    checkpoints.add(0)
    layer_input: dict[int, Tensor] = {}
    with torch.inference_mode():
        hidden = model.model.embed_tokens(source_ids)
        hidden = forward_layers(
            model,
            hidden,
            0,
            observer=observer,
            save_inputs=layer_input,
            save_layers=checkpoints,
        )
        response_hidden = hidden[0].index_select(0, query.to(device))
        runner_parts = []
        target_logprob_parts = []
        entropy_parts = []
        for begin in range(0, len(query), READOUT_CHUNK):
            end = min(begin + READOUT_CHUNK, len(query))
            logits = F.linear(response_hidden[begin:end], model.lm_head.weight)
            chunk_target = target[begin:end]
            float_logits = logits.float()
            log_normalizer = torch.logsumexp(float_logits, dim=1)
            target_logprob_parts.append(
                float_logits.gather(1, chunk_target[:, None])[:, 0] - log_normalizer
            )
            probability = float_logits.softmax(dim=1)
            probability.mul_(float_logits)
            entropy_parts.append(log_normalizer - probability.sum(dim=1))
            logits.scatter_(1, chunk_target[:, None], -torch.inf)
            runner_parts.append(logits.argmax(dim=1))
        runner = torch.cat(runner_parts)
        target_logprob = torch.cat(target_logprob_parts)
        entropy = torch.cat(entropy_parts)
        direction = model.lm_head.weight.index_select(0, target).float()
        direction = direction - model.lm_head.weight.index_select(0, runner).float()
        margin = torch.einsum("td,td->t", response_hidden.float(), direction.float())

    return ForwardCache(
        layer_input=layer_input,
        layer_count=layer_count,
        query=query,
        target=target.cpu(),
        runner=runner.cpu(),
        readout_direction=direction.cpu(),
        full_margin=margin.cpu(),
        baseline_target_logprob=target_logprob.cpu(),
        baseline_entropy=entropy.cpu(),
    )


def first_changed_layer(gate: RelayGate, layer_count: int) -> int | None:
    """Find the earliest layer whose Value sum differs from the baseline."""

    starts: list[int] = []
    if gate.cut_evidence and gate.evidence_mask.any():
        starts.append(0)
    if gate.cut_upstream and gate.upstream_edges.any() and gate.split_layer > 0:
        starts.append(0)
    if (
        gate.cut_downstream
        and gate.downstream_edges.any()
        and gate.split_layer < layer_count
    ):
        starts.append(max(gate.split_layer, 0))
    return min(starts) if starts else None


def gate_to(gate: RelayGate, device: torch.device) -> RelayGate:
    """Move only the three small gate masks needed by the active sample."""

    return RelayGate(
        upstream_edges=gate.upstream_edges.to(device=device, dtype=torch.bool),
        downstream_edges=gate.downstream_edges.to(device=device, dtype=torch.bool),
        split_layer=gate.split_layer,
        cut_evidence=gate.cut_evidence,
        cut_upstream=gate.cut_upstream,
        cut_downstream=gate.cut_downstream,
        evidence_mask=gate.evidence_mask.to(device=device, dtype=torch.bool),
        evidence_targets=(
            None
            if gate.evidence_targets is None
            else gate.evidence_targets.to(device=device, dtype=torch.bool)
        ),
    )


def rerun_gate(
    model: Any,
    cache: ForwardCache,
    gate: RelayGate,
    observer: AttentionObserver | None = None,
) -> Tensor:
    """Return fixed-target/fixed-runner margin changes for all response tokens."""

    start_layer = first_changed_layer(gate, cache.layer_count)
    if start_layer is None:
        return torch.zeros_like(cache.full_margin)

    install_attention_backend(model)
    model.eval()
    device = model.get_input_embeddings().weight.device
    device_gate = gate_to(gate, device)
    checkpoint = start_layer if start_layer in cache.layer_input else 0
    hidden = cache.layer_input[checkpoint].to(device)[None]

    with torch.inference_mode():
        hidden = forward_layers(
            model,
            hidden,
            checkpoint,
            gate=device_gate,
            observer=observer,
        )
        response_hidden = hidden.index_select(1, cache.query.to(device))
        margin = torch.einsum(
            "btd,td->bt",
            response_hidden.float(),
            cache.readout_direction.to(device).float(),
        )[0]
    return margin.cpu() - cache.full_margin
