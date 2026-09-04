"""Exact Llama message observation and intervention.

The implementation runs Llama decoder layers directly from their q/k/v/o_proj,
normalization, residual, and MLP modules. It does not depend on Transformers'
version-specific attention backend registries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

READOUT_CHUNK = 64
VALIDATION_TOKENS = 8
VALIDATED_ATTRIBUTE = "_manual_message_forward_validated"


@dataclass(frozen=True)
class ForwardCache:
    """CPU state and fixed readout needed by exact suffix reruns."""

    layer_input: dict[int, Tensor]
    layer_count: int
    query: Tensor
    target: Tensor
    runner: Tensor
    readout_direction: Tensor
    readout_bias: Tensor
    full_margin: Tensor
    baseline_target_logprob: Tensor
    baseline_entropy: Tensor
    attention_query_chunk: int | None


@dataclass(frozen=True)
class MessageGate:
    """Messages removed after softmax and before the Value sum.

    ``early_edges`` applies below ``split_layer`` and ``late_edges`` at or above
    it. ``source_mask`` applies in every layer, optionally only to
    ``source_targets``. Edge masks use model query/source coordinates.
    """

    split_layer: int
    early_edges: Tensor | None = None
    late_edges: Tensor | None = None
    source_mask: Tensor | None = None
    source_targets: Tensor | None = None


AttentionObserver = Callable[[int, Tensor, Tensor, Tensor], None]


def repeat_kv(states: Tensor, groups: int) -> Tensor:
    """Map KV heads to their grouped query heads."""

    if groups == 1:
        return states
    batch, kv_heads, tokens, head_dim = states.shape
    return (
        states[:, :, None]
        .expand(batch, kv_heads, groups, tokens, head_dim)
        .reshape(batch, kv_heads * groups, tokens, head_dim)
    )


def rotate_half(states: Tensor) -> Tensor:
    half = states.shape[-1] // 2
    return torch.cat((-states[..., half:], states[..., :half]), dim=-1)


def apply_rotary(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    if cos.ndim == 2:
        cos = cos[None]
        sin = sin[None]
    cos = cos[:, None]
    sin = sin[:, None]
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def causal_mask(tokens: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    mask = torch.full(
        (tokens, tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.triu_(diagonal=1)[None, None]


def position_embeddings(model: Any, hidden: Tensor) -> tuple[Tensor, Tensor]:
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[0].self_attn.rotary_emb
    return rotary(hidden, positions)


def deleted_edges(
    gate: MessageGate | None,
    layer: int,
    query_begin: int,
    query_end: int,
    sources: int,
    device: torch.device,
) -> Tensor | None:
    if gate is None:
        return None
    deleted = torch.zeros(
        query_end - query_begin,
        sources,
        dtype=torch.bool,
        device=device,
    )
    if gate.source_mask is not None:
        source_edges = gate.source_mask[:sources].to(
            device=device,
            dtype=torch.bool,
        )[None]
        if gate.source_targets is not None:
            targets = gate.source_targets[query_begin:query_end].to(
                device=device,
                dtype=torch.bool,
            )[:, None]
            source_edges = targets & source_edges
        deleted |= source_edges
    if layer < gate.split_layer and gate.early_edges is not None:
        deleted |= gate.early_edges[
            query_begin:query_end, :sources
        ].to(device=device, dtype=torch.bool)
    if layer >= gate.split_layer and gate.late_edges is not None:
        deleted |= gate.late_edges[
            query_begin:query_end, :sources
        ].to(device=device, dtype=torch.bool)
    return deleted if deleted.any() else None


def gated_attention(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    *,
    gate: MessageGate | None = None,
    observer: AttentionObserver | None = None,
    query_chunk: int | None = None,
) -> Tensor:
    """Compute eager attention and optionally remove selected Value messages."""

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    queries = query.shape[-2]
    sources = key.shape[-2]
    chunk = queries if query_chunk is None else int(query_chunk)
    if chunk < 1:
        raise ValueError("attention query chunk must be positive")

    outputs = []
    source_position = torch.arange(sources, device=query.device)
    for begin in range(0, queries, chunk):
        end = min(begin + chunk, queries)
        probability = torch.matmul(
            query[:, :, begin:end], key.transpose(2, 3)
        ) * scaling
        if attention_mask is None:
            future = source_position[None] > torch.arange(
                begin, end, device=query.device
            )[:, None]
            probability = probability.masked_fill(
                future[None, None], torch.finfo(probability.dtype).min
            )
        else:
            if attention_mask.shape[-2] == 1:
                chunk_mask = attention_mask[..., :, :sources]
            else:
                chunk_mask = attention_mask[..., begin:end, :sources]
            probability = probability + chunk_mask
        probability = probability.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        probability = F.dropout(
            probability,
            p=float(getattr(module, "attention_dropout", 0.0)),
            training=module.training,
        )

        if observer is not None:
            observe_chunk = getattr(observer, "observe_chunk", None)
            if callable(observe_chunk):
                observe_chunk(
                    module.layer_idx,
                    begin,
                    probability,
                    value,
                    module.o_proj.weight,
                )
            elif begin == 0 and end == queries:
                observer(module.layer_idx, probability, value, module.o_proj.weight)
            else:
                raise TypeError(
                    "chunked attention requires an observer with observe_chunk()"
                )

        deleted = deleted_edges(
            gate,
            module.layer_idx,
            begin,
            end,
            sources,
            query.device,
        )
        if deleted is not None:
            probability = probability.masked_fill(
                deleted[None, None], 0
            )
        outputs.append(torch.matmul(probability, value))
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()


def llama_attention(
    module: Any,
    hidden: Tensor,
    attention_mask: Tensor,
    rotary: tuple[Tensor, Tensor],
    gate: MessageGate | None,
    observer: AttentionObserver | None,
    attention_query_chunk: int | None,
) -> Tensor:
    """Run one Llama attention block without a Transformers backend registry."""

    batch, tokens, _ = hidden.shape
    head_dim = int(
        getattr(
            module,
            "head_dim",
            module.q_proj.out_features // module.config.num_attention_heads,
        )
    )
    query_heads = module.q_proj.out_features // head_dim
    kv_heads = module.k_proj.out_features // head_dim

    query = module.q_proj(hidden).view(
        batch, tokens, query_heads, head_dim
    ).transpose(1, 2)
    key = module.k_proj(hidden).view(
        batch, tokens, kv_heads, head_dim
    ).transpose(1, 2)
    value = module.v_proj(hidden).view(
        batch, tokens, kv_heads, head_dim
    ).transpose(1, 2)
    query, key = apply_rotary(query, key, *rotary)
    scaling = float(getattr(module, "scaling", head_dim**-0.5))
    output = gated_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        gate=gate,
        observer=observer,
        query_chunk=attention_query_chunk,
    )
    return module.o_proj(output.reshape(batch, tokens, query_heads * head_dim))


def forward_layers(
    model: Any,
    hidden: Tensor,
    start_layer: int,
    *,
    gate: MessageGate | None = None,
    observer: AttentionObserver | None = None,
    save_inputs: dict[int, Tensor] | None = None,
    save_layers: set[int] | None = None,
    attention_query_chunk: int | None = None,
) -> Tensor:
    """Run an exact full-sequence Llama decoder suffix."""

    mask = (
        None
        if attention_query_chunk is not None
        else causal_mask(hidden.shape[1], hidden.dtype, hidden.device)
    )
    rotary = position_embeddings(model, hidden)
    for layer_index, layer in enumerate(
        model.model.layers[start_layer:],
        start=start_layer,
    ):
        if save_inputs is not None and (
            save_layers is None or layer_index in save_layers
        ):
            save_inputs[layer_index] = hidden[0].detach().cpu()

        residual = hidden
        attention_output = llama_attention(
            layer.self_attn,
            layer.input_layernorm(hidden),
            mask,
            rotary,
            gate,
            observer,
            attention_query_chunk,
        )
        hidden = residual + attention_output
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.model.norm(hidden)


def validate_manual_forward(
    model: Any,
    token_ids: Tensor | Sequence[int],
) -> float:
    """Validate the registry-free decoder against native forward once."""

    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(token_ids, dtype=torch.long).flatten()
    ids = ids[:VALIDATION_TOKENS].to(device)[None]
    if ids.shape[1] < 2:
        raise ValueError("manual-forward validation requires at least two tokens")

    model.eval()
    with torch.inference_mode():
        native = model.model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        manual = forward_layers(model, model.model.embed_tokens(ids), 0)

    tolerance = {
        torch.float32: (5e-5, 5e-6),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tolerance.get(model.dtype, (2e-2, 2e-2))
    if not torch.allclose(native, manual, rtol=rtol, atol=atol):
        error = float((native.float() - manual.float()).abs().max())
        raise RuntimeError(f"manual Llama forward mismatch: max_abs={error:.6g}")

    error = float((native.float() - manual.float()).abs().max())
    setattr(model, VALIDATED_ATTRIBUTE, True)
    return error


def baseline_forward(
    model: Any,
    full_token_ids: Tensor | Sequence[int],
    response_start: int,
    observer: AttentionObserver | None = None,
    checkpoint_layers: Sequence[int] = (0,),
    attention_query_chunk: int | None = None,
) -> ForwardCache:
    """Capture a baseline and fixed target-versus-runner readout."""

    if not getattr(model, VALIDATED_ATTRIBUTE, False):
        validate_manual_forward(model, full_token_ids)
    model.eval()
    device = model.get_input_embeddings().weight.device
    full_ids = torch.as_tensor(full_token_ids, dtype=torch.long)
    source_ids = full_ids[:-1].to(device)[None]
    query = torch.arange(response_start - 1, len(full_ids) - 1)
    target = full_ids[response_start:].to(device)

    checkpoints = {0, *(int(layer) for layer in checkpoint_layers)}
    layer_input: dict[int, Tensor] = {}
    with torch.inference_mode():
        hidden = forward_layers(
            model,
            model.model.embed_tokens(source_ids),
            0,
            observer=observer,
            save_inputs=layer_input,
            save_layers=checkpoints,
            attention_query_chunk=attention_query_chunk,
        )
        response_hidden = hidden[0].index_select(0, query.to(device))
        runner_parts: list[Tensor] = []
        logprob_parts: list[Tensor] = []
        entropy_parts: list[Tensor] = []
        for begin in range(0, len(query), READOUT_CHUNK):
            end = min(begin + READOUT_CHUNK, len(query))
            logits = F.linear(
                response_hidden[begin:end],
                model.lm_head.weight,
                getattr(model.lm_head, "bias", None),
            )
            chunk_target = target[begin:end]
            float_logits = logits.float()
            log_normalizer = torch.logsumexp(float_logits, dim=1)
            logprob_parts.append(
                float_logits.gather(1, chunk_target[:, None])[:, 0]
                - log_normalizer
            )
            probability = float_logits.softmax(dim=1)
            entropy_parts.append(
                log_normalizer - (probability * float_logits).sum(dim=1)
            )
            logits.scatter_(1, chunk_target[:, None], -torch.inf)
            runner_parts.append(logits.argmax(dim=1))

        runner = torch.cat(runner_parts)
        direction = model.lm_head.weight.index_select(0, target).float()
        direction -= model.lm_head.weight.index_select(0, runner).float()
        if getattr(model.lm_head, "bias", None) is None:
            readout_bias = torch.zeros(len(target), device=device)
        else:
            readout_bias = model.lm_head.bias.index_select(0, target).float()
            readout_bias -= model.lm_head.bias.index_select(0, runner).float()
        margin = (
            torch.einsum("td,td->t", response_hidden.float(), direction)
            + readout_bias
        )

    return ForwardCache(
        layer_input=layer_input,
        layer_count=len(model.model.layers),
        query=query,
        target=target.cpu(),
        runner=runner.cpu(),
        readout_direction=direction.cpu(),
        readout_bias=readout_bias.cpu(),
        full_margin=margin.cpu(),
        baseline_target_logprob=torch.cat(logprob_parts).cpu(),
        baseline_entropy=torch.cat(entropy_parts).cpu(),
        attention_query_chunk=attention_query_chunk,
    )


def first_changed_layer(gate: MessageGate, layer_count: int) -> int | None:
    starts: list[int] = []
    if gate.source_mask is not None and gate.source_mask.any():
        starts.append(0)
    if (
        gate.early_edges is not None
        and gate.early_edges.any()
        and gate.split_layer > 0
    ):
        starts.append(0)
    if (
        gate.late_edges is not None
        and gate.late_edges.any()
        and gate.split_layer < layer_count
    ):
        starts.append(max(gate.split_layer, 0))
    return min(starts) if starts else None


def gate_to(gate: MessageGate, device: torch.device) -> MessageGate:
    def move(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.to(device=device, dtype=torch.bool)

    return MessageGate(
        split_layer=gate.split_layer,
        early_edges=move(gate.early_edges),
        late_edges=move(gate.late_edges),
        source_mask=move(gate.source_mask),
        source_targets=move(gate.source_targets),
    )


def rerun_gate(
    model: Any,
    cache: ForwardCache,
    gate: MessageGate,
    observer: AttentionObserver | None = None,
) -> Tensor:
    """Return fixed-target/fixed-runner margin changes for all response tokens."""

    start_layer = first_changed_layer(gate, cache.layer_count)
    if start_layer is None:
        return torch.zeros_like(cache.full_margin)

    model.eval()
    device = model.get_input_embeddings().weight.device
    checkpoint = start_layer if start_layer in cache.layer_input else 0
    hidden = cache.layer_input[checkpoint].to(device)[None]
    with torch.inference_mode():
        hidden = forward_layers(
            model,
            hidden,
            checkpoint,
            gate=gate_to(gate, device),
            observer=observer,
            attention_query_chunk=cache.attention_query_chunk,
        )
        response_hidden = hidden.index_select(1, cache.query.to(device))
        margin = torch.einsum(
            "btd,td->bt",
            response_hidden.float(),
            cache.readout_direction.to(device),
        )[0]
        margin += cache.readout_bias.to(device)
    return margin.cpu() - cache.full_margin
