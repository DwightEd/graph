"""One-pass native Llama observer for the shortcut-route audit.

The observer runs one complete teacher-forced sequence.  It intentionally has
no KV-cache/chunked mode yet: splitting the sequence would require a rooted
cache with a separately tested chunk-invariance contract.  The returned
operators contain no hallucination labels and keep model weights as references
to the frozen observer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .route_pipeline import CapturedRouteOperators
from .route_shortcut import (
    EVIDENCE,
    NUMERIC,
    QUESTION,
    RESPONSE,
    PredictionEvents,
    prediction_events,
)
from .route_suffix import ObservedSuffix, symmetric_swiglu_root_write

CLOSURE_RELATIVE_LIMIT = 1e-3
_ROOT_TOKEN_BLOCK_SIZE = 128
_ROOT_INTERMEDIATE_BLOCK_SIZE = 1024


def endpoint_boundary_error(boundary_error: list[Tensor], query: Tensor) -> Tensor:
    """Return the largest local operator mismatch at each predictor endpoint.

    Cross-token numerical ancestry is already carried by the explicit N root
    and its total variation.  A prefix maximum here would additionally let a
    terminal-stage error at an older position invalidate unrelated later
    predictors even though that stage has no forward path to them.
    """

    if not boundary_error:
        raise ValueError("at least one boundary error is required")
    stage_error = torch.stack(boundary_error)
    if stage_error.ndim != 2 or query.ndim != 1:
        raise ValueError("boundary errors and query positions must be aligned")
    if (query < 0).any() or (query >= stage_error.shape[1]).any():
        raise ValueError("query positions must index the boundary-error sequence")
    return stage_error.index_select(1, query.long()).max(dim=0).values


class NativeRouteObserver:
    """Capture observed E/Q/R/N operators from one frozen Llama forward."""

    def __init__(
        self,
        model: Any,
        *,
        root_token_block_size: int = _ROOT_TOKEN_BLOCK_SIZE,
        root_intermediate_block_size: int = _ROOT_INTERMEDIATE_BLOCK_SIZE,
    ) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        elif getattr(self.model.config, "_attn_implementation", None) != "eager":
            raise ValueError("the route observer requires eager attention")

        self.layers = tuple(self.model.model.layers)
        config = self.model.config
        if str(getattr(config, "model_type", "")) != "llama":
            raise ValueError("the current observer is defined for native Llama only")
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.hidden = int(config.hidden_size)
        self.head_dim = self.layers[0].self_attn.v_proj.out_features // self.kv_heads
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            raise ValueError("pretraining_tp must be 1 for exact projection blocks")
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if str(getattr(config, "hidden_act", "silu")) != "silu":
            raise ValueError("the symmetric observed MLP operator requires SiLU")
        if (
            isinstance(root_token_block_size, bool)
            or not isinstance(root_token_block_size, int)
            or root_token_block_size <= 0
            or isinstance(root_intermediate_block_size, bool)
            or not isinstance(root_intermediate_block_size, int)
            or root_intermediate_block_size <= 0
        ):
            raise ValueError("root block sizes must be positive integers")
        # These blocks partition only the auxiliary FP32 root ledger.  They do
        # not split or approximate the native full-sequence model forward.
        self._root_token_block_size = root_token_block_size
        self._root_intermediate_block_size = root_intermediate_block_size
        self._validate_bias_free()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        root_token_block_size: int = _ROOT_TOKEN_BLOCK_SIZE,
        root_intermediate_block_size: int = _ROOT_INTERMEDIATE_BLOCK_SIZE,
    ) -> NativeRouteObserver:
        """Load a local frozen causal LM with native eager attention."""

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        return cls(
            model,
            root_token_block_size=root_token_block_size,
            root_intermediate_block_size=root_intermediate_block_size,
        )

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def capture(
        self,
        token_ids: Sequence[int] | Tensor,
        response_start: int,
        evidence_mask: Sequence[bool] | Tensor,
    ) -> CapturedRouteOperators:
        """Observe all predictors ``q=P-1+t`` in one full-sequence call.

        ``token_ids`` includes every response target.  The model receives only
        ``token_ids[:-1]``, so the final response token and every event's own
        target embedding are absent from that event's causal predictor.
        ``evidence_mask`` aligns to the prompt, not to response tokens.
        """

        full_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        prompt_evidence = torch.as_tensor(evidence_mask, dtype=torch.bool, device="cpu")
        if full_ids.ndim != 1 or not 0 < response_start < len(full_ids):
            raise ValueError("token_ids and response_start must define a response")
        if prompt_evidence.shape != (response_start,):
            raise ValueError("evidence_mask must align exactly with the prompt")

        source_ids = full_ids[:-1]
        source_count = len(source_ids)
        source_evidence = torch.zeros(source_count, dtype=torch.bool)
        source_evidence[:response_start] = prompt_evidence
        cpu_events = prediction_events(full_ids, response_start)
        events = PredictionEvents(
            query_position=cpu_events.query_position.to(self.device),
            prediction_position=cpu_events.prediction_position.to(self.device),
            target_token_id=cpu_events.target_token_id.to(self.device),
        )
        query = events.query_position
        model_ids = source_ids.to(self.device).unsqueeze(0)

        roots: Tensor | None = None
        initial_roots: Tensor | None = None
        terminal_roots: Tensor | None = None
        final_multiplier: Tensor | None = None
        boundary_error: list[Tensor] = []

        root_values: list[Tensor] = []
        event_attention: list[Tensor] = []
        attention_multiplier: list[Tensor] = []
        mlp_multiplier: list[Tensor] = []
        native_gate: list[Tensor] = []
        native_up: list[Tensor] = []
        self_value_numeric_input: list[Tensor] = []
        post_attention_numeric_write: list[Tensor] = []
        layer_numeric_write: list[Tensor] = []
        final_rms_numeric_write: Tensor | None = None

        layer_input: list[Tensor | None] = [None] * len(self.layers)
        attention_norm: list[Tensor | None] = [None] * len(self.layers)
        value_output: list[Tensor | None] = [None] * len(self.layers)
        attention_write: list[Tensor | None] = [None] * len(self.layers)
        attention_weight: list[Tensor | None] = [None] * len(self.layers)
        mlp_input: list[Tensor | None] = [None] * len(self.layers)
        mlp_norm: list[Tensor | None] = [None] * len(self.layers)
        gate_value: list[Tensor | None] = [None] * len(self.layers)
        up_value: list[Tensor | None] = [None] * len(self.layers)
        mlp_write: list[Tensor | None] = [None] * len(self.layers)
        handles = []

        def closure_delta(reconstructed: Tensor, native: Tensor) -> Tensor:
            target = native.float()
            estimate = reconstructed.float()
            difference = target - estimate
            target_norm = target.flatten(1).norm(dim=1)
            estimate_norm = estimate.flatten(1).norm(dim=1)
            scale = torch.maximum(target_norm, estimate_norm)
            scale = scale.clamp_min(1.0)
            boundary_error.append(difference.flatten(1).norm(dim=1) / scale)
            return difference

        def close_numeric(parts: Tensor, native: Tensor) -> tuple[Tensor, Tensor]:
            closed = parts.float().clone()
            difference = closure_delta(closed.sum(dim=1), native)
            closed[:, NUMERIC] += difference
            return closed, difference

        def embedding_hook(_module: Any, _args: Any, output: Tensor) -> None:
            nonlocal roots, initial_roots
            embedded = output[0].detach().float()
            current = torch.zeros(
                source_count,
                4,
                self.hidden,
                dtype=torch.float32,
                device=self.device,
            )
            evidence = source_evidence.to(self.device)
            prompt = torch.arange(source_count, device=self.device) < response_start
            current[evidence, EVIDENCE] = embedded[evidence]
            current[prompt & ~evidence, QUESTION] = embedded[prompt & ~evidence]
            current[~prompt, RESPONSE] = embedded[~prompt]
            roots = current
            initial_roots = current.index_select(0, query).detach().cpu()

        def layer_input_hook(index: int):
            def hook(_module: Any, args: tuple[Tensor, ...]) -> None:
                layer_input[index] = args[0][0].detach()

            return hook

        def norm_hook(index: int, *, attention: bool):
            def hook(_module: Any, args: tuple[Tensor, ...], output: Tensor) -> None:
                if attention:
                    attention_norm[index] = output[0].detach()
                else:
                    mlp_input[index] = args[0][0].detach()
                    mlp_norm[index] = output[0].detach()

            return hook

        def attention_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> Any:
                if not isinstance(output, tuple) or output[1] is None:
                    raise RuntimeError("eager attention weights were not returned")
                attention_write[index] = output[0][0].detach()
                attention_weight[index] = output[1][0].detach()
                parts = list(output)
                parts[1] = None
                return tuple(parts)

            return hook

        def projection_hook(storage: list[Tensor | None], index: int):
            def hook(_module: Any, _args: Any, output: Tensor) -> None:
                storage[index] = output[0].detach()

            return hook

        def layer_output_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                nonlocal roots
                captured = (
                    layer_input[index],
                    attention_norm[index],
                    value_output[index],
                    attention_write[index],
                    attention_weight[index],
                    mlp_input[index],
                    mlp_norm[index],
                    gate_value[index],
                    up_value[index],
                    mlp_write[index],
                )
                if roots is None or any(value is None for value in captured):
                    raise RuntimeError("decoder hooks did not capture native operators")
                (
                    native_input,
                    native_attention_norm,
                    native_value,
                    native_attention_write,
                    weights,
                    native_mlp_input,
                    native_mlp_norm,
                    gate,
                    up,
                    native_mlp_write,
                ) = captured
                assert native_input is not None
                assert native_attention_norm is not None
                assert native_value is not None
                assert native_attention_write is not None
                assert weights is not None
                assert native_mlp_input is not None
                assert native_mlp_norm is not None
                assert gate is not None
                assert up is not None
                assert native_mlp_write is not None

                layer = self.layers[index]
                closure_delta(roots.sum(dim=1), native_input)
                attention_scale = rms_multiplier(layer.input_layernorm, native_input)
                normalized_roots = roots * attention_scale.unsqueeze(1)
                closure_delta(normalized_roots.sum(dim=1), native_attention_norm)

                raw_values = F.linear(
                    normalized_roots,
                    layer.self_attn.v_proj.weight.detach().float(),
                ).reshape(source_count, 4, self.kv_heads, self.head_dim)
                native_values = native_value.float().reshape(
                    source_count, self.kv_heads, self.head_dim
                )
                values, value_delta = close_numeric(raw_values, native_values)

                repeats = self.heads // self.kv_heads
                head_to_kv = torch.arange(self.heads, device=self.device) // repeats
                values_by_head = values[:, :, head_to_kv].permute(2, 0, 1, 3)
                context = torch.einsum(
                    "hqs,hsrd->qrhd", weights.float(), values_by_head
                )
                rooted_attention_write = F.linear(
                    context.flatten(2),
                    layer.self_attn.o_proj.weight.detach().float(),
                )
                closure_delta(rooted_attention_write.sum(dim=1), native_attention_write)
                post_attention_roots, post_attention_delta = close_numeric(
                    roots + rooted_attention_write, native_mlp_input
                )

                event_value_delta = value_delta.index_select(0, query)
                event_value_delta = event_value_delta[:, head_to_kv]
                self_attention = weights[
                    torch.arange(self.heads, device=self.device)[:, None],
                    query[None],
                    query[None],
                ].T
                weighted_value_delta = self_attention[..., None] * event_value_delta
                root_values.append(values.detach().cpu())
                self_value_numeric_input.append(weighted_value_delta.detach().cpu())
                post_attention_numeric_write.append(
                    post_attention_delta.index_select(0, query).detach().cpu()
                )
                event_attention.append(
                    weights.index_select(1, query).permute(1, 0, 2).detach().cpu()
                )
                attention_multiplier.append(
                    attention_scale.index_select(0, query).detach().cpu()
                )

                # Only the post-attention root state enters the MLP.  Release
                # full-sequence attention scratch before allocating its FP32
                # SwiGLU ledger; completed records are already on CPU.
                roots = post_attention_roots
                for values_list in (
                    layer_input,
                    attention_norm,
                    value_output,
                    attention_write,
                    attention_weight,
                ):
                    values_list[index] = None
                del (
                    captured,
                    native_input,
                    native_attention_norm,
                    native_value,
                    native_attention_write,
                    weights,
                    attention_scale,
                    normalized_roots,
                    raw_values,
                    native_values,
                    values,
                    value_delta,
                    values_by_head,
                    context,
                    rooted_attention_write,
                    post_attention_delta,
                    event_value_delta,
                    self_attention,
                    weighted_value_delta,
                )

                mlp_scale = rms_multiplier(
                    layer.post_attention_layernorm, native_mlp_input
                )
                normalized_mlp_roots = post_attention_roots * mlp_scale.unsqueeze(1)
                closure_delta(normalized_mlp_roots.sum(dim=1), native_mlp_norm)
                rooted_mlp_write = symmetric_swiglu_root_write(
                    normalized_mlp_roots,
                    gate,
                    up,
                    layer.mlp.gate_proj.weight.detach(),
                    layer.mlp.up_proj.weight.detach(),
                    layer.mlp.down_proj.weight.detach(),
                    token_block_size=self._root_token_block_size,
                    intermediate_block_size=self._root_intermediate_block_size,
                )
                closure_delta(rooted_mlp_write.sum(dim=1), native_mlp_write)
                native_output = output[0] if isinstance(output, tuple) else output
                roots, output_delta = close_numeric(
                    post_attention_roots + rooted_mlp_write,
                    native_output[0].detach(),
                )
                layer_numeric_write.append(
                    output_delta.index_select(0, query).detach().cpu()
                )
                mlp_multiplier.append(mlp_scale.index_select(0, query).detach().cpu())
                native_gate.append(gate.index_select(0, query).detach().cpu())
                native_up.append(up.index_select(0, query).detach().cpu())

                for values_list in (
                    mlp_input,
                    mlp_norm,
                    gate_value,
                    up_value,
                    mlp_write,
                ):
                    values_list[index] = None

            return hook

        def final_norm_hook(
            _module: Any, args: tuple[Tensor, ...], output: Tensor
        ) -> None:
            nonlocal roots, terminal_roots, final_multiplier
            nonlocal final_rms_numeric_write
            if roots is None:
                raise RuntimeError("root ledger was not initialized")
            native_input = args[0][0].detach()
            closure_delta(roots.sum(dim=1), native_input)
            multiplier = rms_multiplier(self.model.model.norm, native_input)
            normalized_roots = roots * multiplier.unsqueeze(1)
            closed_roots, final_delta = close_numeric(
                normalized_roots, output[0].detach()
            )
            terminal_roots = closed_roots.index_select(0, query).detach().cpu()
            final_multiplier = multiplier.index_select(0, query).detach().cpu()
            final_rms_numeric_write = final_delta.index_select(0, query).detach().cpu()
            roots = None

        try:
            handles.append(
                self.model.get_input_embeddings().register_forward_hook(embedding_hook)
            )
            for index, layer in enumerate(self.layers):
                handles.extend(
                    (
                        layer.register_forward_pre_hook(layer_input_hook(index)),
                        layer.input_layernorm.register_forward_hook(
                            norm_hook(index, attention=True)
                        ),
                        layer.self_attn.v_proj.register_forward_hook(
                            projection_hook(value_output, index)
                        ),
                        layer.self_attn.register_forward_hook(attention_hook(index)),
                        layer.post_attention_layernorm.register_forward_hook(
                            norm_hook(index, attention=False)
                        ),
                        layer.mlp.gate_proj.register_forward_hook(
                            projection_hook(gate_value, index)
                        ),
                        layer.mlp.up_proj.register_forward_hook(
                            projection_hook(up_value, index)
                        ),
                        layer.mlp.register_forward_hook(
                            projection_hook(mlp_write, index)
                        ),
                        layer.register_forward_hook(layer_output_hook(index)),
                    )
                )
            handles.append(self.model.model.norm.register_forward_hook(final_norm_hook))
            with torch.inference_mode():
                output = self.model(
                    input_ids=model_ids,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=query,
                )
        finally:
            for handle in handles:
                handle.remove()

        if (
            initial_roots is None
            or terminal_roots is None
            or final_multiplier is None
            or final_rms_numeric_write is None
        ):
            raise RuntimeError("native forward did not complete the root ledger")
        if not (
            len(root_values)
            == len(self_value_numeric_input)
            == len(post_attention_numeric_write)
            == len(layer_numeric_write)
            == len(self.layers)
        ):
            raise RuntimeError("native forward did not visit every decoder layer")

        with torch.inference_mode():
            event_logits = output.logits[0].float()
            del output
            target = events.target_token_id
            target_logits = event_logits.gather(1, target[:, None]).squeeze(1)
            log_normalizer = event_logits.logsumexp(dim=1)
            target_logprob = target_logits - log_normalizer
            event_logits.scatter_(1, target[:, None], -torch.inf)
            competitor = event_logits.argmax(dim=1)
            competitor_value = event_logits.gather(1, competitor[:, None]).squeeze(1)
            native_margin = target_logits - competitor_value

            readout = self.model.lm_head.weight.detach()
            direction = readout.index_select(0, target).float()
            direction -= readout.index_select(0, competitor).float()
            terminal_event_roots = terminal_roots.to(self.device)
            terminal_root_margin = torch.einsum(
                "erd,ed->er", terminal_event_roots, direction
            )
            terminal_error = (terminal_root_margin.sum(dim=1) - native_margin).abs()

        competitor_token_id = competitor.detach().cpu()
        target_logprob = target_logprob.detach().cpu()
        direction = direction.detach().cpu()
        event_error = endpoint_boundary_error(boundary_error, query).detach().cpu()
        native_margin = native_margin.detach().cpu()
        terminal_root_margin = terminal_root_margin.detach().cpu()
        terminal_error = terminal_error.detach().cpu()
        del (
            event_logits,
            target,
            target_logits,
            log_normalizer,
            competitor,
            competitor_value,
            readout,
            terminal_event_roots,
        )
        margin_scale = native_margin.abs().clamp_min(1.0)
        native_dtype = self.model.get_input_embeddings().weight.dtype
        closure_limit = max(
            CLOSURE_RELATIVE_LIMIT,
            float(torch.finfo(native_dtype).eps),
        )
        operator_valid = event_error <= closure_limit
        operator_valid &= terminal_error <= closure_limit * margin_scale

        suffix = ObservedSuffix(
            query_position=cpu_events.query_position,
            attention=tuple(event_attention),
            attention_rms_multiplier=tuple(attention_multiplier),
            mlp_rms_multiplier=tuple(mlp_multiplier),
            native_gate=tuple(native_gate),
            native_up=tuple(native_up),
            value_weight=tuple(
                layer.self_attn.v_proj.weight.detach() for layer in self.layers
            ),
            output_weight=tuple(
                layer.self_attn.o_proj.weight.detach() for layer in self.layers
            ),
            gate_weight=tuple(
                layer.mlp.gate_proj.weight.detach() for layer in self.layers
            ),
            up_weight=tuple(layer.mlp.up_proj.weight.detach() for layer in self.layers),
            down_weight=tuple(
                layer.mlp.down_proj.weight.detach() for layer in self.layers
            ),
            final_rms_multiplier=final_multiplier,
            readout_direction=direction,
        )
        return CapturedRouteOperators(
            response_start=int(response_start),
            events=cpu_events,
            source_token_id=source_ids,
            competitor_token_id=competitor_token_id,
            target_logprob=target_logprob,
            source_position=torch.arange(source_count, dtype=torch.long),
            evidence_mask=source_evidence,
            root_values=tuple(root_values),
            input_roots=initial_roots,
            suffix=suffix,
            self_value_numeric_input=torch.stack(self_value_numeric_input, dim=1),
            post_attention_numeric_write=torch.stack(
                post_attention_numeric_write, dim=1
            ),
            layer_numeric_write=torch.stack(layer_numeric_write, dim=1),
            final_rms_numeric_write=final_rms_numeric_write,
            terminal_root_margin=terminal_root_margin,
            native_margin=native_margin,
            operator_error=terminal_error,
            operator_valid=operator_valid,
        )

    def _validate_bias_free(self) -> None:
        projections = [self.model.lm_head]
        for layer in self.layers:
            projections.extend(
                (
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.self_attn.o_proj,
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj,
                )
            )
        if any(projection.bias is not None for projection in projections):
            raise ValueError("the current root ledger supports bias-free Llama only")


def rms_multiplier(module: Any, hidden: Tensor) -> Tensor:
    """Return the native diagonal RMSNorm multiplier in FP32."""

    variance = hidden.float().square().mean(dim=-1, keepdim=True)
    return module.weight.detach().float() * torch.rsqrt(
        variance + float(module.variance_epsilon)
    )
