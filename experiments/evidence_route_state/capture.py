"""One teacher-forced replay of the residual-register route graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .graph import GraphSequence, gram, mlp_relation, route_topology
from .messages import (
    PromptCarriers,
    output_projection_gram,
    prompt_carriers,
    source_write_norms,
)
from .registers import (
    ENDOGENOUS,
    ORIGIN_NAMES,
    add_attention,
    add_mlp,
    final_readout_contributions,
    initialize_registers,
    project_register_values,
    rmsnorm_registers,
    route_register_values,
)


@dataclass(frozen=True)
class PredictionEvents:
    """Physical query positions and the response tokens they predict."""

    query_position: torch.Tensor
    prediction_position: torch.Tensor
    target_id: torch.Tensor


@dataclass(frozen=True)
class PromptRouteControl:
    """The earlier prompt-route collapse geometry, retained as a control."""

    attention: PromptCarriers
    functional: PromptCarriers


@dataclass(frozen=True)
class RegisterGraphTrace:
    """Graph state and native output controls from one observed forward pass."""

    token_ids: torch.Tensor
    response_start: int
    events: PredictionEvents
    graph: GraphSequence
    target_logprob: torch.Tensor
    target_confidence: torch.Tensor
    target_margin: torch.Tensor
    prompt_route: PromptRouteControl
    attention_write_error: torch.Tensor
    register_closure_error: torch.Tensor


def prediction_events(
    token_ids: torch.Tensor,
    response_start: int,
) -> PredictionEvents:
    """Map response token ``p`` to its causal predictor query ``p - 1``."""

    ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    prediction = torch.arange(response_start, len(ids), dtype=torch.long)
    return PredictionEvents(prediction - 1, prediction, ids[prediction])


def relative_error(derived: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Per-row relative L2 error used to expose, not hide, ledger closure."""

    difference = derived.float() - native.float()
    return difference.norm(dim=-1) / native.float().norm(dim=-1).clamp_min(1e-12)


class RegisterGraphReplay:
    """Stream an additive origin ledger through a frozen bias-free Llama."""

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.layers = tuple(model.model.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.head_dim = self.layers[0].self_attn.v_proj.out_features // self.kv_heads
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            raise ValueError("pretraining_tp must be one")
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if any(
            layer.self_attn.v_proj.bias is not None
            or layer.self_attn.o_proj.bias is not None
            for layer in self.layers
        ):
            raise ValueError("the additive ledger requires bias-free V and W_O")
        self.output_grams = tuple(
            output_projection_gram(
                layer.self_attn.o_proj.weight,
                self.heads,
                self.head_dim,
            )
            for layer in self.layers
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> RegisterGraphReplay:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        return cls(model)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def score_targets(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the native LM head to score observed response tokens."""

        logits = self.model.lm_head(hidden.to(self.model.lm_head.weight.dtype)).float()
        target = target.to(logits.device)
        selected = logits.gather(1, target[:, None]).squeeze(1)
        alternatives = logits.clone()
        alternatives.scatter_(1, target[:, None], -torch.inf)
        competitor = alternatives.argmax(1)
        competitor_logit = logits.gather(1, competitor[:, None]).squeeze(1)
        logprob = selected - logits.logsumexp(1)
        return logprob, logprob.exp(), selected - competitor_logit, competitor

    def capture(
        self,
        token_ids: torch.Tensor,
        response_start: int,
        evidence_mask: torch.Tensor,
        *,
        predictor_chunk: int = 16,
    ) -> RegisterGraphTrace:
        """Build one graph sequence without reading a hallucination label.

        The dense source axis is consumed while attention and dynamic values are
        live. The saved object contains the exact register ledger plus the graph
        geometry fixed by the method; no top-k edge controls its score.
        """

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        evidence_mask = torch.as_tensor(evidence_mask, dtype=torch.bool, device="cpu")
        if token_ids.ndim != 1 or not 0 < response_start < len(token_ids):
            raise ValueError("token_ids and response_start do not define a response")
        if evidence_mask.shape != (response_start,):
            raise ValueError("evidence_mask must describe the complete prompt")
        if predictor_chunk <= 0:
            raise ValueError("predictor_chunk must be positive")

        events = prediction_events(token_ids, response_start)
        response_tokens = len(events.target_id)
        source_tokens = len(token_ids) - 1
        layer_count = len(self.layers)
        channels = len(ORIGIN_NAMES)
        hidden_size = int(self.model.config.hidden_size)
        ids = token_ids.to(self.device)

        value_cache = [
            torch.empty(
                source_tokens,
                channels,
                self.kv_heads,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            for _ in self.layers
        ]

        node_embedding = torch.empty(response_tokens, channels, hidden_size)
        residual_gram = torch.empty(
            response_tokens, layer_count + 1, channels, channels
        )
        head_write_gram = torch.empty(
            response_tokens,
            layer_count,
            self.heads,
            channels,
            channels,
        )
        topology = torch.empty(
            response_tokens,
            layer_count,
            self.heads,
            channels,
            7,
        )
        mlp_geometry = torch.empty(response_tokens, layer_count, channels + 1)
        margin_contribution = torch.empty(response_tokens, channels)
        target_logprob = torch.empty(response_tokens)
        target_confidence = torch.empty(response_tokens)
        target_margin = torch.empty(response_tokens)
        attention_write_error = torch.empty(layer_count, response_tokens)
        register_closure_error = torch.empty(layer_count + 1, response_tokens)

        def empty_carriers() -> PromptCarriers:
            return PromptCarriers(
                effective_sources=torch.empty(layer_count, response_tokens),
                effective_rank=torch.empty(layer_count, response_tokens),
                anchor_source=torch.empty(
                    layer_count, response_tokens, self.heads, dtype=torch.int32
                ),
            )

        attention_prompt = empty_carriers()
        functional_prompt = empty_carriers()

        current: torch.Tensor | None = None
        layer_input: torch.Tensor | None = None
        final_registers: torch.Tensor | None = None
        native_head_context: list[torch.Tensor | None] = [None] * layer_count
        query_start = query_stop = 0
        handles = []

        def response_rows() -> tuple[torch.Tensor, torch.Tensor]:
            position = torch.arange(query_start, query_stop, device=self.device)
            local = torch.nonzero(position >= response_start - 1, as_tuple=True)[0]
            slots = position[local] - response_start + 1
            return local, slots

        def input_norm_hook(index: int):
            def hook(module: Any, args: tuple[torch.Tensor, ...], output: Any) -> None:
                nonlocal current, layer_input
                native = args[0][0]
                position = torch.arange(query_start, query_stop, device=self.device)
                if index == 0:
                    current = initialize_registers(
                        native,
                        position,
                        evidence_mask,
                        response_start,
                    )
                assert current is not None
                current[:, ENDOGENOUS] += native.float() - current.sum(1)
                layer_input = native
                normalized = rmsnorm_registers(current, native, module)
                value_projection = self.layers[index].self_attn.v_proj
                native_value = value_projection(output[0])
                value_cache[index][query_start:query_stop].copy_(
                    project_register_values(
                        normalized,
                        value_projection,
                        self.kv_heads,
                        self.head_dim,
                        native_value,
                    )
                )
                local, slots = response_rows()
                if len(local):
                    slot = slots.cpu()
                    residual_gram[slot, index] = gram(current[local]).detach().cpu()
                    register_closure_error[index, slot] = (
                        relative_error(current[local].sum(1), native[local])
                        .detach()
                        .cpu()
                    )

            return hook

        def attention_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                nonlocal current
                assert current is not None and layer_input is not None
                attention = output[1][0, :, :, :query_stop]
                native_write = output[0][0]
                prefix_values = value_cache[index][:query_stop]
                writes, head_context = route_register_values(
                    attention,
                    prefix_values,
                    self.layers[index].self_attn.o_proj.weight,
                    native_head_context[index],
                )
                native_mid = layer_input + native_write
                local, slots = response_rows()
                if len(local):
                    slot = slots.cpu()
                    metric = self.output_grams[index]
                    head_gram = torch.einsum(
                        "qhcd,hde,qhfe->qhcf",
                        head_context.float(),
                        metric.float(),
                        head_context.float(),
                    )
                    head_write_gram[slot, index] = head_gram[local].detach().cpu()
                    response_position = torch.arange(
                        query_start, query_stop, device=self.device
                    )[local]
                    topology[slot, index] = (
                        route_topology(
                            attention[:, local],
                            prefix_values,
                            metric,
                            response_position,
                            response_start,
                        )
                        .detach()
                        .cpu()
                    )
                    attention_write_error[index, slot] = (
                        relative_error(writes[local].sum(1), native_write[local])
                        .detach()
                        .cpu()
                    )

                    raw = prompt_carriers(
                        attention[:, local].float().permute(1, 0, 2),
                        response_position,
                        response_start,
                    )
                    total_values = prefix_values.sum(1)
                    source_norm = source_write_norms(total_values, metric)
                    capacity = (
                        attention[:, local].float().permute(1, 0, 2)
                        * source_norm[None, :, :query_stop]
                    )
                    functional = prompt_carriers(
                        capacity,
                        response_position,
                        response_start,
                    )
                    for target, observed in (
                        (attention_prompt, raw),
                        (functional_prompt, functional),
                    ):
                        target.effective_sources[index, slot] = (
                            observed.effective_sources.detach().cpu()
                        )
                        target.effective_rank[index, slot] = (
                            observed.effective_rank.detach().cpu()
                        )
                        target.anchor_source[index, slot] = (
                            observed.anchor_source.to(torch.int32).detach().cpu()
                        )

                current = add_attention(current, writes, native_mid)

            return hook

        def output_projection_hook(index: int):
            def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
                native_head_context[index] = args[0][0].reshape(
                    query_stop - query_start,
                    self.heads,
                    self.head_dim,
                )

            return hook

        def mlp_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                nonlocal current
                assert current is not None
                native_mlp = output[0]
                local, slots = response_rows()
                if len(local):
                    mlp_geometry[slots.cpu(), index] = (
                        mlp_relation(current[local], native_mlp[local]).detach().cpu()
                    )
                native_mid = current.sum(1).to(native_mlp.dtype)
                native_output = (native_mid + native_mlp).float()
                current = add_mlp(current, native_mlp, native_output)
                if len(local):
                    slot = slots.cpu()
                    residual_gram[slot, index + 1] = gram(current[local]).detach().cpu()
                    register_closure_error[index + 1, slot] = (
                        relative_error(current[local].sum(1), native_output[local])
                        .detach()
                        .cpu()
                    )

            return hook

        def final_norm_hook(
            module: Any,
            args: tuple[torch.Tensor, ...],
            _output: Any,
        ) -> None:
            nonlocal current, final_registers
            assert current is not None
            native = args[0][0]
            current[:, ENDOGENOUS] += native.float() - current.sum(1)
            final_registers = rmsnorm_registers(current, native, module)
            local, slots = response_rows()
            if len(local):
                node_embedding[slots.cpu()] = final_registers[local].detach().cpu()

        for index, layer in enumerate(self.layers):
            handles.extend(
                (
                    layer.input_layernorm.register_forward_hook(input_norm_hook(index)),
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        output_projection_hook(index)
                    ),
                    layer.self_attn.register_forward_hook(attention_hook(index)),
                    layer.mlp.register_forward_hook(mlp_hook(index)),
                )
            )
        handles.append(self.model.model.norm.register_forward_hook(final_norm_hook))

        try:
            from transformers.cache_utils import DynamicCache

            past = DynamicCache()
            with torch.inference_mode():
                for query_start in range(0, source_tokens, predictor_chunk):
                    query_stop = min(query_start + predictor_chunk, source_tokens)
                    current = None
                    final_registers = None
                    output = self.model.model(
                        input_ids=ids[None, query_start:query_stop],
                        attention_mask=torch.ones(
                            1, query_stop, dtype=torch.long, device=self.device
                        ),
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    past = output.past_key_values
                    local, slots = response_rows()
                    if not len(local):
                        continue
                    assert final_registers is not None
                    slot = slots.cpu()
                    target = events.target_id[slot].to(self.device)
                    logprob, confidence, margin, competitor = self.score_targets(
                        output.last_hidden_state[0, local],
                        target,
                    )
                    target_logprob[slot] = logprob.detach().cpu()
                    target_confidence[slot] = confidence.detach().cpu()
                    target_margin[slot] = margin.detach().cpu()
                    contribution = final_readout_contributions(
                        final_registers[local],
                        self.model.lm_head.weight,
                        target,
                        competitor,
                    )
                    contribution[:, ENDOGENOUS] += margin - contribution.sum(1)
                    margin_contribution[slot] = contribution.detach().cpu()
        finally:
            for handle in handles:
                handle.remove()

        valid = torch.zeros(response_tokens, dtype=torch.bool)
        valid[2:] = True
        graph = GraphSequence(
            query_position=events.query_position,
            prediction_position=events.prediction_position,
            node_embedding=node_embedding,
            residual_gram=residual_gram,
            head_write_gram=head_write_gram,
            route_topology=topology,
            mlp_relation=mlp_geometry,
            margin_contribution=margin_contribution,
            valid=valid,
        )
        return RegisterGraphTrace(
            token_ids=token_ids,
            response_start=response_start,
            events=events,
            graph=graph,
            target_logprob=target_logprob,
            target_confidence=target_confidence,
            target_margin=target_margin,
            prompt_route=PromptRouteControl(attention_prompt, functional_prompt),
            attention_write_error=attention_write_error,
            register_closure_error=register_closure_error,
        )


# Narrow import compatibility alias; there is still only one implementation.
RouteMessageReplay = RegisterGraphReplay
