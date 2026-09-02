"""Single-pass teacher-forced capture of exact local route messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .messages import (
    MessageStatistics,
    MLPDiagnostics,
    PromptCarriers,
    message_statistics,
    mlp_diagnostics,
    output_projection_blocks,
    output_projection_gram,
    prompt_carriers,
    reconstruction_error,
    selected_chunk_messages,
    source_write_norms,
)


@dataclass(frozen=True)
class PredictionEvents:
    """Physical predictor coordinates and the response tokens they predict."""

    query_position: torch.Tensor
    prediction_position: torch.Tensor
    target_id: torch.Tensor


@dataclass(frozen=True)
class RouteMessageChunk:
    """Short-lived exact route factors emitted once per layer and query chunk."""

    layer: int
    query_position: torch.Tensor
    prediction_position: torch.Tensor
    statistics: MessageStatistics
    post_attention_state: torch.Tensor
    reconstruction_max_abs: torch.Tensor
    reconstruction_relative_l2: torch.Tensor
    mlp: MLPDiagnostics
    attention: torch.Tensor
    values: torch.Tensor
    output_blocks: torch.Tensor

    def selected_messages(
        self,
        query: torch.Tensor,
        head: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize selected vectors; ``query`` indexes this local chunk."""

        return selected_chunk_messages(
            self.attention,
            self.values,
            self.output_blocks,
            query,
            head,
            source,
        )


@dataclass(frozen=True)
class ResponseTrace:
    """Compact response-only diagnostics from one frozen observer replay."""

    token_ids: torch.Tensor
    response_start: int
    events: PredictionEvents
    target_logprob: torch.Tensor
    target_confidence: torch.Tensor
    target_margin: torch.Tensor
    reconstruction_max_abs: torch.Tensor
    reconstruction_relative_l2: torch.Tensor
    mlp_write_norm: torch.Tensor
    mlp_relative_norm: torch.Tensor
    mlp_state_cosine: torch.Tensor
    attention_prompt: PromptCarriers
    functional_prompt: PromptCarriers


def prediction_events(
    token_ids: Sequence[int] | torch.Tensor,
    response_start: int,
) -> PredictionEvents:
    """Map response token ``p`` to its physical predictor query ``q=p-1``."""

    ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    prediction = torch.arange(response_start, len(ids), dtype=torch.long)
    return PredictionEvents(
        query_position=prediction - 1,
        prediction_position=prediction,
        target_id=ids[prediction],
    )


class RouteMessageReplay:
    """Replay a frozen bias-free Llama once and stream exact route chunks."""

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.layers = tuple(model.model.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.head_dim = self.layers[0].self_attn.v_proj.out_features // self.kv_heads
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            raise ValueError("pretraining_tp must be one for exact value hooks")
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if any(layer.self_attn.o_proj.bias is not None for layer in self.layers):
            raise ValueError("exact edge reconstruction requires bias-free W_O")
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
    ) -> RouteMessageReplay:
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

    def target_scores(
        self,
        hidden: torch.Tensor,
        target_id: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score observed targets with the model's native LM-head arithmetic."""

        logprob, confidence, margin = [], [], []
        weight_dtype = self.model.lm_head.weight.dtype
        for start in range(0, len(hidden), chunk_size):
            stop = min(start + chunk_size, len(hidden))
            logits = self.model.lm_head(hidden[start:stop].to(weight_dtype)).float()
            target = target_id[start:stop]
            selected = logits.gather(1, target[:, None]).squeeze(1)
            competitor = logits.scatter(1, target[:, None], -torch.inf).max(1).values
            selected_logprob = selected - logits.logsumexp(1)
            logprob.append(selected_logprob.cpu())
            confidence.append(selected_logprob.exp().cpu())
            margin.append((selected - competitor).cpu())
        return torch.cat(logprob), torch.cat(confidence), torch.cat(margin)

    def capture(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        response_start: int,
        *,
        predictor_chunk: int = 64,
        logit_chunk: int = 32,
        consume_chunk: Callable[[RouteMessageChunk], None] | None = None,
    ) -> ResponseTrace:
        """Capture every predictor row and retain only response diagnostics.

        ``consume_chunk`` runs synchronously while dense scalar accounts and
        AVWO factors are live.  It must sparsify/propagate prompt rows before
        returning; retaining the chunk would defeat bounded-memory replay.
        """

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        if token_ids.ndim != 1 or not 0 < response_start < len(token_ids):
            raise ValueError("token_ids/response_start do not define a response")
        if min(predictor_chunk, logit_chunk) <= 0:
            raise ValueError("chunk sizes must be positive")

        events = prediction_events(token_ids, response_start)
        response_tokens = len(events.target_id)
        source_tokens = len(token_ids) - 1
        layer_count = len(self.layers)
        ids = token_ids.to(self.device)
        value_dtype = self.model.get_input_embeddings().weight.dtype

        values = [
            torch.empty(
                source_tokens,
                self.kv_heads,
                self.head_dim,
                dtype=value_dtype,
                device=self.device,
            )
            for _ in self.layers
        ]
        source_norm = [
            torch.empty(
                self.heads,
                source_tokens,
                dtype=torch.float32,
                device=self.device,
            )
            for _ in self.layers
        ]

        response_shape = (layer_count, response_tokens)
        reconstruction_max_abs = torch.empty(response_shape)
        reconstruction_relative_l2 = torch.empty(response_shape)
        mlp_write_norm = torch.empty(response_shape)
        mlp_relative_norm = torch.empty(response_shape)
        mlp_state_cosine = torch.empty(response_shape)

        def empty_carriers() -> PromptCarriers:
            return PromptCarriers(
                effective_sources=torch.empty(response_shape),
                effective_rank=torch.empty(response_shape),
                anchor_source=torch.empty(
                    layer_count, response_tokens, self.heads, dtype=torch.int32
                ),
            )

        attention_prompt = empty_carriers()
        functional_prompt = empty_carriers()
        target_logprob = torch.empty(response_tokens)
        target_confidence = torch.empty(response_tokens)
        target_margin = torch.empty(response_tokens)

        attention_by_layer: list[torch.Tensor | None] = [None] * layer_count
        native_write_by_layer: list[torch.Tensor | None] = [None] * layer_count
        mid_by_layer: list[torch.Tensor | None] = [None] * layer_count
        blocks_by_layer: list[torch.Tensor | None] = [None] * layer_count
        query_start = query_stop = 0
        handles = []

        def response_window() -> tuple[int, int, int] | None:
            local_start = max(response_start - 1 - query_start, 0)
            if query_start + local_start >= query_stop:
                return None
            row_start = query_start + local_start - (response_start - 1)
            row_stop = row_start + query_stop - query_start - local_start
            return local_start, row_start, row_stop

        def value_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                current = output[0].reshape(
                    query_stop - query_start, self.kv_heads, self.head_dim
                )
                values[index][query_start:query_stop].copy_(current)
                blocks = output_projection_blocks(
                    self.layers[index].self_attn.o_proj.weight,
                    self.heads,
                    self.head_dim,
                )
                blocks_by_layer[index] = blocks
                source_norm[index][:, query_start:query_stop].copy_(
                    source_write_norms(current, self.output_grams[index])
                )

            return hook

        def attention_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                native_write_by_layer[index] = output[0][0]
                attention_by_layer[index] = output[1][0, :, :, :query_stop]

            return hook

        def post_attention_hook(index: int):
            def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
                mid_by_layer[index] = args[0][0]

            return hook

        def mlp_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                attention = attention_by_layer[index]
                native_write = native_write_by_layer[index]
                mid = mid_by_layer[index]
                blocks = blocks_by_layer[index]
                if any(
                    value is None for value in (attention, native_write, mid, blocks)
                ):
                    raise RuntimeError("decoder hooks did not capture one route layer")

                prefix_values = values[index][:query_stop]
                statistics = message_statistics(
                    attention,
                    prefix_values,
                    blocks,
                    mid,
                    source_norm[index][:, :query_stop],
                )
                maximum, relative = reconstruction_error(
                    statistics.attention_write, native_write
                )
                mlp = mlp_diagnostics(mid, output[0])
                query = torch.arange(
                    query_start, query_stop, dtype=torch.long, device=self.device
                )

                if consume_chunk is not None:
                    consume_chunk(
                        RouteMessageChunk(
                            layer=index,
                            query_position=query,
                            prediction_position=query + 1,
                            statistics=statistics,
                            post_attention_state=mid.float(),
                            reconstruction_max_abs=maximum,
                            reconstruction_relative_l2=relative,
                            mlp=mlp,
                            attention=attention,
                            values=prefix_values,
                            output_blocks=blocks,
                        )
                    )

                window = response_window()
                if window is not None:
                    local, row_start, row_stop = window
                    location = (index, slice(row_start, row_stop))
                    reconstruction_max_abs[location].copy_(maximum[local:].cpu())
                    reconstruction_relative_l2[location].copy_(relative[local:].cpu())
                    mlp_write_norm[location].copy_(mlp.write_norm[local:].cpu())
                    mlp_relative_norm[location].copy_(mlp.relative_norm[local:].cpu())
                    mlp_state_cosine[location].copy_(mlp.state_cosine[local:].cpu())

                    response_query = query[local:]
                    families = (
                        (
                            attention_prompt,
                            prompt_carriers(
                                attention[:, local:].float().permute(1, 0, 2),
                                response_query,
                                response_start,
                            ),
                        ),
                        (
                            functional_prompt,
                            prompt_carriers(
                                statistics.capacity[local:],
                                response_query,
                                response_start,
                            ),
                        ),
                    )
                    for target, observed in families:
                        target.effective_sources[location].copy_(
                            observed.effective_sources.cpu()
                        )
                        target.effective_rank[location].copy_(
                            observed.effective_rank.cpu()
                        )
                        target.anchor_source[location].copy_(
                            observed.anchor_source.to(dtype=torch.int32, device="cpu")
                        )

                attention_by_layer[index] = None
                native_write_by_layer[index] = None
                mid_by_layer[index] = None
                blocks_by_layer[index] = None

            return hook

        for index, layer in enumerate(self.layers):
            handles.extend(
                (
                    layer.self_attn.v_proj.register_forward_hook(value_hook(index)),
                    layer.self_attn.register_forward_hook(attention_hook(index)),
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        post_attention_hook(index)
                    ),
                    layer.mlp.register_forward_hook(mlp_hook(index)),
                )
            )

        try:
            from transformers.cache_utils import DynamicCache

            past = DynamicCache()
            with torch.inference_mode():
                for query_start in range(0, source_tokens, predictor_chunk):
                    query_stop = min(query_start + predictor_chunk, source_tokens)
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

                    window = response_window()
                    if window is None:
                        continue
                    local, row_start, row_stop = window
                    target = ids[response_start + row_start : response_start + row_stop]
                    scores = self.target_scores(
                        output.last_hidden_state[0, local:], target, logit_chunk
                    )
                    target_logprob[row_start:row_stop].copy_(scores[0])
                    target_confidence[row_start:row_stop].copy_(scores[1])
                    target_margin[row_start:row_stop].copy_(scores[2])
        finally:
            for handle in handles:
                handle.remove()

        return ResponseTrace(
            token_ids=token_ids,
            response_start=response_start,
            events=events,
            target_logprob=target_logprob,
            target_confidence=target_confidence,
            target_margin=target_margin,
            reconstruction_max_abs=reconstruction_max_abs,
            reconstruction_relative_l2=reconstruction_relative_l2,
            mlp_write_norm=mlp_write_norm,
            mlp_relative_norm=mlp_relative_norm,
            mlp_state_cosine=mlp_state_cosine,
            attention_prompt=attention_prompt,
            functional_prompt=functional_prompt,
        )
