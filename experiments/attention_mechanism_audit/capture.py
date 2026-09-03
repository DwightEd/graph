"""Independent one-token replay for exact functional message attribution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .graph import GraphBuilder, source_roles


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return the frozen prefix K/V tensors in the stable legacy layout."""

    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        cache = ((layer.keys, layer.values) for layer in cache.layers)
    elif hasattr(cache, "key_cache"):
        cache = zip(cache.key_cache, cache.value_cache)
    return tuple((key.detach(), value.detach()) for key, value in cache)


def _batched_prefix(
    cache: tuple[tuple[torch.Tensor, torch.Tensor], ...], predictors: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build independent exact prefixes for one predictor microbatch."""

    batch = len(predictors)
    past = int(predictors.max().item())
    if batch == 1:
        layers = [(key[:, :, :past], value[:, :, :past]) for key, value in cache]
    else:
        valid = torch.arange(past, device=predictors.device)[None] < predictors[:, None]
        layers = []
        for key, value in cache:
            key = key[:, :, :past].expand(batch, -1, -1, -1).clone()
            value = value[:, :, :past].expand(batch, -1, -1, -1).clone()
            key.masked_fill_(~valid[:, None, :, None], 0)
            value.masked_fill_(~valid[:, None, :, None], 0)
            layers.append((key, value))

    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(layers))
    return DynamicCache(tuple(layers))


class FunctionalMessageReplay:
    """Build a graph whose attention edges are exact ``A·V·W_O`` messages.

    Every batch element is an independent one-token decoding problem with its
    own frozen prefix. Summing the batch log probabilities therefore yields one
    uncontaminated gradient per target token. The default microbatch is one
    because an 8B model plus eight replicated long KV prefixes exceeds a 24 GB
    device even though the underlying attribution is token-local.
    """

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.layers = tuple(model.model.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.head_dim = self.layers[0].self_attn.q_proj.out_features // self.heads
        repeats = self.heads // self.kv_heads
        self.q_to_kv = torch.arange(self.heads, device=self.device) // repeats
        self.output_grams = tuple(self._output_gram(layer) for layer in self.layers)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FunctionalMessageReplay":
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

    def _output_gram(self, layer: Any) -> torch.Tensor:
        weight = layer.self_attn.o_proj.weight.detach().float()
        block = weight.reshape(weight.shape[0], self.heads, self.head_dim)
        return torch.einsum("ohd,ohe->hde", block, block)

    def capture(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        response_start: int,
        evidence_mask: Sequence[bool] | torch.Tensor,
        *,
        predictor_batch: int = 1,
        edge_cover: float = 0.95,
        edge_budget: int = 64,
    ) -> dict[str, Any]:
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        evidence = torch.as_tensor(evidence_mask, dtype=torch.bool, device=self.device)
        response = len(ids) - response_start
        builder = GraphBuilder(
            ids.detach().cpu(),
            response_start,
            len(self.layers),
            self.heads,
            self.head_dim,
            edge_cover=edge_cover,
            edge_budget=edge_budget,
        )

        current_value: list[torch.Tensor | None] = [None] * len(self.layers)
        current_context: list[torch.Tensor | None] = [None] * len(self.layers)
        current_mlp: list[torch.Tensor | None] = [None] * len(self.layers)
        handles = []

        def value_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                current_value[index] = output.reshape(
                    output.shape[0], output.shape[1], self.kv_heads, self.head_dim
                )

            return hook

        def context_hook(index: int):
            def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
                current_context[index] = args[0]

            return hook

        def mlp_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                current_mlp[index] = output

            return hook

        for index, layer in enumerate(self.layers):
            handles.extend(
                (
                    layer.self_attn.v_proj.register_forward_hook(value_hook(index)),
                    layer.self_attn.o_proj.register_forward_pre_hook(context_hook(index)),
                    layer.mlp.register_forward_hook(mlp_hook(index)),
                )
            )

        try:
            with torch.no_grad():
                full = self.model.model(
                    input_ids=ids[:-1][None],
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            prefix_cache = _legacy_cache(full.past_key_values)
            value_bank = tuple(value[0].detach() for value in current_value)
            current_value[:] = [None] * len(self.layers)
            current_context[:] = [None] * len(self.layers)
            current_mlp[:] = [None] * len(self.layers)
            del full

            for start in range(0, response, predictor_batch):
                stop = min(start + predictor_batch, response)
                targets = torch.arange(start, stop, device=self.device)
                predictors = response_start - 1 + targets
                past = _batched_prefix(prefix_cache, predictors)
                past_length = int(predictors.max().item())
                positions = torch.arange(past_length + 1, device=self.device)
                attention_mask = positions[None] < predictors[:, None]
                attention_mask[:, -1] = True

                embedding = self.model.model.embed_tokens(ids[predictors][:, None]).detach()
                embedding.requires_grad_(True)
                output = self.model.model(
                    inputs_embeds=embedding,
                    attention_mask=attention_mask.long(),
                    position_ids=predictors[:, None],
                    past_key_values=past,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
                logits = self.model.lm_head(output.last_hidden_state[:, -1]).float()
                target_ids = ids[predictors + 1]
                selected = logits.gather(1, target_ids[:, None]).squeeze(1)
                logprob = selected - logits.logsumexp(1)
                masked = logits.scatter(1, target_ids[:, None], -torch.inf)
                margin = selected - masked.max(1).values

                captured = tuple(current_context) + tuple(current_mlp)
                gradients = torch.autograd.grad(logprob.sum(), captured)
                context_gradient = gradients[: len(self.layers)]
                mlp_gradient = gradients[len(self.layers) :]

                for row, target in enumerate(targets.tolist()):
                    predictor = int(predictors[row])
                    roles = source_roles(
                        predictor + 1, response_start, predictor, evidence
                    )
                    for layer in range(len(self.layers)):
                        attention = output.attentions[layer][row, :, 0]
                        attention = torch.cat((attention[:, :predictor], attention[:, -1:]), 1)
                        value = torch.cat(
                            (
                                value_bank[layer][:predictor],
                                current_value[layer][row, 0, None].detach(),
                            )
                        )
                        builder.add_layer(
                            target=target,
                            predictor=predictor,
                            layer=layer,
                            attention=attention,
                            value=value,
                            head_gradient=context_gradient[layer][row, 0],
                            output_gram=self.output_grams[layer],
                            q_to_kv=self.q_to_kv,
                            roles=roles,
                            mlp_output=current_mlp[layer][row, 0],
                            mlp_gradient=mlp_gradient[layer][row, 0],
                        )
                    builder.add_target_score(target, logprob[row], margin[row])

                current_value[:] = [None] * len(self.layers)
                current_context[:] = [None] * len(self.layers)
                current_mlp[:] = [None] * len(self.layers)
                del output, gradients, captured, embedding, past
                del logits, selected, logprob, masked, margin
                del context_gradient, mlp_gradient, attention_mask, positions, target_ids
        finally:
            for handle in handles:
                handle.remove()

        graph = builder.finish()
        return {
            "schema": "functional-message-graph-v2",
            "objective": "teacher_forced_target_logprob",
            "evidence_mask": evidence.detach().cpu(),
            "gradient_scope": "independent one-token predictor; no later-target loss",
            **graph.__dict__,
        }