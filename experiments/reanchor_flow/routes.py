"""Layer-resolved routing summaries from native attention messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ROLE_NAMES = ("evidence", "other_prompt", "history")
GRAM_HEAD_CHUNK = 4
GRAM_CACHE_ATTRIBUTE = "_reanchor_output_gram_cpu"


@dataclass(frozen=True)
class RouteTrace:
    """Small CPU traces with shape ``[layer, response_event, role]``."""

    functional_share: Tensor
    attention_share: Tensor
    functional_mass: Tensor
    functional_null: Tensor
    attention_null: Tensor


class RouteAccumulator:
    """Reduce ``A`` and ``A * ||W_O[h]V[h,s]||`` during the forward.

    Each row describes messages read at query ``q`` to predict token
    ``p=q+1``. Layer order is retained; no artificial token-to-token path is
    constructed.
    """

    def __init__(self, model, response_start: int, prompt_evidence_mask) -> None:
        prompt = torch.as_tensor(prompt_evidence_mask, dtype=torch.bool).flatten()
        if len(prompt) != response_start:
            raise ValueError("evidence mask must cover the complete prompt")
        self.row_start = response_start - 1
        self.response_start = response_start
        self.layer_count = int(model.config.num_hidden_layers)
        self.prompt_evidence = prompt
        self._role_mask: Tensor | None = None
        self._functional: Tensor | None = None
        self._attention: Tensor | None = None
        self._mass: Tensor | None = None
        self._functional_null: Tensor | None = None
        self._attention_null: Tensor | None = None
        self._covered: Tensor | None = None
        self._source_norm: dict[int, Tensor] = {}
        self._staging: dict[int, Tensor] = {}
        self._next_query = [0] * self.layer_count
        self._flushed: set[int] = set()
        gram_cache = getattr(model, GRAM_CACHE_ATTRIBUTE, None)
        if gram_cache is None:
            gram_cache = {}
            setattr(model, GRAM_CACHE_ATTRIBUTE, gram_cache)
        self._gram_cache: dict[int, Tensor] = gram_cache

    @staticmethod
    def output_gram(output_weight: Tensor, heads: int, head_dim: int) -> Tensor:
        """Cacheable CPU Gram matrices for the query-head blocks of ``W_O``."""

        hidden = output_weight.shape[0]
        if output_weight.shape != (hidden, heads * head_dim):
            raise ValueError("W_O input width does not match the query heads")
        weight = output_weight.view(hidden, heads, head_dim).permute(1, 2, 0)
        gram_device = torch.empty(
            (heads, head_dim, head_dim),
            device=output_weight.device,
            dtype=torch.float32,
        )
        for begin in range(0, heads, GRAM_HEAD_CHUNK):
            end = min(begin + GRAM_HEAD_CHUNK, heads)
            block = weight[begin:end].float()
            chunk = torch.bmm(block, block.transpose(1, 2))
            gram_device[begin:end].copy_(chunk)
            del block, chunk
        return gram_device.cpu()

    @classmethod
    def source_norm(
        cls,
        value: Tensor,
        output_weight: Tensor,
        gram: Tensor | None = None,
    ) -> Tensor:
        """Norm of every query-head Value after its matching ``W_O`` block."""

        heads, sources, head_dim = value.shape
        hidden = output_weight.shape[0]
        if output_weight.shape != (hidden, heads * head_dim):
            raise ValueError("W_O input width does not match the query heads")
        if gram is None:
            gram = cls.output_gram(output_weight, heads, head_dim)
        if gram.shape != (heads, head_dim, head_dim):
            raise ValueError("cached W_O Gram matrices have the wrong shape")

        # Four heads at a time avoids full W_O/value float32 copies while
        # keeping the quadratic-form evaluation vectorized.
        gram_device = gram.to(value.device)
        norm = torch.empty(
            (heads, sources), device=value.device, dtype=torch.float32
        )
        for begin in range(0, heads, GRAM_HEAD_CHUNK):
            end = min(begin + GRAM_HEAD_CHUNK, heads)
            head_value = value[begin:end].float()
            squared = torch.einsum(
                "hsd,hde,hse->hs",
                head_value,
                gram_device[begin:end],
                head_value,
            )
            norm[begin:end] = squared.clamp_min(0).sqrt()
            del head_value, squared
        return norm

    def _initialize(self, sources: int, device: torch.device) -> None:
        events = sources - self.row_start
        if events < 1:
            raise ValueError("response has no prediction events")
        role = torch.zeros(sources, len(ROLE_NAMES), device=device)
        evidence = torch.zeros(sources, dtype=torch.bool, device=device)
        evidence[: self.response_start] = self.prompt_evidence.to(device)
        role[evidence, 0] = 1
        role[: self.response_start, 1] = (~evidence[: self.response_start]).float()
        role[self.response_start :, 2] = 1
        self._role_mask = role
        shape = (self.layer_count, events, len(ROLE_NAMES))
        self._functional = torch.zeros(shape, dtype=torch.float32)
        self._attention = torch.zeros(shape, dtype=torch.float32)
        self._mass = torch.zeros((self.layer_count, events), dtype=torch.float32)
        self._functional_null = torch.zeros(shape, dtype=torch.float32)
        self._attention_null = torch.zeros(shape, dtype=torch.float32)
        self._covered = torch.zeros((self.layer_count, events), dtype=torch.bool)

    def _stage_layer(self, layer: int, norm: Tensor) -> Tensor:
        """Allocate one O(events) GPU buffer and fill its availability nulls."""

        roles = len(ROLE_NAMES)
        events = norm.shape[1] - self.row_start
        stage = torch.empty(
            (events, 4 * roles + 1), device=norm.device, dtype=torch.float32
        )

        role = self._role_mask
        visible_count = role.cumsum(dim=0)[self.row_start :]
        stage[:, 3 * roles : 4 * roles] = visible_count / visible_count.sum(
            dim=1, keepdim=True
        )

        source_capacity = norm.sum(dim=0)
        visible_capacity = (source_capacity[:, None] * role).cumsum(dim=0)[
            self.row_start :
        ]
        stage[:, 2 * roles : 3 * roles] = visible_capacity / visible_capacity.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        self._staging[layer] = stage
        return stage

    def _flush_layer(self, layer: int) -> None:
        """Transfer all summaries for one layer to CPU in one operation."""

        if not bool(self._covered[layer].all()):
            raise RuntimeError(f"route layer {layer} has missing query chunks")
        roles = len(ROLE_NAMES)
        host = self._staging.pop(layer).cpu()
        self._functional[layer].copy_(host[:, :roles])
        self._attention[layer].copy_(host[:, roles : 2 * roles])
        self._functional_null[layer].copy_(host[:, 2 * roles : 3 * roles])
        self._attention_null[layer].copy_(host[:, 3 * roles : 4 * roles])
        self._mass[layer].copy_(host[:, 4 * roles])
        self._source_norm.pop(layer, None)
        self._flushed.add(layer)
        if len(self._flushed) == self.layer_count:
            self._role_mask = None

    def observe_chunk(
        self,
        layer: int,
        query_start: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        """Consume one absolute query chunk emitted by the manual forward."""

        probability = probability.detach()[0]
        value = repeated_value.detach()[0]
        heads, chunk_queries, sources = probability.shape
        if chunk_queries < 1:
            raise ValueError("attention query chunk must not be empty")
        if value.shape[:2] != (heads, sources):
            raise ValueError("attention Value rows do not match query heads")
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid layer index: {layer}")
        if query_start != self._next_query[layer]:
            raise ValueError(
                f"layer {layer} query chunks must be contiguous and non-overlapping"
            )
        end = query_start + chunk_queries
        if end > sources:
            raise ValueError("attention query chunk exceeds the source sequence")
        self._next_query[layer] = end
        if self._role_mask is None:
            self._initialize(sources, probability.device)
        elif len(self._role_mask) != sources:
            raise ValueError("attention source count changed between layers")

        norm = self._source_norm.get(layer)
        if norm is None:
            gram = self._gram_cache.get(layer)
            if gram is None:
                gram = self.output_gram(output_weight.detach(), heads, value.shape[-1])
                self._gram_cache[layer] = gram
            norm = self.source_norm(value, output_weight.detach(), gram)
            self._source_norm[layer] = norm
        stage = self._staging.get(layer)
        if stage is None:
            stage = self._stage_layer(layer, norm)

        begin = max(query_start, self.row_start)
        if begin < end:
            local = probability[:, begin - query_start :].float()
            capacity = local * norm[:, None]
            role = self._role_mask
            attention = torch.einsum("hqs,sr->qr", local, role) / heads
            functional = torch.einsum("hqs,sr->qr", capacity, role) / heads
            total = functional.sum(dim=1)
            functional = functional / total[:, None].clamp_min(1e-12)
            event = slice(begin - self.row_start, end - self.row_start)
            roles = len(ROLE_NAMES)
            stage[event, :roles] = functional
            stage[event, roles : 2 * roles] = attention
            stage[event, 4 * roles] = total
            self._covered[layer, event] = True

        if end == sources:
            self._flush_layer(layer)

    def observe(
        self,
        layer: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        """Compatibility entry point for an unchunked forward."""

        self.observe_chunk(layer, 0, probability, repeated_value, output_weight)

    def finish(self) -> RouteTrace:
        if self._covered is None or not bool(self._covered.all()):
            raise RuntimeError("route accumulator did not receive every response row")
        return RouteTrace(
            functional_share=self._functional,
            attention_share=self._attention,
            functional_mass=self._mass,
            functional_null=self._functional_null,
            attention_null=self._attention_null,
        )
