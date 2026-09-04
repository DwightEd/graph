"""One-pass token routing traces for prompt revisits and future anchors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .message_norm import model_gram_cache, output_gram, source_norm


@dataclass(frozen=True)
class RouteTrace:
    """Compact CPU traces, normally shaped ``[layer, response_event]``."""

    prompt_share: Tensor
    evidence_share: Tensor
    history_share: Tensor
    nonlocality: Tensor
    prompt_breadth: Tensor
    route_change: Tensor
    future_influence: Tensor
    detail: dict[str, Tensor] | None


class RouteAccumulator:
    """Observe native attention and ``A * ||W_O V||`` without storing all heads.

    ``nonlocality`` is a continuous clipped expected source distance. It does
    not require a source to be farther than a hard token threshold.
    """

    def __init__(
        self,
        model,
        response_start: int,
        prompt_evidence_mask,
        *,
        route_window: int = 4,
        future_horizon: int = 16,
        distance_scale: int = 16,
        detail: bool = False,
    ) -> None:
        if route_window < 1 or future_horizon < 1 or distance_scale < 1:
            raise ValueError("routing windows must be positive")
        prompt = torch.as_tensor(prompt_evidence_mask, dtype=torch.bool).flatten()
        if len(prompt) != response_start:
            raise ValueError("evidence mask must cover the complete prompt")

        self.response_start = int(response_start)
        self.row_start = self.response_start - 1
        self.layer_count = int(model.config.num_hidden_layers)
        self.route_window = int(route_window)
        self.future_horizon = int(future_horizon)
        self.distance_scale = int(distance_scale)
        self.prompt_evidence = prompt
        self.keep_detail = bool(detail)

        self._next_query = [0] * self.layer_count
        self._states: dict[int, dict[str, object]] = {}
        self._shape: tuple[int, int, int] | None = None
        self._fields: dict[str, Tensor] = {}
        self._detail: dict[str, Tensor] | None = None
        self._gram_cache = model_gram_cache(model)
        self._norm_cache: dict[int, Tensor] = {}

    def _initialize(self, heads: int, sources: int) -> None:
        events = sources - self.row_start
        if events < 1:
            raise ValueError("response has no prediction events")
        self._shape = (heads, sources, events)
        shape = (self.layer_count, events)
        self._fields = {
            name: torch.full(shape, float("nan"), dtype=torch.float32)
            for name in (
                "prompt_share",
                "evidence_share",
                "history_share",
                "nonlocality",
                "prompt_breadth",
                "route_change",
                "future_influence",
            )
        }
        if self.keep_detail:
            head_shape = (self.layer_count, heads, events)
            self._detail = {
                "prompt_head": torch.full(head_shape, float("nan")),
                "nonlocal_head": torch.full(head_shape, float("nan")),
                "route_change_head": torch.full(head_shape, float("nan")),
                "future_head": torch.full(head_shape, float("nan")),
                "edge_map": torch.zeros((events, sources), dtype=torch.float32),
            }

    def _layer_state(
        self,
        layer: int,
        heads: int,
        sources: int,
        device: torch.device,
    ) -> dict[str, object]:
        state = self._states.get(layer)
        if state is None:
            events = sources - self.row_start
            state = {
                "rolling": [],
                "future_sum": torch.zeros((heads, events), device=device),
                "future_count": torch.zeros(events, device=device),
            }
            self._states[layer] = state
        return state

    @staticmethod
    def _js(current: Tensor, reference: Tensor) -> Tensor:
        eps = 1e-12
        middle = 0.5 * (current + reference)
        first = current * (
            current.clamp_min(eps).log() - middle.clamp_min(eps).log()
        )
        second = reference * (
            reference.clamp_min(eps).log() - middle.clamp_min(eps).log()
        )
        return 0.5 * (first.sum(-1) + second.sum(-1)) / torch.log(
            current.new_tensor(2.0)
        )

    def observe_chunk(
        self,
        layer: int,
        query_start: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        """Consume one contiguous absolute-query chunk emitted by the forward."""

        probability = probability.detach()[0].float()
        value = repeated_value.detach()[0]
        heads, chunk_queries, sources = probability.shape
        if query_start != self._next_query[layer]:
            raise ValueError("query chunks must be contiguous and non-overlapping")
        query_end = query_start + chunk_queries
        self._next_query[layer] = query_end

        if self._shape is None:
            self._initialize(heads, sources)
        elif self._shape[:2] != (heads, sources):
            raise ValueError("attention shape changed between layers")

        gram = self._gram_cache.get(layer)
        if gram is None:
            gram = output_gram(output_weight.detach(), heads, value.shape[-1])
            self._gram_cache[layer] = gram
        norm = self._norm_cache.get(layer)
        if norm is None:
            norm = source_norm(value, output_weight.detach(), gram)
            self._norm_cache[layer] = norm
        state = self._layer_state(layer, heads, sources, probability.device)

        begin = max(query_start, self.row_start)
        if begin < query_end:
            local = probability[:, begin - query_start : query_end - query_start]
            capacity = local * norm[:, None, :]
            distribution = capacity / capacity.sum(-1, keepdim=True).clamp_min(1e-12)

            query = torch.arange(begin, query_end, device=probability.device)
            source = torch.arange(sources, device=probability.device)
            event = slice(begin - self.row_start, query_end - self.row_start)

            prompt = distribution[:, :, : self.response_start]
            prompt_share = prompt.sum(-1)
            evidence_mask = self.prompt_evidence.to(probability.device)
            evidence_share = prompt[:, :, evidence_mask].sum(-1)
            history_share = distribution[:, :, self.response_start :].sum(-1)

            distance = (query[:, None] - source[None]).clamp_min(0).float()
            distance_weight = (distance / self.distance_scale).clamp_max(1.0)
            nonlocality = (distribution * distance_weight[None]).sum(-1)

            prompt_probability = prompt / prompt_share[..., None].clamp_min(1e-12)
            entropy = -(
                prompt_probability * prompt_probability.clamp_min(1e-12).log()
            ).sum(-1)
            if self.response_start > 1:
                entropy = entropy / torch.log(
                    entropy.new_tensor(float(self.response_start))
                )
            entropy = entropy.masked_fill(prompt_share <= 1e-12, 0)

            change_head = torch.full(
                (heads, len(query)),
                float("nan"),
                device=probability.device,
            )
            rolling: list[Tensor] = state["rolling"]  # type: ignore[assignment]
            future_sum: Tensor = state["future_sum"]  # type: ignore[assignment]
            future_count: Tensor = state["future_count"]  # type: ignore[assignment]
            response_sources = source[self.response_start :]

            for offset, absolute_query in enumerate(query.tolist()):
                current = distribution[:, offset]
                if rolling:
                    reference = torch.stack(rolling, dim=0).mean(0)
                    change_head[:, offset] = self._js(current, reference)
                rolling.append(current.detach())
                if len(rolling) > self.route_window:
                    rolling.pop(0)

                generated_position = absolute_query + 1
                future_lag = generated_position - response_sources
                valid = (future_lag >= 1) & (future_lag <= self.future_horizon)
                if bool(valid.any()):
                    future_sum[:, : len(response_sources)] += (
                        current[:, self.response_start :] * valid[None]
                    )
                    future_count[: len(response_sources)] += valid

            self._fields["prompt_share"][layer, event] = prompt_share.mean(0).cpu()
            self._fields["evidence_share"][layer, event] = evidence_share.mean(0).cpu()
            self._fields["history_share"][layer, event] = history_share.mean(0).cpu()
            self._fields["nonlocality"][layer, event] = nonlocality.mean(0).cpu()
            self._fields["prompt_breadth"][layer, event] = entropy.mean(0).cpu()
            self._fields["route_change"][layer, event] = change_head.mean(0).cpu()

            if self._detail is not None:
                self._detail["prompt_head"][layer, :, event] = prompt_share.cpu()
                self._detail["nonlocal_head"][layer, :, event] = nonlocality.cpu()
                self._detail["route_change_head"][layer, :, event] = change_head.cpu()
                self._detail["edge_map"][event] += (
                    distribution.mean(0).cpu() / self.layer_count
                )

        if query_end == sources:
            self._norm_cache.pop(layer, None)
            state = self._states.pop(layer)
            future_sum = state["future_sum"]  # type: ignore[assignment]
            future_count = state["future_count"]  # type: ignore[assignment]
            future = future_sum / future_count[None].clamp_min(1)
            future[:, future_count <= 0] = float("nan")
            self._fields["future_influence"][layer] = future.mean(0).cpu()
            if self._detail is not None:
                self._detail["future_head"][layer] = future.cpu()

    def observe(
        self,
        layer: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        self.observe_chunk(layer, 0, probability, repeated_value, output_weight)

    def finish(self) -> RouteTrace:
        if self._shape is None or self._states:
            raise RuntimeError("route accumulator did not receive a complete forward")
        if self._next_query != [self._shape[1]] * self.layer_count:
            raise RuntimeError("one or more layers are incomplete")
        return RouteTrace(
            prompt_share=self._fields["prompt_share"],
            evidence_share=self._fields["evidence_share"],
            history_share=self._fields["history_share"],
            nonlocality=self._fields["nonlocality"],
            prompt_breadth=self._fields["prompt_breadth"],
            route_change=self._fields["route_change"],
            future_influence=self._fields["future_influence"],
            detail=self._detail,
        )
