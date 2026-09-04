"""One-pass token routing traces for re-anchor mechanism discovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .message_norm import model_gram_cache, output_gram, source_norm


@dataclass(frozen=True)
class RouteTrace:
    """CPU traces aligned to prediction events.

    Population summaries are ``[layer, event]``.  ``head`` preserves the
    corresponding ``[layer, head, event]`` tensors for event discovery; these
    must not be reconstructed from the layer/head means.
    """

    prompt_share: Tensor
    evidence_share: Tensor
    history_share: Tensor
    prompt_lift: Tensor
    evidence_lift: Tensor
    history_lift: Tensor
    nonlocality: Tensor
    prompt_breadth: Tensor
    route_change: Tensor
    predictor_reuse: Tensor
    future_influence: Tensor
    head: dict[str, Tensor]
    detail: dict[str, Tensor] | None


class RouteAccumulator:
    """Observe native attention and head-preserving ``A * ||W_O V||`` routes."""

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
        self._head_fields: dict[str, Tensor] = {}
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
                "prompt_lift",
                "evidence_lift",
                "history_lift",
                "nonlocality",
                "prompt_breadth",
                "route_change",
                "predictor_reuse",
                "future_influence",
            )
        }
        head_shape = (self.layer_count, heads, events)
        self._head_fields = {
            name: torch.full(head_shape, float("nan"), dtype=torch.float16)
            for name in (
                "attention_prompt_mass",
                "attention_evidence_mass",
                "attention_history_mass",
                "prompt_share",
                "evidence_share",
                "history_share",
                "nonlocality",
                "route_change",
                "predictor_reuse",
                "future_influence",
            )
        }
        if self.keep_detail:
            self._detail = {
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
                "predictor_sum": torch.zeros((heads, events), device=device),
                "predictor_count": torch.zeros(events, device=device),
                "emitted_sum": torch.zeros((heads, events), device=device),
                "emitted_count": torch.zeros(events, device=device),
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
            norm = source_norm(value, output_weight.detach(), gram).float()
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
            evidence_mask = self.prompt_evidence.to(probability.device)

            prompt = distribution[:, :, : self.response_start]
            prompt_share = prompt.sum(-1)
            evidence_share = prompt[:, :, evidence_mask].sum(-1)
            history_share = distribution[:, :, self.response_start :].sum(-1)
            attention_prompt_mass = local[:, :, : self.response_start].sum(-1)
            attention_evidence_mass = local[:, :, : self.response_start][
                :, :, evidence_mask
            ].sum(-1)
            attention_history_mass = local[:, :, self.response_start :].sum(-1)

            visible = source[None] <= query[:, None]
            available = norm[:, None, :] * visible[None]
            available = available / available.sum(-1, keepdim=True).clamp_min(1e-12)
            prompt_null = available[:, :, : self.response_start].sum(-1)
            evidence_null = available[:, :, : self.response_start][
                :, :, evidence_mask
            ].sum(-1)
            history_null = available[:, :, self.response_start :].sum(-1)
            prompt_lift = torch.log((prompt_share + 1e-12) / (prompt_null + 1e-12))
            evidence_lift = torch.log(
                (evidence_share + 1e-12) / (evidence_null + 1e-12)
            )
            history_lift = torch.log((history_share + 1e-12) / (history_null + 1e-12))

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
                (heads, len(query)), float("nan"), device=probability.device
            )
            rolling: list[Tensor] = state["rolling"]  # type: ignore[assignment]
            predictor_sum: Tensor = state["predictor_sum"]  # type: ignore[assignment]
            predictor_count: Tensor = state["predictor_count"]  # type: ignore[assignment]
            emitted_sum: Tensor = state["emitted_sum"]  # type: ignore[assignment]
            emitted_count: Tensor = state["emitted_count"]  # type: ignore[assignment]
            predictor_sources = source[self.row_start :]
            response_sources = source[self.response_start :]

            for offset, absolute_query in enumerate(query.tolist()):
                current = distribution[:, offset]
                if rolling:
                    reference = torch.stack(rolling, dim=0).mean(0)
                    change_head[:, offset] = self._js(current, reference)
                rolling.append(current.detach())
                if len(rolling) > self.route_window:
                    rolling.pop(0)

                predictor_lag = absolute_query - predictor_sources
                predictor_valid = (predictor_lag >= 1) & (
                    predictor_lag <= self.future_horizon
                )
                if bool(predictor_valid.any()):
                    predictor_sum += (
                        current[:, self.row_start :] * predictor_valid[None]
                    )
                    predictor_count += predictor_valid

                # Event e predicts the emitted token p=q+1.  Once p exists in
                # the teacher-forced prefix, later prediction rows may use its
                # token state.  This is distinct from reusing predictor q.
                emitted_lag = absolute_query + 1 - response_sources
                emitted_valid = (emitted_lag >= 1) & (
                    emitted_lag <= self.future_horizon
                )
                if bool(emitted_valid.any()):
                    emitted_sum[:, : len(response_sources)] += (
                        current[:, self.response_start :] * emitted_valid[None]
                    )
                    emitted_count[: len(response_sources)] += emitted_valid

            values = {
                "prompt_share": prompt_share,
                "evidence_share": evidence_share,
                "history_share": history_share,
                "prompt_lift": prompt_lift,
                "evidence_lift": evidence_lift,
                "history_lift": history_lift,
                "nonlocality": nonlocality,
                "prompt_breadth": entropy,
                "route_change": change_head,
            }
            for name, value in values.items():
                self._fields[name][layer, event] = value.mean(0).cpu()

            for name in (
                "prompt_share",
                "evidence_share",
                "history_share",
                "nonlocality",
                "route_change",
            ):
                self._head_fields[name][layer, :, event] = values[name].to(
                    device="cpu", dtype=torch.float16
                )
            for name, value in (
                ("attention_prompt_mass", attention_prompt_mass),
                ("attention_evidence_mass", attention_evidence_mass),
                ("attention_history_mass", attention_history_mass),
            ):
                self._head_fields[name][layer, :, event] = value.to(
                    device="cpu", dtype=torch.float16
                )

            if self._detail is not None:
                self._detail["edge_map"][event] += (
                    distribution.mean(0).cpu() / self.layer_count
                )

        if query_end == sources:
            self._norm_cache.pop(layer, None)
            state = self._states.pop(layer)
            for field, prefix in (
                ("predictor_reuse", "predictor"),
                ("future_influence", "emitted"),
            ):
                total: Tensor = state[f"{prefix}_sum"]  # type: ignore[assignment]
                count: Tensor = state[f"{prefix}_count"]  # type: ignore[assignment]
                average = total / count[None].clamp_min(1)
                average[:, count <= 0] = float("nan")
                self._fields[field][layer] = average.mean(0).cpu()
                self._head_fields[field][layer] = average.to(
                    device="cpu", dtype=torch.float16
                )

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
            prompt_lift=self._fields["prompt_lift"],
            evidence_lift=self._fields["evidence_lift"],
            history_lift=self._fields["history_lift"],
            nonlocality=self._fields["nonlocality"],
            prompt_breadth=self._fields["prompt_breadth"],
            route_change=self._fields["route_change"],
            predictor_reuse=self._fields["predictor_reuse"],
            future_influence=self._fields["future_influence"],
            head=self._head_fields,
            detail=(
                None
                if self._detail is None
                else {
                    **self._detail,
                    "prompt_head": self._head_fields["prompt_share"],
                    "evidence_head": self._head_fields["evidence_share"],
                    "nonlocal_head": self._head_fields["nonlocality"],
                    "route_change_head": self._head_fields["route_change"],
                    "predictor_reuse_head": self._head_fields["predictor_reuse"],
                    "future_head": self._head_fields["future_influence"],
                }
            ),
        )
