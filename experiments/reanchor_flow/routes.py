"""Streaming attention and exact residual-message magnitude maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RouteMaps:
    row_start: int
    functional: Tensor
    attention: Tensor
    functional_all: Tensor
    attention_all: Tensor
    layer_start: int
    layer_stop: int


class RouteAccumulator:
    """Build attention-only and ``A * ||W_O[h] V[h,s]||`` maps together."""

    def __init__(
        self,
        model,
        response_start: int,
        *,
        query_chunk: int = 128,
        layer_start: int | None = None,
        layer_stop: int | None = None,
    ) -> None:
        layers = int(model.config.num_hidden_layers)
        self.row_start = response_start - 1
        self.query_chunk = int(query_chunk)
        self.layer_start = layers // 3 if layer_start is None else int(layer_start)
        self.layer_stop = max(self.layer_start + 1, 2 * layers // 3)
        if layer_stop is not None:
            self.layer_stop = int(layer_stop)
        if not 0 <= self.layer_start < self.layer_stop <= layers:
            raise ValueError("flow layer band is outside the decoder")
        self.layer_count = layers
        self._seen: set[int] = set()
        self._functional = None
        self._attention = None
        self._functional_all = None
        self._attention_all = None
        self._selected = 0

    @staticmethod
    def source_norm(value: Tensor, output_weight: Tensor) -> Tensor:
        heads, _, head_dim = value.shape
        hidden = output_weight.shape[0]
        block = output_weight.float().reshape(hidden, heads, head_dim)
        block = block.permute(1, 2, 0)
        gram = block @ block.transpose(1, 2)
        squared = torch.einsum("hsd,hde,hse->hs", value.float(), gram, value.float())
        return squared.clamp_min(0).sqrt()

    def observe(
        self,
        layer: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        probability = probability.detach()[0]
        value = repeated_value.detach()[0]
        heads, queries, sources = probability.shape
        rows = queries - self.row_start
        if rows <= 0 or value.shape[:2] != (heads, sources):
            raise ValueError("attention observer received incompatible shapes")
        if layer in self._seen:
            raise ValueError(f"attention layer observed twice: {layer}")
        self._seen.add(int(layer))

        shape = (rows, sources)
        if self._functional is None:
            self._functional = torch.zeros(shape, device=probability.device)
            self._attention = torch.zeros_like(self._functional)
            self._functional_all = torch.zeros_like(self._functional)
            self._attention_all = torch.zeros_like(self._functional)

        source_norm = self.source_norm(value, output_weight.detach())
        selected = self.layer_start <= layer < self.layer_stop
        for begin in range(self.row_start, queries, self.query_chunk):
            end = min(begin + self.query_chunk, queries)
            row = slice(begin - self.row_start, end - self.row_start)
            attention = probability[:, begin:end].float().mean(0)
            functional = (
                probability[:, begin:end].float() * source_norm[:, None, :]
            ).mean(0)
            self._attention_all[row] += attention
            self._functional_all[row] += functional
            if selected:
                self._attention[row] += attention
                self._functional[row] += functional
        self._selected += int(selected)

    def finish(self) -> RouteMaps:
        if self._seen != set(range(self.layer_count)):
            raise RuntimeError("baseline observer did not receive every decoder layer")
        if self._functional is None or self._selected == 0:
            raise RuntimeError("no route map was accumulated")
        return RouteMaps(
            row_start=self.row_start,
            functional=(self._functional / self._selected).cpu(),
            attention=(self._attention / self._selected).cpu(),
            functional_all=(self._functional_all / self.layer_count).cpu(),
            attention_all=(self._attention_all / self.layer_count).cpu(),
            layer_start=self.layer_start,
            layer_stop=self.layer_stop,
        )
