"""Stream attention-only and exact residual-message magnitude maps together."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RouteMaps:
    row_start: int
    functional: Tensor
    attention: Tensor
    functional_middle: Tensor
    attention_middle: Tensor
    middle_start: int
    middle_stop: int


class RouteAccumulator:
    """Reduce ``A`` and ``A * ||W_O[h] V[g(h),s]||`` in one model pass."""

    def __init__(self, model, response_start: int, *, query_chunk: int = 128) -> None:
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.row_start = response_start - 1
        self.layer_count = int(model.config.num_hidden_layers)
        self.middle_start = self.layer_count // 3
        self.middle_stop = max(self.middle_start + 1, 2 * self.layer_count // 3)
        self.query_chunk = int(query_chunk)
        self._seen: set[int] = set()
        self._functional = None
        self._attention = None
        self._functional_middle = None
        self._attention_middle = None
        self._middle_count = 0

    @staticmethod
    def source_norm(value: Tensor, output_weight: Tensor) -> Tensor:
        """Exact norm of each query-head Value after its matching W_O block."""

        heads, _, head_dim = value.shape
        hidden = output_weight.shape[0]
        block = output_weight.float().reshape(hidden, heads, head_dim)
        block = block.permute(1, 2, 0)
        gram = block @ block.transpose(1, 2)
        squared = torch.einsum(
            "hsd,hde,hse->hs", value.float(), gram, value.float()
        )
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
            self._functional_middle = torch.zeros_like(self._functional)
            self._attention_middle = torch.zeros_like(self._functional)
        elif self._functional.shape != shape:
            raise ValueError("attention shape changed between layers")

        source_norm = self.source_norm(value, output_weight.detach())
        middle = self.middle_start <= layer < self.middle_stop
        for begin in range(self.row_start, queries, self.query_chunk):
            end = min(begin + self.query_chunk, queries)
            row = slice(begin - self.row_start, end - self.row_start)
            attention = probability[:, begin:end].float().mean(0)
            functional = (
                probability[:, begin:end].float() * source_norm[:, None, :]
            ).mean(0)
            self._attention[row] += attention
            self._functional[row] += functional
            if middle:
                self._attention_middle[row] += attention
                self._functional_middle[row] += functional
        self._middle_count += int(middle)

    def finish(self) -> RouteMaps:
        if self._seen != set(range(self.layer_count)):
            raise RuntimeError("baseline observer did not receive every decoder layer")
        if self._functional is None or self._middle_count == 0:
            raise RuntimeError("no route map was accumulated")
        return RouteMaps(
            row_start=self.row_start,
            functional=(self._functional / self.layer_count).cpu(),
            attention=(self._attention / self.layer_count).cpu(),
            functional_middle=(self._functional_middle / self._middle_count).cpu(),
            attention_middle=(self._attention_middle / self._middle_count).cpu(),
            middle_start=self.middle_start,
            middle_stop=self.middle_stop,
        )
