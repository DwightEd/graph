"""Streaming functional route maps from native attention messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FunctionalRoutes:
    row_start: int
    split_layer: int
    absolute_map: Tensor
    all_map: Tensor
    early_absolute_map: Tensor
    early_map: Tensor
    late_absolute_map: Tensor
    late_map: Tensor
    local_map: Tensor
    global_map: Tensor


class FunctionalRouteAccumulator:
    """Reduce exact ``A * ||W_O[h] V[h,s]||`` routes during one forward."""

    def __init__(
        self,
        model,
        response_start: int,
        head_quantile: float = 0.3,
        query_chunk: int = 128,
        split_layer: int | None = None,
    ) -> None:
        if not 0 < head_quantile <= 0.5:
            raise ValueError("head_quantile must be in (0, 0.5]")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.row_start = response_start - 1
        self.head_quantile = head_quantile
        self.query_chunk = query_chunk
        layer_count = int(model.config.num_hidden_layers)
        self._layer_count = layer_count
        self.split_layer = layer_count // 2 if split_layer is None else split_layer
        self._early: Tensor | None = None
        self._early_absolute: Tensor | None = None
        self._late: Tensor | None = None
        self._late_absolute: Tensor | None = None
        self._local: Tensor | None = None
        self._global: Tensor | None = None
        self._count = 0
        self._early_count = 0
        self._late_count = 0
        self._seen_layers: set[int] = set()
        self._attention_shape: tuple[int, int, int] | None = None

    @staticmethod
    def _source_norm(repeated_value: Tensor, output_weight: Tensor) -> Tensor:
        heads, _, head_dim = repeated_value.shape
        hidden = output_weight.shape[0]
        blocks = output_weight.float().reshape(hidden, heads, head_dim)
        blocks = blocks.permute(1, 2, 0)
        gram = blocks @ blocks.transpose(1, 2)
        value = repeated_value.float()
        squared = torch.einsum(
            "hsd,hde,hse->hs",
            value,
            gram,
            value,
        )
        return squared.clamp_min(0).sqrt()

    @staticmethod
    def _capacity(probability: Tensor, source_norm: Tensor) -> tuple[Tensor, Tensor]:
        capacity = probability.float() * source_norm[:, None]
        normalized = capacity / capacity.sum(-1, keepdim=True).clamp_min(1e-12)
        return capacity, normalized

    def observe(
        self,
        layer: int,
        probability: Tensor,
        repeated_value: Tensor,
        output_weight: Tensor,
    ) -> None:
        """Consume one native callback; GQA values must already match Q heads."""

        if probability.shape[0] != 1 or repeated_value.shape[0] != 1:
            raise ValueError("functional routes require a single example")
        probability = probability.detach()[0]
        repeated_value = repeated_value.detach()[0]
        heads, queries, sources = probability.shape
        if repeated_value.shape[:2] != (heads, sources):
            raise ValueError(
                "repeated_value must have one row per query head and source"
            )
        head_dim = repeated_value.shape[-1]
        if output_weight.shape[1] != heads * head_dim:
            raise ValueError(
                "output projection does not match the repeated query heads"
            )
        if layer not in range(self._layer_count) or layer in self._seen_layers:
            raise ValueError(f"invalid or repeated attention layer: {layer}")
        attention_shape = (heads, queries, sources)
        if self._attention_shape is None:
            self._attention_shape = attention_shape
        elif self._attention_shape != attention_shape:
            raise ValueError(
                "query-head, query, or source count changed between layers"
            )
        self._seen_layers.add(layer)

        rows = queries - self.row_start
        shape = (rows, sources)
        if self._local is None:
            self._local = torch.zeros(shape, device=probability.device)
            self._global = torch.zeros_like(self._local)
            self._early = torch.zeros_like(self._local)
            self._early_absolute = torch.zeros_like(self._local)
            self._late = torch.zeros_like(self._local)
            self._late_absolute = torch.zeros_like(self._local)
        elif self._local.shape != shape:
            raise ValueError("attention shape changed between layers")

        if layer < self.split_layer:
            band_map, band_absolute = self._early, self._early_absolute
            self._early_count += 1
        else:
            band_map, band_absolute = self._late, self._late_absolute
            self._late_count += 1

        source_norm = self._source_norm(repeated_value, output_weight.detach())
        distance = torch.zeros(heads, device=probability.device)
        source_position = torch.arange(sources, device=probability.device)

        for begin in range(self.row_start, queries, self.query_chunk):
            end = min(begin + self.query_chunk, queries)
            capacity, route = self._capacity(probability[:, begin:end], source_norm)
            row = slice(begin - self.row_start, end - self.row_start)
            band_absolute[row] += capacity.mean(0)
            band_map[row] += route.mean(0)
            lookback = torch.arange(begin, end, device=probability.device)[:, None]
            lookback = lookback - source_position
            distance += (route * lookback[None]).sum((1, 2))

        distance /= rows
        group_size = max(1, int(heads * self.head_quantile))
        order = distance.argsort()
        local_head = order[:group_size]
        global_head = order[-group_size:]

        for begin in range(self.row_start, queries, self.query_chunk):
            end = min(begin + self.query_chunk, queries)
            _, route = self._capacity(probability[:, begin:end], source_norm)
            row = slice(begin - self.row_start, end - self.row_start)
            self._local[row] += route.index_select(0, local_head).mean(0)
            self._global[row] += route.index_select(0, global_head).mean(0)

        self._count += 1

    def finish(self) -> FunctionalRoutes:
        """Return the accumulated functional maps on CPU."""

        missing = set(range(self._layer_count)) - self._seen_layers
        if missing:
            raise RuntimeError(f"attention observer missed layers: {sorted(missing)}")
        if (
            self._early is None
            or self._early_absolute is None
            or self._late is None
            or self._late_absolute is None
            or self._local is None
            or self._global is None
        ):
            raise RuntimeError("no attention layers were observed")
        scale = 1.0 / self._count
        early_scale = 1.0 / self._early_count if self._early_count else 0.0
        late_scale = 1.0 / self._late_count if self._late_count else 0.0
        return FunctionalRoutes(
            row_start=self.row_start,
            split_layer=self.split_layer,
            absolute_map=((self._early_absolute + self._late_absolute) * scale).cpu(),
            all_map=((self._early + self._late) * scale).cpu(),
            early_absolute_map=(self._early_absolute * early_scale).cpu(),
            early_map=(self._early * early_scale).cpu(),
            late_absolute_map=(self._late_absolute * late_scale).cpu(),
            late_map=(self._late * late_scale).cpu(),
            local_map=(self._local * scale).cpu(),
            global_map=(self._global * scale).cpu(),
        )
