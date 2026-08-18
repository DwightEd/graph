"""Sparse RR attention to route and received-support source sets.

The canonical data interface remains the only source. One graph stores sparse
RR events by Transformer layer and materializes a bounded layer at a time,
avoiding a persistent ``[layer, head, token, source]`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SourceSetConfig


@dataclass(frozen=True)
class SparseRRLayer:
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor

    def validate(self, *, num_heads: int, response_count: int) -> "SparseRRLayer":
        arrays = (self.head, self.query, self.source, self.weight)
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("sparse RR arrays must be one-dimensional")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("sparse RR arrays are not aligned")
        if len(self.head):
            if bool((self.head < 0).any()) or bool((self.head >= num_heads).any()):
                raise ValueError("RR head is outside model geometry")
            if bool((self.query < 0).any()) or bool((self.query >= response_count).any()):
                raise ValueError("RR query is outside the response")
            if bool((self.source < 0).any()) or bool((self.source >= self.query).any()):
                raise ValueError("causal RR events require source < query")
            if bool((self.weight <= 0).any()) or not bool(torch.isfinite(self.weight).all()):
                raise ValueError("retained RR weights must be positive and finite")
        return self


@dataclass(frozen=True)
class LayerSourceSets:
    """Current route and prefix received-support memory for each token/head."""

    route_source: torch.Tensor
    route_weight: torch.Tensor
    route_received: torch.Tensor
    route_received_delta: torch.Tensor
    route_mask: torch.Tensor
    memory_source: torch.Tensor
    memory_received: torch.Tensor
    memory_received_delta: torch.Tensor
    memory_current_weight: torch.Tensor
    memory_mask: torch.Tensor
    total_mass: torch.Tensor
    tail_mass: torch.Tensor
    edge_count: torch.Tensor

    @property
    def response_count(self) -> int:
        return int(self.total_mass.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.total_mass.shape[1])

    def validate(self) -> "LayerSourceSets":
        if self.total_mass.ndim != 2:
            raise ValueError("row summaries must have shape [token, head]")
        row_shape = self.total_mass.shape
        if any(value.shape != row_shape for value in (self.tail_mass, self.edge_count)):
            raise ValueError("row summaries are not aligned")
        route_shape = self.route_source.shape
        memory_shape = self.memory_source.shape
        if route_shape[:2] != row_shape or memory_shape[:2] != row_shape:
            raise ValueError("source-set rows do not match token/head geometry")
        if any(
            value.shape != route_shape
            for value in (
                self.route_weight,
                self.route_received,
                self.route_received_delta,
                self.route_mask,
            )
        ):
            raise ValueError("route source-set fields are not aligned")
        if any(
            value.shape != memory_shape
            for value in (
                self.memory_received,
                self.memory_received_delta,
                self.memory_current_weight,
                self.memory_mask,
            )
        ):
            raise ValueError("memory source-set fields are not aligned")
        numeric = (
            self.route_weight,
            self.route_received,
            self.route_received_delta,
            self.memory_received,
            self.memory_received_delta,
            self.memory_current_weight,
            self.total_mass,
            self.tail_mass,
            self.edge_count,
        )
        if any(not bool(torch.isfinite(value).all()) for value in numeric):
            raise FloatingPointError("source-set materialization produced non-finite values")
        if bool((self.tail_mass < -1e-6).any()):
            raise ValueError("selected route mass exceeds the retained row mass")
        return self


@dataclass(frozen=True)
class CausalSourceSetGraph:
    layers: tuple[SparseRRLayer, ...]
    num_heads: int
    response_count: int
    attention_floor: float

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def validate(self) -> "CausalSourceSetGraph":
        if self.num_layers < 1 or self.num_heads < 1 or self.response_count < 1:
            raise ValueError("invalid causal source-set graph geometry")
        if not 0.0 < float(self.attention_floor) < 1.0:
            raise ValueError("attention_floor must be in (0,1)")
        for layer in self.layers:
            layer.validate(num_heads=self.num_heads, response_count=self.response_count)
        return self

    def materialize_layer(
        self,
        layer_index: int,
        config: SourceSetConfig,
        *,
        device: str | torch.device,
    ) -> LayerSourceSets:
        """Build one layer's route and persistence-memory sets."""

        config.validate()
        layer_index = int(layer_index)
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("layer index is outside source-set graph")
        device = torch.device(device)
        layer = self.layers[layer_index]
        heads, tokens = self.num_heads, self.response_count
        dtype = torch.float32

        current = torch.zeros((heads, tokens, tokens), dtype=dtype, device=device)
        if len(layer.weight):
            current.index_put_(
                (
                    layer.head.to(device=device, dtype=torch.long),
                    layer.query.to(device=device, dtype=torch.long),
                    layer.source.to(device=device, dtype=torch.long),
                ),
                layer.weight.to(device=device, dtype=dtype),
                accumulate=True,
            )

        cumulative = current.cumsum(dim=1)
        target = torch.arange(tokens, device=device, dtype=dtype)[:, None]
        source = torch.arange(tokens, device=device, dtype=dtype)[None, :]
        age = (target - source + 1.0).clamp_min(1.0)
        causal = source < target
        received = torch.where(
            causal[None, :, :],
            cumulative / age[None, :, :],
            torch.zeros_like(cumulative),
        )
        previous = torch.zeros_like(received)
        if tokens > 1:
            previous[:, 1:] = received[:, :-1]
        received_delta = received - previous

        total_mass = current.sum(dim=2)
        edge_count = (current > 0).sum(dim=2).float()

        route_weight, route_source = _topk_padded(current, config.max_route_sources)
        route_received = torch.gather(received, 2, route_source)
        route_delta = torch.gather(received_delta, 2, route_source)
        before = route_weight.cumsum(dim=2) - route_weight
        route_keep = (route_weight > 0) & (
            before
            < float(config.route_mass_coverage)
            * total_mass[:, :, None].clamp_min(config.epsilon)
        )
        selected_mass = (route_weight * route_keep).sum(dim=2)
        tail_mass = (total_mass - selected_mass).clamp_min(0.0)

        memory_received, memory_source = _topk_padded(
            received, config.max_memory_sources
        )
        memory_delta = torch.gather(received_delta, 2, memory_source)
        memory_current = torch.gather(current, 2, memory_source)
        memory_keep = memory_received > 0

        transpose = lambda value: value.permute(1, 0, 2).contiguous()
        return LayerSourceSets(
            route_source=transpose(route_source),
            route_weight=transpose(route_weight),
            route_received=transpose(route_received),
            route_received_delta=transpose(route_delta),
            route_mask=transpose(route_keep),
            memory_source=transpose(memory_source),
            memory_received=transpose(memory_received),
            memory_received_delta=transpose(memory_delta),
            memory_current_weight=transpose(memory_current),
            memory_mask=transpose(memory_keep),
            total_mass=total_mass.transpose(0, 1).contiguous(),
            tail_mass=tail_mass.transpose(0, 1).contiguous(),
            edge_count=edge_count.transpose(0, 1).contiguous(),
        ).validate()


def extract_causal_source_set_graph(
    sample,
    config: SourceSetConfig | None = None,
) -> CausalSourceSetGraph:
    """Extract exact retained causal RR events through ResearchSample views."""

    config = SourceSetConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    storage = [[[], [], [], []] for _ in range(layers)]

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        rr = block.source >= prompt_count
        if not bool(rr.any()):
            continue
        layer = block.layer[rr].long()
        head = block.head[rr].long()
        query = block.query[rr].long()
        source = (block.source[rr] - prompt_count).long()
        weight = block.weight[rr].float()
        causal = source < query
        if not bool(causal.any()):
            continue
        layer, head, query, source, weight = (
            value[causal] for value in (layer, head, query, source, weight)
        )
        for current_layer in torch.unique(layer).tolist():
            selected = layer == int(current_layer)
            for bucket, value in zip(
                storage[int(current_layer)],
                (head[selected], query[selected], source[selected], weight[selected]),
                strict=True,
            ):
                bucket.append(value.detach().cpu())

    result: list[SparseRRLayer] = []
    for head_parts, query_parts, source_parts, weight_parts in storage:
        if weight_parts:
            layer = SparseRRLayer(
                head=torch.cat(head_parts).long(),
                query=torch.cat(query_parts).long(),
                source=torch.cat(source_parts).long(),
                weight=torch.cat(weight_parts).float(),
            )
        else:
            layer = SparseRRLayer(
                head=torch.empty(0, dtype=torch.long),
                query=torch.empty(0, dtype=torch.long),
                source=torch.empty(0, dtype=torch.long),
                weight=torch.empty(0, dtype=torch.float32),
            )
        result.append(layer)

    return CausalSourceSetGraph(
        layers=tuple(result),
        num_heads=heads,
        response_count=response_count,
        attention_floor=float(attention.attention_floor),
    ).validate()


def _topk_padded(values: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(count)
    keep = min(count, int(values.shape[-1]))
    top_value, top_index = torch.topk(
        values, k=keep, dim=-1, largest=True, sorted=True
    )
    if keep == count:
        return top_value, top_index
    pad_shape = (*values.shape[:-1], count - keep)
    return (
        torch.cat(
            (
                top_value,
                torch.zeros(pad_shape, device=values.device, dtype=values.dtype),
            ),
            dim=-1,
        ),
        torch.cat(
            (
                top_index,
                torch.zeros(pad_shape, device=values.device, dtype=torch.long),
            ),
            dim=-1,
        ),
    )
