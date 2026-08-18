"""Sparse RR attention to exact route and received-support source sets.

The canonical data interface remains the only source. One graph stores sparse
RR events by Transformer layer. A layer is materialized in bounded query
chunks, so the implementation never creates persistent ``[head, token, token]``
current/cumulative/received tensors for the complete response.

Chunking is an execution detail only. For every token/head, the returned route
set, received-support memory set, mass coverage, and received-support delta are
the same quantities as the dense definition up to floating-point summation
roundoff.
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
            if bool((self.weight <= 0).any()) or not bool(
                torch.isfinite(self.weight).all()
            ):
                raise ValueError("retained RR weights must be positive and finite")
            if len(self.query) > 1 and bool((self.query[1:] < self.query[:-1]).any()):
                raise ValueError("sparse RR events must be sorted by query")
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
        if any(
            value.shape != row_shape for value in (self.tail_mass, self.edge_count)
        ):
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
            raise FloatingPointError(
                "source-set materialization produced non-finite values"
            )
        if bool((self.tail_mass < -1e-6).any()):
            raise ValueError("selected route mass exceeds the retained row mass")
        return self

    def tensor_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.route_source,
            self.route_weight,
            self.route_received,
            self.route_received_delta,
            self.route_mask,
            self.memory_source,
            self.memory_received,
            self.memory_received_delta,
            self.memory_current_weight,
            self.memory_mask,
            self.total_mass,
            self.tail_mass,
            self.edge_count,
        )

    @classmethod
    def from_tensor_tuple(cls, values: tuple[torch.Tensor, ...]) -> "LayerSourceSets":
        if len(values) != 13:
            raise ValueError("LayerSourceSets tensor tuple has the wrong length")
        return cls(*values)


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
            layer.validate(
                num_heads=self.num_heads, response_count=self.response_count
            )
        return self

    @torch.no_grad()
    def materialize_layer(
        self,
        layer_index: int,
        config: SourceSetConfig,
        *,
        device: str | torch.device,
    ) -> LayerSourceSets:
        """Build one layer's exact bounded source sets in query chunks.

        The dense reference definition is

        ``received[h,t,j] = sum_{u<=t} A[h,u,j] / (t-j+1)`` for ``j<t``.

        This implementation evaluates the same definition with a running
        ``[head, source]`` cumulative state and a temporary
        ``[head, query_chunk, source]`` block. Peak materialization memory is
        therefore ``O(H * C * T)`` instead of ``O(H * T^2)``, where ``C`` is
        ``materialize_query_chunk_size``.
        """

        config.validate()
        layer_index = int(layer_index)
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("layer index is outside source-set graph")
        device = torch.device(device)
        layer = self.layers[layer_index]
        heads, tokens = self.num_heads, self.response_count
        dtype = torch.float32
        route_count = int(config.max_route_sources)
        memory_count = int(config.max_memory_sources)
        chunk_size = min(int(config.materialize_query_chunk_size), tokens)

        route_source_out = torch.zeros(
            (tokens, heads, route_count), dtype=torch.int32, device=device
        )
        route_weight_out = torch.zeros(
            (tokens, heads, route_count), dtype=dtype, device=device
        )
        route_received_out = torch.zeros_like(route_weight_out)
        route_delta_out = torch.zeros_like(route_weight_out)
        route_mask_out = torch.zeros(
            (tokens, heads, route_count), dtype=torch.bool, device=device
        )
        memory_source_out = torch.zeros(
            (tokens, heads, memory_count), dtype=torch.int32, device=device
        )
        memory_received_out = torch.zeros(
            (tokens, heads, memory_count), dtype=dtype, device=device
        )
        memory_delta_out = torch.zeros_like(memory_received_out)
        memory_current_out = torch.zeros_like(memory_received_out)
        memory_mask_out = torch.zeros(
            (tokens, heads, memory_count), dtype=torch.bool, device=device
        )
        total_mass_out = torch.zeros((tokens, heads), dtype=dtype, device=device)
        tail_mass_out = torch.zeros_like(total_mass_out)
        edge_count_out = torch.zeros_like(total_mass_out)

        running = torch.zeros((heads, tokens), dtype=dtype, device=device)
        source_axis = torch.arange(tokens, device=device, dtype=torch.long)
        query_cpu = layer.query

        for start in range(0, tokens, chunk_size):
            end = min(tokens, start + chunk_size)
            width = end - start
            current = torch.zeros(
                (heads, width, tokens), dtype=dtype, device=device
            )
            if len(query_cpu):
                left = int(
                    torch.searchsorted(
                        query_cpu,
                        torch.tensor(start, dtype=query_cpu.dtype),
                        right=False,
                    ).item()
                )
                right = int(
                    torch.searchsorted(
                        query_cpu,
                        torch.tensor(end, dtype=query_cpu.dtype),
                        right=False,
                    ).item()
                )
                if right > left:
                    edge_head = layer.head[left:right].to(
                        device=device, dtype=torch.long, non_blocking=True
                    )
                    edge_query = (
                        layer.query[left:right].to(
                            device=device, dtype=torch.long, non_blocking=True
                        )
                        - start
                    )
                    edge_source = layer.source[left:right].to(
                        device=device, dtype=torch.long, non_blocking=True
                    )
                    edge_weight = layer.weight[left:right].to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                    current.index_put_(
                        (edge_head, edge_query, edge_source),
                        edge_weight,
                        accumulate=True,
                    )

            cumulative = running[:, None, :] + current.cumsum(dim=1)
            query_values = torch.arange(
                start, end, device=device, dtype=torch.long
            )
            query_axis = query_values[:, None]
            causal = source_axis[None, :] < query_axis
            age = (query_axis - source_axis[None, :] + 1).clamp_min(1).to(dtype)
            received = cumulative / age[None, :, :]
            received.masked_fill_(~causal[None, :, :], 0.0)
            query_for_selected = query_values.view(1, width, 1)

            route_weight, route_source = _topk_padded(current, route_count)
            route_received = torch.gather(received, 2, route_source)
            route_previous_age = (
                query_for_selected - route_source
            ).clamp_min(1).to(dtype)
            route_previous = (
                torch.gather(cumulative, 2, route_source) - route_weight
            ) / route_previous_age
            route_delta = route_received - route_previous

            total_mass = current.sum(dim=2)
            edge_count = (current > 0).sum(dim=2).to(dtype)
            before = route_weight.cumsum(dim=2) - route_weight
            route_keep = (route_weight > 0) & (
                before
                < float(config.route_mass_coverage)
                * total_mass[:, :, None].clamp_min(config.epsilon)
            )
            selected_mass = (route_weight * route_keep).sum(dim=2)
            tail_mass = (total_mass - selected_mass).clamp_min(0.0)

            memory_received, memory_source = _topk_padded(
                received, memory_count
            )
            memory_current = torch.gather(current, 2, memory_source)
            memory_previous_age = (
                query_for_selected - memory_source
            ).clamp_min(1).to(dtype)
            memory_previous = (
                torch.gather(cumulative, 2, memory_source) - memory_current
            ) / memory_previous_age
            memory_delta = memory_received - memory_previous
            memory_keep = memory_received > 0

            route_source_out[start:end] = route_source.permute(1, 0, 2).to(
                torch.int32
            )
            route_weight_out[start:end] = route_weight.permute(1, 0, 2)
            route_received_out[start:end] = route_received.permute(1, 0, 2)
            route_delta_out[start:end] = route_delta.permute(1, 0, 2)
            route_mask_out[start:end] = route_keep.permute(1, 0, 2)
            memory_source_out[start:end] = memory_source.permute(1, 0, 2).to(
                torch.int32
            )
            memory_received_out[start:end] = memory_received.permute(1, 0, 2)
            memory_delta_out[start:end] = memory_delta.permute(1, 0, 2)
            memory_current_out[start:end] = memory_current.permute(1, 0, 2)
            memory_mask_out[start:end] = memory_keep.permute(1, 0, 2)
            total_mass_out[start:end] = total_mass.transpose(0, 1)
            tail_mass_out[start:end] = tail_mass.transpose(0, 1)
            edge_count_out[start:end] = edge_count.transpose(0, 1)
            # Clone the last cumulative row so the full chunk storage can be
            # released before the next query block.
            running = cumulative[:, -1].clone()

        return LayerSourceSets(
            route_source=route_source_out,
            route_weight=route_weight_out,
            route_received=route_received_out,
            route_received_delta=route_delta_out,
            route_mask=route_mask_out,
            memory_source=memory_source_out,
            memory_received=memory_received_out,
            memory_received_delta=memory_delta_out,
            memory_current_weight=memory_current_out,
            memory_mask=memory_mask_out,
            total_mass=total_mass_out,
            tail_mass=tail_mass_out,
            edge_count=edge_count_out,
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
            head = torch.cat(head_parts).long()
            query = torch.cat(query_parts).long()
            source = torch.cat(source_parts).long()
            weight = torch.cat(weight_parts).float()
            key = (
                query * (heads * response_count)
                + head * response_count
                + source
            )
            order = torch.argsort(key, stable=True)
            layer = SparseRRLayer(
                head=head[order],
                query=query[order],
                source=source[order],
                weight=weight[order],
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


def _topk_padded(
    values: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
                torch.zeros(
                    pad_shape, device=values.device, dtype=values.dtype
                ),
            ),
            dim=-1,
        ),
        torch.cat(
            (
                top_index,
                torch.zeros(
                    pad_shape, device=values.device, dtype=torch.long
                ),
            ),
            dim=-1,
        ),
    )