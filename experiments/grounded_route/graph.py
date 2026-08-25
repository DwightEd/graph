"""Build one exact sparse token graph from a cached attention sample."""

from dataclasses import dataclass, replace
from functools import cached_property

import torch

from experiment_protocol import canonical_source_group

from .config import GraphConfig

PROMPT = 0
RESPONSE = 1


@dataclass(frozen=True)
class TokenEdges:
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    weight: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.source.numel())

    @property
    def lag(self) -> torch.Tensor:
        return self.target - self.source

    def select(self, index) -> "TokenEdges":
        return TokenEdges(
            source=self.source[index],
            target=self.target[index],
            layer=self.layer[index],
            head=self.head[index],
            weight=self.weight[index],
        )

    def to(self, device) -> "TokenEdges":
        return TokenEdges(
            source=self.source.to(device),
            target=self.target.to(device),
            layer=self.layer.to(device),
            head=self.head.to(device),
            weight=self.weight.to(device),
        )


@dataclass(frozen=True)
class TokenGraph:
    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    token_count: int
    response_count: int
    layer_count: int
    head_count: int
    attention_floor: float
    edges: TokenEdges
    diagonal: torch.Tensor
    unresolved: torch.Tensor
    token_ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @property
    def edge_count(self) -> int:
        return self.edges.count

    @property
    def edge_response_target(self) -> torch.Tensor:
        return self.edges.target - self.response_start

    @property
    def edge_source_role(self) -> torch.Tensor:
        return (self.edges.source >= self.response_start).long()

    @cached_property
    def layer_offsets(self) -> tuple[int, ...]:
        """Offsets into canonical layer-major edge storage."""

        counts = torch.bincount(
            self.edges.layer.detach().cpu(),
            minlength=self.layer_count,
        )
        return (0, *counts.cumsum(0).tolist())

    @cached_property
    def edges_are_canonical(self) -> bool:
        return canonical_edge_order(self)

    @property
    def response_token_ids(self) -> torch.Tensor:
        return self.token_ids[self.response_start :]

    def layer_edges(self, layer: int, device=None) -> TokenEdges:
        """Materialize one layer of edges on the requested compute device."""

        start, stop = self.layer_offsets[layer : layer + 2]
        edges = self.edges.select(slice(start, stop))
        return edges if device is None else edges.to(device)

    def to(self, device) -> "TokenGraph":
        """Move dense node state to ``device`` while retaining sparse edges on CPU."""

        graph = TokenGraph(
            sample_id=self.sample_id,
            source_id=self.source_id,
            task_type=self.task_type,
            response_start=self.response_start,
            token_count=self.token_count,
            response_count=self.response_count,
            layer_count=self.layer_count,
            head_count=self.head_count,
            attention_floor=self.attention_floor,
            edges=self.edges.to("cpu"),
            diagonal=self.diagonal.to(device),
            unresolved=self.unresolved.to(device),
            token_ids=self.token_ids.to("cpu"),
        )
        if self.edges_are_canonical:
            graph.__dict__["edges_are_canonical"] = True
        return graph

    def truncate_response(self, count: int) -> "TokenGraph":
        """Return the exact prefix graph containing ``count`` response nodes."""

        count = int(count)
        if not 0 <= count <= self.response_count:
            raise ValueError("response prefix is outside the graph")
        token_count = self.response_start + count
        keep = self.edges.target < token_count
        return TokenGraph(
            sample_id=self.sample_id,
            source_id=self.source_id,
            task_type=self.task_type,
            response_start=self.response_start,
            token_count=token_count,
            response_count=count,
            layer_count=self.layer_count,
            head_count=self.head_count,
            attention_floor=self.attention_floor,
            edges=self.edges.select(keep),
            diagonal=self.diagonal[:count],
            unresolved=self.unresolved[:count],
            token_ids=self.token_ids[:token_count],
        ).check().canonicalize()

    def canonicalize(self) -> "TokenGraph":
        """Return canonical layer/head/target/source edge storage."""

        if self.edges_are_canonical:
            return self
        order = torch.argsort(endpoint_storage_keys(self), stable=True)
        graph = replace(self, edges=self.edges.select(order)).check()
        graph.__dict__["edges_are_canonical"] = True
        return graph

    def check(self) -> "TokenGraph":
        edge_shape = (self.edge_count,)
        assert self.edges.source.shape == edge_shape
        assert self.edges.target.shape == edge_shape
        assert self.edges.layer.shape == edge_shape
        assert self.edges.head.shape == edge_shape
        assert self.edges.weight.shape == edge_shape
        assert self.diagonal.shape == (
            self.response_count,
            self.layer_count,
            self.head_count,
        )
        assert self.unresolved.shape == self.diagonal.shape
        assert self.token_ids.shape == (self.token_count,)
        assert self.response_start + self.response_count == self.token_count
        if self.edge_count:
            assert bool((self.edges.source < self.edges.target).all())
            assert bool((self.edges.target >= self.response_start).all())
            assert bool((self.edges.target < self.token_count).all())
            assert bool((self.edges.layer < self.layer_count).all())
            assert bool((self.edges.head < self.head_count).all())
        return self


def endpoint_storage_keys(graph: TokenGraph) -> torch.Tensor:
    """Packed keys in canonical ``layer, head, target, source`` order."""

    row = (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )
    return row * graph.token_count + graph.edges.source


def canonical_edge_order(graph: TokenGraph, block_size: int = 1_000_000) -> bool:
    """Check canonical order with bounded temporary memory."""

    edges = graph.edges
    for start in range(1, graph.edge_count, block_size):
        stop = min(start + block_size, graph.edge_count)
        left = slice(start - 1, stop - 1)
        right = slice(start, stop)
        layer_equal = edges.layer[right] == edges.layer[left]
        head_equal = layer_equal & (edges.head[right] == edges.head[left])
        target_equal = head_equal & (edges.target[right] == edges.target[left])
        invalid = edges.layer[right] < edges.layer[left]
        invalid |= layer_equal & (edges.head[right] < edges.head[left])
        invalid |= head_equal & (edges.target[right] < edges.target[left])
        invalid |= target_equal & (edges.source[right] < edges.source[left])
        if bool(invalid.any()):
            return False
    return True


def read_edges(sample, block_rows: int) -> TokenEdges:
    """Decode CSR blocks into one preallocated CPU edge table."""

    capacity = int(sample.attention().response_values.numel())
    columns = {
        "source": torch.empty(capacity, dtype=torch.long),
        "target": torch.empty(capacity, dtype=torch.long),
        "layer": torch.empty(capacity, dtype=torch.long),
        "head": torch.empty(capacity, dtype=torch.long),
        "weight": torch.empty(capacity, dtype=torch.float32),
    }
    cursor = 0
    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        keep = block.source < block.target
        if not bool(keep.any()):
            continue
        count = int(keep.sum().item())
        stop = cursor + count
        columns["source"][cursor:stop].copy_(block.source[keep].to("cpu", torch.long))
        columns["target"][cursor:stop].copy_(block.target[keep].to("cpu", torch.long))
        columns["layer"][cursor:stop].copy_(block.layer[keep].to("cpu", torch.long))
        columns["head"][cursor:stop].copy_(block.head[keep].to("cpu", torch.long))
        columns["weight"][cursor:stop].copy_(
            block.weight[keep].to("cpu", torch.float32).clamp_min_(0.0)
        )
        cursor = stop

    return TokenEdges(*(columns[name][:cursor] for name in columns))


def conserve_rows(
    edges: TokenEdges,
    diagonal: torch.Tensor,
    response_start: int,
    tolerance: float,
) -> tuple[TokenEdges, torch.Tensor, torch.Tensor]:
    """Conserve retained, diagonal and censored mass in every attention row."""

    response_count, layer_count, head_count = diagonal.shape
    retained = diagonal.new_zeros(diagonal.shape)
    if edges.count:
        retained.index_put_(
            (
                edges.target - response_start,
                edges.layer,
                edges.head,
            ),
            edges.weight,
            accumulate=True,
        )

    known = retained + diagonal
    if known.numel() and float((known - 1.0).clamp_min(0.0).max().item()) > tolerance:
        raise ValueError("attention row mass exceeds the cache tolerance")

    scale = torch.where(known > 1.0, known.reciprocal(), torch.ones_like(known))
    diagonal = diagonal * scale
    weight = edges.weight
    for start in range(0, edges.count, 1_000_000):
        stop = min(start + 1_000_000, edges.count)
        weight[start:stop].mul_(
            scale[
                edges.target[start:stop] - response_start,
                edges.layer[start:stop],
                edges.head[start:stop],
            ]
        )
    unresolved = (1.0 - retained * scale - diagonal).clamp_min(0.0)
    normalized = TokenEdges(
        edges.source,
        edges.target,
        edges.layer,
        edges.head,
        weight,
    )
    return normalized, diagonal, unresolved


@torch.no_grad()
def build_graph(sample, config: GraphConfig | None = None) -> TokenGraph:
    config = GraphConfig() if config is None else config
    attention = sample.attention()
    response_start = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    layer_count = int(attention.num_layers)
    head_count = int(attention.num_heads)

    edges = read_edges(sample, config.block_rows)
    diagonal = (
        attention.attention_diagonal[:, :, response_start:]
        .to(device="cpu", dtype=torch.float32)
        .permute(2, 0, 1)
        .contiguous()
    )
    edges, diagonal, unresolved = conserve_rows(
        edges,
        diagonal,
        response_start,
        config.numerical_tolerance,
    )

    return TokenGraph(
        sample_id=str(sample.sample_id),
        source_id=canonical_source_group(sample),
        task_type=str(sample.task_type or ""),
        response_start=response_start,
        token_count=int(attention.num_tokens),
        response_count=response_count,
        layer_count=layer_count,
        head_count=head_count,
        attention_floor=float(attention.attention_floor),
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
        token_ids=attention.token_ids.to(device="cpu", dtype=torch.long),
    ).check().canonicalize()
