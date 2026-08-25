"""Build one sparse multiplex attention graph per prompt-response sample.

Tokens are graph nodes. Every retained attention incidence keeps its exact
source, response target, Transformer layer, head and weight. Missing sparse
entries are represented by row-level unresolved mass instead of fake zeros.
"""

from dataclasses import dataclass

import torch

from .config import GraphConfig


@dataclass(frozen=True)
class AttentionEdges:
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    weight: torch.Tensor
    layer_pointer: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.source.numel())

    def layer_slice(self, layer: int) -> slice:
        start = int(self.layer_pointer[layer].item())
        stop = int(self.layer_pointer[layer + 1].item())
        return slice(start, stop)

    def to(self, device) -> "AttentionEdges":
        return AttentionEdges(
            source=self.source.to(device),
            target=self.target.to(device),
            layer=self.layer.to(device),
            head=self.head.to(device),
            weight=self.weight.to(device),
            layer_pointer=self.layer_pointer.to(device),
        )


@dataclass(frozen=True)
class AttentionGraph:
    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    token_count: int
    response_count: int
    layer_count: int
    head_count: int
    attention_floor: float
    edges: AttentionEdges
    diagonal: torch.Tensor
    unresolved: torch.Tensor
    response_token_ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.edges.weight.device

    @property
    def edge_count(self) -> int:
        return self.edges.count

    def to(self, device) -> "AttentionGraph":
        return AttentionGraph(
            sample_id=self.sample_id,
            source_id=self.source_id,
            task_type=self.task_type,
            response_start=self.response_start,
            token_count=self.token_count,
            response_count=self.response_count,
            layer_count=self.layer_count,
            head_count=self.head_count,
            attention_floor=self.attention_floor,
            edges=self.edges.to(device),
            diagonal=self.diagonal.to(device),
            unresolved=self.unresolved.to(device),
            response_token_ids=self.response_token_ids.to(device),
        )

    def check(self) -> "AttentionGraph":
        edge_shape = (self.edge_count,)
        assert self.edges.source.shape == edge_shape
        assert self.edges.target.shape == edge_shape
        assert self.edges.layer.shape == edge_shape
        assert self.edges.head.shape == edge_shape
        assert self.edges.weight.shape == edge_shape
        assert self.edges.layer_pointer.shape == (self.layer_count + 1,)
        assert self.diagonal.shape == (
            self.response_count,
            self.layer_count,
            self.head_count,
        )
        assert self.unresolved.shape == self.diagonal.shape
        assert not self.edge_count or bool((self.edges.source < self.edges.target).all())
        return self


def read_sparse_entries(sample, config: GraphConfig) -> tuple[torch.Tensor, ...]:
    columns: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("source", "target", "layer", "head", "weight")
    }
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        keep = (
            (block.source < block.target)
            & (block.source != block.target)
            & (block.weight > config.minimum_edge_weight)
        )
        if not bool(keep.any()):
            continue
        columns["source"].append(block.source[keep].long())
        columns["target"].append(block.target[keep].long())
        columns["layer"].append(block.layer[keep].long())
        columns["head"].append(block.head[keep].long())
        columns["weight"].append(block.weight[keep].float().clamp_min(0.0))

    if columns["source"]:
        return tuple(torch.cat(columns[name]) for name in columns)

    device = sample.attention().response_values.device
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_float = torch.empty(0, dtype=torch.float32, device=device)
    return empty_long, empty_long, empty_long, empty_long, empty_float


def normalize_rows(
    attention,
    source: torch.Tensor,
    target: torch.Tensor,
    layer: torch.Tensor,
    head: torch.Tensor,
    weight: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    response_start = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    layer_count = int(attention.num_layers)
    head_count = int(attention.num_heads)

    diagonal = (
        attention.attention_diagonal[:, :, response_start:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    retained = torch.zeros(
        (response_count, layer_count, head_count),
        dtype=torch.float32,
        device=weight.device,
    )
    if weight.numel():
        retained.index_put_(
            (target - response_start, layer, head),
            weight,
            accumulate=True,
        )

    known = retained + diagonal
    excess = (known - 1.0).clamp_min(0.0)
    if excess.numel() and float(excess.max().item()) > tolerance:
        raise ValueError("attention row mass exceeds the cache tolerance")

    scale = torch.where(known > 1.0, known.reciprocal(), torch.ones_like(known))
    if weight.numel():
        weight = weight * scale[target - response_start, layer, head]
    diagonal = diagonal * scale
    unresolved = (1.0 - retained * scale - diagonal).clamp_min(0.0)
    return weight, diagonal, unresolved


def sort_edges(
    source: torch.Tensor,
    target: torch.Tensor,
    layer: torch.Tensor,
    head: torch.Tensor,
    weight: torch.Tensor,
    token_count: int,
    response_start: int,
    response_count: int,
    layer_count: int,
    head_count: int,
) -> AttentionEdges:
    if not weight.numel():
        pointer = torch.zeros(layer_count + 1, dtype=torch.long, device=weight.device)
        return AttentionEdges(source, target, layer, head, weight, pointer)

    response_target = target - response_start
    key = (
        (((layer * response_count) + response_target) * head_count + head)
        * token_count
        + source
    )
    order = torch.argsort(key)
    source = source[order]
    target = target[order]
    layer = layer[order]
    head = head[order]
    weight = weight[order]

    counts = torch.bincount(layer, minlength=layer_count)
    pointer = torch.zeros(layer_count + 1, dtype=torch.long, device=weight.device)
    pointer[1:] = torch.cumsum(counts, dim=0)
    return AttentionEdges(source, target, layer, head, weight, pointer)


@torch.no_grad()
def build_graph(sample, config: GraphConfig | None = None) -> AttentionGraph:
    config = GraphConfig() if config is None else config
    attention = sample.attention()
    source, target, layer, head, weight = read_sparse_entries(sample, config)
    weight, diagonal, unresolved = normalize_rows(
        attention,
        source,
        target,
        layer,
        head,
        weight,
        config.numerical_tolerance,
    )
    edges = sort_edges(
        source,
        target,
        layer,
        head,
        weight,
        int(attention.num_tokens),
        int(attention.response_idx),
        int(attention.num_response_tokens),
        int(attention.num_layers),
        int(attention.num_heads),
    )
    return AttentionGraph(
        sample_id=str(sample.sample_id),
        source_id=str(sample.source_id),
        task_type=str(sample.task_type or ""),
        response_start=int(attention.response_idx),
        token_count=int(attention.num_tokens),
        response_count=int(attention.num_response_tokens),
        layer_count=int(attention.num_layers),
        head_count=int(attention.num_heads),
        attention_floor=float(attention.attention_floor),
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
        response_token_ids=attention.token_ids[int(attention.response_idx):].long(),
    ).check()
