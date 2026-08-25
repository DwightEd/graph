from dataclasses import replace
from types import SimpleNamespace

import torch

from research_dataset import SparseAttentionBlock

from experiments.grounded_route.graph import TokenEdges, TokenGraph


def make_graph(
    *,
    layers: int = 3,
    heads: int = 4,
    response_start: int = 3,
    response_count: int = 7,
) -> TokenGraph:
    token_count = response_start + response_count
    source: list[int] = []
    target: list[int] = []
    layer: list[int] = []
    head: list[int] = []
    weight: list[float] = []

    for current_layer in range(layers):
        for current_head in range(heads):
            for current_target in range(response_start, token_count):
                prompt_source = (current_target + current_layer + current_head) % response_start
                source.append(prompt_source)
                target.append(current_target)
                layer.append(current_layer)
                head.append(current_head)
                weight.append(0.11 + 0.01 * (current_head % 2))

                if current_target >= response_start + 2:
                    source.append(current_target - 2)
                    target.append(current_target)
                    layer.append(current_layer)
                    head.append(current_head)
                    weight.append(0.16)

    edges = TokenEdges(
        source=torch.tensor(source, dtype=torch.long),
        target=torch.tensor(target, dtype=torch.long),
        layer=torch.tensor(layer, dtype=torch.long),
        head=torch.tensor(head, dtype=torch.long),
        weight=torch.tensor(weight, dtype=torch.float32),
    )
    retained = torch.zeros((response_count, layers, heads), dtype=torch.float32)
    retained.index_put_(
        (edges.target - response_start, edges.layer, edges.head),
        edges.weight,
        accumulate=True,
    )
    diagonal = torch.full_like(retained, 0.05)
    unresolved = 1.0 - retained - diagonal
    return TokenGraph(
        sample_id="synthetic",
        source_id="source",
        task_type="QA",
        response_start=response_start,
        token_count=token_count,
        response_count=response_count,
        layer_count=layers,
        head_count=heads,
        attention_floor=0.01,
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
        token_ids=torch.arange(100, 100 + token_count),
    ).check()


def make_rewirable_graph() -> TokenGraph:
    graph = make_graph(layers=1, heads=1)
    edges = TokenEdges(
        source=torch.tensor([0, 1, 3, 4]),
        target=torch.tensor([6, 7, 8, 9]),
        layer=torch.zeros(4, dtype=torch.long),
        head=torch.zeros(4, dtype=torch.long),
        weight=torch.tensor([0.11, 0.17, 0.19, 0.23]),
    )
    retained = torch.zeros((graph.response_count, 1, 1))
    retained.index_put_(
        (edges.target - graph.response_start, edges.layer, edges.head),
        edges.weight,
        accumulate=True,
    )
    diagonal = torch.full_like(retained, 0.05)
    return replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=1.0 - retained - diagonal,
    ).check()


def make_weight_shuffle_graph() -> TokenGraph:
    graph = make_graph(layers=1, heads=1)
    edges = TokenEdges(
        source=torch.tensor([0, 1, 3, 4]),
        target=torch.tensor([8, 8, 9, 9]),
        layer=torch.zeros(4, dtype=torch.long),
        head=torch.zeros(4, dtype=torch.long),
        weight=torch.tensor([0.07, 0.17, 0.13, 0.23]),
    )
    retained = torch.zeros((graph.response_count, 1, 1))
    retained.index_put_(
        (edges.target - graph.response_start, edges.layer, edges.head),
        edges.weight,
        accumulate=True,
    )
    diagonal = torch.full_like(retained, 0.05)
    return replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=1.0 - retained - diagonal,
    ).check()


def permute_edge_storage(graph: TokenGraph, order: torch.Tensor) -> TokenGraph:
    edges = graph.edges
    return replace(
        graph,
        edges=TokenEdges(
            source=edges.source[order],
            target=edges.target[order],
            layer=edges.layer[order],
            head=edges.head[order],
            weight=edges.weight[order],
        ),
    ).check()


def replace_edge_weights(graph: TokenGraph, selected: torch.Tensor, value: float) -> TokenGraph:
    weight = graph.edges.weight.clone()
    weight[selected] = value
    return replace(
        graph,
        edges=TokenEdges(
            source=graph.edges.source,
            target=graph.edges.target,
            layer=graph.edges.layer,
            head=graph.edges.head,
            weight=weight,
        ),
    ).check()


class SparseSample:
    def __init__(self, *, layers: int = 32, heads: int = 32) -> None:
        self.sample_id = "csr-1024"
        self.source_id = "source"
        self.task_type = "QA"
        self.response_start = 2
        self.response_count = 2
        self.token_count = self.response_start + self.response_count

        channel = torch.arange(layers * heads, dtype=torch.long)
        layer = torch.div(channel, heads, rounding_mode="floor").repeat_interleave(
            self.response_count
        )
        head = channel.remainder(heads).repeat_interleave(self.response_count)
        query = torch.arange(self.response_count).repeat(layers * heads)
        target = self.response_start + query
        source = torch.zeros_like(target)
        weight = 0.10 + (layer * heads + head).float() * 1e-6
        self.block = SparseAttentionBlock(
            row=torch.arange(len(weight)),
            layer=layer,
            head=head,
            query=query,
            target=target,
            source=source,
            weight=weight,
        )
        diagonal = torch.full((layers, heads, self.token_count), 0.05)
        self._attention = SimpleNamespace(
            response_idx=self.response_start,
            num_response_tokens=self.response_count,
            num_tokens=self.token_count,
            num_layers=layers,
            num_heads=heads,
            num_channels=layers * heads,
            attention_floor=0.01,
            attention_diagonal=diagonal,
            response_values=weight,
            token_ids=torch.arange(self.token_count),
        )

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        yield self.block
