from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.source_reuse_contrast.data import SourceReuseGraph


def synthetic_graph() -> SourceReuseGraph:
    """Four response tokens with prompt relay and repeated source coalitions."""

    prompt = 3
    tokens = 4
    layers = 2
    heads = 2
    events = [
        (0, 0, 0, 0, 0.40), (0, 1, 0, 1, 0.30),
        (0, 0, 1, 0, 0.25), (0, 1, 1, 1, 0.20),
        (1, 0, 0, 0, 0.30), (1, 3, 0, 1, 0.35),
        (1, 0, 1, 0, 0.20), (1, 3, 1, 1, 0.40),
        (2, 3, 0, 0, 0.25), (2, 4, 0, 1, 0.35), (2, 1, 0, 0, 0.15),
        (2, 3, 1, 0, 0.30), (2, 4, 1, 1, 0.30), (2, 1, 1, 1, 0.10),
        (3, 3, 0, 0, 0.20), (3, 4, 0, 1, 0.30), (3, 5, 0, 0, 0.25), (3, 2, 0, 1, 0.10),
        (3, 3, 1, 0, 0.20), (3, 4, 1, 1, 0.25), (3, 5, 1, 0, 0.25), (3, 2, 1, 1, 0.10),
    ]
    events.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    query = torch.tensor([row[0] for row in events], dtype=torch.long)
    source = torch.tensor([row[1] for row in events], dtype=torch.long)
    layer = torch.tensor([row[2] for row in events], dtype=torch.long)
    head = torch.tensor([row[3] for row in events], dtype=torch.long)
    weight = torch.tensor([row[4] for row in events], dtype=torch.float32)
    counts = torch.bincount(query, minlength=tokens)
    query_ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    diagonal = torch.full((tokens, layers, heads), 0.1, dtype=torch.float32)
    return SourceReuseGraph(
        sample_id="synthetic",
        source_id="source-synthetic",
        task_type="QA",
        response_idx=prompt,
        num_response_tokens=tokens,
        num_tokens=prompt + tokens,
        num_layers=layers,
        num_heads=heads,
        attention_floor=0.01,
        layer=layer,
        head=head,
        query=query,
        source=source,
        weight=weight,
        query_ptr=query_ptr,
        diagonal=diagonal,
    )


class SyntheticBlock:
    def __init__(self, graph: SourceReuseGraph):
        self.layer = graph.layer
        self.head = graph.head
        self.query = graph.query
        self.source = graph.source
        self.weight = graph.weight


class SyntheticSample:
    def __init__(self, sample_id: str, labels: list[int]):
        graph = synthetic_graph()
        self.sample_id = sample_id
        self.source_id = f"source-{sample_id}"
        self.task_type = "QA"
        self.labels = torch.tensor(labels, dtype=torch.int8)
        self._graph = SourceReuseGraph(
            sample_id=sample_id,
            source_id=self.source_id,
            task_type=self.task_type,
            response_idx=graph.response_idx,
            num_response_tokens=graph.num_response_tokens,
            num_tokens=graph.num_tokens,
            num_layers=graph.num_layers,
            num_heads=graph.num_heads,
            attention_floor=graph.attention_floor,
            layer=graph.layer,
            head=graph.head,
            query=graph.query,
            source=graph.source,
            weight=graph.weight,
            query_ptr=graph.query_ptr,
            diagonal=graph.diagonal,
        )
        self._attention = SimpleNamespace(
            response_idx=graph.response_idx,
            num_response_tokens=graph.num_response_tokens,
            num_tokens=graph.num_tokens,
            num_layers=graph.num_layers,
            num_heads=graph.num_heads,
            attention_floor=graph.attention_floor,
            response_values=graph.weight,
            attention_diagonal=torch.cat(
                (
                    torch.zeros(graph.num_layers, graph.num_heads, graph.response_idx),
                    graph.diagonal.permute(1, 2, 0).contiguous(),
                ),
                dim=2,
            ),
        )

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=8192):
        yield SyntheticBlock(self._graph)

    def release_attention(self):
        return None


class SyntheticLabelStore:
    def response_labels(self, sample):
        return sample.labels


class SyntheticDataset:
    def __init__(self, samples: list[SyntheticSample]):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return SyntheticLabelStore()
