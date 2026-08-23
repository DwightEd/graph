from dataclasses import replace
from types import SimpleNamespace

import torch

from experiments.graph_structure_audit.config import RecoveryConfig
from experiments.source_reuse_contrast.data import SourceReuseGraph


def raw_graph(sample_id="sample", source_id="source", labels=None):
    response_idx = 3
    response_tokens = 4
    layers = 3
    heads = 2
    events = []
    # Each response token reads multiple exact prompt/response sources with
    # different layer-head patterns.
    sources_by_token = [
        [0, 1],
        [1, 2, 3],
        [0, 3, 4],
        [1, 4, 5],
    ]
    for query, sources in enumerate(sources_by_token):
        for source in sources:
            for layer in range(layers):
                head = (source + query + layer) % heads
                weight = 0.08 + 0.03 * (layer + 1) + 0.01 * (head + 1)
                events.append((query, source, layer, head, weight))
                if (source + layer) % 2 == 0:
                    other = 1 - head
                    events.append((query, source, layer, other, weight * 0.5))
    events.sort()
    query = torch.tensor([item[0] for item in events], dtype=torch.long)
    source = torch.tensor([item[1] for item in events], dtype=torch.long)
    layer = torch.tensor([item[2] for item in events], dtype=torch.long)
    head = torch.tensor([item[3] for item in events], dtype=torch.long)
    weight = torch.tensor([item[4] for item in events], dtype=torch.float32)
    counts = torch.bincount(query, minlength=response_tokens)
    query_ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    diagonal = torch.full((response_tokens, layers, heads), 0.08)
    graph = SourceReuseGraph(
        sample_id=sample_id,
        source_id=source_id,
        task_type="Data2txt",
        response_idx=response_idx,
        num_response_tokens=response_tokens,
        num_tokens=response_idx + response_tokens,
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
    return graph, [0, 1, 0, 0] if labels is None else labels


class Sample:
    def __init__(self, graph, labels):
        self.graph = graph
        self.sample_id = graph.sample_id
        self.source_id = graph.source_id
        self.task_type = graph.task_type
        self.labels = torch.tensor(labels, dtype=torch.int8)
        self._attention = None
        self.release_calls = 0

    def attention(self):
        if self._attention is None:
            diagonal = torch.zeros(
                self.graph.num_layers,
                self.graph.num_heads,
                self.graph.num_tokens,
            )
            diagonal[:, :, self.graph.response_idx :] = self.graph.diagonal.permute(
                1, 2, 0
            )
            self._attention = SimpleNamespace(
                response_idx=self.graph.response_idx,
                response_values=torch.empty(0),
                num_tokens=self.graph.num_tokens,
                num_response_tokens=self.graph.num_response_tokens,
                num_layers=self.graph.num_layers,
                num_heads=self.graph.num_heads,
                attention_floor=self.graph.attention_floor,
                attention_diagonal=diagonal,
            )
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=8192):
        del block_rows
        yield SimpleNamespace(
            layer=self.graph.layer,
            head=self.graph.head,
            query=self.graph.query,
            source=self.graph.source,
            weight=self.graph.weight,
        )

    def release_attention(self):
        self._attention = None
        self.release_calls += 1


class LabelStore:
    def response_labels(self, sample):
        return sample.labels


class Dataset:
    def __init__(self, samples):
        self.items = {item.sample_id: item for item in samples}
        self.sample_ids = list(self.items)

    def __getitem__(self, item):
        return self.items[str(item)]

    def prepare_evaluation_labels(self):
        return LabelStore()


def dataset(prefix, count, *, labels=False):
    samples = []
    for index in range(count):
        graph, default_labels = raw_graph(
            sample_id=f"{prefix}-{index}",
            source_id=f"{prefix}-source-{index}",
            labels=[0, index % 2, 0, 0] if labels else None,
        )
        samples.append(Sample(graph, default_labels))
    return Dataset(samples)


def tiny_config(**changes):
    config = RecoveryConfig(
        hidden_dim=16,
        role_dim=4,
        position_dim=4,
        lag_bins=8,
        channel_mask_rate=0.3,
        pair_layer_mask_rate=0.2,
        diagonal_mask_rate=0.3,
        dropout=0.0,
        epochs=1,
        validation_fraction=0.25,
        patience=1,
        score_rounds=1,
        show_progress=False,
    )
    return replace(config, **changes)
