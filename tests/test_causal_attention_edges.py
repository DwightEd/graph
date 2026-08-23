from types import SimpleNamespace

import torch

from experiments.causal_attention_edges import collect_causal_attention_edges


class Sample:
    def __init__(self):
        self.loaded = SimpleNamespace(
            response_idx=2,
            response_values=torch.empty(6),
        )

    def attention(self):
        return self.loaded

    def iter_sparse_attention_blocks(self, block_rows):
        assert block_rows == 2
        yield SimpleNamespace(
            layer=torch.tensor([0, 0, 1]),
            head=torch.tensor([0, 1, 0]),
            query=torch.tensor([0, 0, 1]),
            source=torch.tensor([0, 2, 2]),
            weight=torch.tensor([0.2, 0.9, 0.3]),
        )
        yield SimpleNamespace(
            layer=torch.tensor([1, 1, 1]),
            head=torch.tensor([1, 0, 1]),
            query=torch.tensor([0, 1, 1]),
            source=torch.tensor([1, 3, 0]),
            weight=torch.tensor([0.4, 0.8, -0.1]),
        )


def test_causal_attention_edges_preallocate_and_filter_across_blocks():
    edges = collect_causal_attention_edges(Sample(), block_rows=2)

    assert edges.num_edges == 4
    torch.testing.assert_close(edges.layer, torch.tensor([0, 1, 1, 1]))
    torch.testing.assert_close(edges.head, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(edges.query, torch.tensor([0, 1, 0, 1]))
    torch.testing.assert_close(edges.source, torch.tensor([0, 2, 1, 0]))
    torch.testing.assert_close(edges.weight, torch.tensor([0.2, 0.3, 0.4, 0.0]))
