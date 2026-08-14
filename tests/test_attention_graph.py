import unittest

import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.statistics import TOKEN_FEATURES, direct_lookback, token_statistics


class Sample:
    sample_id = "r1"
    source_id = "s1"
    response_idx = 2
    attention_floor = 0.01
    token_ids = torch.tensor([10, 11, 12, 13])
    attention_diagonal = torch.tensor([
        [[0.8, 0.7, 0.6, 0.5], [0.7, 0.6, 0.5, 0.4]]
    ], dtype=torch.float16)
    response_row_ptr = torch.tensor([0, 2, 4, 5, 6])
    response_column_indices = torch.tensor([0, 1, 0, 2, 0, 2], dtype=torch.int32)
    response_values = torch.tensor([0.2, 0.4, 0.1, 0.3, 0.6, 0.5], dtype=torch.float16)
    num_layers = 1
    num_heads = 2
    num_tokens = 4
    num_response_tokens = 2
    num_channels = 2


class AttentionGraphTests(unittest.TestCase):
    def test_undefined_legacy_lookback_uses_attention_floor(self):
        sample = Sample()
        sample.attention_diagonal = torch.zeros_like(sample.attention_diagonal)
        sample.response_row_ptr = torch.zeros(5, dtype=torch.int64)
        sample.response_column_indices = torch.empty(0, dtype=torch.int32)
        sample.response_values = torch.empty(0, dtype=torch.float16)

        anomaly = direct_lookback(sample)

        torch.testing.assert_close(anomaly, torch.full((2,), .99))

    def test_graph_preserves_pair_topology_and_channel_traces(self):
        graph = build_attention_graph(Sample(), GraphBuildConfig())
        self.assertEqual(graph.edge_index.tolist(), [[0, 1, 0, 2], [2, 2, 3, 3]])
        self.assertEqual(graph.edge_type.tolist(), [0, 0, 0, 1])
        self.assertEqual(graph.trace_edge_id.numel(), 6)
        self.assertEqual(graph.trace_channel.numel(), 6)
        self.assertEqual(graph.node_attr.shape, (4, 2))

    def test_statistics_remain_finite_diagnostics(self):
        graph = build_attention_graph(Sample())
        values = token_statistics(graph)
        self.assertEqual(values.shape, (2, len(TOKEN_FEATURES)))
        self.assertTrue(torch.isfinite(values).all())

if __name__ == "__main__":
    unittest.main()
