import unittest

import torch

from cache import AttentionSample
from hypergraph import build_attention_hypergraph


class AttentionHypergraphTests(unittest.TestCase):
    def make_sample(self) -> AttentionSample:
        # N=5 with two prompt tokens and three generated response tokens.
        # Sparse rows are channel-major, then response-target-major.
        row_ptr = torch.tensor([0, 2, 4, 5, 6, 7, 9], dtype=torch.int64)
        columns = torch.tensor([0, 1, 0, 2, 1, 1, 2, 0, 3], dtype=torch.int32)
        values = torch.tensor([0.6, 0.5, 0.7, 0.8, 0.5, 0.9, 0.7, 0.9, 0.9])
        diagonal = torch.tensor(
            [[[10.0, 11.0, 12.0, 13.0, 14.0], [20.0, 21.0, 22.0, 23.0, 24.0]]]
        )
        return AttentionSample(
            sample_id="sample-1",
            source_id="source-1",
            response_idx=2,
            token_ids=torch.tensor([101, 102, 201, 202, 203], dtype=torch.int64),
            attention_diagonal=diagonal,
            response_row_ptr=row_ptr,
            response_column_indices=columns,
            response_values=values,
            attention_floor=0.5,
        )

    def test_builds_typed_thresholded_hyperedges_in_deterministic_order(self) -> None:
        graph = build_attention_hypergraph(self.make_sample(), tau=0.5)

        self.assertEqual(graph.sample_id, "sample-1")
        self.assertEqual(graph.source_id, "source-1")
        self.assertEqual(graph.response_idx, 2)
        torch.testing.assert_close(
            graph.node_attr,
            torch.tensor(
                [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0], [14.0, 24.0]]
            ),
        )
        torch.testing.assert_close(
            graph.incidence_index,
            torch.tensor(
                [[0, 2, 0, 3, 2, 3, 1, 2, 2, 3, 0, 4, 3, 4],
                 [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]],
                dtype=torch.int64,
            ),
        )
        torch.testing.assert_close(
            graph.incidence_weight,
            torch.tensor([0.6, 12.0, 0.7, 13.0, 0.8, 13.0, 0.9, 22.0,
                          0.7, 23.0, 0.9, 24.0, 0.9, 24.0]),
        )
        torch.testing.assert_close(
            graph.hyperedge_target,
            torch.tensor([2, 3, 3, 2, 3, 4, 4], dtype=torch.int64),
        )
        torch.testing.assert_close(
            graph.hyperedge_channel,
            torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            graph.hyperedge_type,
            torch.tensor([0, 0, 1, 0, 1, 0, 1], dtype=torch.int8),
        )

    def test_preserves_dtypes_device_and_excludes_labels_from_serialization(self) -> None:
        sample = self.make_sample()
        graph = build_attention_hypergraph(sample, tau=0.5)

        self.assertEqual(graph.node_attr.dtype, sample.attention_diagonal.dtype)
        self.assertEqual(graph.incidence_index.dtype, torch.int64)
        self.assertEqual(graph.incidence_weight.dtype, sample.response_values.dtype)
        self.assertEqual(graph.hyperedge_target.dtype, torch.int64)
        self.assertEqual(graph.hyperedge_channel.dtype, torch.int32)
        self.assertEqual(graph.hyperedge_type.dtype, torch.int8)
        for value in graph.to_dict().values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device, sample.token_ids.device)
        self.assertEqual(
            list(graph.to_dict()),
            [
                "sample_id", "source_id", "response_idx", "token_ids", "node_attr",
                "incidence_index", "incidence_weight", "hyperedge_target",
                "hyperedge_channel", "hyperedge_type",
            ],
        )
        self.assertNotIn("label", graph.to_dict())

    def test_rejects_a_threshold_below_the_sparse_attention_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention_floor"):
            build_attention_hypergraph(self.make_sample(), tau=0.49)


if __name__ == "__main__":
    unittest.main()
