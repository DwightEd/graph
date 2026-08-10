import unittest
from types import SimpleNamespace

import torch

from graphs import build_original_graph, build_relation_topk_graph


def make_sample(device: torch.device | str = "cpu") -> SimpleNamespace:
    """A two-channel retained-attention cache for three response tokens."""
    values_by_row = [
        [(0, 0.1), (1, 0.7)],
        [(0, 0.4), (1, 0.2), (2, 0.9)],
        [(0, 0.3), (1, 0.5), (2, 0.1), (3, 0.8)],
        [(0, 0.5), (1, 0.6)],
        [(0, 0.8), (1, 0.1), (2, 0.2)],
        [(0, 0.1), (1, 0.9), (2, 0.7), (3, 0.6)],
    ]
    row_ptr = [0]
    columns = []
    values = []
    for row in values_by_row:
        columns.extend(source for source, _ in row)
        values.extend(value for _, value in row)
        row_ptr.append(len(columns))

    return SimpleNamespace(
        sample_id="sample-1",
        source_id="source-1",
        response_idx=2,
        token_ids=torch.tensor([11, 12, 13, 14, 15], device=device),
        attention_diagonal=torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]],
            device=device,
        ),
        response_row_ptr=torch.tensor(row_ptr, device=device),
        response_column_indices=torch.tensor(columns, device=device),
        response_values=torch.tensor(values, device=device),
        attention_floor=0.05,
    )


class OriginalGraphTests(unittest.TestCase):
    def test_builds_exact_tau_thresholded_edges_in_deterministic_order(self) -> None:
        graph = build_original_graph(make_sample(), tau=0.65)

        self.assertTrue(
            torch.equal(
                graph.edge_index,
                torch.tensor([[1, 0, 2, 1, 2, 3], [2, 3, 3, 4, 4, 4]]),
            )
        )
        self.assertTrue(torch.equal(graph.edge_type, torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int8)))
        torch.testing.assert_close(
            graph.edge_attr,
            torch.tensor(
                [[0.7, 0.0], [0.0, 0.8], [0.9, 0.0], [0.0, 0.9], [0.0, 0.7], [0.8, 0.0]]
            ),
        )
        torch.testing.assert_close(
            graph.node_attr,
            torch.tensor([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=torch.float32),
        )
        self.assertIsNone(graph.edge_weight)
        self.assertIsNone(graph.trace_ptr)
        self.assertNotIn("trace_ptr", graph.to_dict())

    def test_rejects_tau_below_the_retained_cache_floor(self) -> None:
        with self.assertRaises(ValueError):
            build_original_graph(make_sample(), tau=0.04)


class RelationTopKGraphTests(unittest.TestCase):
    def test_selects_prompt_and_history_topk_with_retained_mean_weights(self) -> None:
        graph = build_relation_topk_graph(make_sample(), k_prompt=1, k_history=1)

        self.assertTrue(
            torch.equal(graph.edge_index, torch.tensor([[1, 0, 2, 1, 3], [2, 3, 3, 4, 4]]))
        )
        self.assertTrue(torch.equal(graph.edge_type, torch.tensor([0, 0, 1, 0, 1], dtype=torch.int8)))
        torch.testing.assert_close(graph.edge_weight, torch.tensor([0.65, 0.6, 0.55, 0.7, 0.7]))
        self.assertIsNone(graph.edge_attr)
        self.assertIsNone(graph.trace_ptr)

    def test_channel_trace_recovers_each_selected_edge_channel_value(self) -> None:
        graph = build_relation_topk_graph(make_sample(), k_prompt=1, k_history=1, with_channels=True)

        self.assertTrue(torch.equal(graph.trace_ptr, torch.tensor([0, 2, 4, 6, 8, 10])))
        self.assertTrue(torch.equal(graph.trace_channel, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32)))
        torch.testing.assert_close(
            graph.trace_value,
            torch.tensor([0.7, 0.6, 0.4, 0.8, 0.9, 0.2, 0.5, 0.9, 0.8, 0.6]),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_outputs_remain_on_the_sample_device(self) -> None:
        graph = build_relation_topk_graph(make_sample("cuda"), k_prompt=1, k_history=1, with_channels=True)

        self.assertEqual(graph.node_attr.device.type, "cuda")
        self.assertEqual(graph.edge_index.device.type, "cuda")
        self.assertEqual(graph.edge_weight.device.type, "cuda")
        self.assertEqual(graph.trace_ptr.device.type, "cuda")
        self.assertEqual(graph.trace_channel.device.type, "cuda")
        self.assertEqual(graph.trace_value.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
