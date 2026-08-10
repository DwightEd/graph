import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
            [[[0.01, 0.02, 0.03, 0.04, 0.05], [0.10, 0.20, 0.30, 0.40, 0.50]]],
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
            torch.tensor(
                [[0.01, 0.10], [0.02, 0.20], [0.03, 0.30], [0.04, 0.40], [0.05, 0.50]],
                dtype=torch.float32,
            ),
        )
        self.assertIsNone(graph.edge_weight)
        self.assertIsNone(graph.trace_ptr)
        self.assertNotIn("trace_ptr", graph.to_dict())

    def test_rejects_tau_below_the_retained_cache_floor(self) -> None:
        with self.assertRaises(ValueError):
            build_original_graph(make_sample(), tau=0.04)

    def test_compares_half_values_in_float32_before_thresholding(self) -> None:
        sample = make_sample()
        sample.attention_floor = 0.005
        sample.response_values = torch.tensor([0.01], dtype=torch.float16)
        sample.response_row_ptr = torch.tensor([0, 1, 1, 1, 1, 1, 1])
        sample.response_column_indices = torch.tensor([0], dtype=torch.int32)

        graph = build_original_graph(sample, tau=0.01)

        self.assertTrue(torch.equal(graph.edge_index, torch.tensor([[0], [2]])))
        self.assertEqual(graph.edge_attr.dtype, torch.float16)

    def test_rejects_nonfinite_or_out_of_range_tau(self) -> None:
        for tau in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(tau=tau), self.assertRaisesRegex(ValueError, "finite"):
                build_original_graph(make_sample(), tau=tau)


class RelationTopKGraphTests(unittest.TestCase):
    def test_rejects_negative_relation_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_relation_topk_graph(make_sample(), k_prompt=-1, k_history=1)
    def test_selects_prompt_and_history_topk_with_retained_mean_weights(self) -> None:
        graph = build_relation_topk_graph(make_sample(), k_prompt=1, k_history=1)

        self.assertTrue(
            torch.equal(graph.edge_index, torch.tensor([[1, 0, 2, 1, 3], [2, 3, 3, 4, 4]]))
        )
        self.assertTrue(torch.equal(graph.edge_type, torch.tensor([0, 0, 1, 0, 1], dtype=torch.int8)))
        torch.testing.assert_close(graph.edge_weight, torch.tensor([0.65, 0.6, 0.55, 0.7, 0.7]))
        self.assertIsNone(graph.edge_attr)
        self.assertIsNone(graph.trace_ptr)

    def test_computes_half_precision_retained_means_in_float32(self) -> None:
        sample = make_sample()
        sample.response_values = sample.response_values.to(torch.float16)

        graph = build_relation_topk_graph(sample, k_prompt=1, k_history=1)

        self.assertEqual(graph.edge_weight.dtype, torch.float32)

    def test_channel_trace_recovers_each_selected_edge_channel_value(self) -> None:
        graph = build_relation_topk_graph(make_sample(), k_prompt=1, k_history=1, with_channels=True)

        self.assertTrue(torch.equal(graph.trace_ptr, torch.tensor([0, 2, 4, 6, 8, 10])))
        self.assertTrue(torch.equal(graph.trace_channel, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32)))
        torch.testing.assert_close(
            graph.trace_value,
            torch.tensor([0.7, 0.6, 0.4, 0.8, 0.9, 0.2, 0.5, 0.9, 0.8, 0.6]),
        )

    def test_breaks_topk_ties_by_source_and_orders_prompt_before_history(self) -> None:
        sample = make_sample()
        sample.response_row_ptr = torch.tensor([0, 2, 4, 4, 6, 8, 8])
        sample.response_column_indices = torch.tensor([0, 1, 0, 2, 0, 1, 0, 2])
        sample.response_values = torch.tensor([0.6, 0.6, 0.2, 0.9, 0.4, 0.4, 0.2, 0.9])
        graph = build_relation_topk_graph(sample, k_prompt=1, k_history=1)

        self.assertTrue(
            torch.equal(graph.edge_index, torch.tensor([[0, 0, 2], [2, 3, 3]]))
        )
        self.assertTrue(torch.equal(graph.edge_type, torch.tensor([0, 0, 1], dtype=torch.int8)))

    def test_handles_an_empty_relation_and_an_empty_cache(self) -> None:
        prompt_only = make_sample()
        prompt_only.response_row_ptr = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        prompt_only.response_column_indices = torch.zeros(6, dtype=torch.int64)
        prompt_only.response_values = torch.full((6,), 0.5)
        graph = build_relation_topk_graph(prompt_only, k_prompt=1, k_history=1)
        self.assertTrue(torch.equal(graph.edge_type, torch.zeros(3, dtype=torch.int8)))

        empty = make_sample()
        empty.response_row_ptr = torch.zeros(7, dtype=torch.int64)
        empty.response_column_indices = torch.empty(0, dtype=torch.int64)
        empty.response_values = torch.empty(0)
        graph = build_relation_topk_graph(empty, k_prompt=1, k_history=1, with_channels=True)
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(graph.edge_weight.numel(), 0)
        self.assertTrue(torch.equal(graph.trace_ptr, torch.tensor([0], dtype=torch.int64)))

    def test_selects_topk_without_per_target_nonzero_scans(self) -> None:
        with patch("graphs.torch.nonzero", side_effect=AssertionError("per-target nonzero scan")):
            graph = build_relation_topk_graph(make_sample(), k_prompt=1, k_history=1)

        self.assertEqual(graph.edge_index.shape[1], 5)

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
