"""Behavioral regression tests for incremental RR-prefix accumulation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache import AttentionSample, index_row, save_attention_sample, write_split_index
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    causal_prefix_edge_batches,
    prefix_causal_attention_modes,
)
from research_dataset import ResearchDataset


def _sparse_multichannel_sample() -> AttentionSample:
    """Two channels, three response tokens, and deliberately sparse RR rows."""
    return AttentionSample(
        sample_id="sample",
        source_id="source",
        response_idx=2,
        token_ids=torch.tensor([10, 11, 12, 13, 14]),
        attention_diagonal=torch.tensor(
            [[[0.8, 0.6, 0.35, 0.25, 0.15], [0.7, 0.5, 0.3, 0.2, 0.1]]],
            dtype=torch.float16,
        ),
        # Rows are (channel, response-query), with only a few RR entries.
        response_row_ptr=torch.tensor([0, 0, 1, 3, 3, 4, 6], dtype=torch.int32),
        response_column_indices=torch.tensor([2, 2, 3, 2, 2, 3], dtype=torch.int32),
        response_values=torch.tensor(
            [0.17, 0.13, 0.23, 0.19, 0.11, 0.29], dtype=torch.float16
        ),
        attention_floor=0.01,
    )


def _write_dataset(root: Path, attention: AttentionSample) -> ResearchDataset:
    path = root / "attention" / "sample.npz"
    save_attention_sample(attention, path)
    write_split_index(
        root,
        [
            index_row(
                root,
                attention,
                path,
                metadata={
                    "split": "train",
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                },
            )
        ],
        attention_floor=0.01,
        num_layers=1,
        num_heads=2,
        alignment="post_token_query_at_same_position",
    )
    return ResearchDataset(root)


def _naive_modes(sample, positions: list[int], top_k: int):
    """Independent dense reference for the documented age-normalized formula."""
    attention = sample.attention()
    response_count = attention.num_response_tokens
    channels = attention.num_channels
    diagonal = (
        attention.attention_diagonal[:, :, attention.response_idx :]
        .float()
        .reshape(channels, response_count)
        .cpu()
        .numpy()
    )
    output = np.zeros((len(positions), channels, top_k), dtype=np.float32)
    source_output = np.full(output.shape, -1, dtype=np.int32)
    lag_output = np.full(output.shape, -1, dtype=np.int32)
    edges = []
    for block in sample.iter_sparse_attention_blocks(block_rows=4096):
        for layer, head, query, source, weight in zip(
            block.layer.cpu().tolist(),
            block.head.cpu().tolist(),
            block.query.cpu().tolist(),
            block.source.cpu().tolist(),
            block.weight.cpu().tolist(),
        ):
            if source >= attention.response_idx:
                edges.append(
                    (layer * attention.num_heads + head, query, source - attention.response_idx, weight)
                )

    for row, prefix in enumerate(positions):
        received = diagonal.copy()
        for channel, query, source, weight in edges:
            if query <= prefix:
                received[channel, source] += weight
        active = prefix + 1
        denominator = np.arange(prefix + 1, 0, -1, dtype=np.float32)
        coordinates = received[:, :active] / denominator - diagonal[:, :active]
        for channel in range(channels):
            order = np.argsort(-np.abs(coordinates[channel]), kind="stable")[:top_k]
            output[row, channel, : len(order)] = coordinates[channel, order]
            source_output[row, channel, : len(order)] = order
            lag_output[row, channel, : len(order)] = prefix - order
    return output, source_output, lag_output


class _NoRREdges:
    def __init__(self, attention: AttentionSample) -> None:
        self._attention = attention

    def attention(self) -> AttentionSample:
        return self._attention

    def iter_sparse_attention_blocks(self, *, block_rows: int):
        del block_rows
        return iter(())


class SpectralPrefixPerformanceTests(unittest.TestCase):
    def test_incremental_prefixes_match_the_dense_formula_on_sparse_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), _sparse_multichannel_sample())
            sample = dataset["sample"]
            positions = [0, 2]
            expected = _naive_modes(sample, positions, top_k=2)

            actual = prefix_causal_attention_modes(
                sample, positions=positions, config=SpectralConfig(top_k=2)
            )

            np.testing.assert_allclose(actual.values, expected[0], rtol=0, atol=2e-6)
            np.testing.assert_array_equal(actual.source_index, expected[1])
            np.testing.assert_array_equal(actual.lag, expected[2])

    def test_incremental_batches_consume_each_eligible_edge_once(self):
        queries = torch.tensor([3, 0, 4, 1, 3, 2, 4])
        positions = np.asarray([1, 3, 4], dtype=np.int64)

        batches = list(causal_prefix_edge_batches(queries, positions))
        consumed = torch.cat([edge_ids for _, edge_ids in batches]).tolist()

        self.assertEqual(consumed, [1, 3, 5, 0, 4, 2, 6])
        self.assertEqual(sorted(consumed), list(range(len(queries))))
        self.assertEqual(len(consumed), len(set(consumed)))
        for prefix, edge_ids in batches:
            self.assertTrue(bool((queries[edge_ids] <= prefix).all()))

    def test_no_rr_edges_keeps_the_documented_diagonal_only_coordinates(self):
        attention = _sparse_multichannel_sample()
        sample = _NoRREdges(attention)
        expected = _naive_modes(sample, [0, 2], top_k=2)
        modes = prefix_causal_attention_modes(
            sample,
            positions=[0, 2],
            config=SpectralConfig(top_k=2),
        )
        np.testing.assert_allclose(modes.values, expected[0], rtol=0, atol=2e-6)
        np.testing.assert_array_equal(modes.source_index, expected[1])
        np.testing.assert_array_equal(modes.lag, expected[2])


if __name__ == "__main__":
    unittest.main()
