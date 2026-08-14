import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from research_dataset import ResearchDataset


def _dataset_root(directory: str) -> Path:
    root = Path(directory)
    (root / "attention").mkdir()
    sample = AttentionSample(
        "r1",
        "s1",
        2,
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([[[0.8, 0.7, 0.6, 0.5]]], dtype=torch.float16),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0.2, 0.3], dtype=torch.float16),
        0.01,
    )
    path = root / "attention" / "r1.npz"
    save_attention_sample(sample, path)
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"sample_id": "r1", "positive_runs": [[1, 2]]}) + "\n",
        encoding="utf-8",
    )
    write_split_index(
        root,
        [index_row(root, sample, path, metadata={"split": "test", "quality": "good"})],
        attention_floor=0.01,
        num_layers=1,
        num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={"split": "test", "labels_sha256": sha256(labels)},
    )
    return root


class ResearchViewTests(unittest.TestCase):
    def test_sparse_blocks_decode_canonical_csr(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = ResearchDataset(_dataset_root(directory))
            sample = dataset["r1"]
            blocks = list(sample.iter_sparse_attention_blocks(block_rows=1))
            self.assertEqual(len(blocks), 2)
            self.assertEqual(
                torch.cat([block.source for block in blocks]).tolist(),
                [0, 2],
            )
            self.assertEqual(
                torch.cat([block.target for block in blocks]).tolist(),
                [2, 3],
            )
            self.assertEqual(
                torch.cat([block.query for block in blocks]).tolist(),
                [0, 1],
            )

    def test_dense_channel_and_mean_view_share_zero_fill_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = ResearchDataset(_dataset_root(directory))
            sample = dataset["r1"]
            dense = sample.dense_response_channel(0, 0, include_diagonal=True)
            expected = torch.tensor(
                [[0.2, 0.0, 0.6, 0.0], [0.0, 0.0, 0.3, 0.5]],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(dense, expected, atol=1e-3))

            mean = sample.mean_response_attention(include_diagonal=False)
            expected_without_diagonal = expected.clone()
            expected_without_diagonal[0, 2] = 0.0
            expected_without_diagonal[1, 3] = 0.0
            self.assertTrue(
                torch.allclose(mean, expected_without_diagonal, atol=1e-3)
            )


if __name__ == "__main__":
    unittest.main()
