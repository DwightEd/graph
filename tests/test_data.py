import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionSample, index_row, save_attention_sample, sha256, verify_split, write_split_index
from research_dataset import ResearchDataset


class DataTests(unittest.TestCase):
    def test_canonical_labels_stay_sidecar_and_align_to_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attention").mkdir()
            sample = AttentionSample(
                "r1", "s1", 2,
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([[[0.8, 0.7, 0.6, 0.5]]], dtype=torch.float16),
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 2], dtype=torch.int32),
                torch.tensor([0.2, 0.3], dtype=torch.float16),
                0.01,
            )
            path = root / "attention/r1.npz"
            save_attention_sample(sample, path)
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({"sample_id": "r1", "positive_runs": [[1, 2]]}) + "\n")
            write_split_index(
                root,
                [index_row(root, sample, path, metadata={"split": "test", "quality": "good"})],
                attention_floor=0.01,
                num_layers=1,
                num_heads=1,
                alignment="post_token_query_at_same_position",
                extra={"split": "test", "labels_sha256": sha256(labels)},
            )
            self.assertEqual(verify_split(root), 1)
            dataset = ResearchDataset(root, verify_hashes=True)
            restored = dataset["r1"]
            self.assertEqual(restored.attention().num_response_tokens, 2)
            self.assertEqual(dataset.labels().response_labels(restored).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
