import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from cache import AttentionSample, index_row, save_attention_sample, sha256, verify_split, write_split_index
from formal_cache import FORMAL_CACHE_SCHEMA, formal_fingerprint
from research_dataset import ResearchDataset, open_research_dataset


class DataTests(unittest.TestCase):
    def test_attention_validation_checks_csr_in_bounded_row_blocks(self):
        response_tokens = 600
        sample = AttentionSample(
            "bounded", "source", 1,
            torch.arange(response_tokens + 1, dtype=torch.int32),
            torch.zeros((1, 1, response_tokens + 1), dtype=torch.float16),
            torch.arange(response_tokens + 1, dtype=torch.int32),
            torch.zeros(response_tokens, dtype=torch.int32),
            torch.full((response_tokens,), .1, dtype=torch.float16),
            .01,
        )
        original = torch.repeat_interleave
        row_block_sizes = []

        def bounded_repeat(rows, lengths):
            row_block_sizes.append(rows.numel())
            return original(rows, lengths)

        with patch("cache.torch.repeat_interleave", side_effect=bounded_repeat):
            sample.validate()

        self.assertEqual(row_block_sizes, [response_tokens])
        self.assertLessEqual(max(row_block_sizes), 4096)

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

    def test_formal_sparse_pt_is_read_directly_without_npz_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "attention_r1.pt"
            spec = {
                "attention_cache_schema": FORMAL_CACHE_SCHEMA,
                "split": "test",
                "cache_dtype": "torch.float16",
                "attention_floor": 0.01,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "model_path": "/models/observer",
            }
            fingerprint = formal_fingerprint(spec)
            payload = {
                "attention_cache_schema": FORMAL_CACHE_SCHEMA,
                "attention_cache_fingerprint": fingerprint,
                "response_id": "r1",
                "source_id": "s1",
                "split": "test",
                "cache_dtype": "torch.float16",
                "num_attention_layers": 1,
                "num_attention_heads": 1,
                "quality": "good",
                "was_truncated": False,
                "response_idx": 2,
                "token_ids": torch.tensor([1, 2, 3, 4]),
                "attention_diagonal": torch.tensor(
                    [[[0.8, 0.7, 0.6, 0.5]]], dtype=torch.float16
                ),
                "response_row_ptr": torch.tensor([0, 1, 2]),
                "response_column_indices": torch.tensor([0, 2]),
                "response_values": torch.tensor([0.2, 0.3], dtype=torch.float16),
                "attention_floor": 0.01,
                "y_token": torch.tensor([0, 0, 0, 1]),
                "task_type": "QA",
                "data_source": "MARCO",
            }
            path = root / name
            torch.save(payload, path)
            manifest = {
                "state": "complete",
                "cache_file_names": [name],
                "matched_samples": 1,
                "cache_files": 1,
                "cache_files_sha256": {name: sha256(path)},
                "attention_cache_spec": spec,
                "attention_cache_fingerprint": fingerprint,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with patch("formal_cache.sha256") as file_hash:
                dataset = open_research_dataset(
                    root, verify_hashes=False, retain_embedded_labels=True
                )
                sample = dataset["r1"]
                self.assertEqual(sample.attention().num_response_tokens, 2)
                self.assertEqual(sample.task_type, "QA")
                self.assertEqual(
                    dataset.labels().response_labels(sample).tolist(), [0, 1]
                )
                file_hash.assert_not_called()
            self.assertEqual(list(root.glob("*.npz")), [])


if __name__ == "__main__":
    unittest.main()
