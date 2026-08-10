import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionSample, load_attention_sample, save_attention_sample


def sample() -> AttentionSample:
    return AttentionSample(
        sample_id="response-1",
        source_id="source-1",
        response_idx=2,
        token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        attention_diagonal=torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
        response_row_ptr=torch.tensor([0, 1, 3], dtype=torch.int64),
        response_column_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        response_values=torch.tensor([0.4, 0.2, 0.3]),
        attention_floor=0.01,
    )


class AttentionSampleTests(unittest.TestCase):
    def test_save_load_round_trip_only_persists_fixed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            save_attention_sample(sample(), path)
            payload = torch.load(path, weights_only=True)
            restored = load_attention_sample(path, map_location="cpu")

        self.assertEqual(payload["schema"], "attention-response-csr-v1")
        self.assertEqual(set(payload), {
            "schema", "sample_id", "source_id", "response_idx", "token_ids",
            "attention_diagonal", "response_row_ptr", "response_column_indices",
            "response_values", "attention_floor",
        })
        self.assertEqual(restored.num_layers, 1)
        self.assertEqual(restored.num_heads, 1)
        self.assertEqual(restored.num_tokens, 4)
        self.assertEqual(restored.num_response_tokens, 2)
        self.assertEqual(restored.num_channels, 1)

    def test_loads_old_schema_and_ignores_labels_and_metadata(self) -> None:
        payload = {
            "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
            "response_id": "old-response", "source_id": "source-1", "response_idx": 2,
            "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "attention_diagonal": torch.ones((1, 1, 4)),
            "response_row_ptr": torch.tensor([0, 0, 1], dtype=torch.int64),
            "response_column_indices": torch.tensor([0], dtype=torch.int32),
            "response_values": torch.tensor([0.5]), "attention_floor": 0.01,
            "y_token": torch.ones(4), "generator_model": "ignored",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save(payload, path)
            restored = load_attention_sample(path)

        self.assertEqual(restored.sample_id, "old-response")
        self.assertEqual(restored.source_id, "source-1")

    def test_rejects_noncausal_csr_entries(self) -> None:
        invalid = sample()
        invalid.response_column_indices = torch.tensor([0, 0, 3], dtype=torch.int32)
        with self.assertRaises(ValueError):
            invalid.validate()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map_location coverage")
    def test_validates_tensors_loaded_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            save_attention_sample(sample(), path)
            restored = load_attention_sample(path, map_location="cuda")

        self.assertEqual(restored.response_row_ptr.device.type, "cuda")
