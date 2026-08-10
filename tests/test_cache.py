import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionSample, load_attention_sample, save_attention_sample
from extract import AttentionCollector


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
        floor = 0.9
        quantized_floor = torch.tensor(floor, dtype=torch.bfloat16)
        payload = {
            "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
            "response_id": "old-response", "source_id": "source-1", "response_idx": 2,
            "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "attention_diagonal": torch.ones((1, 1, 4), dtype=torch.bfloat16),
            "response_row_ptr": torch.tensor([0, 0, 1], dtype=torch.int64),
            "response_column_indices": torch.tensor([0], dtype=torch.int32),
            "response_values": quantized_floor.reshape(1), "attention_floor": floor,
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

    def test_accepts_float16_and_bfloat16_values_equal_to_quantized_floor(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                floor = 0.9
                quantized_floor = torch.tensor(floor, dtype=dtype)
                valid = AttentionSample(
                    sample_id="response-1",
                    source_id="source-1",
                    response_idx=2,
                    token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
                    attention_diagonal=torch.ones((1, 1, 4), dtype=dtype),
                    response_row_ptr=torch.tensor([0, 1, 2], dtype=torch.int64),
                    response_column_indices=torch.tensor([0, 0], dtype=torch.int32),
                    response_values=quantized_floor.repeat(2),
                    attention_floor=floor,
                )

                valid.validate()

    def test_rejects_values_below_the_quantized_attention_floor(self) -> None:
        invalid = sample()
        floor = 0.9
        quantized_floor = torch.tensor(floor, dtype=torch.float16)
        invalid.attention_diagonal = torch.ones((1, 1, 4), dtype=torch.float16)
        invalid.response_values = torch.tensor(
            [
                0.95,
                torch.nextafter(
                    quantized_floor,
                    torch.tensor(0.0, dtype=torch.float16),
                ).item(),
                0.95,
            ],
            dtype=torch.float16,
        )
        invalid.attention_floor = floor
        with self.assertRaisesRegex(ValueError, "attention_floor"):
            invalid.validate()

    def test_collector_artifact_saves_when_cast_value_equals_quantized_floor(self) -> None:
        collector = AttentionCollector(
            num_layers=1,
            num_heads=1,
            num_tokens=3,
            response_idx=2,
            floor=0.9,
            dtype=torch.float16,
        )
        attention = torch.zeros((1, 3, 3), dtype=torch.float32)
        attention[0, 2, 0] = 0.9001
        collector.consume(0, attention)
        diagonal, row_ptr, columns, values = collector.finalize()
        artifact = AttentionSample(
            sample_id="response-1",
            source_id="source-1",
            response_idx=2,
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            attention_diagonal=diagonal,
            response_row_ptr=row_ptr,
            response_column_indices=columns,
            response_values=values,
            attention_floor=0.9,
        )

        with tempfile.TemporaryDirectory() as directory:
            save_attention_sample(artifact, Path(directory) / "sample.pt")

    def test_rejects_attention_floor_outside_zero_to_one_range(self) -> None:
        for floor in (float("nan"), float("inf"), -0.01, 0, 1.01):
            with self.subTest(floor=floor):
                invalid = sample()
                invalid.attention_floor = floor
                with self.assertRaisesRegex(ValueError, "attention_floor"):
                    invalid.validate()

    def test_rejects_duplicate_or_unsorted_columns_within_a_csr_row(self) -> None:
        for columns in (
            torch.tensor([0, 1, 1], dtype=torch.int32),
            torch.tensor([0, 1, 0], dtype=torch.int32),
        ):
            with self.subTest(columns=columns.tolist()):
                invalid = sample()
                invalid.response_column_indices = columns
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    invalid.validate()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map_location coverage")
    def test_validates_tensors_loaded_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            save_attention_sample(sample(), path)
            restored = load_attention_sample(path, map_location="cuda")

        self.assertEqual(restored.response_row_ptr.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device consistency coverage")
    def test_rejects_cache_tensors_on_different_devices(self) -> None:
        invalid = sample()
        invalid.token_ids = invalid.token_ids.to("cuda")
        with self.assertRaisesRegex(ValueError, "same device"):
            invalid.validate()
