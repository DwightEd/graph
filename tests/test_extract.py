import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from extract import AttentionCollector, AttentionExtractor, ExtractionConfig


class FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "P:"

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {"input_ids": [1, 2, 3, 4], "offset_mapping": [(0, 1), (1, 2), (2, 3), (3, 4)]}


class FakeSelfAttention(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, output_attentions=False):
        return (hidden, self.attention, "cache-slot")


class FakeLayer(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = FakeSelfAttention(attention)
        self.last_attention = "not called"
        self.last_cache = "not called"

    def forward(self, hidden, output_attentions=False):
        hidden, self.last_attention, self.last_cache = self.self_attn(
            hidden, output_attentions=output_attentions
        )
        return hidden


class FakeBackbone(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, return_dict=True, use_cache=False, output_attentions=False):
        self.calls.append({"return_dict": return_dict, "use_cache": use_cache, "output_attentions": output_attentions})
        hidden = input_ids.float()
        for layer in self.layers:
            hidden = layer(hidden, output_attentions=output_attentions)
        return hidden


class FakeModel(torch.nn.Module):
    def __init__(self, max_position_embeddings=4):
        super().__init__()
        first = torch.tensor([[[[.1, .2, .3, .4], [.2, .2, .3, .3], [.3, .2, .2, .3], [.4, .3, .2, .1]]]])
        second = first * .5
        self.model = FakeBackbone([FakeLayer(first), FakeLayer(second)])
        self.config = type("Config", (), {"num_attention_heads": 1, "max_position_embeddings": max_position_embeddings})()
        self.forward_called = False

    def forward(self, *args, **kwargs):
        self.forward_called = True
        return self.model(*args, **kwargs)


class ExtractTests(unittest.TestCase):
    def test_collector_compacts_diagonal_and_causal_response_values(self) -> None:
        collector = AttentionCollector(num_layers=1, num_heads=1, num_tokens=4, response_idx=2, floor=.25, dtype=torch.float32)
        attention = torch.tensor([[[.1, .2, .3, .4], [.2, .2, .3, .3], [.3, .2, .2, .3], [.4, .3, .2, .1]]])
        collector.consume(0, attention)
        diagonal, row_ptr, columns, values = collector.finalize()

        self.assertEqual(diagonal.shape, (1, 1, 4))
        self.assertEqual(row_ptr.tolist(), [0, 1, 3])
        self.assertEqual(columns.tolist(), [0, 0, 1])
        self.assertEqual(values.tolist(), [0.30000001192092896, 0.4000000059604645, 0.30000001192092896])

    def test_collector_compares_half_attention_in_float32(self) -> None:
        collector = AttentionCollector(num_layers=1, num_heads=1, num_tokens=3, response_idx=2, floor=.01, dtype=torch.float16)
        attention = torch.tensor([[[0, 0, 0], [0, 0, 0], [.01, 0, 0]]], dtype=torch.float16)
        collector.consume(0, attention)
        _, row_ptr, columns, _ = collector.finalize()

        self.assertEqual(row_ptr.tolist(), [0, 1])
        self.assertEqual(columns.tolist(), [0])

    def test_extractor_writes_manifest_index_and_label_free_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "source_info.jsonl").write_text(json.dumps({"source_id": "s", "prompt": "q", "task_type": "qa"}) + "\n")
            (dataset / "response.jsonl").write_text(json.dumps({"id": "r", "source_id": "s", "response": "a", "split": "test", "model": "llama-2-7b-chat", "quality": "good", "labels": [1]}) + "\n")
            output = root / "out"
            config = ExtractionConfig(model_path="observer", dataset_path=dataset, output_dir=output, split="test", generator_model="llama-2-7b-chat", task_type="all", floor=.01, dtype="float32", device="cpu")
            model = FakeModel()
            with patch("extract.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch("extract.AutoModelForCausalLM.from_pretrained", return_value=model):
                AttentionExtractor(config).run()
            manifest = json.loads((output / "manifest.json").read_text())
            index = json.loads((output / "index.jsonl").read_text())

        self.assertEqual(manifest["schema"], "attention-response-csr-v1")
        self.assertEqual(manifest["observer_model"], "observer")
        self.assertEqual(manifest["input_policy"], "full_context_no_truncation")
        self.assertEqual(manifest["count"], 1)
        self.assertEqual(index["sample_id"], "r")
        self.assertNotIn("label", index)
        self.assertFalse(model.forward_called)
        self.assertEqual(model.model.calls, [{"return_dict": True, "use_cache": False, "output_attentions": True}])
        self.assertIsNone(model.model.layers[0].last_attention)
        self.assertIsNone(model.model.layers[1].last_attention)
        self.assertEqual(model.model.layers[0].last_cache, "cache-slot")
        self.assertEqual(model.model.layers[1].last_cache, "cache-slot")

    def test_extractor_rejects_over_context_input_without_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "source_info.jsonl").write_text(json.dumps({"source_id": "s", "prompt": "q", "task_type": "qa"}) + "\n")
            (dataset / "response.jsonl").write_text(json.dumps({"id": "r", "source_id": "s", "response": "a", "split": "test", "model": "llama-2-7b-chat", "quality": "good"}) + "\n")
            config = ExtractionConfig(model_path="observer", dataset_path=dataset, output_dir=root / "out", split="test", dtype="float32", device="cpu")
            with patch("extract.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch("extract.AutoModelForCausalLM.from_pretrained", return_value=FakeModel(max_position_embeddings=3)):
                with self.assertRaisesRegex(ValueError, "context"):
                    AttentionExtractor(config).run()

    def test_extractor_rejects_unknown_dtype(self) -> None:
        config = ExtractionConfig(model_path="observer", dataset_path="dataset", output_dir="out", split="test", dtype="float64")
        with self.assertRaisesRegex(ValueError, "dtype"):
            AttentionExtractor(config).run()
