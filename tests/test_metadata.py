import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionDataset, AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from metadata import RESEARCH_INDEX_FIELDS, enrich_ragtruth_indices
from research_dataset import LabelStore, ResearchDataset


class MetadataTests(unittest.TestCase):
    def test_enrich_existing_json_and_access_research_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "canonical" / "train"
            (split / "attention").mkdir(parents=True)

            sample = AttentionSample(
                "r1",
                "s1",
                2,
                torch.tensor([10, 11, 12, 13]),
                torch.tensor([[[1.0, 0.5, 0.2, 0.1]]], dtype=torch.float16),
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 0], dtype=torch.int32),
                torch.tensor([0.08, 0.10], dtype=torch.float16),
                0.01,
            )
            attention_path = split / "attention" / "r1.npz"
            save_attention_sample(sample, attention_path)
            write_split_index(
                split,
                [index_row(split, sample, attention_path)],
                attention_floor=0.01,
                num_layers=1,
                num_heads=1,
                alignment="post_token_query_at_same_position",
                extra={"observer_model": "Meta-Llama-3.1-8B-Instruct"},
            )
            (split / "labels.jsonl").write_text(
                json.dumps({"sample_id": "r1", "positive_runs": [[0, 1]]}) + "\n",
                encoding="utf-8",
            )

            graph_split = root / "graphs" / "train"
            graph_split.mkdir(parents=True)
            graph_index = graph_split / "index.jsonl"
            graph_index.write_text(json.dumps({"sample_id": "r1", "path": "graphs/r1.pt"}) + "\n")
            (graph_split / "manifest.json").write_text(json.dumps({
                "count": 1,
                "index_sha256": sha256(graph_index),
                "input_manifest_sha256": sha256(split / "manifest.json"),
                "input_index_sha256": sha256(split / "index.jsonl"),
            }))

            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "source_info.jsonl").write_text(json.dumps({
                "source_id": "s1",
                "task_type": "Summary",
                "source": "CNN/DM",
                "source_info": "article",
                "prompt": "summarize",
            }) + "\n", encoding="utf-8")
            (dataset / "response.jsonl").write_text(json.dumps({
                "id": "r1",
                "source_id": "s1",
                "model": "llama-2-7b-chat",
                "temperature": 0.7,
                "labels": [],
                "split": "train",
                "quality": "good",
                "response": "answer",
            }) + "\n", encoding="utf-8")

            result = enrich_ragtruth_indices(root / "canonical", dataset, root / "graphs")
            self.assertEqual(result["splits"], {"train": 1})
            self.assertEqual(result["graphs"], {"train": 1})

            row = json.loads((split / "index.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(tuple(row), RESEARCH_INDEX_FIELDS)
            self.assertEqual(row["task_type"], "Summary")
            self.assertEqual(row["data_source"], "CNN/DM")
            self.assertEqual(row["generator_model"], "llama-2-7b-chat")
            self.assertEqual(row["temperature"], 0.7)

            graph_manifest = json.loads((graph_split / "manifest.json").read_text())
            self.assertEqual(graph_manifest["input_manifest_sha256"], sha256(split / "manifest.json"))
            self.assertEqual(graph_manifest["input_index_sha256"], sha256(split / "index.jsonl"))

            self.assertEqual(len(AttentionDataset(split)), 1)
            research = ResearchDataset(split)
            item = research["r1"]
            self.assertEqual(item.data_source, "CNN/DM")
            self.assertEqual(item.generator_model, "llama-2-7b-chat")
            self.assertEqual(item.observer_model, "Meta-Llama-3.1-8B-Instruct")
            self.assertEqual(item.metadata["task_type"], "Summary")

            labels = LabelStore(split / "labels.jsonl")
            self.assertEqual(labels.positive_runs("r1"), [[0, 1]])
            self.assertEqual(labels.token_labels(item).tolist(), [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
