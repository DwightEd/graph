import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionDataset, AttentionSample, index_row, save_attention_sample, write_split_index
from research_analysis import GRAPH_FEATURE_NAMES, SampleBehaviorVisualizer


def make_sample(sample_id, source_id):
    return AttentionSample(
        sample_id,
        source_id,
        2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[1.0, 0.5, 0.2, 0.1]]], dtype=torch.float16),
        torch.tensor([0, 2, 4]),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([0.02, 0.08, 0.10, 0.04], dtype=torch.float16),
        0.01,
    )


def write_dataset(root):
    (root / "attention").mkdir(parents=True)
    rows = []
    specs = (
        ("e1", "s1", "QA", "MARCO"),
        ("c1", "s1", "QA", "MARCO"),
        ("c2", "s2", "Summary", "CNN/DM"),
    )
    for sample_id, source_id, task_type, data_source in specs:
        sample = make_sample(sample_id, source_id)
        path = root / "attention" / f"{sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(
            root,
            sample,
            path,
            metadata={
                "split": "test",
                "task_type": task_type,
                "data_source": data_source,
                "generator_model": "llama-2-7b-chat",
                "temperature": 0.7,
                "quality": "good",
            },
        ))
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=1,
        num_heads=1,
        alignment="post_token_query_at_same_position",
    )
    (root / "labels.jsonl").write_text(
        "".join([
            json.dumps({"sample_id": "e1", "positive_runs": [[0, 1]]}) + "\n",
            json.dumps({"sample_id": "c1", "positive_runs": []}) + "\n",
            json.dumps({"sample_id": "c2", "positive_runs": []}) + "\n",
        ]),
        encoding="utf-8",
    )


class ResearchAnalysisTests(unittest.TestCase):
    def test_enriched_index_rows_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            dataset = AttentionDataset(root)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.rows[0]["task_type"], "QA")

    def test_visualizer_lists_and_analyzes_error_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            viewer = SampleBehaviorVisualizer(root)
            self.assertEqual(viewer.error_sample_ids, ["e1"])
            self.assertEqual(set(viewer.correct_sample_ids), {"c1", "c2"})
            result = viewer.analyze("e1")
            self.assertEqual(result["response_features"].shape, (2, len(GRAPH_FEATURE_NAMES)))
            self.assertEqual(result["positive_runs"], [[0, 1]])
            self.assertEqual(viewer.match_correct("e1"), "c1")


if __name__ == "__main__":
    unittest.main()
