import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import AttentionSample, index_row, save_attention_sample, write_split_index
from node_tsne import NodeTSNEVisualizer
from research_dataset import STRUCTURAL_FEATURE_NAMES


def sample(sample_id, source_id):
    return AttentionSample(
        sample_id,
        source_id,
        2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[1.0, 0.8, 0.5, 0.4]]], dtype=torch.float16),
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([0.2, 0.4, 0.1, 0.3], dtype=torch.float16),
        0.01,
    )


class NodeTSNETests(unittest.TestCase):
    def _write_split(self, root):
        (root / "attention").mkdir(parents=True)
        rows = []
        for sample_id, source_id in (("r1", "s1"), ("r2", "s2")):
            value = sample(sample_id, source_id)
            path = root / "attention" / f"{sample_id}.npz"
            save_attention_sample(value, path)
            rows.append(
                index_row(
                    root,
                    value,
                    path,
                    metadata={
                        "split": "test",
                        "task_type": "QA",
                        "data_source": "MARCO",
                        "generator_model": "llama-2-7b-chat",
                        "temperature": 0.7,
                        "quality": "good",
                    },
                )
            )
        write_split_index(
            root,
            rows,
            attention_floor=0.01,
            num_layers=1,
            num_heads=1,
            alignment="post_token_query_at_same_position",
        )
        (root / "labels.jsonl").write_text(
            "".join(
                [
                    json.dumps({"sample_id": "r1", "positive_runs": [[1, 2]]}) + "\n",
                    json.dumps({"sample_id": "r2", "positive_runs": []}) + "\n",
                ]
            ),
            encoding="utf-8",
        )

    def test_collect_pools_response_nodes_across_sample_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_split(root)
            viewer = NodeTSNEVisualizer(root, random_state=0)
            result = viewer.collect()

            self.assertEqual(result["features"].shape, (4, len(STRUCTURAL_FEATURE_NAMES)))
            self.assertEqual(result["labels"].tolist(), [0, 1, 0, 0])
            self.assertEqual(result["sample_id"].tolist(), ["r1", "r1", "r2", "r2"])
            self.assertEqual(result["response_position"].tolist(), [0, 1, 0, 1])
            self.assertEqual(result["total_nodes_before_sampling"], 4)
            self.assertEqual(result["selected_nodes"], 4)

    def test_uniform_node_limit_is_label_independent_and_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_split(root)
            viewer = NodeTSNEVisualizer(root, random_state=3)
            result = viewer.collect(max_nodes=3)

            self.assertEqual(result["features"].shape, (3, len(STRUCTURAL_FEATURE_NAMES)))
            self.assertEqual(len(result["labels"]), 3)
            self.assertEqual(len(result["sample_id"]), 3)
            self.assertEqual(len(result["response_position"]), 3)
            self.assertEqual(result["total_nodes_before_sampling"], 4)
            self.assertEqual(result["selected_nodes"], 3)


if __name__ == "__main__":
    unittest.main()
