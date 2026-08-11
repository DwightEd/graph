import json
import tempfile
import unittest
from pathlib import Path

import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    write_split_index,
)
from research_dataset import (
    ResearchDataset,
    STRUCTURAL_FEATURE_NAMES,
)


class ResearchDatasetGraphViewTests(unittest.TestCase):
    def _write_split(self, root: Path):
        sample = AttentionSample(
            "r1",
            "s1",
            2,
            torch.tensor([10, 11, 12, 13]),
            torch.tensor(
                [
                    [
                        [1.0, 0.8, 0.5, 0.4],
                        [1.0, 0.7, 0.4, 0.3],
                    ]
                ],
                dtype=torch.float16,
            ),
            torch.tensor([0, 2, 4, 5, 6], dtype=torch.int32),
            torch.tensor([0, 1, 0, 2, 0, 2], dtype=torch.int32),
            torch.tensor([0.2, 0.4, 0.1, 0.3, 0.6, 0.5], dtype=torch.float16),
            0.01,
        )
        (root / "attention").mkdir(parents=True)
        path = root / "attention" / "r1.npz"
        save_attention_sample(sample, path)
        row = index_row(
            root,
            sample,
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
        write_split_index(
            root,
            [row],
            attention_floor=0.01,
            num_layers=1,
            num_heads=2,
            alignment="post_token_query_at_same_position",
        )
        (root / "labels.jsonl").write_text(
            json.dumps({"sample_id": "r1", "positive_runs": [[1, 2]]}) + "\n",
            encoding="utf-8",
        )

    def test_graph_view_uses_enriched_index_and_aggregates_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_split(root)
            dataset = ResearchDataset(root)
            labels = dataset.labels()
            view = dataset["r1"].graph_view(labels)

            self.assertEqual(view["metadata"]["task_type"], "QA")
            self.assertEqual(view["structural_feature_names"], STRUCTURAL_FEATURE_NAMES)
            self.assertEqual(view["response_features"].shape, (2, 12))
            self.assertEqual(view["response_labels"].tolist(), [0, 1])

            relations = view["relations"]
            pairs = list(
                zip(
                    relations["source"].tolist(),
                    relations["target"].tolist(),
                    relations["channel_count"].tolist(),
                )
            )
            self.assertEqual(pairs, [(0, 2, 2), (1, 2, 1), (0, 3, 1), (2, 3, 2)])

            weights = relations["weight"]
            torch.testing.assert_close(
                weights,
                torch.tensor([0.4, 0.2, 0.05, 0.4]),
                atol=2e-3,
                rtol=0,
            )

    def test_structural_features_capture_prompt_history_and_density(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_split(root)
            sample = ResearchDataset(root)["r1"]
            features = sample.structural_features()

            prompt_share = STRUCTURAL_FEATURE_NAMES.index("prompt_mass_share")
            history_share = STRUCTURAL_FEATURE_NAMES.index("history_edge_share")
            in_degree = STRUCTURAL_FEATURE_NAMES.index("in_degree")
            channel_density = STRUCTURAL_FEATURE_NAMES.index("channel_edge_density")

            self.assertAlmostEqual(float(features[0, prompt_share]), 1.0, places=5)
            self.assertAlmostEqual(float(features[1, prompt_share]), 1.0 / 9.0, places=4)
            self.assertAlmostEqual(float(features[1, history_share]), 0.5, places=5)
            self.assertEqual(features[:, in_degree].tolist(), [2.0, 2.0])
            self.assertAlmostEqual(float(features[0, channel_density]), 0.75, places=5)
            self.assertAlmostEqual(float(features[1, channel_density]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
