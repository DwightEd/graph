import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from archive import AttentionArchiveConverter, ArchiveConfig
from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionDataset, AttentionSample, NPZ_FIELDS, save_attention_sample
from graphs import build_original_graph


def sample():
    return AttentionSample(
        "r1",
        "s1",
        2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[1.0, 0.5, 0.2, 0.1]]], dtype=torch.float16),
        torch.tensor([0, 2, 4]),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([0.02, 0.08, 0.10, 0.04], dtype=torch.float16),
        0.01,
    )


class CoreTests(unittest.TestCase):
    def test_npz_has_exactly_six_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            save_attention_sample(sample(), path)
            with np.load(path) as arrays:
                self.assertEqual(set(arrays.files), set(NPZ_FIELDS))

    def test_original_graph_matches_threshold_union(self):
        graph = build_original_graph(sample(), 0.05)
        self.assertEqual(graph.edge_index.tolist(), [[1, 0], [2, 3]])
        self.assertEqual(graph.edge_type.tolist(), [0, 0])
        self.assertEqual(graph.edge_attr.shape, (2, 1))

    def test_formal_converter_and_builder_use_same_canonical_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal"
            for split in ("train", "test"):
                (formal / split).mkdir(parents=True)
                value = sample()
                torch.save(
                    {
                        "response_id": f"{split}-1",
                        "source_id": f"source-{split}",
                        "response_idx": value.response_idx,
                        "token_ids": value.token_ids,
                        "attention_diagonal": value.attention_diagonal,
                        "response_row_ptr": value.response_row_ptr,
                        "response_column_indices": value.response_column_indices,
                        "response_values": value.response_values,
                        "attention_floor": value.attention_floor,
                        "y_token": torch.tensor([0, 0, 1, 0]),
                        "unused_metadata": "ignored",
                    },
                    formal / split / f"attention_{split}-1.pt",
                )

            archive = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal, archive)).run()
            dataset = AttentionDataset(archive / "train")
            restored = next(iter(dataset))
            self.assertEqual(restored.sample_id, "train-1")
            self.assertTrue((archive / "train" / "labels.jsonl").is_file())

            output = root / "graphs"
            GraphDatasetBuilder(
                BuildConfig(
                    archive / "train",
                    output,
                    kind="original",
                    tau=0.05,
                    device="cpu",
                )
            ).run()
            graph = torch.load(output / "graphs" / "train-1.pt", weights_only=True)
            self.assertEqual(
                set(graph),
                {"response_idx", "token_ids", "x", "edge_index", "edge_type", "edge_attr"},
            )


if __name__ == "__main__":
    unittest.main()
