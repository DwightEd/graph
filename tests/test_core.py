import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from archive import (
    AttentionArchiveConverter,
    ArchiveConfig,
    TraceArchiveConfig,
    TraceArchiveConverter,
)
from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionDataset, AttentionSample, NPZ_FIELDS, index_row, save_attention_sample, sha256, write_split_index
from features import (
    HIDDEN_FIELDS,
    STAT_FIELDS,
    load_hidden_features,
    load_node_features,
    load_token_stats,
    save_hidden_features,
    save_token_stats,
    teacher_forced_stats,
)
from graphs import build_original_graph, dense_edge_attr


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


def write_split(root: Path):
    s = sample()
    (root / "attention").mkdir(parents=True)
    path = root / "attention" / "r1.npz"
    save_attention_sample(s, path)
    write_split_index(
        root, [index_row(root, s, path)], attention_floor=0.01, num_layers=1, num_heads=1,
        alignment="post_token_query_at_same_position",
    )
    return s


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class CoreTests(unittest.TestCase):
    def test_npz_has_exactly_six_attention_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            save_attention_sample(sample(), path)
            with np.load(path) as arrays:
                self.assertEqual(set(arrays.files), set(NPZ_FIELDS))

    def test_hidden_and_stats_are_small_separate_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = write_split(root)
            hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
            save_hidden_features(root / "hidden/r1.npz", s.token_ids, [1, 3], hidden)
            save_token_stats(
                root / "token_stats/r1.npz",
                s.token_ids,
                torch.tensor([0.0, -1.0, -2.0, -3.0]),
                torch.tensor([0.0, 1.0, 2.0, 3.0]),
            )
            with np.load(root / "hidden/r1.npz") as arrays:
                self.assertEqual(set(arrays.files), set(HIDDEN_FIELDS))
            with np.load(root / "token_stats/r1.npz") as arrays:
                self.assertEqual(set(arrays.files), set(STAT_FIELDS))
            restored = next(iter(AttentionDataset(root)))
            x = load_node_features(root, restored, "all")
            self.assertEqual(x.shape, (4, 1 + 6 + 2))

    def test_sidecar_loaders_reject_inconsistent_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / "hidden.npz"
            np.savez_compressed(
                hidden,
                token_ids=np.zeros(3, dtype=np.int32),
                hidden_layer_ids=np.zeros(2, dtype=np.int16),
                hidden_states=np.zeros((1, 3, 2), dtype=np.float16),
            )
            with self.assertRaisesRegex(ValueError, "hidden shapes"):
                load_hidden_features(hidden)

            stats = root / "stats.npz"
            np.savez_compressed(
                stats,
                token_ids=np.zeros(3, dtype=np.int32),
                token_log_prob=np.zeros((3, 1), dtype=np.float32),
                entropy=np.zeros(3, dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "token-stat shapes"):
                load_token_stats(stats)

    def test_teacher_forced_stats_match_direct_distribution(self):
        logits = torch.tensor([
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 2.0]]
        ])
        tokens = torch.tensor([0, 1, 2])
        log_prob, entropy = teacher_forced_stats(logits, tokens, chunk_size=1)
        direct = torch.log_softmax(logits[0, :2], dim=-1)
        expected_lp = torch.tensor([0.0, direct[0, 1], direct[1, 2]])
        expected_h = torch.tensor([
            0.0,
            -(direct[0].exp() * direct[0]).sum(),
            -(direct[1].exp() * direct[1]).sum(),
        ])
        self.assertTrue(torch.allclose(log_prob, expected_lp, atol=1e-6))
        self.assertTrue(torch.allclose(entropy, expected_h, atol=1e-6))

    def test_original_graph_keeps_old_semantics_without_dense_storage(self):
        graph = build_original_graph(sample(), 0.05)
        self.assertEqual(graph.edge_index.tolist(), [[1, 0], [2, 3]])
        self.assertEqual(graph.edge_type.tolist(), [0, 0])
        dense = dense_edge_attr(graph, 1)
        self.assertEqual(dense.shape, (2, 1))
        self.assertTrue(torch.allclose(dense[:, 0].float(), torch.tensor([0.08, 0.10]), atol=1e-3))

    def test_formal_converter_and_builder_use_same_canonical_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal"
            for split in ("train", "test"):
                (formal / split).mkdir(parents=True)
                value = sample()
                torch.save(
                    {
                        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
                        "response_id": f"{split}-1",
                        "source_id": f"source-{split}",
                        "split": split,
                        "cache_dtype": "torch.float16",
                        "attention_cache_fingerprint": fingerprint({
                            "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
                            "split": split, "cache_dtype": "torch.float16", "attention_floor": value.attention_floor,
                            "num_hidden_layers": 1, "num_attention_heads": 1,
                            "model_path": "/models/Meta-Llama-3.1-8B-Instruct", "generator_model": "llama-2-7b-chat",
                        }),
                        "num_attention_layers": 1,
                        "num_attention_heads": 1,
                        "quality": "good",
                        "was_truncated": False,
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
                filename = f"attention_{split}-1.pt"
                spec = {
                    "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
                    "split": split,
                    "cache_dtype": "torch.float16",
                    "attention_floor": value.attention_floor,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "model_path": "/models/Meta-Llama-3.1-8B-Instruct",
                    "generator_model": "llama-2-7b-chat",
                }
                (formal / split / "manifest.json").write_text(json.dumps({
                    "state": "complete", "cache_file_names": [filename], "matched_samples": 1,
                    "cache_files": 1, "cache_files_sha256": {filename: sha256(formal / split / filename)},
                    "attention_cache_spec": spec, "attention_cache_fingerprint": fingerprint(spec),
                }))

            archive = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal, archive)).run()
            dataset = AttentionDataset(archive / "train")
            restored = next(iter(dataset))
            self.assertEqual(restored.sample_id, "train-1")
            self.assertTrue((archive / "train" / "labels.jsonl").is_file())

            output = root / "graphs"
            GraphDatasetBuilder(BuildConfig(
                archive / "train", output, kind="original", tau=0.05,
                device="cpu"
            )).run()
            graph = torch.load(output / "graphs" / "train-1.pt", weights_only=True)
            self.assertEqual(
                set(graph),
                {"num_nodes", "response_idx", "edge_index", "edge_type", "edge_ptr", "edge_channel", "edge_value"},
            )

    def test_legacy_trace_converter_keeps_only_hidden_and_token_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "traces"
            traces.mkdir()
            torch.save(
                {
                    "example_id": "e1",
                    "input_ids": torch.tensor([1, 2, 3]),
                    "selected_hidden_layers": torch.tensor([4, 8]),
                    "hidden_states": torch.randn(2, 3, 5),
                    "token_log_prob": torch.tensor([0.0, -1.0, -2.0]),
                    "next_token_entropy": torch.tensor([0.0, 0.5, 0.7]),
                    "feature_record": {"unused": 1},
                },
                traces / "legacy.pt",
            )
            out = root / "features"
            TraceArchiveConverter(TraceArchiveConfig(traces, out)).run()
            self.assertTrue((out / "hidden/e1.npz").is_file())
            self.assertTrue((out / "token_stats/e1.npz").is_file())


if __name__ == "__main__":
    unittest.main()
