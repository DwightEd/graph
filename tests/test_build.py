import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch

from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionSample, load_attention_sample, save_attention_sample


def make_sample(sample_id: str, source_id: str) -> AttentionSample:
    return AttentionSample(
        sample_id=sample_id,
        source_id=source_id,
        response_idx=2,
        token_ids=torch.tensor([101, 102, 201, 202], dtype=torch.int64),
        attention_diagonal=torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
        response_row_ptr=torch.tensor([0, 1, 3], dtype=torch.int64),
        response_column_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        response_values=torch.tensor([0.7, 0.8, 0.6]),
        attention_floor=0.01,
    )


class GraphDatasetBuilderTests(unittest.TestCase):
    def write_cache(self, root: Path) -> Path:
        cache_dir = root / "cache" / "train"
        cache_dir.mkdir(parents=True)
        save_attention_sample(make_sample("first", "source-a"), cache_dir / "first.pt")
        save_attention_sample(make_sample("second", "source-b"), cache_dir / "second.pt")
        (cache_dir / "manifest.json").write_text(json.dumps({"ignored": True}))
        (cache_dir / "index.jsonl").write_text(json.dumps({"label": "ignored"}) + "\n")
        return cache_dir

    def test_builds_each_graph_kind_from_pt_files_and_writes_label_free_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)
            for kind, schema, count_name in (
                ("original", "token-graph-v1", "num_edges"),
                ("relation_topk", "token-graph-v1", "num_edges"),
                ("relation_topk_channels", "token-graph-v1", "num_edges"),
                ("hypergraph", "attention-hypergraph-v1", "num_hyperedges"),
            ):
                output = root / kind
                summary = GraphDatasetBuilder(BuildConfig(
                    cache_dir=cache_dir,
                    output_dir=output,
                    kind=kind,
                    device="cpu",
                )).run()
                manifest = json.loads((output / "manifest.json").read_text())
                index_rows = [json.loads(line) for line in (output / "index.jsonl").read_text().splitlines()]

                self.assertEqual(summary["count"], 2)
                self.assertEqual(manifest["kind"], kind)
                self.assertEqual(manifest["count"], 2)
                self.assertNotIn("label", manifest)
                self.assertEqual(len(index_rows), 2)
                for row in index_rows:
                    self.assertEqual(set(row), {"sample_id", "source_id", "path", "num_nodes", count_name})
                    self.assertNotIn("label", row)
                    payload = torch.load(output / row["path"], weights_only=True)
                    self.assertEqual(set(payload), {"schema", "graph"})
                    self.assertEqual(payload["schema"], schema)
                    self.assertNotIn("label", payload["graph"])
                    self.assertEqual(payload["graph"]["token_ids"].numel(), row["num_nodes"])

            original_manifest = json.loads((root / "original" / "manifest.json").read_text())
            self.assertTrue(original_manifest["compatibility_dense_channel_mode"])

    def test_limit_and_device_are_applied_to_each_cache_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)
            output = root / "output"
            with patch("build.load_attention_sample", wraps=load_attention_sample) as loader:
                summary = GraphDatasetBuilder(BuildConfig(
                    cache_dir=cache_dir,
                    output_dir=output,
                    kind="relation_topk_channels",
                    device="cpu",
                    limit=1,
                )).run()

        self.assertEqual(summary["count"], 1)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")

    def test_rejects_invalid_cache_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            for cache_dir in (root / "missing", root / "empty"):
                if cache_dir.name == "empty":
                    cache_dir.mkdir()
                with self.subTest(cache_dir=cache_dir), self.assertRaisesRegex(ValueError, "cache_dir"):
                    GraphDatasetBuilder(BuildConfig(cache_dir, output, device="cpu")).run()
                self.assertFalse(output.exists())

    def test_rejects_same_cache_and_output_without_changing_input_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)
            index_before = (cache_dir / "index.jsonl").read_bytes()
            manifest_before = (cache_dir / "manifest.json").read_bytes()

            with self.assertRaisesRegex(ValueError, "must differ"):
                GraphDatasetBuilder(BuildConfig(cache_dir, cache_dir, device="cpu")).run()

            self.assertEqual((cache_dir / "index.jsonl").read_bytes(), index_before)
            self.assertEqual((cache_dir / "manifest.json").read_bytes(), manifest_before)

    def test_rejects_stale_output_and_invalid_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)
            output = root / "output"
            output.mkdir()
            (output / "old.txt").write_text("stale")
            with self.assertRaises(FileExistsError):
                GraphDatasetBuilder(BuildConfig(cache_dir, output, device="cpu")).run()

            for config in (
                BuildConfig(cache_dir, root / "tau", tau=float("nan"), device="cpu"),
                BuildConfig(cache_dir, root / "negative-k", k_prompt=-1, device="cpu"),
                BuildConfig(cache_dir, root / "zero-limit", limit=0, device="cpu"),
            ):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    GraphDatasetBuilder(config).run()

    def test_rejects_tau_below_a_cache_floor_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)
            output = root / "output"

            with patch("build.load_attention_sample", wraps=load_attention_sample) as loader:
                with self.assertRaisesRegex(ValueError, "attention_floor"):
                    GraphDatasetBuilder(BuildConfig(
                        cache_dir,
                        output,
                        kind="original",
                        tau=0.005,
                        device="cpu",
                    )).run()

            self.assertFalse(output.exists())
            loader.assert_called_once_with(cache_dir / "first.pt", map_location="cpu")

    def test_preflights_only_the_first_threshold_cache_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = self.write_cache(root)

            def load_until_gpu(path: Path, map_location: str) -> AttentionSample:
                if map_location == "cpu":
                    return load_attention_sample(path, map_location)
                raise RuntimeError("stop after preflight")

            with patch("build.load_attention_sample", side_effect=load_until_gpu) as loader:
                with self.assertRaisesRegex(RuntimeError, "stop after preflight"):
                    GraphDatasetBuilder(BuildConfig(
                        cache_dir,
                        root / "output",
                        kind="original",
                        tau=0.05,
                        device="cuda",
                    )).run()

            cpu_calls = [
                call for call in loader.call_args_list if call.kwargs["map_location"] == "cpu"
            ]
            self.assertEqual(cpu_calls, [
                call(cache_dir / "first.pt", map_location="cpu"),
            ])


if __name__ == "__main__":
    unittest.main()
