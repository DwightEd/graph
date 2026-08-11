import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from features import save_hidden_features, save_token_stats
from graphs import build_original_graph
from metadata import RESEARCH_INDEX_FIELDS, enrich_ragtruth_indices
from research_dataset import ResearchDataset


def make_archive(root: Path) -> tuple[Path, Path]:
    split = root / "canonical" / "train"
    (split / "attention").mkdir(parents=True)
    sample = AttentionSample(
        "r1", "s1", 2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[1.0, 0.5, 0.2, 0.1]]], dtype=torch.float16),
        torch.tensor([0, 1, 2]), torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0.08, 0.10], dtype=torch.float16), 0.01,
    )
    attention_path = split / "attention" / "r1.npz"
    save_attention_sample(sample, attention_path)
    write_split_index(
        split, [index_row(split, sample, attention_path)], attention_floor=0.01,
        num_layers=1, num_heads=1, alignment="post_token_query_at_same_position",
        extra={"observer_model": "Meta-Llama-3.1-8B-Instruct"},
    )
    (split / "labels.jsonl").write_text(
        json.dumps({"sample_id": "r1", "positive_runs": [[0, 1]]}) + "\n", encoding="utf-8"
    )
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["labels_sha256"] = sha256(split / "labels.jsonl")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    save_hidden_features(split / "hidden" / "r1.npz", sample.token_ids, [0], torch.ones((1, 4, 2)))
    save_token_stats(split / "token_stats" / "r1.npz", sample.token_ids, torch.zeros(4), torch.ones(4))
    return split, sample.token_ids


def make_graph(split: Path, root: Path) -> Path:
    graph_split = root / "graphs" / "train"
    (graph_split / "graphs").mkdir(parents=True)
    sample = ResearchDataset(split)["r1"].attention()
    graph_path = graph_split / "graphs" / "r1.pt"
    torch.save(build_original_graph(sample, 0.01).to_dict(), graph_path)
    row = {
        "sample_id": "r1", "source_id": "s1", "path": "graphs/r1.pt",
        "num_nodes": 4, "num_edges": 2, "sha256": sha256(graph_path),
        "bytes": graph_path.stat().st_size,
    }
    index_path = graph_split / "index.jsonl"
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (graph_split / "manifest.json").write_text(json.dumps({
        "count": 1, "index_sha256": sha256(index_path),
        "input_manifest_sha256": sha256(split / "manifest.json"),
        "input_index_sha256": sha256(split / "index.jsonl"),
    }), encoding="utf-8")
    return graph_split


def make_ragtruth_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "source_info.jsonl").write_text(json.dumps({
        "source_id": "s1", "task_type": "Summary", "source": "CNN/DM",
    }) + "\n", encoding="utf-8")
    (dataset / "response.jsonl").write_text(json.dumps({
        "id": "r1", "source_id": "s1", "model": "llama-2-7b-chat",
        "temperature": 0.7, "split": "train", "quality": "good",
    }) + "\n", encoding="utf-8")
    return dataset


class MetadataTests(unittest.TestCase):
    def test_dataset_binds_graph_sidecars_and_labels_to_canonical_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, token_ids = make_archive(root)
            graph_split = make_graph(split, root)

            research = ResearchDataset(split, {"original": graph_split})
            item = research["r1"]
            self.assertEqual(research.graph_manifests["original"]["count"], 1)
            self.assertEqual(item.attention().token_ids.tolist(), token_ids.tolist())
            self.assertEqual(item.graph("original")["num_nodes"], 4)
            original = item.original_graph()
            self.assertEqual(original.num_nodes, 4)
            self.assertEqual(original.edge_index.tolist(), [[0, 0], [2, 3]])
            self.assertEqual(item.hidden()[0].tolist(), token_ids.tolist())
            self.assertEqual(item.stats()[0].tolist(), token_ids.tolist())
            self.assertEqual(research.labels().token_labels(item).tolist(), [0, 0, 1, 0])

    def test_dataset_rejects_graph_with_stale_canonical_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            graph_split = make_graph(split, root)
            manifest_path = graph_split / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["input_index_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                ResearchDataset(split, {"original": graph_split})

    def test_dataset_rejects_hidden_and_stats_with_misaligned_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, token_ids = make_archive(root)
            save_hidden_features(split / "hidden" / "r1.npz", token_ids + 1, [0], torch.ones((1, 4, 2)))
            item = ResearchDataset(split)["r1"]
            with self.assertRaisesRegex(ValueError, "hidden token_ids"):
                item.hidden()
            save_hidden_features(split / "hidden" / "r1.npz", token_ids, [0], torch.ones((1, 4, 2)))
            save_token_stats(split / "token_stats" / "r1.npz", token_ids + 1, torch.zeros(4), torch.ones(4))
            with self.assertRaisesRegex(ValueError, "token_stats token_ids"):
                item.stats()

    def test_dataset_rejects_attention_file_with_wrong_byte_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            (split / "attention" / "r1.npz").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "attention sample byte count"):
                ResearchDataset(split)["r1"].attention()

    def test_dataset_rejects_attention_geometry_that_disagrees_with_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            manifest_path = split / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["num_heads"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "attention geometry"):
                ResearchDataset(split)["r1"].attention()

    def test_dataset_rejects_graph_response_boundary_that_disagrees_with_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            graph_split = make_graph(split, root)
            item = ResearchDataset(split, {"original": graph_split})["r1"]
            bad_graph = {"num_nodes": 4, "response_idx": 1}

            with patch("research_dataset.torch.load", return_value=bad_graph):
                with self.assertRaisesRegex(ValueError, "response boundaries"):
                    item.graph("original")

    def test_labels_reject_sample_from_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            other_split, _ = make_archive(root / "other")
            labels = ResearchDataset(split).labels()
            foreign = ResearchDataset(other_split)["r1"]
            with self.assertRaisesRegex(ValueError, "different dataset"):
                labels.token_labels(foreign)

    def test_metadata_graph_preflight_leaves_all_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            graph_split = make_graph(split, root)
            graph_manifest = json.loads((graph_split / "manifest.json").read_text())
            graph_manifest["input_manifest_sha256"] = "0" * 64
            (graph_split / "manifest.json").write_text(json.dumps(graph_manifest), encoding="utf-8")
            before = {
                path: path.read_bytes()
                for path in (split / "index.jsonl", split / "manifest.json", graph_split / "manifest.json")
            }
            with self.assertRaisesRegex(ValueError, "provenance"):
                enrich_ragtruth_indices(root / "canonical", make_ragtruth_dataset(root), root / "graphs")
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_metadata_canonical_preflight_leaves_indices_unchanged_without_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            attention = split / "attention" / "r1.npz"
            altered = bytearray(attention.read_bytes())
            altered[-1] ^= 1
            attention.write_bytes(altered)
            before = {path: path.read_bytes() for path in (split / "index.jsonl", split / "manifest.json")}

            with self.assertRaisesRegex(ValueError, "SHA256"):
                enrich_ragtruth_indices(root / "canonical", make_ragtruth_dataset(root))

            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_metadata_rejects_generator_mismatch_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            manifest_path = split / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["generator_model"] = "expected-model"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            before = {path: path.read_bytes() for path in (split / "index.jsonl", manifest_path)}

            with self.assertRaisesRegex(ValueError, "generator_model"):
                enrich_ragtruth_indices(root / "canonical", make_ragtruth_dataset(root))

            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_enrich_existing_json_and_access_research_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, _ = make_archive(root)
            graph_split = make_graph(split, root)
            result = enrich_ragtruth_indices(root / "canonical", make_ragtruth_dataset(root), root / "graphs")
            self.assertEqual(result["splits"], {"train": 1})
            self.assertEqual(result["graphs"], {"train": 1})
            row = json.loads((split / "index.jsonl").read_text())
            self.assertEqual(tuple(row), RESEARCH_INDEX_FIELDS)
            research = ResearchDataset(split, {"original": graph_split})
            item = research["r1"]
            self.assertEqual(item.data_source, "CNN/DM")
            self.assertEqual(item.generator_model, "llama-2-7b-chat")
            self.assertEqual(item.observer_model, "Meta-Llama-3.1-8B-Instruct")


if __name__ == "__main__":
    unittest.main()
