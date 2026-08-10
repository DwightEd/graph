import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from archive import ArtifactInspector, AttentionArchiveConverter, AttentionArchiveVerifier, ArchiveConfig
from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionSample, index_row, load_attention_sample, save_attention_sample, sha256, write_split_index
from extract import AttentionExtractor, ExtractionConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: dict[str, object]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_payload(sample_id: str) -> dict[str, object]:
    split = sample_id.split("-", 1)[0]
    spec = formal_spec(split)
    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "response_id": sample_id,
        "source_id": f"source-{sample_id}",
        "split": split,
        "cache_dtype": "torch.float16",
        "attention_cache_fingerprint": fingerprint(spec),
        "num_attention_layers": 1,
        "num_attention_heads": 1,
        "quality": "good",
        "was_truncated": False,
        "response_idx": 2,
        "token_ids": torch.tensor([10, 11, 12, 13]),
        "attention_diagonal": torch.ones((1, 1, 4), dtype=torch.float16),
        "response_row_ptr": torch.tensor([0, 2, 4]),
        "response_column_indices": torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        "response_values": torch.tensor([0.02, 0.08, 0.10, 0.04], dtype=torch.float16),
        "attention_floor": 0.01,
        "y_token": torch.tensor([0, 0, 1, 0]),
    }


def formal_spec(split: str) -> dict[str, object]:
    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "split": split,
        "cache_dtype": "torch.float16",
        "attention_floor": 0.01,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "model_path": "/models/Meta-Llama-3.1-8B-Instruct",
        "generator_model": "llama-2-7b-chat",
        "system_prompt": "中文",
        "dataset_files_sha256": {"response.jsonl": "a" * 64},
        "model_files_sha256": {"config.json": "b" * 64},
    }


def write_formal_root(root: Path) -> Path:
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir(parents=True)
        filename = f"attention_{split}-1.pt"
        path = directory / filename
        torch.save(sample_payload(f"{split}-1"), path)
        spec = formal_spec(split)
        (directory / "manifest.json").write_text(json.dumps({
            "state": "complete",
            "attention_cache_fingerprint": fingerprint(spec),
            "cache_file_names": [filename],
            "matched_samples": 1,
            "cache_files": 1,
            "cache_files_sha256": {filename: sha256(path)},
            "attention_cache_spec": spec,
        }))
    return root


def attention_sample(*, row_ptr=None, columns=None) -> AttentionSample:
    return AttentionSample(
        sample_id="r1",
        source_id="s1",
        response_idx=2,
        token_ids=torch.tensor([10, 11, 12, 13]),
        attention_diagonal=torch.ones((1, 1, 4), dtype=torch.float16),
        response_row_ptr=torch.tensor(row_ptr or [0, 2, 4]),
        response_column_indices=torch.tensor(columns or [0, 1, 0, 2], dtype=torch.int32),
        response_values=torch.tensor([0.02, 0.08, 0.10, 0.04], dtype=torch.float16),
        attention_floor=0.01,
    )


class IntegrityTests(unittest.TestCase):
    def test_converter_writes_bound_canonical_inventory_and_verifier_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            formal = write_formal_root(root / "formal")
            source_manifest_sha256 = sha256(formal / "train" / "manifest.json")
            AttentionArchiveConverter(ArchiveConfig(formal, archive)).run()

            manifest = json.loads((archive / "train" / "manifest.json").read_text())
            row = json.loads((archive / "train" / "index.jsonl").read_text())
            self.assertIn("schema", manifest)
            self.assertIn("count", manifest)
            self.assertIn("index_sha256", manifest)
            self.assertIn("labels_sha256", manifest)
            self.assertEqual(manifest["source_manifest_sha256"], source_manifest_sha256)
            self.assertEqual(manifest["source_attention_fingerprint"], fingerprint(formal_spec("train")))
            self.assertEqual(manifest["observer_model"], "Meta-Llama-3.1-8B-Instruct")
            self.assertEqual(manifest["generator_model"], "llama-2-7b-chat")
            self.assertIn("sha256", row)
            self.assertIn("bytes", row)

            path = archive / "train" / row["path"]
            original = path.read_bytes()
            path.write_bytes(original + b"tampered")
            with self.assertRaises(ValueError):
                AttentionArchiveVerifier(archive).run()
            path.write_bytes(original)

            labels = archive / "train" / "labels.jsonl"
            labels.write_text(labels.read_text() + "{}\n")
            with self.assertRaises(ValueError):
                AttentionArchiveVerifier(archive).run()

    def test_verifier_rejects_deleted_index_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(write_formal_root(root / "formal"), archive)).run()
            (archive / "train" / "index.jsonl").write_text("")

            with self.assertRaises(ValueError):
                AttentionArchiveVerifier(archive).run()

    def test_verifier_accepts_direct_split_without_labels_and_enforces_declared_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "split"
            sample = attention_sample()
            path = split / "attention" / "r1.npz"
            save_attention_sample(sample, path)
            manifest = write_split_index(
                split, [index_row(split, sample, path)], attention_floor=0.01, num_layers=1, num_heads=1,
                alignment="post_token_query_at_same_position",
            )
            self.assertEqual(AttentionArchiveVerifier(split).run()["count"], 1)

            manifest["labels_sha256"] = "0" * 64
            (split / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                AttentionArchiveVerifier(split).run()

    def test_converter_rejects_cross_split_geometry_and_prompt_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = write_formal_root(root / "formal")
            path = formal / "test" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["attention_cache_spec"]["num_attention_heads"] = 2
            manifest["attention_cache_fingerprint"] = fingerprint(manifest["attention_cache_spec"])
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "geometry"):
                AttentionArchiveConverter(ArchiveConfig(formal, root / "archive")).run()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = write_formal_root(root / "formal")
            manifest_path = formal / "test" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["attention_cache_spec"]["model_path"] = "/models/another-observer"
            changed_fingerprint = fingerprint(manifest["attention_cache_spec"])
            manifest["attention_cache_fingerprint"] = changed_fingerprint
            payload_path = formal / "test" / "attention_test-1.pt"
            payload = torch.load(payload_path, weights_only=True)
            payload["attention_cache_fingerprint"] = changed_fingerprint
            torch.save(payload, payload_path)
            manifest["cache_files_sha256"][payload_path.name] = sha256(payload_path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "model"):
                AttentionArchiveConverter(ArchiveConfig(formal, root / "archive")).run()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = write_formal_root(root / "formal")
            payload_path = formal / "train" / "attention_train-1.pt"
            payload = torch.load(payload_path, weights_only=True)
            payload["y_token"][0] = 1
            torch.save(payload, payload_path)
            self._rewrite_manifest(formal, lambda item: item["cache_files_sha256"].update({payload_path.name: sha256(payload_path)}))
            with self.assertRaisesRegex(ValueError, "y_token"):
                AttentionArchiveConverter(ArchiveConfig(formal, root / "archive")).run()

    def test_inspector_returns_bounded_summary_without_source_hash_map(self):
        with tempfile.TemporaryDirectory() as directory:
            report = ArtifactInspector(write_formal_root(Path(directory) / "formal")).run()
        encoded = json.dumps(report)
        self.assertNotIn("cache_files_sha256", encoded)
        self.assertNotIn("dataset_files_sha256", encoded)
        self.assertNotIn("model_files_sha256", encoded)
        self.assertEqual(report["splits"]["train"]["sample"]["tensors"]["token_ids"], {"shape": [4], "dtype": "int64"})

    def test_extractor_rejects_invalid_limit_before_model_loading(self):
        for limit in (True, 1.5):
            with self.subTest(limit=limit), patch("extract.AutoTokenizer.from_pretrained", side_effect=AssertionError("model loaded")):
                with self.assertRaises(ValueError):
                    AttentionExtractor(ExtractionConfig("model", "data", "output", "train", limit=limit)).run()

    def test_extractor_rejects_empty_dataset_before_model_loading(self):
        with patch("extract.load_ragtruth_samples", return_value=[]), \
                patch("extract.AutoTokenizer.from_pretrained", side_effect=AssertionError("model loaded")):
            with self.assertRaises(ValueError):
                AttentionExtractor(ExtractionConfig("model", "data", "output", "train", limit=1)).run()

    def test_attention_validation_rejects_nonmonotone_or_noncausal_csr(self):
        with self.subTest("nonmonotone_row_ptr"):
            with self.assertRaises(ValueError):
                attention_sample(row_ptr=[0, 3, 2]).validate()
        with self.subTest("future_column"):
            with self.assertRaises(ValueError):
                attention_sample(columns=[0, 2, 0, 3]).validate()
        with self.subTest("repeated_column"):
            with self.assertRaises(ValueError):
                attention_sample(columns=[0, 0, 0, 2]).validate()
        with self.subTest("zero_layers"):
            with self.assertRaises(ValueError):
                AttentionSample(
                    "r1", "s1", 2, torch.tensor([10, 11, 12, 13]),
                    torch.empty((0, 1, 4), dtype=torch.float16), torch.tensor([0]),
                    torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.float16), 0.01,
                ).validate()
        with self.subTest("zero_heads"):
            with self.assertRaises(ValueError):
                AttentionSample(
                    "r1", "s1", 2, torch.tensor([10, 11, 12, 13]),
                    torch.empty((1, 0, 4), dtype=torch.float16), torch.tensor([0]),
                    torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.float16), 0.01,
                ).validate()

    def test_npz_loader_rejects_wrong_canonical_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez_compressed(
                path,
                token_ids=np.array([10, 11, 12, 13], dtype=np.float32),
                response_idx=np.array(2, dtype=np.int32),
                attention_diagonal=np.ones((1, 1, 4), dtype=np.float16),
                response_row_ptr=np.array([0, 2, 4], dtype=np.int32),
                response_column_indices=np.array([0, 1, 0, 2], dtype=np.int32),
                response_values=np.ones(4, dtype=np.float16),
            )
            with self.assertRaises(ValueError):
                load_attention_sample(path, sample_id="r1", source_id="s1", attention_floor=0.01)

    def test_converter_rejects_invalid_production_source(self):
        for name, mutate in (
            ("state", lambda root: self._rewrite_manifest(root, lambda item: item.update(state="writing"))),
            ("hash", lambda root: self._rewrite_manifest(root, lambda item: item["cache_files_sha256"].update({"attention_train-1.pt": "0" * 64}))),
            ("schema", lambda root: self._rewrite_payload(root, lambda item: item.update(attention_cache_schema="other"))),
            ("dtype", lambda root: self._rewrite_payload(root, lambda item: item.update(cache_dtype=torch.float32))),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                formal = write_formal_root(root / "formal")
                mutate(formal)

                with self.assertRaises(ValueError):
                    AttentionArchiveConverter(ArchiveConfig(formal, root / "archive")).run()
                self.assertFalse((root / "archive").exists())

    @staticmethod
    def _rewrite_manifest(root: Path, mutate) -> None:
        path = root / "train" / "manifest.json"
        manifest = json.loads(path.read_text())
        mutate(manifest)
        path.write_text(json.dumps(manifest))

    @staticmethod
    def _rewrite_payload(root: Path, mutate) -> None:
        path = root / "train" / "attention_train-1.pt"
        payload = torch.load(path, weights_only=True)
        mutate(payload)
        torch.save(payload, path)
        manifest_path = root / "train" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cache_files_sha256"][path.name] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))

    def test_converter_and_builder_never_write_into_existing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = write_formal_root(root / "formal")
            (root / "archive").mkdir()
            with self.assertRaises((FileExistsError, ValueError)):
                AttentionArchiveConverter(ArchiveConfig(formal, root / "archive")).run()
            with self.assertRaises((FileExistsError, ValueError)):
                AttentionArchiveConverter(ArchiveConfig(formal, formal)).run()

            archive = root / "archive-good"
            AttentionArchiveConverter(ArchiveConfig(formal, archive)).run()
            cache = archive / "train"
            with self.assertRaises(ValueError):
                GraphDatasetBuilder(BuildConfig(cache, cache, device="cpu")).run()


if __name__ == "__main__":
    unittest.main()
