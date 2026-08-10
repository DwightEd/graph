import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from archive import AttentionArchiveConverter, AttentionArchiveStore, AttentionArchiveVerifier, ArchiveConfig, ArtifactInspector
from build import BuildConfig, GraphDatasetBuilder
from graphs import build_original_graph, build_relation_topk_graph
from hypergraph import build_attention_hypergraph


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replay_spec(split: str) -> dict[str, object]:
    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "cache_dtype": "torch.float16",
        "attention_floor": 0.01,
        "input_policy": "full_context_no_truncation",
        "quality_policy": "official_good_only",
        "label_policy": "all_ragtruth_spans_including_implicit_true",
        "system_prompt": "You are a helpful assistant.",
        "tokenization_policy": "test-tokenization",
        "truncation": False,
        "all_layers": True,
        "all_heads": True,
        "dataset_dir": "/data/RAGTruth",
        "dataset_files_sha256": {"response.jsonl": "a" * 64},
        "model_path": "/models/Meta-Llama-3.1-8B-Instruct",
        "model_files_sha256": {"config.json": "b" * 64},
        "model_class": "LlamaForCausalLM",
        "tokenizer_class": "LlamaTokenizerFast",
        "transformers_version": "4.46.3",
        "tokenizers_version": "0.20.3",
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "max_position_embeddings": 8192,
        "split": split,
        "generator_model": "llama-2-7b-chat",
        "task_type": "all",
        "dtype": "float16",
        "attn_implementation": "eager",
        "torch_version": "2.6.0",
    }


def formal_payload(sample_id: str, source_id: str, split: str, task_type: str = "qa") -> dict[str, object]:
    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "response_id": sample_id,
        "source_id": source_id,
        "split": split,
        "attention_cache_fingerprint": canonical_fingerprint(replay_spec(split)),
        "cache_dtype": torch.float16,
        "generator_model": "llama-2-7b-chat",
        "quality": "good",
        "input_policy": "full_context_no_truncation",
        "was_truncated": False,
        "task_type": task_type,
        "response_idx": 2,
        "token_ids": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        "attention_diagonal": torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float16),
        "response_row_ptr": torch.tensor([0, 1, 3], dtype=torch.int64),
        "response_column_indices": torch.tensor([0, 0, 1], dtype=torch.int32),
        "response_values": torch.tensor([0.7, 0.8, 0.6], dtype=torch.float16),
        "num_attention_layers": 1,
        "num_attention_heads": 1,
        "attention_floor": 0.01,
        "y_token": torch.tensor([0, 0, 1, 1], dtype=torch.float32),
    }


class AttentionArchiveTests(unittest.TestCase):
    def write_formal_root(self, root: Path) -> Path:
        for split, sample_id in (("train", "train-1"), ("test", "test-1")):
            split_dir = root / split
            split_dir.mkdir(parents=True)
            filename = f"{sample_id}.pt"
            torch.save(formal_payload(sample_id, f"source-{split}", split), split_dir / filename)
            manifest = {
                "state": "complete",
                "attention_cache_dir": str(split_dir),
                "attention_cache_fingerprint": canonical_fingerprint(replay_spec(split)),
                "cache_file_names": [filename],
                "cache_files": 1,
                "matched_samples": 1,
                "cache_files_sha256": {filename: sha256(split_dir / filename)},
                "attention_cache_spec": replay_spec(split),
                "saved": 1,
                "reused": 0,
            }
            (split_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        return root

    def test_inspect_reports_safe_inventory_without_manifest_hash_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal_root = self.write_formal_root(Path(directory) / "formal")
            expected_fields = sorted(json.loads((formal_root / "train" / "manifest.json").read_text()))
            report = ArtifactInspector(formal_root).run()

        train = report["splits"]["train"]
        self.assertEqual(train["manifest_fields"], expected_fields)
        self.assertEqual(train["state"], "complete")
        self.assertEqual(train["declared_count"], 1)
        self.assertNotIn("cache_files_sha256", train)
        self.assertFalse(report["payload_hashes_verified"])
        self.assertEqual(train["sample"]["tensors"]["token_ids"], {"shape": [4], "dtype": "int64"})

    def test_converts_verifies_and_preserves_formal_tensors_for_every_graph_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            archive_root = root / "archive"
            summary = AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            verified = AttentionArchiveVerifier(archive_root).run()
            manifest = json.loads((archive_root / "manifest.json").read_text())
            rows = [json.loads(line) for line in (archive_root / "index.jsonl").read_text().splitlines()]
            self.assertEqual([row["split"] for row in rows], ["train", "test"])
            self.assertEqual(set(rows[0]), {"sample_id", "source_id", "split", "attention_path", "N", "R", "response_idx", "nnz", "sha256", "bytes", "task_type"})
            with np.load(archive_root / rows[0]["attention_path"], allow_pickle=False) as payload:
                self.assertEqual(payload["attention_diagonal"].dtype, np.dtype("float16"))
                self.assertEqual(payload["response_values"].dtype, np.dtype("float16"))
            raw, _ = __import__("archive")._formal_sample(torch.load(formal_root / "train" / "train-1.pt", weights_only=True))
            canonical = next(iter(AttentionArchiveStore(archive_root, "train")))
            for kind in ("original", "relation_topk", "relation_topk_channels", "hypergraph"):
                expected = self._build(kind, raw)
                actual = self._build(kind, canonical)
                self._assert_graph_equal(expected.to_dict(), actual.to_dict())
                GraphDatasetBuilder(BuildConfig(archive_root, root / f"graphs-{kind}", kind=kind, split="train", device="cpu")).run()

        self.assertEqual(summary["count"], 2)
        self.assertEqual(verified["count"], 2)
        self.assertEqual(manifest["modalities"], {
            "attention": "present", "hidden_states": "absent", "full_logits": "absent",
            "token_logprob": "absent", "lm_entropy": "absent",
        })
        self.assertEqual(manifest["source_replay_spec"], {key: value for key, value in replay_spec("train").items() if key != "split"})
        self.assertEqual(manifest["cache_dtype"], "torch.float16")
        self.assertIn("index_sha256", manifest)
        self.assertIn("train", manifest["label_sha256"])

    def test_rejects_invalid_source_integrity_and_dtype_without_creating_output(self) -> None:
        cases = (
            ("state", lambda manifest: manifest.update(state="writing"), "state"),
            ("hash", lambda manifest: manifest["cache_files_sha256"].update({"train-1.pt": "0" * 64}), "SHA256"),
            ("spec", lambda manifest: manifest["attention_cache_spec"].update(cache_dtype="torch.float32"), "cache_dtype"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                formal_root = self.write_formal_root(root / "formal")
                path = formal_root / "train" / "manifest.json"
                manifest = json.loads(path.read_text())
                mutate(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    AttentionArchiveConverter(ArchiveConfig(formal_root, root / "archive")).run()
                self.assertFalse((root / "archive").exists())

    def test_rejects_non_production_spec_sample_quality_and_manifest_totals(self) -> None:
        cases = (
            ("schema", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(attention_cache_schema="other")), "schema"),
            ("spec", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(truncation=True)), "truncation"),
            ("layers", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(all_layers=False)), "all_layers"),
            ("heads", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(all_heads=False)), "all_layers/all_heads"),
            ("input", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(input_policy="short")), "input_policy"),
            ("policy", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(quality_policy="other")), "quality_policy"),
            ("task", lambda root: self._rewrite_manifest(root, "train", lambda item: item["attention_cache_spec"].update(task_type="qa")), "task_type"),
            ("quality", lambda root: self._rewrite_payload(root, "train", lambda item: item.update(quality="bad")), "quality"),
            ("totals", lambda root: self._rewrite_manifest(root, "train", lambda item: item.update(saved=0, reused=0)), "saved and reused"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                formal_root = self.write_formal_root(root / "formal")
                mutate(formal_root)
                with self.assertRaisesRegex(ValueError, message):
                    AttentionArchiveConverter(ArchiveConfig(formal_root, root / "archive")).run()

    @staticmethod
    def _rewrite_manifest(root: Path, split: str, mutate: object) -> None:
        path = root / split / "manifest.json"
        manifest = json.loads(path.read_text())
        mutate(manifest)
        manifest["attention_cache_fingerprint"] = canonical_fingerprint(manifest["attention_cache_spec"])
        path.write_text(json.dumps(manifest))

    @staticmethod
    def _rewrite_payload(root: Path, split: str, mutate: object) -> None:
        path = root / split / f"{split}-1.pt"
        payload = torch.load(path, weights_only=True)
        mutate(payload)
        torch.save(payload, path)
        manifest_path = root / split / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cache_files_sha256"][path.name] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))

    def test_rejects_non_float16_payload_and_tampering_and_untrusted_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            payload_path = formal_root / "train" / "train-1.pt"
            payload = torch.load(payload_path, weights_only=True)
            payload["response_values"] = payload["response_values"].float()
            torch.save(payload, payload_path)
            manifest_path = formal_root / "train" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["cache_files_sha256"]["train-1.pt"] = sha256(payload_path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "float16"):
                AttentionArchiveConverter(ArchiveConfig(formal_root, root / "archive")).run()

            formal_root = self.write_formal_root(root / "formal-good")
            archive_root = root / "archive-good"
            AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            rows = [json.loads(line) for line in (archive_root / "index.jsonl").read_text().splitlines()]
            rows[0]["attention_path"] = "attention/train/../../outside.npz"
            (archive_root / "index.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "index_sha256|attention_path"):
                AttentionArchiveVerifier(archive_root).run()

    def test_rejects_semantically_replaced_canonical_npz_before_store_or_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            archive_root = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            path = archive_root / "attention" / "train" / "train-1.npz"
            with np.load(path, allow_pickle=False) as arrays:
                replacement = {name: arrays[name].copy() for name in arrays.files}
            replacement["response_values"][0] = np.float16(0.5)
            np.savez_compressed(path, **replacement)

            with self.assertRaisesRegex(ValueError, "SHA256"):
                AttentionArchiveVerifier(archive_root).run()
            store = AttentionArchiveStore(archive_root, "train")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                next(iter(store))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                GraphDatasetBuilder(
                    BuildConfig(archive_root, root / "graphs", split="train", device="cpu")
                ).run()

    def test_rejects_index_dimensions_that_do_not_bind_the_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            archive_root = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            index_path = archive_root / "index.jsonl"
            rows = [json.loads(line) for line in index_path.read_text().splitlines()]
            rows[0]["nnz"] += 1
            index_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            manifest_path = archive_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["index_sha256"] = sha256(index_path)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "dimensions"):
                next(iter(AttentionArchiveStore(archive_root, "train")))

    def test_rejects_source_replacement_after_inventory_before_conversion_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            converter = AttentionArchiveConverter(ArchiveConfig(formal_root, root / "archive"))
            inventory = converter._source_inventory(formal_root)
            path = formal_root / "train" / "train-1.pt"
            payload = torch.load(path, weights_only=True)
            payload["response_values"][0] = torch.tensor(0.5, dtype=torch.float16)
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "SHA256"):
                converter._write_archive(formal_root, root / "staging", inventory)

    def test_canonical_builder_passes_requested_device_to_one_pass_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            archive_root = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            with patch("build.AttentionArchiveStore", wraps=AttentionArchiveStore) as store:
                GraphDatasetBuilder(BuildConfig(archive_root, root / "graphs", split="train", device="cpu", limit=1)).run()

        store.assert_called_once_with(archive_root, "train", device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_canonical_store_moves_every_tensor_to_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal_root = self.write_formal_root(root / "formal")
            archive_root = root / "archive"
            AttentionArchiveConverter(ArchiveConfig(formal_root, archive_root)).run()
            sample = next(iter(AttentionArchiveStore(archive_root, "train", device="cuda")))

        for tensor in (sample.token_ids, sample.attention_diagonal, sample.response_row_ptr,
                       sample.response_column_indices, sample.response_values):
            self.assertEqual(tensor.device.type, "cuda")

    @staticmethod
    def _build(kind: str, sample: object):
        if kind == "original":
            return build_original_graph(sample, .05)
        if kind == "hypergraph":
            return build_attention_hypergraph(sample, .05)
        return build_relation_topk_graph(sample, 8, 8, with_channels=kind == "relation_topk_channels")

    def _assert_graph_equal(self, expected: dict[str, object], actual: dict[str, object]) -> None:
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            else:
                self.assertEqual(expected[key], actual[key])


if __name__ == "__main__":
    unittest.main()
