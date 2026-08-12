"""Validated lazy access to the existing formal sparse attention cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from cache import AttentionSample, sha256


FORMAL_CACHE_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"


def formal_fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_formal_manifest(split_root):
    split_root = Path(split_root)
    path = split_root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"formal split has no manifest: {split_root}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "state",
        "cache_file_names",
        "matched_samples",
        "cache_files",
        "cache_files_sha256",
        "attention_cache_spec",
        "attention_cache_fingerprint",
    }
    if required.difference(manifest) or manifest["state"] != "complete":
        raise ValueError("formal manifest is incomplete")

    names = manifest["cache_file_names"]
    hashes = manifest["cache_files_sha256"]
    if (
        not isinstance(names, list)
        or manifest["matched_samples"] != len(names)
        or manifest["cache_files"] != len(names)
        or not isinstance(hashes, dict)
        or set(hashes) != set(names)
    ):
        raise ValueError("formal manifest cache inventory is invalid")

    spec = manifest["attention_cache_spec"]
    split = str(spec.get("split", "")).casefold()
    if (
        spec.get("attention_cache_schema") != FORMAL_CACHE_SCHEMA
        or split not in {"train", "test"}
        or spec.get("cache_dtype") != "torch.float16"
        or not 0.0 < float(spec.get("attention_floor", -1)) <= 1.0
        or int(spec.get("num_hidden_layers", 0)) < 1
        or int(spec.get("num_attention_heads", 0)) < 1
    ):
        raise ValueError("formal attention_cache_spec is invalid")
    if manifest["attention_cache_fingerprint"] != formal_fingerprint(spec):
        raise ValueError("formal attention fingerprint does not match its spec")

    files = [(split_root / name, str(hashes[name])) for name in names]
    if any(file.parent != split_root or not file.is_file() for file, _ in files):
        raise ValueError("formal manifest references an invalid cache file")
    return manifest, spec, files, split


def load_formal_sample(path, expected_hash, *, split, spec):
    path = Path(path)
    if sha256(path) != expected_hash:
        raise ValueError(f"formal cache SHA256 mismatch: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("formal cache must contain a dictionary")
    required = {
        "attention_cache_schema",
        "attention_cache_fingerprint",
        "response_id",
        "source_id",
        "split",
        "cache_dtype",
        "num_attention_layers",
        "num_attention_heads",
        "quality",
        "was_truncated",
        "response_idx",
        "token_ids",
        "attention_diagonal",
        "response_row_ptr",
        "response_column_indices",
        "response_values",
        "attention_floor",
        "y_token",
    }
    if required.difference(payload):
        raise ValueError("formal cache is missing required fields")
    if (
        payload["attention_cache_schema"] != FORMAL_CACHE_SCHEMA
        or str(payload["split"]).casefold() != split
        or str(payload["cache_dtype"]) != spec["cache_dtype"]
        or str(payload["quality"]).casefold() != "good"
        or bool(payload["was_truncated"])
        or payload["attention_cache_fingerprint"] != formal_fingerprint(spec)
    ):
        raise ValueError("formal cache metadata does not match its manifest")

    sample = AttentionSample(
        str(payload["response_id"]),
        str(payload["source_id"]),
        int(payload["response_idx"]),
        torch.as_tensor(payload["token_ids"]),
        torch.as_tensor(payload["attention_diagonal"]),
        torch.as_tensor(payload["response_row_ptr"]),
        torch.as_tensor(payload["response_column_indices"]),
        torch.as_tensor(payload["response_values"]),
        float(payload["attention_floor"]),
    )
    sample.validate()
    if (
        sample.num_layers != int(payload["num_attention_layers"])
        or sample.num_heads != int(payload["num_attention_heads"])
        or sample.num_layers != int(spec["num_hidden_layers"])
        or sample.num_heads != int(spec["num_attention_heads"])
        or sample.attention_floor != float(spec["attention_floor"])
    ):
        raise ValueError("formal sample attention geometry does not match manifest")

    labels = torch.as_tensor(payload["y_token"]).flatten()
    if (
        labels.numel() != sample.num_tokens
        or bool((~((labels == 0) | (labels == 1))).any())
        or bool(labels[: sample.response_idx].any())
    ):
        raise ValueError("formal y_token must be binary and response-aligned")
    return sample, labels, payload
