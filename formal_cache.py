"""Validated lazy access to the existing formal sparse attention cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from cache import AttentionSample, sha256


FORMAL_CACHE_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"

FORMAL_TENSOR_FIELDS = {
    "token_ids",
    "attention_diagonal",
    "response_row_ptr",
    "response_column_indices",
    "response_values",
    "y_token",
}
FORMAL_REQUIRED_METADATA_FIELDS = {
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
    "attention_floor",
}
FORMAL_OPTIONAL_METADATA_FIELDS = {
    "task_type",
    "data_source",
    "source",
    "generator_model",
    "temperature",
}
FORMAL_REQUIRED_FIELDS = (
    FORMAL_REQUIRED_METADATA_FIELDS | FORMAL_TENSOR_FIELDS | {"response_idx"}
)


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


def _load_formal_payload(path, *, mmap):
    """Use PyTorch's restricted loader and optionally leave storages memory-mapped."""

    load_kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        # Keep the historical full-load path compatible with PyTorch releases
        # that support ``weights_only`` but predate the ``mmap`` keyword.  The
        # metadata firewall cannot safely fall back to eager tensor reads.
        try:
            payload = torch.load(Path(path), mmap=True, **load_kwargs)
        except TypeError as error:
            if "mmap" not in str(error):
                raise
            raise RuntimeError(
                "formal metadata-only access requires torch.load(mmap=True) support"
            ) from error
    else:
        payload = torch.load(Path(path), **load_kwargs)
    if not isinstance(payload, dict):
        raise ValueError("formal cache must contain a dictionary")
    return payload


def _scalar_metadata(payload, name, *, required):
    if name not in payload:
        if required:
            raise ValueError(f"formal cache is missing metadata field: {name}")
        return None
    value = payload[name]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"formal cache metadata field {name} must be a scalar")
        value = value.item()
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"formal cache metadata field {name} must be a scalar")
    return value


def _validated_formal_metadata(payload, *, split, spec):
    """Copy only allow-listed scalar metadata; never dereference tensor fields."""

    missing = FORMAL_REQUIRED_METADATA_FIELDS.difference(payload)
    if missing:
        raise ValueError("formal cache is missing required metadata fields")
    metadata = {
        name: _scalar_metadata(payload, name, required=True)
        for name in FORMAL_REQUIRED_METADATA_FIELDS
    }
    for name in FORMAL_OPTIONAL_METADATA_FIELDS:
        if name in payload:
            metadata[name] = _scalar_metadata(payload, name, required=False)
    if (
        metadata["attention_cache_schema"] != FORMAL_CACHE_SCHEMA
        or str(metadata["split"]).casefold() != split
        or str(metadata["cache_dtype"]) != spec["cache_dtype"]
        or str(metadata["quality"]).casefold() != "good"
        or bool(metadata["was_truncated"])
        or metadata["attention_cache_fingerprint"] != formal_fingerprint(spec)
        or int(metadata["num_attention_layers"]) != int(spec["num_hidden_layers"])
        or int(metadata["num_attention_heads"]) != int(spec["num_attention_heads"])
        or float(metadata["attention_floor"]) != float(spec["attention_floor"])
    ):
        raise ValueError("formal cache metadata does not match its manifest")
    return metadata


def read_formal_sample_metadata(
    path,
    expected_hash,
    *,
    split,
    spec,
    verify_hash=True,
):
    """Read only allow-listed scalar metadata from a memory-mapped archive.

    Tensor storages are mapped by :func:`torch.load`, but this function neither
    indexes nor converts any attention, token, or ``y_token`` tensor. Hash
    verification, when requested, still scans the archive's raw bytes.
    """

    path = Path(path)
    if verify_hash and sha256(path) != expected_hash:
        raise ValueError(f"formal cache SHA256 mismatch: {path.name}")
    payload = _load_formal_payload(path, mmap=True)
    if FORMAL_REQUIRED_FIELDS.difference(payload):
        raise ValueError("formal cache is missing required fields")
    return _validated_formal_metadata(payload, split=split, spec=spec)


def load_formal_sample(
    path,
    expected_hash,
    *,
    split,
    spec,
    verify_hash=True,
    retain_labels=True,
):
    """Load one attention sample, optionally leaving embedded labels sealed.

    The default preserves the original API and returns validated labels plus
    the full payload. With ``retain_labels=False``, the archive is memory-mapped,
    ``y_token`` is checked only for key presence, and the returned label is
    ``None``; its values and shape are deliberately not read or validated.
    """

    path = Path(path)
    if verify_hash and sha256(path) != expected_hash:
        raise ValueError(f"formal cache SHA256 mismatch: {path.name}")
    retain_labels = bool(retain_labels)
    payload = _load_formal_payload(path, mmap=not retain_labels)
    if FORMAL_REQUIRED_FIELDS.difference(payload):
        raise ValueError("formal cache is missing required fields")
    metadata = _validated_formal_metadata(payload, split=split, spec=spec)

    sample = AttentionSample(
        str(metadata["response_id"]),
        str(metadata["source_id"]),
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
        sample.num_layers != int(metadata["num_attention_layers"])
        or sample.num_heads != int(metadata["num_attention_heads"])
        or sample.num_layers != int(spec["num_hidden_layers"])
        or sample.num_heads != int(spec["num_attention_heads"])
        or sample.attention_floor != float(spec["attention_floor"])
    ):
        raise ValueError("formal sample attention geometry does not match manifest")

    if not retain_labels:
        return sample, None, metadata

    labels = torch.as_tensor(payload["y_token"]).flatten()
    if (
        labels.numel() != sample.num_tokens
        or bool((~((labels == 0) | (labels == 1))).any())
        or bool(labels[: sample.response_idx].any())
    ):
        raise ValueError("formal y_token must be binary and response-aligned")
    return sample, labels, payload
