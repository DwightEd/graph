"""Convert verified formal RAGTruth attention caches to canonical NPZ splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    verify_split,
    write_split_index,
)

FORMAL_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"


@dataclass(frozen=True)
class ArchiveConfig:
    formal_root: str | Path
    output_root: str | Path


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _positive_runs(y_token, response_idx):
    labels = torch.as_tensor(y_token).flatten()
    positions = torch.nonzero(labels[response_idx:] > 0, as_tuple=False).flatten().tolist()
    runs = []
    for position in positions:
        if not runs or position != runs[-1][1]:
            runs.append([position, position + 1])
        else:
            runs[-1][1] += 1
    return runs


def _formal_manifest(split_root: Path, split: str):
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
    if (
        spec.get("attention_cache_schema") != FORMAL_SCHEMA
        or spec.get("split") != split
        or spec.get("cache_dtype") != "torch.float16"
        or not 0.0 < float(spec.get("attention_floor", -1)) <= 1.0
        or int(spec.get("num_hidden_layers", 0)) < 1
        or int(spec.get("num_attention_heads", 0)) < 1
    ):
        raise ValueError("formal attention_cache_spec is invalid")
    if manifest["attention_cache_fingerprint"] != _fingerprint(spec):
        raise ValueError("formal attention fingerprint does not match its spec")
    files = [(split_root / name, str(hashes[name])) for name in names]
    if any(path.parent != split_root or not path.is_file() for path, _ in files):
        raise ValueError("formal manifest references an invalid cache file")
    return manifest, spec, files


def _load_formal(path: Path, expected_hash: str, *, split: str, spec: dict):
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
        payload["attention_cache_schema"] != FORMAL_SCHEMA
        or str(payload["split"]).casefold() != split
        or str(payload["cache_dtype"]) != spec["cache_dtype"]
        or str(payload["quality"]).casefold() != "good"
        or bool(payload["was_truncated"])
        or payload["attention_cache_fingerprint"] != _fingerprint(spec)
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
    return sample, labels


class AttentionArchiveConverter:
    def __init__(self, config: ArchiveConfig):
        self.config = config

    def run(self):
        formal_root = Path(self.config.formal_root)
        output_root = Path(self.config.output_root)
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError("output_root must be empty")
        summary = {"output_root": str(output_root), "splits": {}}

        for split in ("train", "test"):
            manifest, spec, files = _formal_manifest(formal_root / split, split)
            output = output_root / split
            (output / "attention").mkdir(parents=True, exist_ok=True)
            rows, label_rows = [], []
            for path, digest in files:
                sample, labels = _load_formal(path, digest, split=split, spec=spec)
                attention_path = output / "attention" / f"{sample.sample_id}.npz"
                save_attention_sample(sample, attention_path)
                rows.append(
                    index_row(
                        output,
                        sample,
                        attention_path,
                        metadata={"split": split, "quality": "good"},
                    )
                )
                label_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "positive_runs": _positive_runs(labels, sample.response_idx),
                    }
                )

            label_path = output / "labels.jsonl"
            label_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
                encoding="utf-8",
            )
            observer_model = Path(str(spec.get("model_path", ""))).name or None
            generator_model = spec.get("generator_model")
            extra = {
                "dataset": "RAGTruth",
                "split": split,
                "labels_sha256": sha256(label_path),
                "source_formal_manifest_sha256": sha256(formal_root / split / "manifest.json"),
                "source_attention_cache_fingerprint": manifest["attention_cache_fingerprint"],
            }
            if observer_model:
                extra["observer_model"] = observer_model
            if generator_model is not None:
                extra["generator_model"] = str(generator_model)
            write_split_index(
                output,
                rows,
                attention_floor=float(spec["attention_floor"]),
                num_layers=int(spec["num_hidden_layers"]),
                num_heads=int(spec["num_attention_heads"]),
                alignment="post_token_query_at_same_position",
                extra=extra,
            )
            summary["splits"][split] = verify_split(output)

        summary["count"] = sum(summary["splits"].values())
        return summary


class AttentionArchiveVerifier:
    def __init__(self, archive_root):
        self.root = Path(archive_root)

    def run(self):
        roots = (
            [self.root]
            if (self.root / "manifest.json").is_file()
            else [
                self.root / split
                for split in ("train", "test")
                if (self.root / split / "manifest.json").is_file()
            ]
        )
        if not roots:
            raise ValueError("archive_root has no canonical split")
        splits = {root.name: verify_split(root) for root in roots}
        return {
            "archive_root": str(self.root),
            "splits": splits,
            "count": sum(splits.values()),
        }
