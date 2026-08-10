"""Convert verified formal attention caches into a canonical, label-isolated archive."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm

from cache import AttentionSample, LEGACY_SCHEMA


FORMAL_SCHEMA = LEGACY_SCHEMA
CANONICAL_SCHEMA = "ragtruth-attention-archive-v1"
SPLITS = ("train", "test")
NPZ_FIELDS = {
    "token_ids", "response_idx", "attention_diagonal", "response_row_ptr",
    "response_column_indices", "response_values",
}


@dataclass(frozen=True)
class ArchiveConfig:
    formal_root: str | Path
    output_root: str | Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=path.parent) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_weights(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Hash and load one formal sample through one opened file handle."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError(f"formal cache SHA256 does not match {path.name}")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a tensor dictionary")
    return payload


def _dtype_name(value: Any) -> str:
    return str(value) if isinstance(value, torch.dtype) else str(value)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_id(payload: dict[str, Any]) -> str:
    value = str(payload["response_id"])
    if not value or Path(value).name != value:
        raise ValueError("sample_id must be a simple filename")
    return value


def _require_float16(payload: dict[str, Any]) -> None:
    if _dtype_name(payload.get("cache_dtype")) != "torch.float16":
        raise ValueError("formal payload cache_dtype must be torch.float16")
    for name in ("attention_diagonal", "response_values"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
            raise ValueError(f"{name} must be float16 in formal production cache")


def _formal_sample(payload: dict[str, Any]) -> tuple[AttentionSample, torch.Tensor]:
    if payload.get("attention_cache_schema") != FORMAL_SCHEMA:
        raise ValueError("unsupported formal attention cache schema")
    fields = {
        "source_id", "response_idx", "token_ids", "attention_diagonal",
        "response_row_ptr", "response_column_indices", "response_values",
        "attention_floor", "y_token", "split", "attention_cache_fingerprint", "cache_dtype",
        "input_policy", "was_truncated", "num_attention_layers", "num_attention_heads",
        "task_type", "generator_model", "quality",
    }
    missing = fields.difference(payload)
    if missing:
        raise ValueError(f"formal cache is missing fields: {sorted(missing)}")
    _require_float16(payload)
    if payload["quality"] != "good":
        raise ValueError("formal payload quality must be good")
    sample = AttentionSample(
        sample_id=_sample_id(payload), source_id=str(payload["source_id"]),
        response_idx=int(payload["response_idx"]), token_ids=payload["token_ids"],
        attention_diagonal=payload["attention_diagonal"],
        response_row_ptr=payload["response_row_ptr"],
        response_column_indices=payload["response_column_indices"],
        response_values=payload["response_values"], attention_floor=float(payload["attention_floor"]),
    )
    sample.validate()
    if sample.num_layers != int(payload["num_attention_layers"]) or sample.num_heads != int(payload["num_attention_heads"]):
        raise ValueError("formal payload attention layer/head counts do not match attention_diagonal")
    labels = payload["y_token"]
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != sample.num_tokens:
        raise ValueError("y_token must be a token-length vector")
    if not bool(((labels == 0) | (labels == 1)).all()) or not bool((labels[:sample.response_idx] == 0).all()):
        raise ValueError("y_token must be binary with zero prompt values")
    return sample, labels.to(dtype=torch.int64)


def _int32(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.numel():
        lower, upper = int(tensor.min()), int(tensor.max())
        info = np.iinfo(np.int32)
        if lower < info.min or upper > info.max:
            raise ValueError(f"{name} cannot be downcast to int32")
    return tensor.detach().cpu().to(dtype=torch.int32).numpy()


def _arrays(sample: AttentionSample) -> dict[str, np.ndarray]:
    return {
        "token_ids": _int32(sample.token_ids, "token_ids"),
        "response_idx": np.asarray(sample.response_idx, dtype=np.int32),
        "attention_diagonal": sample.attention_diagonal.detach().cpu().numpy(),
        "response_row_ptr": _int32(sample.response_row_ptr, "response_row_ptr"),
        "response_column_indices": _int32(sample.response_column_indices, "response_column_indices"),
        "response_values": sample.response_values.detach().cpu().numpy(),
    }


def _positive_runs(labels: torch.Tensor, response_idx: int) -> list[list[int]]:
    positions = torch.nonzero(labels[response_idx:] == 1, as_tuple=False).flatten().tolist()
    runs: list[list[int]] = []
    for position in positions:
        if not runs or position != runs[-1][1]:
            runs.append([position, position + 1])
        else:
            runs[-1][1] += 1
    return runs


class ArtifactInspector:
    """Report manifest-declared formal-artifact facts without scanning every payload."""

    def __init__(self, formal_root: str | Path) -> None:
        self.formal_root = Path(formal_root)

    def run(self) -> dict[str, Any]:
        source = AttentionArchiveConverter._source_inventory(self.formal_root)
        report: dict[str, Any] = {
            "root": str(self.formal_root),
            "payload_hashes_verified": False,
            "splits": {},
        }
        for split in SPLITS:
            item = source[split]
            sample_report: dict[str, Any] | None = None
            if item["paths"]:
                payload = _load_weights(item["paths"][0])
                sample_report = {
                    "path": item["paths"][0].relative_to(self.formal_root).as_posix(),
                    "fields": sorted(payload),
                    "tensors": {
                        name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
                        for name, value in payload.items() if isinstance(value, torch.Tensor)
                    },
                }
            report["splits"][split] = {
                "manifest_fields": sorted(item["manifest"]), "state": item["manifest"]["state"],
                "declared_count": item["manifest"]["matched_samples"], "sample_count": len(item["paths"]),
                "sample": sample_report,
            }
        return report


class AttentionArchiveConverter:
    def __init__(self, config: ArchiveConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        formal_root, output_root = Path(self.config.formal_root), Path(self.config.output_root)
        source = self._source_inventory(formal_root)
        if output_root.exists():
            raise FileExistsError("output_root must not already exist")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging = output_root.parent / f".{output_root.name}.staging-{uuid4().hex}"
        try:
            summary = self._write_archive(formal_root, staging, source)
            AttentionArchiveVerifier(staging).run()
            staging.replace(output_root)
            summary["output_root"] = str(output_root)
            return summary
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _write_archive(self, formal_root: Path, root: Path, source: dict[str, dict[str, Any]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        labels: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        source_bytes = sum(item["source_bytes"] for item in source.values())
        for split in SPLITS:
            for path in tqdm(source[split]["paths"], desc=f"archive attention {split}"):
                payload = _load_weights(path, source[split]["hashes"][path.name])
                sample, y_token = _formal_sample(payload)
                self._validate_payload_spec(payload, source[split]["spec"], split)
                relative = Path("attention") / split / f"{sample.sample_id}.npz"
                target = root / relative
                _atomic_npz(target, _arrays(sample))
                rows.append({
                    "sample_id": sample.sample_id, "source_id": sample.source_id, "split": split,
                    "attention_path": relative.as_posix(), "N": sample.num_tokens,
                    "R": sample.num_response_tokens, "response_idx": sample.response_idx,
                    "nnz": int(sample.response_values.numel()), "sha256": _sha256(target),
                    "bytes": target.stat().st_size, "task_type": str(payload["task_type"]),
                })
                labels[split].append({"sample_id": sample.sample_id, "positive_runs": _positive_runs(y_token, sample.response_idx)})
            ids = [row["sample_id"] for row in rows if row["split"] == split]
            if len(ids) != source[split]["matched_samples"]:
                raise ValueError(f"{split} matched_samples do not match formal samples")
            _atomic_text(root / "labels" / f"{split}.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels[split]))
        _atomic_text(root / "index.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        train_spec, test_spec = source["train"]["spec"], source["test"]["spec"]
        replay = {key: value for key, value in train_spec.items() if key != "split"}
        if replay != {key: value for key, value in test_spec.items() if key != "split"}:
            raise ValueError("train/test attention_cache_spec differ beyond split")
        payload_bytes = (
            sum(row["bytes"] for row in rows)
            + (root / "index.jsonl").stat().st_size
            + sum((root / "labels" / f"{split}.jsonl").stat().st_size for split in SPLITS)
        )
        manifest = {
            "schema": CANONICAL_SCHEMA, "source_formal_root": str(formal_root),
            "source_replay_spec": replay, "cache_dtype": "torch.float16",
            "attention_floor": float(replay["attention_floor"]), "num_layers": int(replay["num_hidden_layers"]),
            "num_heads": int(replay["num_attention_heads"]), "channel_order": "layer,head", "state": "complete",
            "splits": {
                split: {"fingerprint": source[split]["manifest"]["attention_cache_fingerprint"], "manifest_sha256": source[split]["manifest_sha256"], "count": len(source[split]["paths"]), "state": "complete"}
                for split in SPLITS
            },
            "modalities": {"attention": "present", "hidden_states": "absent", "full_logits": "absent", "token_logprob": "absent", "lm_entropy": "absent"},
            "label_coordinate": "response_relative_[start,end)", "index_sha256": _sha256(root / "index.jsonl"),
            "label_sha256": {split: _sha256(root / "labels" / f"{split}.jsonl") for split in SPLITS},
            "source_bytes": source_bytes, "payload_bytes": payload_bytes, "size_ratio": payload_bytes / source_bytes,
            "manifest_bytes": 0,
        }
        while True:
            rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            rendered_size = len(rendered.encode("utf-8"))
            if manifest["manifest_bytes"] == rendered_size:
                break
            manifest["manifest_bytes"] = rendered_size
        _atomic_text(root / "manifest.json", rendered)
        return {"count": len(rows), "source_bytes": source_bytes, "payload_bytes": payload_bytes,
                "manifest_bytes": manifest["manifest_bytes"], "size_ratio": payload_bytes / source_bytes}

    @staticmethod
    def _validate_payload_spec(payload: dict[str, Any], spec: dict[str, Any], split: str) -> None:
        for key in ("split", "input_policy", "generator_model"):
            if payload[key] != spec[key]:
                raise ValueError(f"formal payload {key} does not match {split} attention_cache_spec")
        if payload["attention_cache_fingerprint"] != _fingerprint(spec):
            raise ValueError("formal payload attention_cache_fingerprint does not match attention_cache_spec")
        if bool(payload["was_truncated"]):
            raise ValueError("formal payload was_truncated must be false")
        if _dtype_name(payload["cache_dtype"]) != spec["cache_dtype"]:
            raise ValueError("formal payload cache_dtype does not match attention_cache_spec")
        if (int(payload["num_attention_layers"]) != int(spec["num_hidden_layers"])
                or int(payload["num_attention_heads"]) != int(spec["num_attention_heads"])
                or float(payload["attention_floor"]) != float(spec["attention_floor"])):
            raise ValueError("formal payload attention geometry does not match attention_cache_spec")

    @staticmethod
    def _source_inventory(formal_root: Path) -> dict[str, dict[str, Any]]:
        if not formal_root.is_dir():
            raise ValueError("formal_root must be an existing directory")
        inventory: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            directory, manifest_path = formal_root / split, formal_root / split / "manifest.json"
            if not directory.is_dir() or not manifest_path.is_file():
                raise ValueError(f"formal_root must contain {split}/manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = sorted(directory.glob("*.pt"))
            names = [path.name for path in paths]
            if manifest.get("state") != "complete":
                raise ValueError(f"{split} manifest state must be complete")
            required_manifest = {
                "state", "attention_cache_dir", "attention_cache_fingerprint", "attention_cache_spec",
                "matched_samples", "cache_files", "cache_file_names", "cache_files_sha256", "saved", "reused",
            }
            if set(manifest) != required_manifest:
                raise ValueError(f"{split} manifest fields are invalid")
            if not isinstance(manifest["attention_cache_dir"], str):
                raise ValueError(f"{split} manifest attention_cache_dir must be a string")
            if not paths:
                raise ValueError(f"{split} must contain top-level .pt files")
            for key in ("matched_samples", "cache_files", "saved", "reused"):
                if isinstance(manifest[key], bool) or not isinstance(manifest[key], int):
                    raise ValueError(f"{split} manifest {key} must be an integer")
            if manifest["matched_samples"] != len(paths) or manifest["cache_files"] != len(paths):
                raise ValueError(f"{split} manifest counts do not match top-level .pt files")
            if manifest["saved"] + manifest["reused"] != manifest["matched_samples"]:
                raise ValueError(f"{split} manifest saved and reused must equal matched_samples")
            if not isinstance(manifest["cache_file_names"], list) or manifest["cache_file_names"] != names:
                raise ValueError(f"{split} cache_file_names do not match top-level .pt files")
            hashes = manifest.get("cache_files_sha256")
            if not isinstance(hashes, dict) or set(hashes) != set(names):
                raise ValueError(f"{split} cache_files_sha256 keys do not match top-level .pt files")
            for path in paths:
                value = hashes[path.name]
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"{split} cache_files_sha256 format is invalid")
            spec = manifest.get("attention_cache_spec")
            if not isinstance(spec, dict) or spec.get("split") != split:
                raise ValueError(f"{split} attention_cache_spec is invalid")
            required_spec = {
                "attention_cache_schema", "cache_dtype", "attention_floor", "input_policy", "quality_policy",
                "label_policy", "system_prompt", "tokenization_policy", "truncation", "all_layers", "all_heads",
                "dataset_dir", "dataset_files_sha256", "model_path", "model_files_sha256", "model_class",
                "tokenizer_class", "transformers_version", "tokenizers_version", "num_hidden_layers",
                "num_attention_heads", "max_position_embeddings", "split", "generator_model", "task_type",
                "dtype", "attn_implementation", "torch_version",
            }
            if set(spec) != required_spec or spec["cache_dtype"] != "torch.float16":
                raise ValueError(f"{split} attention_cache_spec cache_dtype must be torch.float16")
            if spec["attention_cache_schema"] != FORMAL_SCHEMA:
                raise ValueError(f"{split} attention_cache_spec schema is invalid")
            if spec["truncation"] is not False:
                raise ValueError(f"{split} attention_cache_spec truncation must be false")
            if spec["all_layers"] is not True or spec["all_heads"] is not True:
                raise ValueError(f"{split} attention_cache_spec all_layers/all_heads must be true")
            if spec["input_policy"] != "full_context_no_truncation":
                raise ValueError(f"{split} attention_cache_spec input_policy is invalid")
            if spec["quality_policy"] != "official_good_only":
                raise ValueError(f"{split} attention_cache_spec quality_policy is invalid")
            if spec["task_type"] != "all":
                raise ValueError(f"{split} attention_cache_spec task_type must be all")
            if manifest["attention_cache_fingerprint"] != _fingerprint(spec):
                raise ValueError(f"{split} attention_cache_fingerprint does not match attention_cache_spec")
            inventory[split] = {
                "paths": paths,
                "manifest": manifest,
                "manifest_sha256": _sha256(manifest_path),
                "source_bytes": sum(path.stat().st_size for path in paths),
                "spec": spec,
                "hashes": hashes,
                "matched_samples": manifest["matched_samples"],
            }
        return inventory


class AttentionArchiveStore:
    """Load exactly one canonical NPZ sample at a time from fixed archive paths."""

    def __init__(self, root: str | Path, split: str, device: str | torch.device = "cpu") -> None:
        self.root, self.split, self.device = Path(root), split, device
        if split not in SPLITS:
            raise ValueError("split must be train or test")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CANONICAL_SCHEMA or self.manifest.get("state") != "complete":
            raise ValueError("cache_dir is not a canonical attention archive")
        if self.manifest.get("index_sha256") != _sha256(self.root / "index.jsonl"):
            raise ValueError("index_sha256 does not match index.jsonl")
        self.attention_floor = float(self.manifest["attention_floor"])
        self.rows = self._rows()

    def _rows(self) -> list[dict[str, Any]]:
        rows = [json.loads(line) for line in (self.root / "index.jsonl").read_text(encoding="utf-8").splitlines()]
        result, seen = [], set()
        required_fields = {"sample_id", "source_id", "split", "attention_path", "N", "R", "response_idx", "nnz", "sha256", "bytes", "task_type"}
        for row in rows:
            if row.get("split") != self.split:
                continue
            if set(row) != required_fields:
                raise ValueError("index fields are invalid")
            sample_id = str(row.get("sample_id", ""))
            expected = f"attention/{self.split}/{sample_id}.npz"
            if not sample_id or Path(sample_id).name != sample_id or row.get("attention_path") != expected or sample_id in seen:
                raise ValueError("index attention_path is invalid")
            seen.add(sample_id)
            result.append(row)
        return result

    def __iter__(self) -> Iterator[AttentionSample]:
        for row in self.rows:
            yield self._load(row)

    def _load(self, row: dict[str, Any]) -> AttentionSample:
        sample_id = str(row["sample_id"])
        path = (self.root / "attention" / self.split / f"{sample_id}.npz").resolve()
        base = (self.root / "attention" / self.split).resolve()
        if path.parent != base or not path.is_file():
            raise ValueError("canonical attention_path escapes archive")
        if row["bytes"] != path.stat().st_size:
            raise ValueError("canonical attention byte count does not match index")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("canonical attention SHA256 does not match index")
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as arrays:
                if set(arrays.files) != NPZ_FIELDS:
                    raise ValueError("canonical attention NPZ fields are invalid")
                expected = {
                    "token_ids": np.dtype("int32"),
                    "response_idx": np.dtype("int32"),
                    "attention_diagonal": np.dtype("float16"),
                    "response_row_ptr": np.dtype("int32"),
                    "response_column_indices": np.dtype("int32"),
                    "response_values": np.dtype("float16"),
                }
                if any(arrays[name].dtype != dtype for name, dtype in expected.items()) or arrays["response_idx"].ndim != 0:
                    raise ValueError("canonical attention NPZ dtypes are invalid")
                sample = AttentionSample(
                    sample_id=sample_id,
                    source_id=str(row["source_id"]),
                    response_idx=int(arrays["response_idx"]),
                    token_ids=torch.from_numpy(arrays["token_ids"].astype(np.int64, copy=False)),
                    attention_diagonal=torch.from_numpy(arrays["attention_diagonal"]),
                    response_row_ptr=torch.from_numpy(arrays["response_row_ptr"].astype(np.int64, copy=False)),
                    response_column_indices=torch.from_numpy(arrays["response_column_indices"]),
                    response_values=torch.from_numpy(arrays["response_values"]),
                    attention_floor=self.attention_floor,
                )
        sample.validate()
        index_shape = (row["N"], row["R"], row["response_idx"], row["nnz"])
        artifact_shape = (
            sample.num_tokens,
            sample.num_response_tokens,
            sample.response_idx,
            sample.response_values.numel(),
        )
        if index_shape != artifact_shape:
            raise ValueError("canonical attention dimensions do not match index")
        return AttentionSample(sample_id=sample.sample_id, source_id=sample.source_id, response_idx=sample.response_idx,
            token_ids=sample.token_ids.to(self.device), attention_diagonal=sample.attention_diagonal.to(self.device),
            response_row_ptr=sample.response_row_ptr.to(self.device), response_column_indices=sample.response_column_indices.to(self.device),
            response_values=sample.response_values.to(self.device), attention_floor=sample.attention_floor)


class AttentionArchiveVerifier:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run(self) -> dict[str, Any]:
        manifest_path, index_path = self.root / "manifest.json", self.root / "index.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != CANONICAL_SCHEMA or manifest.get("state") != "complete"
                or manifest.get("cache_dtype") != "torch.float16"):
            raise ValueError("unsupported canonical archive schema or cache_dtype")
        replay = manifest.get("source_replay_spec")
        if not isinstance(replay, dict) or "split" in replay or replay.get("cache_dtype") != "torch.float16":
            raise ValueError("source_replay_spec is invalid")
        if (manifest.get("attention_floor") != replay.get("attention_floor")
                or manifest.get("num_layers") != replay.get("num_hidden_layers")
                or manifest.get("num_heads") != replay.get("num_attention_heads")):
            raise ValueError("canonical manifest attention metadata is invalid")
        if manifest.get("modalities") != {"attention": "present", "hidden_states": "absent", "full_logits": "absent", "token_logprob": "absent", "lm_entropy": "absent"}:
            raise ValueError("canonical modalities are invalid")
        if manifest.get("index_sha256") != _sha256(index_path):
            raise ValueError("index_sha256 does not match index.jsonl")
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        expected_files, counts, seen = {"manifest.json", "index.jsonl"}, {split: 0 for split in SPLITS}, set()
        stores = {split: AttentionArchiveStore(self.root, split) for split in SPLITS}
        for row in rows:
            split, sample_id = row.get("split"), str(row.get("sample_id", ""))
            if split not in SPLITS or (split, sample_id) in seen:
                raise ValueError("index sample set is invalid")
            required_row = {"sample_id", "source_id", "split", "attention_path", "N", "R", "response_idx", "nnz", "sha256", "bytes", "task_type"}
            if set(row) != required_row:
                raise ValueError("index fields are invalid")
            expected = f"attention/{split}/{sample_id}.npz"
            if row.get("attention_path") != expected:
                raise ValueError("index attention_path is invalid")
            path = self.root / expected
            if not path.is_file():
                raise ValueError("attention artifact does not exist")
            sample = stores[split]._load(row)
            if (row.get("N"), row.get("R"), row.get("response_idx"), row.get("nnz")) != (sample.num_tokens, sample.num_response_tokens, sample.response_idx, sample.response_values.numel()):
                raise ValueError("index counts do not match attention artifact")
            if sample.num_layers != manifest["num_layers"] or sample.num_heads != manifest["num_heads"]:
                raise ValueError("manifest attention dimensions do not match artifact")
            expected_files.add(expected); seen.add((split, sample_id)); counts[split] += 1
        for split in SPLITS:
            label = self.root / "labels" / f"{split}.jsonl"
            expected_files.add(label.relative_to(self.root).as_posix())
            if manifest.get("label_sha256", {}).get(split) != _sha256(label):
                raise ValueError("label_sha256 does not match label sidecar")
            label_rows = [json.loads(line) for line in label.read_text(encoding="utf-8").splitlines()]
            sample_rows = [row for row in rows if row["split"] == split]
            if (len(label_rows) != len({row.get("sample_id") for row in label_rows})
                    or {row.get("sample_id") for row in label_rows} != {row["sample_id"] for row in sample_rows}):
                raise ValueError("label sample set does not match index")
            response_lengths = {row["sample_id"]: row["R"] for row in sample_rows}
            for label_row in label_rows:
                if set(label_row) != {"sample_id", "positive_runs"} or not isinstance(label_row["positive_runs"], list):
                    raise ValueError("label sidecar fields are invalid")
                previous_end = 0
                for run in label_row["positive_runs"]:
                    if (not isinstance(run, list) or len(run) != 2 or not all(isinstance(value, int) for value in run)
                            or not 0 <= run[0] < run[1] <= response_lengths[label_row["sample_id"]] or run[0] < previous_end):
                        raise ValueError("label runs are invalid")
                    previous_end = run[1]
            declared = manifest.get("splits", {}).get(split, {})
            if (set(declared) != {"fingerprint", "manifest_sha256", "count", "state"}
                    or declared.get("state") != "complete" or declared.get("count") != counts[split]):
                raise ValueError("manifest split count or state does not match index")
            if declared["fingerprint"] != _fingerprint({**replay, "split": split}):
                raise ValueError("manifest split fingerprint does not match source_replay_spec")
            if not isinstance(declared["manifest_sha256"], str) or len(declared["manifest_sha256"]) != 64:
                raise ValueError("manifest split manifest_sha256 is invalid")
        actual = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        if actual != expected_files:
            raise ValueError("archive file set does not match manifest and index")
        actual_payload_bytes = (
            sum(row["bytes"] for row in rows) + index_path.stat().st_size
            + sum((self.root / "labels" / f"{split}.jsonl").stat().st_size for split in SPLITS)
        )
        if manifest.get("payload_bytes") != actual_payload_bytes:
            raise ValueError("payload_bytes does not match index")
        if manifest.get("manifest_bytes") != manifest_path.stat().st_size:
            raise ValueError("manifest_bytes does not match manifest.json")
        if manifest.get("size_ratio") != manifest["payload_bytes"] / manifest["source_bytes"]:
            raise ValueError("size_ratio is invalid")
        return {"count": len(rows), "splits": counts, "root": str(self.root)}
