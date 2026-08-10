"""Convert formal attention caches and legacy feature traces."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import torch
from tqdm import tqdm

from cache import AttentionDataset, AttentionSample, index_row, save_attention_sample, sha256, verify_split, write_split_index
from features import save_hidden_features, save_token_stats, teacher_forced_stats


FORMAL_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"


@dataclass(frozen=True)
class ArchiveConfig:
    formal_root: str | Path
    output_root: str | Path


@dataclass(frozen=True)
class TraceArchiveConfig:
    trace_dir: str | Path
    output_dir: str | Path


def _positive_runs(y_token: torch.Tensor, response_idx: int) -> list[list[int]]:
    positions = torch.nonzero(y_token[response_idx:] > 0, as_tuple=False).flatten().tolist()
    runs: list[list[int]] = []
    for position in positions:
        if not runs or position != runs[-1][1]:
            runs.append([position, position + 1])
        else:
            runs[-1][1] += 1
    return runs


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_formal(path: Path, expected_hash: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"formal cache SHA256 does not match {path.name}")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("formal cache must contain a dictionary")
    return payload


def _formal_manifest(directory: Path, split: str) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    required = {"state", "cache_file_names", "matched_samples", "cache_files", "cache_files_sha256",
                "attention_cache_spec", "attention_cache_fingerprint"}
    if required.difference(manifest) or manifest["state"] != "complete":
        raise ValueError("formal manifest is incomplete")
    names = manifest["cache_file_names"]
    if not isinstance(names, list) or manifest["matched_samples"] != len(names) or manifest["cache_files"] != len(names):
        raise ValueError("formal manifest file counts do not match")
    spec = manifest["attention_cache_spec"]
    if (spec.get("attention_cache_schema") != FORMAL_SCHEMA or spec.get("split") != split
            or spec.get("cache_dtype") != "torch.float16" or float(spec.get("attention_floor", -1)) < 0
            or int(spec.get("num_hidden_layers", 0)) < 1 or int(spec.get("num_attention_heads", 0)) < 1):
        raise ValueError("formal manifest attention_cache_spec is invalid")
    if manifest["attention_cache_fingerprint"] != _fingerprint(spec):
        raise ValueError("formal manifest attention fingerprint does not match spec")
    hashes = manifest["cache_files_sha256"]
    if set(hashes) != set(names):
        raise ValueError("formal manifest hash inventory does not match filenames")
    paths = [directory / name for name in names]
    if any(path.parent != directory or not path.is_file() for path in paths):
        raise ValueError("formal manifest references an invalid cache file")
    manifest["_sha256"] = sha256(directory / "manifest.json")
    return manifest, paths


def _formal_sample(payload: dict[str, Any], split: str, spec: dict[str, Any]) -> tuple[AttentionSample, torch.Tensor]:
    required = {"attention_cache_schema", "response_id", "source_id", "split", "cache_dtype",
                "num_attention_layers", "num_attention_heads", "quality", "was_truncated", "response_idx",
                "token_ids", "attention_diagonal", "response_row_ptr", "response_column_indices",
                "response_values", "attention_floor", "y_token", "attention_cache_fingerprint"}
    if required.difference(payload):
        raise ValueError("formal cache is missing required fields")
    if (payload["attention_cache_schema"] != FORMAL_SCHEMA or payload["split"] != split
            or str(payload["cache_dtype"]) != spec["cache_dtype"] or payload["quality"] != "good" or payload["was_truncated"]
            or payload["attention_cache_fingerprint"] != _fingerprint(spec)):
        raise ValueError("formal cache metadata is invalid")
    sample = AttentionSample(
        str(payload["response_id"]), str(payload["source_id"]), int(payload["response_idx"]),
        torch.as_tensor(payload["token_ids"]), torch.as_tensor(payload["attention_diagonal"]),
        torch.as_tensor(payload["response_row_ptr"]), torch.as_tensor(payload["response_column_indices"]),
        torch.as_tensor(payload["response_values"]), float(payload["attention_floor"]),
    )
    sample.validate()
    if (sample.num_layers != payload["num_attention_layers"] or sample.num_heads != payload["num_attention_heads"]
            or sample.num_layers != spec["num_hidden_layers"] or sample.num_heads != spec["num_attention_heads"]
            or sample.attention_floor != float(spec["attention_floor"])):
        raise ValueError("formal cache geometry does not match manifest")
    labels = torch.as_tensor(payload["y_token"])
    if (labels.ndim != 1 or labels.numel() != sample.num_tokens or not bool(((labels == 0) | (labels == 1)).all())
            or not bool((labels[:sample.response_idx] == 0).all())):
        raise ValueError("formal y_token must be binary and token-aligned")
    return sample, labels


class AttentionArchiveConverter:
    def __init__(self, config: ArchiveConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        formal_root, output_root = Path(self.config.formal_root), Path(self.config.output_root)
        if output_root.exists():
            raise FileExistsError("output_root must not already exist")
        source = {split: _formal_manifest(formal_root / split, split) for split in ("train", "test")}
        train_spec, test_spec = source["train"][0]["attention_cache_spec"], source["test"][0]["attention_cache_spec"]
        if {key: value for key, value in train_spec.items() if key != "split"} != {
            key: value for key, value in test_spec.items() if key != "split"
        }:
            raise ValueError("train/test model or attention geometry/specification differs")
        staging = output_root.parent / f".{output_root.name}.staging-{uuid4().hex}"
        try:
            counts = {split: self._write_split(staging / split, split, *source[split]) for split in ("train", "test")}
            AttentionArchiveVerifier(staging).run()
            staging.replace(output_root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return {"count": sum(counts.values()), "splits": counts, "output_root": str(output_root)}

    @staticmethod
    def _write_split(output: Path, split: str, manifest: dict[str, Any], paths: list[Path]) -> int:
        output.mkdir(parents=True)
        spec, rows, labels = manifest["attention_cache_spec"], [], []
        for path in tqdm(paths, desc=f"convert {split}"):
            raw = _load_formal(path, manifest["cache_files_sha256"][path.name])
            sample, y_token = _formal_sample(raw, split, spec)
            target = output / "attention" / f"{sample.sample_id}.npz"
            save_attention_sample(sample, target)
            rows.append(index_row(output, sample, target))
            labels.append({"sample_id": sample.sample_id, "positive_runs": _positive_runs(y_token, sample.response_idx)})
        labels_path = output / "labels.jsonl"
        labels_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in labels), encoding="utf-8")
        write_split_index(output, rows, attention_floor=float(spec["attention_floor"]),
                          num_layers=int(spec["num_hidden_layers"]), num_heads=int(spec["num_attention_heads"]),
                          alignment="post_token_query_at_same_position", extra={
                              "labels_sha256": sha256(labels_path), "source_manifest_sha256": manifest["_sha256"],
                              "source_attention_fingerprint": manifest["attention_cache_fingerprint"],
                              "observer_model": Path(spec["model_path"]).name, "generator_model": spec["generator_model"],
                          })
        return len(rows)


class AttentionArchiveStore(AttentionDataset):
    def __init__(self, root: str | Path, split: str, device: str | torch.device = "cpu") -> None:
        super().__init__(Path(root) / split, device=device)


class AttentionArchiveVerifier:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run(self) -> dict[str, Any]:
        if (self.root / "manifest.json").is_file():
            return {"count": verify_split(self.root)}
        counts = {split: verify_split(self.root / split) for split in ("train", "test")}
        return {"count": sum(counts.values()), "splits": counts}


def _first(raw: dict[str, Any], *names):
    return next((raw[name] for name in names if name in raw), None)


def convert_trace_dir(trace_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    trace_dir, output_dir = Path(trace_dir), Path(output_dir)
    paths = sorted(trace_dir.glob("*.pt"))
    if not paths:
        paths = sorted(trace_dir.rglob("*.pt"))
    if not paths:
        raise ValueError(f"no trace .pt files found in {trace_dir}")
    rows = []
    for path in tqdm(paths, desc="convert feature traces"):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sample_id = str(_first(raw, "example_id", "response_id", "sample_id") or path.stem)
        token_ids = _first(raw, "input_ids", "token_ids")
        if token_ids is None:
            continue
        token_ids = torch.as_tensor(token_ids).flatten()
        hidden, log_prob, entropy = _first(raw, "hidden_states"), _first(raw, "token_log_prob"), _first(raw, "next_token_entropy", "entropy")
        has_hidden = hidden is not None
        if has_hidden:
            hidden = torch.as_tensor(hidden)
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            layer_ids = _first(raw, "selected_hidden_layers", "hidden_layer_ids")
            save_hidden_features(output_dir / "hidden" / f"{sample_id}.npz", token_ids,
                                 torch.as_tensor(layer_ids if layer_ids is not None else torch.arange(hidden.shape[0])).flatten().tolist(), hidden)
        if log_prob is None or entropy is None:
            logits = _first(raw, "logits")
            if logits is not None:
                log_prob, entropy = teacher_forced_stats(logits, token_ids)
        has_stats = log_prob is not None and entropy is not None
        if has_stats:
            save_token_stats(output_dir / "token_stats" / f"{sample_id}.npz", token_ids, log_prob, entropy)
        if has_hidden or has_stats:
            rows.append({"sample_id": sample_id, "hidden": has_hidden, "token_stats": has_stats})
    (output_dir / "feature_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"count": len(rows), "output_dir": str(output_dir)}


class TraceArchiveConverter:
    def __init__(self, config: TraceArchiveConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        return convert_trace_dir(self.config.trace_dir, self.config.output_dir)


class ArtifactInspector:
    def __init__(self, formal_root: str | Path) -> None:
        self.formal_root = Path(formal_root)

    def run(self) -> dict[str, Any]:
        report = {"root": str(self.formal_root), "splits": {}}
        for split in ("train", "test"):
            manifest, paths = _formal_manifest(self.formal_root / split, split)
            raw = _load_formal(paths[0], manifest["cache_files_sha256"][paths[0].name])
            spec = manifest["attention_cache_spec"]
            spec_summary = {
                "attention_cache_schema": spec["attention_cache_schema"],
                "cache_dtype": spec["cache_dtype"],
                "attention_floor": spec["attention_floor"],
                "num_hidden_layers": spec["num_hidden_layers"],
                "num_attention_heads": spec["num_attention_heads"],
                "observer_model": Path(spec["model_path"]).name,
                "generator_model": spec["generator_model"],
                "split": spec["split"],
            }
            if "task_type" in spec:
                spec_summary["task_type"] = spec["task_type"]
            report["splits"][split] = {
                "state": manifest["state"], "count": manifest["matched_samples"],
                "attention_cache_spec": spec_summary,
                "sample": {"fields": sorted(raw), "tensors": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
                    for name, value in raw.items() if isinstance(value, torch.Tensor)
                }},
            }
        return report
