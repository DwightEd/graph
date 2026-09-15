"""Reuse NPZ identity fields and existing inputs/records indexes; no labels."""

import hashlib
import json
from pathlib import Path

import numpy as np


ALIASES = {
    "id": ("id", "response_id", "sample_id"),
    "source_id": ("source_id",),
    "split": ("official_split", "dataset_split", "split"),
    "task": ("task", "task_type"),
    "generator": ("generator", "model"),
    "response_sha256": ("response_sha256",),
    "prompt_length": ("prompt_length", "response_idx"),
    "token_ids": ("token_ids",),
    "offsets": ("offsets",),
}


def identity_fields(data):
    """Read only named safe members, including scalar JSON from saved NPZs."""
    result = {}
    if "record_json" in data:
        result = identity_fields(json.loads(str(np.asarray(data["record_json"]).item())))
    for name, aliases in ALIASES.items():
        for alias in aliases:
            if alias in data:
                value = data[alias]
                value = value.tolist() if isinstance(value, np.ndarray) else value
                if value is not None and not (isinstance(value, str) and not value):
                    value = value if name in ("token_ids", "offsets", "prompt_length") else str(value)
                    if name in result and not np.array_equal(result[name], value):
                        raise ValueError("conflicting identity field: " + name)
                    result[name] = value
    if "response" in data:
        text = np.asarray(data["response"]).item()
        digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        if result.get("response_sha256", digest) != digest:
            raise ValueError("response text and response_sha256 disagree")
        result["response_sha256"] = digest
    return result


def merge_identity(native, extra):
    """Fill missing values; never silently override a different sample."""
    result = dict(native)
    for name, value in extra.items():
        if name in result and not np.array_equal(result[name], value):
            raise ValueError("NPZ/index mismatch: " + name)
        result[name] = value
    return result


def file_stamp(path):
    path = Path(path)
    return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]


class CacheIndex:
    """Read existing inputs.jsonl, records.jsonl or records.json.

    An explicit path may be a population directory or an existing index file.
    Without it, only the cache directory and its parent are searched. Matching
    uses saved IDs or exact paths, never a same-source/different-answer join.
    """

    def __init__(self, root, path=None):
        self.root = Path(root).resolve()
        locations = [Path(path)] if path else [self.root, self.root.parent]
        self.path = None
        for location in locations:
            candidates = [location] if location.is_file() else [location / name for name in
                           ("inputs.jsonl", "records.jsonl", "records.json")]
            self.path = next((p.resolve() for p in candidates if p.is_file()), None)
            if self.path is not None:
                break
        if path and self.path is None:
            raise FileNotFoundError("no existing inputs/records index at " + str(path))
        self.by_id, self.by_path, self.inputs = {}, {}, []
        self.annotations = None
        if self.path is None:
            return
        self.inputs.append(file_stamp(self.path))
        text = self.path.read_text(encoding="utf-8")
        rows = json.loads(text) if self.path.suffix == ".json" else [json.loads(s) for s in text.splitlines() if s.strip()]
        for raw in rows:
            row = identity_fields(raw)
            relative = raw.get("cache", raw.get("trace"))
            if relative:
                if relative in self.by_path:
                    raise ValueError("duplicate cache path in existing index: " + relative)
                self.by_path[relative] = row
            rid = row.get("id")
            if rid:
                if rid in self.by_id:
                    raise ValueError("duplicate response ID in existing index: " + rid)
                self.by_id[rid] = row
        settings = self.path.parent / "settings.json"
        if settings.is_file():
            self.inputs.append(file_stamp(settings))
            dataset = json.loads(settings.read_text(encoding="utf-8")).get("dataset")
            if dataset:
                # Match reuse_detector: dataset paths are interpreted from the
                # repository working directory, not a newly invented layout.
                self.annotations = str((Path(dataset).expanduser() / "response.jsonl").resolve())

    def resolve(self, path, native):
        path = Path(path).resolve()
        relative = path.relative_to(self.root).as_posix()
        row = self.by_path.get(relative)
        explicit = row is not None
        if row is None and path.name in self.by_path:
            matches = list(self.root.rglob(path.name))
            if len(matches) != 1:
                raise ValueError("ambiguous cache basename: " + path.name)
            row, explicit = self.by_path[path.name], True
        # attention_<id>.npz is the existing canonical naming convention.
        rid = native.get("id", path.stem.removeprefix("attention_"))
        if row is None:
            row = self.by_id.get(rid)
        if row is None:
            if self.path is not None and not native.get("id"):
                raise ValueError("cache not found in existing index: " + relative)
            return native, []
        row = dict(row)
        used = []
        # Existing reuse/structural indexes keep offsets in <id>.npz. Only
        # identity members are read: local_attention/values/labels stay unopened.
        if "offsets" not in row or "token_ids" not in row:
            key = row.get("id", rid)
            companion = self.path.parent / (key + ".npz")
            if Path(key).name == key and companion.is_file() and companion.resolve() != path:
                with np.load(companion, allow_pickle=False) as data:
                    row = merge_identity(row, identity_fields(data))
                used.append(file_stamp(companion))
        if "offsets" not in native and "offsets" in row and not explicit:
            # Equal lengths or a matching answer ID do not prove token alignment.
            if "token_ids" not in native or "token_ids" not in row or "prompt_length" not in row:
                raise ValueError("copying offsets by ID requires matching token_ids and prompt_length: " + rid)
        return merge_identity(native, row), used
