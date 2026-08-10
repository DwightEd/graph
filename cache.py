"""Canonical sparse attention split storage."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CANONICAL_SCHEMA = "ragtruth-attention-split-v1"
NPZ_FIELDS = (
    "token_ids", "response_idx", "attention_diagonal", "response_row_ptr",
    "response_column_indices", "response_values",
)
NPZ_DTYPES = {
    "token_ids": np.dtype("int32"), "response_idx": np.dtype("int32"),
    "attention_diagonal": np.dtype("float16"), "response_row_ptr": np.dtype("int32"),
    "response_column_indices": np.dtype("int32"), "response_values": np.dtype("float16"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int32(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.numel():
        lower, upper = int(tensor.min()), int(tensor.max())
        info = np.iinfo(np.int32)
        if lower < info.min or upper > info.max:
            raise ValueError(f"{name} cannot be represented as int32")
    return tensor.detach().cpu().to(torch.int32).numpy()


@dataclass
class AttentionSample:
    sample_id: str
    source_id: str
    response_idx: int
    token_ids: torch.Tensor
    attention_diagonal: torch.Tensor
    response_row_ptr: torch.Tensor
    response_column_indices: torch.Tensor
    response_values: torch.Tensor
    attention_floor: float

    @property
    def num_layers(self) -> int:
        return int(self.attention_diagonal.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.attention_diagonal.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.token_ids.numel())

    @property
    def num_response_tokens(self) -> int:
        return self.num_tokens - self.response_idx

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    def validate(self) -> None:
        integer = (torch.int32, torch.int64)
        if not 0 < self.attention_floor <= 1 or not np.isfinite(self.attention_floor):
            raise ValueError("attention_floor must be finite and in (0,1]")
        if self.token_ids.ndim != 1 or self.token_ids.dtype not in integer:
            raise ValueError("token_ids must be an integer [N] vector")
        if self.attention_diagonal.ndim != 3 or self.attention_diagonal.shape[2] != self.num_tokens:
            raise ValueError("attention_diagonal must be [L,H,N]")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("attention_diagonal must have at least one layer and head")
        if self.attention_diagonal.dtype != torch.float16:
            raise ValueError("attention_diagonal must be float16")
        if not 0 < self.response_idx < self.num_tokens:
            raise ValueError("response_idx must split prompt and response")
        if not (torch.isfinite(self.attention_diagonal).all() and ((self.attention_diagonal >= 0) & (self.attention_diagonal <= 1)).all()):
            raise ValueError("attention_diagonal must be finite probabilities")
        expected_rows = self.num_channels * self.num_response_tokens
        if self.response_row_ptr.ndim != 1 or self.response_row_ptr.numel() != expected_rows + 1:
            raise ValueError("response_row_ptr has the wrong length")
        if self.response_row_ptr.dtype not in integer:
            raise ValueError("response_row_ptr must be integer")
        if self.response_column_indices.ndim != 1 or self.response_values.ndim != 1:
            raise ValueError("CSR columns/values must be vectors")
        if self.response_column_indices.dtype not in integer or self.response_values.dtype != torch.float16:
            raise ValueError("CSR columns must be integer and values float16")
        if self.response_column_indices.numel() != self.response_values.numel():
            raise ValueError("CSR columns and values must align")
        if int(self.response_row_ptr[0]) != 0 or int(self.response_row_ptr[-1]) != self.response_values.numel():
            raise ValueError("response_row_ptr does not span response_values")
        if bool((self.response_row_ptr[1:] < self.response_row_ptr[:-1]).any()):
            raise ValueError("response_row_ptr must be monotone")
        if not (torch.isfinite(self.response_values).all() and ((self.response_values >= self.attention_floor) & (self.response_values <= 1)).all()):
            raise ValueError("response_values must be finite probabilities above attention_floor")
        lengths = self.response_row_ptr[1:] - self.response_row_ptr[:-1]
        rows = torch.repeat_interleave(torch.arange(expected_rows, device=lengths.device), lengths)
        targets = self.response_idx + rows.remainder(self.num_response_tokens)
        if self.response_column_indices.numel() and bool(((self.response_column_indices < 0) | (self.response_column_indices >= targets)).any()):
            raise ValueError("CSR columns must point to earlier tokens")
        same_row = rows[1:] == rows[:-1]
        if bool((same_row & (self.response_column_indices[1:] <= self.response_column_indices[:-1])).any()):
            raise ValueError("CSR columns must be increasing within a row")


def save_attention_sample(sample: AttentionSample, path: str | Path) -> None:
    sample.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_ids=_int32(sample.token_ids, "token_ids"),
        response_idx=np.asarray(sample.response_idx, dtype=np.int32),
        attention_diagonal=sample.attention_diagonal.detach().cpu().numpy(),
        response_row_ptr=_int32(sample.response_row_ptr, "response_row_ptr"),
        response_column_indices=_int32(sample.response_column_indices, "response_column_indices"),
        response_values=sample.response_values.detach().cpu().numpy(),
    )


def load_attention_sample(path: str | Path, *, sample_id: str, source_id: str,
                          attention_floor: float, device: str | torch.device = "cpu") -> AttentionSample:
    with np.load(Path(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(NPZ_FIELDS):
            raise ValueError(f"attention sample must contain exactly {NPZ_FIELDS}")
        if any(arrays[name].dtype != NPZ_DTYPES[name] for name in NPZ_FIELDS):
            raise ValueError("attention sample has non-canonical dtype")
        sample = AttentionSample(
            sample_id, source_id, int(arrays["response_idx"]),
            torch.from_numpy(arrays["token_ids"]), torch.from_numpy(arrays["attention_diagonal"]),
            torch.from_numpy(arrays["response_row_ptr"]), torch.from_numpy(arrays["response_column_indices"]),
            torch.from_numpy(arrays["response_values"]), float(attention_floor),
        )
    sample.validate()
    for name in ("token_ids", "attention_diagonal", "response_row_ptr", "response_column_indices", "response_values"):
        setattr(sample, name, getattr(sample, name).to(device))
    return sample


def index_row(root: Path, sample: AttentionSample, path: Path) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id, "source_id": sample.source_id,
        "path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size,
    }


def write_split_index(root: str | Path, rows: list[dict[str, Any]], *, attention_floor: float,
                      num_layers: int, num_heads: int, alignment: str,
                      extra: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    index = root / "index.jsonl"
    index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": CANONICAL_SCHEMA, "attention_floor": attention_floor,
        "num_layers": num_layers, "num_heads": num_heads, "count": len(rows),
        "index_sha256": sha256(index), "alignment": alignment,
    }
    if extra:
        manifest.update(extra)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class AttentionDataset:
    """Iterate one verified canonical split directory."""

    def __init__(self, root: str | Path, device: str | torch.device = "cpu", verify_hashes: bool = False) -> None:
        self.root, self.device = Path(root), device
        self.verify_hashes = verify_hashes
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        required = {"schema", "attention_floor", "num_layers", "num_heads", "count", "index_sha256", "alignment"}
        if self.manifest.get("schema") != CANONICAL_SCHEMA or required.difference(self.manifest):
            raise ValueError("invalid canonical split manifest")
        index = self.root / "index.jsonl"
        if sha256(index) != self.manifest["index_sha256"]:
            raise ValueError("index_sha256 does not match index.jsonl")
        self.rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(self.rows) != self.manifest["count"] or len({row["sample_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("canonical index count or sample IDs are invalid")
        if any(set(row) != {"sample_id", "source_id", "path", "sha256", "bytes"} for row in self.rows):
            raise ValueError("canonical index rows have the wrong fields")
        self.attention_floor = float(self.manifest["attention_floor"])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            path = self.root / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"]:
                raise ValueError("attention sample byte count does not match index")
            if self.verify_hashes and sha256(path) != row["sha256"]:
                raise ValueError("attention sample SHA256 does not match index")
            sample = load_attention_sample(path, sample_id=str(row["sample_id"]), source_id=str(row["source_id"]),
                                          attention_floor=self.attention_floor, device=self.device)
            if sample.num_layers != self.manifest["num_layers"] or sample.num_heads != self.manifest["num_heads"]:
                raise ValueError("attention geometry does not match manifest")
            yield sample


def verify_split(root: str | Path) -> int:
    """Verify one canonical split, including optional label sidecar integrity."""
    dataset = AttentionDataset(root, verify_hashes=True)
    labels_sha256 = dataset.manifest.get("labels_sha256")
    if labels_sha256 is not None:
        labels = dataset.root / "labels.jsonl"
        if not labels.is_file() or sha256(labels) != labels_sha256:
            raise ValueError("labels_sha256 does not match labels.jsonl")
    for _ in dataset:
        pass
    return len(dataset)
