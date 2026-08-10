"""Canonical sparse attention data used by every graph builder."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


NPZ_FIELDS = (
    "token_ids",
    "response_idx",
    "attention_diagonal",
    "response_row_ptr",
    "response_column_indices",
    "response_values",
)


@dataclass
class AttentionSample:
    """In-memory view of one attention sample.

    Only the six tensors are stored per sample. ``sample_id``, ``source_id``
    and ``attention_floor`` come from the dataset index/manifest.
    """

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
        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be [N]")
        if self.attention_diagonal.ndim != 3 or self.attention_diagonal.shape[2] != self.num_tokens:
            raise ValueError("attention_diagonal must be [L,H,N]")
        if not 0 < self.response_idx < self.num_tokens:
            raise ValueError("response_idx must split prompt and response")
        expected_rows = self.num_channels * self.num_response_tokens
        if self.response_row_ptr.ndim != 1 or self.response_row_ptr.numel() != expected_rows + 1:
            raise ValueError("response_row_ptr has the wrong length")
        if self.response_column_indices.ndim != 1 or self.response_values.ndim != 1:
            raise ValueError("CSR columns/values must be vectors")
        if self.response_column_indices.numel() != self.response_values.numel():
            raise ValueError("CSR columns and values must align")
        if int(self.response_row_ptr[0]) != 0 or int(self.response_row_ptr[-1]) != self.response_values.numel():
            raise ValueError("response_row_ptr does not span response_values")


def save_attention_sample(sample: AttentionSample, path: str | Path) -> None:
    """Write exactly the six canonical per-sample arrays."""
    sample.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_ids=sample.token_ids.detach().cpu().to(torch.int32).numpy(),
        response_idx=np.asarray(sample.response_idx, dtype=np.int32),
        attention_diagonal=sample.attention_diagonal.detach().cpu().to(torch.float16).numpy(),
        response_row_ptr=sample.response_row_ptr.detach().cpu().to(torch.int32).numpy(),
        response_column_indices=sample.response_column_indices.detach().cpu().to(torch.int32).numpy(),
        response_values=sample.response_values.detach().cpu().to(torch.float16).numpy(),
    )


def load_attention_sample(
    path: str | Path,
    *,
    sample_id: str,
    source_id: str,
    attention_floor: float,
    device: str | torch.device = "cpu",
) -> AttentionSample:
    """Load one canonical NPZ sample."""
    with np.load(Path(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(NPZ_FIELDS):
            raise ValueError(f"attention sample must contain exactly {NPZ_FIELDS}")
        sample = AttentionSample(
            sample_id=sample_id,
            source_id=source_id,
            response_idx=int(arrays["response_idx"]),
            token_ids=torch.from_numpy(arrays["token_ids"].astype(np.int64, copy=False)),
            attention_diagonal=torch.from_numpy(arrays["attention_diagonal"]),
            response_row_ptr=torch.from_numpy(arrays["response_row_ptr"].astype(np.int64, copy=False)),
            response_column_indices=torch.from_numpy(arrays["response_column_indices"].astype(np.int32, copy=False)),
            response_values=torch.from_numpy(arrays["response_values"]),
            attention_floor=float(attention_floor),
        )
    sample.validate()
    for name in (
        "token_ids",
        "attention_diagonal",
        "response_row_ptr",
        "response_column_indices",
        "response_values",
    ):
        setattr(sample, name, getattr(sample, name).to(device))
    return sample


class AttentionDataset:
    """Iterate one canonical train or test split directory."""

    def __init__(self, root: str | Path, device: str | torch.device = "cpu") -> None:
        self.root = Path(root)
        self.device = device
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.attention_floor = float(self.manifest["attention_floor"])
        self.rows = [
            json.loads(line)
            for line in (self.root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            yield load_attention_sample(
                self.root / row["path"],
                sample_id=str(row["sample_id"]),
                source_id=str(row["source_id"]),
                attention_floor=self.attention_floor,
                device=self.device,
            )
