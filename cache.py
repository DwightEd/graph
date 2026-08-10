"""Label-free sparse attention cache artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SCHEMA = "attention-response-csr-v1"
LEGACY_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"


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
        if self.token_ids.dtype != torch.int64 or self.token_ids.ndim != 1:
            raise ValueError("token_ids must be an int64 vector")
        if self.attention_diagonal.ndim != 3 or self.attention_diagonal.shape[2] != self.num_tokens:
            raise ValueError("attention_diagonal must have shape [layers, heads, tokens]")
        if not self.attention_diagonal.is_floating_point():
            raise ValueError("attention_diagonal must be floating point")
        if not 0 < self.response_idx < self.num_tokens:
            raise ValueError("response_idx must leave prompt and response tokens")
        if self.response_row_ptr.dtype != torch.int64 or self.response_row_ptr.ndim != 1:
            raise ValueError("response_row_ptr must be an int64 vector")
        if self.response_column_indices.dtype != torch.int32 or self.response_column_indices.ndim != 1:
            raise ValueError("response_column_indices must be an int32 vector")
        if self.response_values.ndim != 1 or not self.response_values.is_floating_point():
            raise ValueError("response_values must be a floating point vector")
        rows = self.num_channels * self.num_response_tokens
        if self.response_row_ptr.numel() != rows + 1:
            raise ValueError("CSR row count does not match attention shape")
        if self.response_column_indices.numel() != self.response_values.numel():
            raise ValueError("CSR columns and values do not align")
        if (self.response_row_ptr[1:] < self.response_row_ptr[:-1]).any() or int(self.response_row_ptr[0]) != 0 or int(self.response_row_ptr[-1]) != self.response_values.numel():
            raise ValueError("CSR row pointers are invalid")
        if self.response_column_indices.numel():
            if int(self.response_column_indices.min()) < 0 or int(self.response_column_indices.max()) >= self.num_tokens:
                raise ValueError("CSR column index is out of range")
            counts = self.response_row_ptr[1:] - self.response_row_ptr[:-1]
            row_indices = torch.repeat_interleave(
                torch.arange(rows, device=self.response_row_ptr.device), counts
            )
            centers = self.response_idx + row_indices.remainder(self.num_response_tokens)
            columns = self.response_column_indices.to(
                device=centers.device, dtype=torch.int64
            )
            if (columns >= centers).any():
                raise ValueError("CSR entries must be causal")

    def save(self, path: str | Path) -> None:
        save_attention_sample(self, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu") -> "AttentionSample":
        return load_attention_sample(path, map_location)


def save_attention_sample(sample: AttentionSample, path: str | Path) -> None:
    sample.validate()
    payload = {
        "schema": SCHEMA,
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "response_idx": sample.response_idx,
        "token_ids": sample.token_ids.detach().cpu(),
        "attention_diagonal": sample.attention_diagonal.detach().cpu(),
        "response_row_ptr": sample.response_row_ptr.detach().cpu(),
        "response_column_indices": sample.response_column_indices.detach().cpu(),
        "response_values": sample.response_values.detach().cpu(),
        "attention_floor": float(sample.attention_floor),
    }
    torch.save(payload, Path(path))


def load_attention_sample(path: str | Path, map_location: Any = "cpu") -> AttentionSample:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    schema = payload.get("schema", payload.get("attention_cache_schema"))
    if schema == SCHEMA:
        sample_id = payload["sample_id"]
    elif schema == LEGACY_SCHEMA:
        sample_id = payload["response_id"]
    else:
        raise ValueError("unsupported attention cache schema")
    sample = AttentionSample(
        sample_id=str(sample_id), source_id=str(payload["source_id"]),
        response_idx=int(payload["response_idx"]), token_ids=payload["token_ids"],
        attention_diagonal=payload["attention_diagonal"],
        response_row_ptr=payload["response_row_ptr"],
        response_column_indices=payload["response_column_indices"],
        response_values=payload["response_values"], attention_floor=float(payload["attention_floor"]),
    )
    sample.validate()
    return sample
