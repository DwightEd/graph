"""Sparse attention data for source-reuse predictability learning."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


PROMPT = 0
RESPONSE = 1


def select_sample_ids(
    dataset,
    *,
    task_type: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Select samples by task before applying an optional count limit."""

    sample_ids = [str(sample_id) for sample_id in dataset.sample_ids]
    if task_type is not None:
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if str(dataset[sample_id].task_type) == task_type
        ]
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        selection = f" for task type {task_type!r}" if task_type is not None else ""
        raise ValueError(f"no samples selected{selection}")
    return sample_ids


@dataclass(frozen=True)
class SourceReuseGraph:
    """One response represented as a causal stream of source incidences."""

    sample_id: str
    source_id: str
    task_type: str
    response_idx: int
    num_response_tokens: int
    num_tokens: int
    num_layers: int
    num_heads: int
    attention_floor: float
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor
    query_ptr: torch.Tensor
    diagonal: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def num_edges(self) -> int:
        return int(self.weight.numel())

    def token_slice(self, token: int) -> slice:
        return slice(
            int(self.query_ptr[token].item()),
            int(self.query_ptr[token + 1].item()),
        )

    def to(self, device: str | torch.device) -> "SourceReuseGraph":
        tensor_fields = {
            name: getattr(self, name).to(device)
            for name in (
                "layer",
                "head",
                "query",
                "source",
                "weight",
                "query_ptr",
                "diagonal",
            )
        }
        return replace(self, **tensor_fields)


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


def collect_source_reuse_graph(sample, *, block_rows: int = 8192) -> SourceReuseGraph:
    """Decode retained causal attention through ``research_dataset`` only."""

    attention = sample.attention()
    response_idx = int(attention.response_idx)
    parts = {name: [] for name in ("layer", "head", "query", "source", "weight")}

    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        target = response_idx + block.query
        off_diagonal = block.source < target
        if not bool(off_diagonal.any()):
            continue
        parts["layer"].append(block.layer[off_diagonal].long())
        parts["head"].append(block.head[off_diagonal].long())
        parts["query"].append(block.query[off_diagonal].long())
        parts["source"].append(block.source[off_diagonal].long())
        parts["weight"].append(block.weight[off_diagonal].float().clamp_min(0.0))

    device = attention.response_values.device
    if parts["weight"]:
        layer, head, query, source, weight = (
            torch.cat(parts[name])
            for name in ("layer", "head", "query", "source", "weight")
        )
        key = (
            ((query * int(attention.num_tokens) + source) * int(attention.num_layers) + layer)
            * int(attention.num_heads)
            + head
        )
        order = key.argsort()
        layer = layer[order]
        head = head[order]
        query = query[order]
        source = source[order]
        weight = weight[order]
    else:
        layer = _empty_long(device)
        head = _empty_long(device)
        query = _empty_long(device)
        source = _empty_long(device)
        weight = torch.empty(0, dtype=torch.float32, device=device)

    response_count = int(attention.num_response_tokens)
    counts = torch.bincount(query, minlength=response_count)
    query_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(dim=0))
    )
    diagonal = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )

    return SourceReuseGraph(
        sample_id=str(sample.sample_id),
        source_id=str(sample.source_id),
        task_type=str(sample.task_type or "unknown"),
        response_idx=response_idx,
        num_response_tokens=response_count,
        num_tokens=int(attention.num_tokens),
        num_layers=int(attention.num_layers),
        num_heads=int(attention.num_heads),
        attention_floor=float(attention.attention_floor),
        layer=layer,
        head=head,
        query=query,
        source=source,
        weight=weight,
        query_ptr=query_ptr,
        diagonal=diagonal,
    )
