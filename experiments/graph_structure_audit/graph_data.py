"""Lossless sample-level multiplex attention graphs."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from attention_lifecycle import loaded_attention
from experiments.source_reuse_contrast.data import (
    SourceReuseGraph,
    collect_source_reuse_graph,
)


@dataclass(frozen=True)
class MultiplexGraph:
    sample_id: str
    source_id: str
    task_type: str
    response_idx: int
    num_nodes: int
    num_response_tokens: int
    num_layers: int
    num_heads: int
    attention_floor: float
    edge_index: torch.Tensor          # [2, E], exact source -> response target
    edge_attr: torch.Tensor           # [E, L, H]
    edge_observed: torch.Tensor       # [E, L, H]
    target_ptr: torch.Tensor          # [T + 1], pairs sorted by response target
    node_role: torch.Tensor           # [N], 0 prompt / 1 response
    node_position: torch.Tensor       # [N]
    diagonal: torch.Tensor            # [N, L, H]
    diagonal_observed: torch.Tensor   # [N, L, H]

    @property
    def device(self) -> torch.device:
        return self.edge_attr.device

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def incoming(self, token: int) -> slice:
        return slice(int(self.target_ptr[token]), int(self.target_ptr[token + 1]))

    def to(self, device: str | torch.device) -> "MultiplexGraph":
        names = (
            "edge_index",
            "edge_attr",
            "edge_observed",
            "target_ptr",
            "node_role",
            "node_position",
            "diagonal",
            "diagonal_observed",
        )
        return replace(self, **{name: getattr(self, name).to(device) for name in names})

    def numpy_dict(self) -> dict[str, np.ndarray]:
        return {
            "edge_index": self.edge_index.cpu().numpy().astype(np.int32),
            "edge_attr": self.edge_attr.cpu().numpy().astype(np.float16),
            "edge_observed": self.edge_observed.cpu().numpy().astype(bool),
            "target_ptr": self.target_ptr.cpu().numpy().astype(np.int64),
            "node_role": self.node_role.cpu().numpy().astype(np.int8),
            "node_position": self.node_position.cpu().numpy().astype(np.float32),
            "diagonal": self.diagonal.cpu().numpy().astype(np.float16),
            "diagonal_observed": self.diagonal_observed.cpu().numpy().astype(bool),
        }


def build_multiplex_graph(raw: SourceReuseGraph) -> MultiplexGraph:
    source = raw.source.long()
    target = raw.response_idx + raw.query.long()
    pair_key = target * raw.num_tokens + source
    unique_key, event_pair = torch.unique(pair_key, sorted=True, return_inverse=True)

    pair_target = torch.div(unique_key, raw.num_tokens, rounding_mode="floor")
    pair_source = unique_key % raw.num_tokens
    edge_index = torch.stack((pair_source, pair_target))

    shape = (len(unique_key), raw.num_layers, raw.num_heads)
    edge_attr = raw.weight.new_zeros(shape)
    edge_observed = torch.zeros(shape, dtype=torch.bool, device=raw.device)
    edge_attr.index_put_((event_pair, raw.layer, raw.head), raw.weight, accumulate=True)
    edge_observed.index_put_(
        (event_pair, raw.layer, raw.head),
        torch.ones_like(raw.weight, dtype=torch.bool),
    )

    response_target = pair_target - raw.response_idx
    counts = torch.bincount(response_target, minlength=raw.num_response_tokens)
    target_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    node_role = torch.zeros(raw.num_tokens, dtype=torch.long, device=raw.device)
    node_role[raw.response_idx :] = 1
    node_position = torch.arange(raw.num_tokens, device=raw.device, dtype=torch.float32)
    node_position /= float(max(raw.num_tokens - 1, 1))

    diagonal = raw.weight.new_zeros((raw.num_tokens, raw.num_layers, raw.num_heads))
    diagonal_observed = torch.zeros_like(diagonal, dtype=torch.bool)
    diagonal[raw.response_idx :] = raw.diagonal
    diagonal_observed[raw.response_idx :] = True

    return MultiplexGraph(
        sample_id=raw.sample_id,
        source_id=raw.source_id,
        task_type=raw.task_type,
        response_idx=raw.response_idx,
        num_nodes=raw.num_tokens,
        num_response_tokens=raw.num_response_tokens,
        num_layers=raw.num_layers,
        num_heads=raw.num_heads,
        attention_floor=raw.attention_floor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_observed=edge_observed,
        target_ptr=target_ptr,
        node_role=node_role,
        node_position=node_position,
        diagonal=diagonal,
        diagonal_observed=diagonal_observed,
    )


def load_multiplex_graph(sample, *, block_rows: int = 8192) -> MultiplexGraph:
    """Materialize one graph and release the source attention immediately."""

    with loaded_attention(sample):
        raw = collect_source_reuse_graph(sample, block_rows=block_rows)
    return build_multiplex_graph(raw)
