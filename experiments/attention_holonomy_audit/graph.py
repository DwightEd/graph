"""Dual-axis causal attention event graph used by the mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GraphConfig

PROMPT_EVENT = 0
RESPONSE_EVENT = 1
EVENT_ROLE_NAMES = ("prompt_source", "response_source")


@dataclass(frozen=True)
class AttentionEventGraph:
    sample_id: str
    source_id: str
    task_type: str
    response_idx: int
    num_tokens: int
    num_response_tokens: int
    num_layers: int
    num_heads: int
    attention_floor: float

    event_source: torch.Tensor
    event_target: torch.Tensor
    event_layer: torch.Tensor
    event_role: torch.Tensor
    event_lag: torch.Tensor
    event_head_value: torch.Tensor
    event_head_observed: torch.Tensor

    depth_edge_index: torch.Tensor
    relay_edge_index: torch.Tensor
    diamond_index: torch.Tensor

    query_event_index: torch.Tensor
    query_ptr: torch.Tensor
    query_target: torch.Tensor
    query_layer: torch.Tensor

    diagonal: torch.Tensor
    unresolved_mass: torch.Tensor
    response_token_ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.event_head_value.device

    @property
    def num_events(self) -> int:
        return int(self.event_source.numel())

    @property
    def event_query(self) -> torch.Tensor:
        return self.event_target - self.response_idx

    @property
    def event_mass(self) -> torch.Tensor:
        return self.event_head_value.sum(dim=-1)

    def validate(self) -> "AttentionEventGraph":
        events = self.num_events
        for column in (
            self.event_source,
            self.event_target,
            self.event_layer,
            self.event_role,
            self.event_lag,
        ):
            if column.shape != (events,):
                raise ValueError("event columns must be aligned")
        if self.event_head_value.shape != (events, self.num_heads):
            raise ValueError("event_head_value must be [event, head]")
        if self.event_head_observed.shape != self.event_head_value.shape:
            raise ValueError("event observation mask must match head values")
        if events and bool((self.event_source >= self.event_target).any()):
            raise ValueError("event graph must remain prefix-causal")
        if events and bool(
            ((self.event_layer < 0) | (self.event_layer >= self.num_layers)).any()
        ):
            raise ValueError("event layer is outside model geometry")
        if self.depth_edge_index.ndim != 2 or self.depth_edge_index.shape[0] != 2:
            raise ValueError("depth_edge_index must be [2, edge]")
        if self.relay_edge_index.ndim != 2 or self.relay_edge_index.shape[0] != 2:
            raise ValueError("relay_edge_index must be [2, edge]")
        if self.diamond_index.ndim != 2 or self.diamond_index.shape[0] != 4:
            raise ValueError("diamond_index must be [4, diamond]")
        if self.diagonal.shape != (
            self.num_response_tokens,
            self.num_layers,
            self.num_heads,
        ):
            raise ValueError("diagonal must be [response, layer, head]")
        if self.unresolved_mass.shape != self.diagonal.shape:
            raise ValueError("unresolved mass must match diagonal geometry")
        return self


def _empty_index(rows: int, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, 0), dtype=torch.long, device=device)


@torch.no_grad()
def build_attention_event_graph(
    sample,
    *,
    config: GraphConfig | None = None,
) -> AttentionEventGraph:
    """Build event, depth, relay, query-set and causal-diamond relations."""

    config = GraphConfig() if config is None else config
    attention = sample.attention()
    response_idx = int(attention.response_idx)
    response_tokens = int(attention.num_response_tokens)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    num_tokens = int(attention.num_tokens)
    device = attention.response_values.device

    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("source", "target", "layer", "head", "weight")
    }
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        selected = (block.source < block.target) & (block.source != block.target)
        if not bool(selected.any()):
            continue
        for name, value in (
            ("source", block.source[selected].long()),
            ("target", block.target[selected].long()),
            ("layer", block.layer[selected].long()),
            ("head", block.head[selected].long()),
            ("weight", block.weight[selected].float().clamp_min(0.0)),
        ):
            parts[name].append(value)

    if parts["source"]:
        source, target, layer, head, raw_weight = (
            torch.cat(parts[name])
            for name in ("source", "target", "layer", "head", "weight")
        )
    else:
        source = torch.empty(0, dtype=torch.long, device=device)
        target = torch.empty_like(source)
        layer = torch.empty_like(source)
        head = torch.empty_like(source)
        raw_weight = torch.empty(0, dtype=torch.float32, device=device)

    diagonal = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    off_diagonal = torch.zeros(
        (response_tokens, layers, heads), dtype=torch.float32, device=device
    )
    if raw_weight.numel():
        off_diagonal.index_put_(
            (target - response_idx, layer, head), raw_weight, accumulate=True
        )
    known = off_diagonal + diagonal
    overshoot = (known - 1.0).clamp_min(0.0)
    if overshoot.numel() and float(overshoot.max().item()) > config.numerical_tolerance:
        raise ValueError("attention row mass exceeds numerical tolerance")
    scale = torch.where(known > 1.0, known.reciprocal(), torch.ones_like(known))
    if raw_weight.numel():
        raw_weight = raw_weight * scale[target - response_idx, layer, head]
    diagonal = diagonal * scale
    unresolved = (1.0 - off_diagonal * scale - diagonal).clamp_min(0.0)

    if raw_weight.numel():
        key = (layer * num_tokens + source) * num_tokens + target
        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        event_layer = torch.div(unique_key, num_tokens * num_tokens, rounding_mode="floor")
        remainder = unique_key.remainder(num_tokens * num_tokens)
        event_source = torch.div(remainder, num_tokens, rounding_mode="floor")
        event_target = remainder.remainder(num_tokens)
        event_head_value = torch.zeros(
            (len(unique_key), heads), dtype=torch.float32, device=device
        )
        event_head_value.index_put_((inverse, head), raw_weight, accumulate=True)
        event_head_observed = torch.zeros(
            (len(unique_key), heads), dtype=torch.bool, device=device
        )
        event_head_observed.index_put_(
            (inverse, head), torch.ones_like(head, dtype=torch.bool), accumulate=False
        )
        keep = event_head_value.sum(dim=-1) > float(config.minimum_event_mass)
        event_source = event_source[keep]
        event_target = event_target[keep]
        event_layer = event_layer[keep]
        event_head_value = event_head_value[keep]
        event_head_observed = event_head_observed[keep]
    else:
        event_source = torch.empty(0, dtype=torch.long, device=device)
        event_target = torch.empty_like(event_source)
        event_layer = torch.empty_like(event_source)
        event_head_value = torch.empty((0, heads), dtype=torch.float32, device=device)
        event_head_observed = torch.empty((0, heads), dtype=torch.bool, device=device)

    event_role = (event_source >= response_idx).long()
    event_lag = event_target - event_source
    event_mass = event_head_value.sum(dim=-1)

    source_list = event_source.detach().cpu().tolist()
    target_list = event_target.detach().cpu().tolist()
    layer_list = event_layer.detach().cpu().tolist()
    mass_list = event_mass.detach().cpu().tolist()
    lookup = {
        (int(source_value), int(target_value), int(layer_value)): index
        for index, (source_value, target_value, layer_value) in enumerate(
            zip(source_list, target_list, layer_list, strict=True)
        )
    }

    depth_left: list[int] = []
    depth_right: list[int] = []
    for index, (s, t, current_layer) in enumerate(
        zip(source_list, target_list, layer_list, strict=True)
    ):
        next_index = lookup.get((int(s), int(t), int(current_layer) + 1))
        if next_index is not None:
            depth_left.append(index)
            depth_right.append(next_index)

    incoming: dict[tuple[int, int], list[int]] = {}
    for index, (t, current_layer) in enumerate(zip(target_list, layer_list, strict=True)):
        incoming.setdefault((int(current_layer), int(t)), []).append(index)

    relay_left: list[int] = []
    relay_right: list[int] = []
    for successor, (middle, current_layer) in enumerate(
        zip(source_list, layer_list, strict=True)
    ):
        if middle < response_idx or current_layer < 1:
            continue
        predecessors = incoming.get((int(current_layer) - 1, int(middle)), [])
        if not predecessors:
            continue
        predecessors = sorted(
            predecessors,
            key=lambda index: mass_list[index],
            reverse=True,
        )[: int(config.max_relay_predecessors)]
        relay_left.extend(predecessors)
        relay_right.extend([successor] * len(predecessors))

    query_groups: dict[tuple[int, int], list[int]] = {}
    for index, (t, current_layer) in enumerate(zip(target_list, layer_list, strict=True)):
        query_groups.setdefault((int(t), int(current_layer)), []).append(index)
    query_event_index: list[int] = []
    query_ptr = [0]
    query_target: list[int] = []
    query_layer: list[int] = []
    for (t, current_layer), indices in sorted(query_groups.items()):
        selected = sorted(indices, key=lambda index: mass_list[index], reverse=True)[
            : int(config.max_query_events)
        ]
        query_event_index.extend(selected)
        query_ptr.append(len(query_event_index))
        query_target.append(t)
        query_layer.append(current_layer)

    diamond_rows = [[], [], [], []]
    for predecessor, successor in zip(relay_left, relay_right, strict=True):
        current_layer = layer_list[predecessor]
        depth_middle = lookup.get(
            (
                source_list[predecessor],
                target_list[predecessor],
                current_layer + 1,
            )
        )
        end = lookup.get(
            (
                source_list[successor],
                target_list[successor],
                current_layer + 2,
            )
        )
        if depth_middle is None or end is None:
            continue
        for row, value in enumerate((predecessor, depth_middle, successor, end)):
            diamond_rows[row].append(value)

    def index_tensor(rows: list[list[int]] | tuple[list[int], ...]) -> torch.Tensor:
        if not rows or not rows[0]:
            return _empty_index(len(rows), device)
        return torch.tensor(rows, dtype=torch.long, device=device)

    return AttentionEventGraph(
        sample_id=str(sample.sample_id),
        source_id=str(sample.source_id),
        task_type=str(sample.task_type or ""),
        response_idx=response_idx,
        num_tokens=num_tokens,
        num_response_tokens=response_tokens,
        num_layers=layers,
        num_heads=heads,
        attention_floor=float(attention.attention_floor),
        event_source=event_source,
        event_target=event_target,
        event_layer=event_layer,
        event_role=event_role,
        event_lag=event_lag,
        event_head_value=event_head_value,
        event_head_observed=event_head_observed,
        depth_edge_index=index_tensor([depth_left, depth_right]),
        relay_edge_index=index_tensor([relay_left, relay_right]),
        diamond_index=index_tensor(diamond_rows),
        query_event_index=torch.tensor(query_event_index, dtype=torch.long, device=device),
        query_ptr=torch.tensor(query_ptr, dtype=torch.long, device=device),
        query_target=torch.tensor(query_target, dtype=torch.long, device=device),
        query_layer=torch.tensor(query_layer, dtype=torch.long, device=device),
        diagonal=diagonal,
        unresolved_mass=unresolved,
        response_token_ids=attention.token_ids[response_idx:].long(),
    ).validate()
