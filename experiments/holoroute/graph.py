"""Build one bounded dual-axis attention event graph per sample.

A node is an exact ``(source, target, layer)`` attention event. Its attribute is
that layer's complete head profile. The graph keeps the strongest events for
each target-layer pair and moves discarded retained mass into ``unresolved`` so
row mass remains conserved.
"""

from dataclasses import dataclass

import torch

from .config import GraphConfig

PROMPT = 0
RESPONSE = 1


@dataclass(frozen=True)
class Events:
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    role: torch.Tensor
    lag: torch.Tensor
    value: torch.Tensor
    observed: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.source.numel())

    @property
    def mass(self) -> torch.Tensor:
        return self.value.sum(dim=-1)

    def select(self, index: torch.Tensor) -> "Events":
        return Events(
            source=self.source[index],
            target=self.target[index],
            layer=self.layer[index],
            role=self.role[index],
            lag=self.lag[index],
            value=self.value[index],
            observed=self.observed[index],
        )

    def to(self, device) -> "Events":
        return Events(
            source=self.source.to(device),
            target=self.target.to(device),
            layer=self.layer.to(device),
            role=self.role.to(device),
            lag=self.lag.to(device),
            value=self.value.to(device),
            observed=self.observed.to(device),
        )


@dataclass(frozen=True)
class QueryGroups:
    events: torch.Tensor
    pointer: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor

    @property
    def count(self) -> int:
        return max(int(self.pointer.numel()) - 1, 0)

    def members(self, group: int) -> torch.Tensor:
        start = int(self.pointer[group].item())
        stop = int(self.pointer[group + 1].item())
        return self.events[start:stop]

    def to(self, device) -> "QueryGroups":
        return QueryGroups(
            events=self.events.to(device),
            pointer=self.pointer.to(device),
            target=self.target.to(device),
            layer=self.layer.to(device),
        )


@dataclass(frozen=True)
class EventGraph:
    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    token_count: int
    response_count: int
    layer_count: int
    head_count: int
    attention_floor: float
    events: Events
    depth_edges: torch.Tensor
    relay_edges: torch.Tensor
    queries: QueryGroups
    diamonds: torch.Tensor
    diagonal: torch.Tensor
    unresolved: torch.Tensor
    response_token_ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.events.value.device

    @property
    def event_count(self) -> int:
        return self.events.count

    @property
    def event_query(self) -> torch.Tensor:
        return self.events.target - self.response_start

    def to(self, device) -> "EventGraph":
        return EventGraph(
            sample_id=self.sample_id,
            source_id=self.source_id,
            task_type=self.task_type,
            response_start=self.response_start,
            token_count=self.token_count,
            response_count=self.response_count,
            layer_count=self.layer_count,
            head_count=self.head_count,
            attention_floor=self.attention_floor,
            events=self.events.to(device),
            depth_edges=self.depth_edges.to(device),
            relay_edges=self.relay_edges.to(device),
            queries=self.queries.to(device),
            diamonds=self.diamonds.to(device),
            diagonal=self.diagonal.to(device),
            unresolved=self.unresolved.to(device),
            response_token_ids=self.response_token_ids.to(device),
        )

    def check(self) -> "EventGraph":
        assert self.events.value.shape == (self.event_count, self.head_count)
        assert self.events.observed.shape == self.events.value.shape
        assert not self.event_count or bool((self.events.source < self.events.target).all())
        assert self.depth_edges.shape[0] == 2
        assert self.relay_edges.shape[0] == 2
        assert self.diamonds.shape[0] == 4
        assert self.diagonal.shape == (self.response_count, self.layer_count, self.head_count)
        assert self.unresolved.shape == self.diagonal.shape
        return self


def empty_index(rows: int, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, 0), dtype=torch.long, device=device)


def read_sparse_entries(sample, config: GraphConfig) -> tuple[torch.Tensor, ...]:
    columns: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("source", "target", "layer", "head", "weight")
    }
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        keep = (block.source < block.target) & (block.source != block.target)
        if not bool(keep.any()):
            continue
        columns["source"].append(block.source[keep].long())
        columns["target"].append(block.target[keep].long())
        columns["layer"].append(block.layer[keep].long())
        columns["head"].append(block.head[keep].long())
        columns["weight"].append(block.weight[keep].float().clamp_min(0.0))

    if columns["source"]:
        return tuple(torch.cat(columns[name]) for name in columns)

    device = sample.attention().response_values.device
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_float = torch.empty(0, dtype=torch.float32, device=device)
    return empty_long, empty_long, empty_long, empty_long, empty_float


def normalize_rows(
    attention,
    source: torch.Tensor,
    target: torch.Tensor,
    layer: torch.Tensor,
    head: torch.Tensor,
    weight: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    response_start = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    layer_count = int(attention.num_layers)
    head_count = int(attention.num_heads)

    diagonal = (
        attention.attention_diagonal[:, :, response_start:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    retained = torch.zeros(
        (response_count, layer_count, head_count),
        dtype=torch.float32,
        device=weight.device,
    )
    if weight.numel():
        retained.index_put_((target - response_start, layer, head), weight, accumulate=True)

    known = retained + diagonal
    if known.numel() and float((known - 1.0).clamp_min(0.0).max().item()) > tolerance:
        raise ValueError("attention row mass exceeds the cache tolerance")

    scale = torch.where(known > 1.0, known.reciprocal(), torch.ones_like(known))
    if weight.numel():
        weight = weight * scale[target - response_start, layer, head]
    diagonal = diagonal * scale
    unresolved = (1.0 - retained * scale - diagonal).clamp_min(0.0)
    return weight, diagonal, unresolved


def group_head_profiles(
    source: torch.Tensor,
    target: torch.Tensor,
    layer: torch.Tensor,
    head: torch.Tensor,
    weight: torch.Tensor,
    token_count: int,
    head_count: int,
    response_start: int,
    minimum_mass: float,
) -> Events:
    if not weight.numel():
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return Events(
            empty,
            empty,
            empty,
            empty,
            empty,
            torch.empty((0, head_count), dtype=torch.float32, device=source.device),
            torch.empty((0, head_count), dtype=torch.bool, device=source.device),
        )

    key = (layer * token_count + source) * token_count + target
    unique, event_of_entry = torch.unique(key, sorted=True, return_inverse=True)
    event_layer = torch.div(unique, token_count * token_count, rounding_mode="floor")
    remainder = unique.remainder(token_count * token_count)
    event_source = torch.div(remainder, token_count, rounding_mode="floor")
    event_target = remainder.remainder(token_count)

    values = torch.zeros((len(unique), head_count), dtype=torch.float32, device=weight.device)
    values.index_put_((event_of_entry, head), weight, accumulate=True)
    observed = torch.zeros_like(values, dtype=torch.bool)
    observed[event_of_entry, head] = True

    keep = values.sum(dim=-1) > minimum_mass
    event_source = event_source[keep]
    event_target = event_target[keep]
    event_layer = event_layer[keep]
    values = values[keep]
    observed = observed[keep]
    return Events(
        source=event_source,
        target=event_target,
        layer=event_layer,
        role=(event_source >= response_start).long(),
        lag=event_target - event_source,
        value=values,
        observed=observed,
    )


def limit_events(
    events: Events,
    response_start: int,
    response_count: int,
    layer_count: int,
    maximum_events: int,
) -> tuple[Events, torch.Tensor]:
    """Keep top-mass events per target-layer and return their discarded mass."""

    discarded = events.value.new_zeros((response_count, layer_count, events.value.shape[-1]))
    if not events.count:
        return events, discarded

    groups: dict[tuple[int, int], list[int]] = {}
    targets = events.target.detach().cpu().tolist()
    layers = events.layer.detach().cpu().tolist()
    masses = events.mass.detach().cpu().tolist()
    for index, key in enumerate(zip(targets, layers, strict=True)):
        groups.setdefault(key, []).append(index)

    keep_indices: list[int] = []
    for members in groups.values():
        members.sort(key=lambda index: masses[index], reverse=True)
        keep_indices.extend(members[:maximum_events])
    keep_indices.sort()

    selected = torch.tensor(keep_indices, dtype=torch.long, device=events.value.device)
    keep = torch.zeros(events.count, dtype=torch.bool, device=events.value.device)
    keep[selected] = True
    dropped = ~keep
    if bool(dropped.any()):
        row = (
            (events.target[dropped] - response_start) * layer_count
            + events.layer[dropped]
        )
        flat = discarded.view(response_count * layer_count, -1)
        flat.index_add_(0, row, events.value[dropped])

    return events.select(selected), discarded


def event_lookup(events: Events) -> dict[tuple[int, int, int], int]:
    return {
        (int(source), int(target), int(layer)): index
        for index, (source, target, layer) in enumerate(
            zip(
                events.source.detach().cpu().tolist(),
                events.target.detach().cpu().tolist(),
                events.layer.detach().cpu().tolist(),
                strict=True,
            )
        )
    }


def build_depth_edges(events: Events, lookup: dict[tuple[int, int, int], int]) -> torch.Tensor:
    left: list[int] = []
    right: list[int] = []
    for index, (source, target, layer) in enumerate(
        zip(
            events.source.detach().cpu().tolist(),
            events.target.detach().cpu().tolist(),
            events.layer.detach().cpu().tolist(),
            strict=True,
        )
    ):
        successor = lookup.get((source, target, layer + 1))
        if successor is not None:
            left.append(index)
            right.append(successor)
    if not left:
        return empty_index(2, events.value.device)
    return torch.tensor([left, right], dtype=torch.long, device=events.value.device)


def incoming_events(events: Events) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for index, (target, layer) in enumerate(
        zip(
            events.target.detach().cpu().tolist(),
            events.layer.detach().cpu().tolist(),
            strict=True,
        )
    ):
        result.setdefault((layer, target), []).append(index)
    return result


def build_relay_edges(
    events: Events,
    response_start: int,
    maximum_predecessors: int,
) -> torch.Tensor:
    incoming = incoming_events(events)
    left: list[int] = []
    right: list[int] = []
    sources = events.source.detach().cpu().tolist()
    layers = events.layer.detach().cpu().tolist()
    masses = events.mass.detach().cpu().tolist()

    for successor, (middle, layer) in enumerate(zip(sources, layers, strict=True)):
        if middle < response_start or layer == 0:
            continue
        predecessors = incoming.get((layer - 1, middle), [])
        predecessors = sorted(predecessors, key=lambda event: masses[event], reverse=True)
        predecessors = predecessors[:maximum_predecessors]
        left.extend(predecessors)
        right.extend([successor] * len(predecessors))

    if not left:
        return empty_index(2, events.value.device)
    return torch.tensor([left, right], dtype=torch.long, device=events.value.device)


def build_query_groups(events: Events, maximum_events: int) -> QueryGroups:
    groups: dict[tuple[int, int], list[int]] = {}
    targets = events.target.detach().cpu().tolist()
    layers = events.layer.detach().cpu().tolist()
    masses = events.mass.detach().cpu().tolist()
    for index, key in enumerate(zip(targets, layers, strict=True)):
        groups.setdefault(key, []).append(index)

    ordered_events: list[int] = []
    pointer = [0]
    query_target: list[int] = []
    query_layer: list[int] = []
    for (target, layer), members in sorted(groups.items()):
        members = sorted(members, key=lambda event: masses[event], reverse=True)[:maximum_events]
        ordered_events.extend(members)
        pointer.append(len(ordered_events))
        query_target.append(target)
        query_layer.append(layer)

    device = events.value.device
    return QueryGroups(
        events=torch.tensor(ordered_events, dtype=torch.long, device=device),
        pointer=torch.tensor(pointer, dtype=torch.long, device=device),
        target=torch.tensor(query_target, dtype=torch.long, device=device),
        layer=torch.tensor(query_layer, dtype=torch.long, device=device),
    )


def build_diamonds(
    events: Events,
    relay_edges: torch.Tensor,
    lookup: dict[tuple[int, int, int], int],
    maximum_per_query_layer: int,
) -> torch.Tensor:
    sources = events.source.detach().cpu().tolist()
    targets = events.target.detach().cpu().tolist()
    layers = events.layer.detach().cpu().tolist()
    masses = events.mass.detach().cpu().tolist()
    candidates: dict[tuple[int, int], list[tuple[float, tuple[int, int, int, int]]]] = {}

    for predecessor, successor in relay_edges.T.detach().cpu().tolist():
        layer = layers[predecessor]
        depth_middle = lookup.get((sources[predecessor], targets[predecessor], layer + 1))
        end = lookup.get((sources[successor], targets[successor], layer + 2))
        if depth_middle is None or end is None:
            continue
        key = (targets[end], layers[end])
        score = masses[predecessor] + masses[successor]
        candidates.setdefault(key, []).append(
            (score, (predecessor, depth_middle, successor, end))
        )

    rows = [[], [], [], []]
    for values in candidates.values():
        values.sort(key=lambda item: item[0], reverse=True)
        for _, diamond in values[:maximum_per_query_layer]:
            for row, event in enumerate(diamond):
                rows[row].append(event)

    if not rows[0]:
        return empty_index(4, events.value.device)
    return torch.tensor(rows, dtype=torch.long, device=events.value.device)


@torch.no_grad()
def build_graph(sample, config: GraphConfig | None = None) -> EventGraph:
    config = GraphConfig() if config is None else config
    attention = sample.attention()
    source, target, layer, head, weight = read_sparse_entries(sample, config)
    weight, diagonal, unresolved = normalize_rows(
        attention,
        source,
        target,
        layer,
        head,
        weight,
        config.numerical_tolerance,
    )
    events = group_head_profiles(
        source,
        target,
        layer,
        head,
        weight,
        int(attention.num_tokens),
        int(attention.num_heads),
        int(attention.response_idx),
        config.minimum_event_mass,
    )
    events, discarded = limit_events(
        events,
        int(attention.response_idx),
        int(attention.num_response_tokens),
        int(attention.num_layers),
        config.max_events_per_query_layer,
    )
    unresolved = (unresolved + discarded).clamp_max(1.0)

    lookup = event_lookup(events)
    depth_edges = build_depth_edges(events, lookup)
    relay_edges = build_relay_edges(
        events,
        int(attention.response_idx),
        config.max_relay_predecessors,
    )
    queries = build_query_groups(events, config.max_query_events)
    diamonds = build_diamonds(
        events,
        relay_edges,
        lookup,
        config.max_diamonds_per_query_layer,
    )

    return EventGraph(
        sample_id=str(sample.sample_id),
        source_id=str(sample.source_id),
        task_type=str(sample.task_type or ""),
        response_start=int(attention.response_idx),
        token_count=int(attention.num_tokens),
        response_count=int(attention.num_response_tokens),
        layer_count=int(attention.num_layers),
        head_count=int(attention.num_heads),
        attention_floor=float(attention.attention_floor),
        events=events,
        depth_edges=depth_edges,
        relay_edges=relay_edges,
        queries=queries,
        diamonds=diamonds,
        diagonal=diagonal,
        unresolved=unresolved,
        response_token_ids=attention.token_ids[int(attention.response_idx):].long(),
    ).check()
