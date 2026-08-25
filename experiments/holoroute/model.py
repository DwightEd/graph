"""Neural layers for HoloRoute.

The forward path is intentionally direct:

``head profile -> event state -> depth/relay/query context -> fused state``.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .graph import EventGraph, PROMPT

EVENT = 0
DEPTH = 1
RELAY = 2
QUERY = 3

DEPTH_CONTEXT = 0
RELAY_CONTEXT = 1
QUERY_CONTEXT = 2

DEPTH_RELATION = 0
PROMPT_RELAY = 1
RESPONSE_RELAY = 2
RELATION_COUNT = 3


def segment_softmax(score: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    if not score.numel():
        return score
    maximum = score.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, target, score, reduce="amax", include_self=True)
    weight = torch.exp(score - maximum[target])
    normalizer = score.new_zeros(size)
    normalizer.index_add_(0, target, weight)
    return weight / normalizer[target].clamp_min(1e-12)


def scatter_sum(message: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    output = message.new_zeros((size, message.shape[-1]))
    if message.numel():
        output.index_add_(0, target, message)
    return output


@dataclass(frozen=True)
class Predictions:
    value: torch.Tensor
    support: torch.Tensor


@dataclass(frozen=True)
class ModelOutput:
    state: torch.Tensor
    predictions: Predictions
    contexts: torch.Tensor
    coverage: torch.Tensor
    holonomy: torch.Tensor
    holonomy_token: torch.Tensor


class HeadEncoder(nn.Module):
    """Encode an event's head set with learned seed queries.

    Seed-to-head attention is linear in the number of heads. The previous
    head-to-head Transformer materialized a square attention matrix for every
    event and dominated GPU memory on long samples.
    """

    def __init__(self, layers: int, heads: int, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.head_id = nn.Embedding(heads, hidden)
        self.layer_id = nn.Embedding(layers, hidden)
        self.role_id = nn.Embedding(2, hidden)
        self.lag_id = nn.Embedding(config.lag_buckets, hidden)
        self.value = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.seeds = nn.Parameter(torch.empty(2, hidden))
        nn.init.normal_(self.seeds, std=0.02)
        self.head_pool = nn.MultiheadAttention(
            hidden,
            num_heads=config.head_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.summary_blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 4 * hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * hidden, hidden),
            )
            for _ in range(config.head_layers)
        )
        self.metadata = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        graph: EventGraph,
        values: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        heads = values.shape[1]
        head = torch.arange(heads, device=values.device)
        tokens = self.value(torch.stack((torch.log1p(values), observed.float()), dim=-1))
        tokens = tokens + self.head_id(head)[None]
        tokens = tokens + self.layer_id(graph.events.layer)[:, None]

        query = self.seeds[None].expand(len(tokens), -1, -1)
        summary, _ = self.head_pool(query, tokens, tokens, need_weights=False)
        state = summary.mean(dim=1)
        for block in self.summary_blocks:
            state = state + block(state)

        lag = torch.floor(torch.log2(graph.events.lag.float().clamp_min(1))).long()
        lag = lag.clamp_max(self.lag_id.num_embeddings - 1)
        query_position = graph.event_query.float() / max(graph.response_count - 1, 1)
        source_position = graph.events.source.float() / max(graph.token_count - 1, 1)
        metadata = torch.stack(
            (
                query_position,
                source_position,
                values.sum(dim=-1),
                observed.float().mean(dim=-1),
            ),
            dim=-1,
        )
        state = state + self.metadata(metadata)
        state = state + self.role_id(graph.events.role)
        state = state + self.lag_id(lag)
        return self.norm(state)


class Transport(nn.Module):
    """Relation-specific low-rank transport before message aggregation."""

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.relation = nn.Embedding(RELATION_COUNT, hidden)
        self.left = nn.Parameter(torch.empty(RELATION_COUNT, hidden, rank))
        self.right = nn.Parameter(torch.empty(RELATION_COUNT, hidden, rank))
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        if not source.numel():
            return source
        relation_state = self.relation(relation)
        coefficient = self.gate(torch.cat((source, target, relation_state), dim=-1))
        latent = torch.einsum("ed,edr->er", source, self.right[relation])
        update = torch.einsum("er,edr->ed", latent * coefficient, self.left[relation])
        return self.norm(source + update)


class RelationMixer(nn.Module):
    """Transport and aggregate messages entering the same event."""

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.transport = Transport(hidden, rank)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        edges: torch.Tensor,
        relation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not edges.shape[1]:
            return (
                state.new_zeros(state.shape),
                torch.zeros(len(state), dtype=torch.bool, device=state.device),
            )
        source, target = edges
        message = self.transport(state[source], state[target], relation)
        score = self.attention(torch.cat((message, state[target]), dim=-1)).squeeze(-1)
        weight = segment_softmax(score, target, len(state))
        context = scatter_sum(message * weight[:, None], target, len(state))
        coverage = torch.zeros(len(state), dtype=torch.bool, device=state.device)
        coverage[target] = True
        return context, coverage


class QueryMixer(nn.Module):
    """Predict each event from the other events in its query-layer set."""

    def __init__(self, hidden: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        state: torch.Tensor,
        graph: EventGraph,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = state.new_zeros(state.shape)
        coverage = torch.zeros(len(state), dtype=torch.bool, device=state.device)

        for group in range(graph.queries.count):
            members = graph.queries.members(group)
            available = members if keep is None else members[keep[members]]
            for target in members:
                sources = available[available != target]
                if not len(sources):
                    continue
                query = state[target].view(1, 1, -1)
                keys = state[sources].unsqueeze(0)
                value, _ = self.attention(query, keys, keys, need_weights=False)
                context[target] = self.norm(value[0, 0])
                coverage[target] = True
        return context, coverage


class HoloRouteLayer(nn.Module):
    """One HoloRoute message-passing layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.depth = RelationMixer(hidden, config.transport_rank)
        self.relay = RelationMixer(hidden, config.transport_rank)
        self.query = QueryMixer(hidden, config.head_attention_heads, config.dropout)
        self.project = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(4))
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        state: torch.Tensor,
        graph: EventGraph,
        relay_keep: torch.Tensor | None = None,
        query_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_relation = torch.full(
            (graph.depth_edges.shape[1],),
            DEPTH_RELATION,
            dtype=torch.long,
            device=state.device,
        )
        depth, depth_coverage = self.depth(state, graph.depth_edges, depth_relation)

        relay_edges = graph.relay_edges if relay_keep is None else graph.relay_edges[:, relay_keep]
        if relay_edges.shape[1]:
            predecessor = relay_edges[0]
            relay_relation = torch.where(
                graph.events.role[predecessor] == PROMPT,
                torch.full_like(predecessor, PROMPT_RELAY),
                torch.full_like(predecessor, RESPONSE_RELAY),
            )
        else:
            relay_relation = torch.empty(0, dtype=torch.long, device=state.device)
        relay, relay_coverage = self.relay(state, relay_edges, relay_relation)
        query, query_coverage = self.query(state, graph, query_keep)

        contexts = torch.stack((depth, relay, query), dim=1)
        coverage = torch.stack((depth_coverage, relay_coverage, query_coverage), dim=1)
        components = torch.stack(
            tuple(
                projection(value)
                for projection, value in zip(
                    self.project,
                    (state, depth, relay, query),
                    strict=True,
                )
            ),
            dim=1,
        )
        weight = torch.softmax(
            self.gate(torch.cat((state, depth, relay, query), dim=-1)),
            dim=-1,
        )
        state = (components * weight[..., None]).sum(dim=1)
        state = self.norm(state + self.update(state))
        return state, contexts, coverage


class EventDecoder(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.value = nn.Linear(hidden, heads)
        self.support = nn.Linear(hidden, heads)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(self.value(state)), self.support(state)


class HoloRoute(nn.Module):
    """Neural encoder on the dual-axis causal attention event graph."""

    def __init__(self, layers: int, heads: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.encoder = HeadEncoder(layers, heads, self.config)
        self.layers = nn.ModuleList(
            HoloRouteLayer(self.config) for _ in range(self.config.message_layers)
        )
        self.decoders = nn.ModuleList(
            EventDecoder(self.config.hidden_dim, heads) for _ in range(4)
        )
        self.curvature = nn.Sequential(
            nn.Linear(4, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

    def diamond_error(
        self,
        state: torch.Tensor,
        graph: EventGraph,
        layer: HoloRouteLayer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not graph.diamonds.shape[1]:
            return (
                state.new_empty(0),
                torch.empty(0, dtype=torch.long, device=state.device),
            )

        start, depth_middle, relay_middle, end = graph.diamonds
        depth_relation = torch.full_like(start, DEPTH_RELATION)
        relay_a = 1 + graph.events.role[start]
        relay_b = 1 + graph.events.role[depth_middle]

        relay_then_depth = layer.relay.transport(
            state[start],
            state[relay_middle],
            relay_a,
        )
        relay_then_depth = layer.depth.transport(
            relay_then_depth,
            state[end],
            depth_relation,
        )
        depth_then_relay = layer.depth.transport(
            state[start],
            state[depth_middle],
            depth_relation,
        )
        depth_then_relay = layer.relay.transport(
            depth_then_relay,
            state[end],
            relay_b,
        )

        metadata = torch.stack(
            (
                graph.events.layer[start].float() / max(graph.layer_count - 1, 1),
                graph.events.lag[start].float().log1p(),
                graph.events.lag[end].float().log1p(),
                graph.events.role[start].float(),
            ),
            dim=-1,
        )
        expected = self.curvature(metadata)
        error = (relay_then_depth - depth_then_relay - expected).square().mean(dim=-1)
        return error, graph.event_query[end]

    def forward(
        self,
        graph: EventGraph,
        values: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        relay_keep: torch.Tensor | None = None,
        query_keep: torch.Tensor | None = None,
    ) -> ModelOutput:
        values = graph.events.value if values is None else values
        observed = graph.events.observed if observed is None else observed
        state = self.encoder(graph, values, observed)
        contexts = state.new_zeros((len(state), 3, state.shape[-1]))
        coverage = torch.zeros((len(state), 3), dtype=torch.bool, device=state.device)
        for layer in self.layers:
            state, contexts, coverage = layer(
                state,
                graph,
                relay_keep,
                query_keep,
            )

        views = (
            state,
            contexts[:, DEPTH_CONTEXT],
            contexts[:, RELAY_CONTEXT],
            contexts[:, QUERY_CONTEXT],
        )
        value_prediction: list[torch.Tensor] = []
        support_prediction: list[torch.Tensor] = []
        for decoder, view in zip(self.decoders, views, strict=True):
            value, support = decoder(view)
            value_prediction.append(value)
            support_prediction.append(support)

        holonomy, holonomy_token = self.diamond_error(
            state,
            graph,
            self.layers[-1],
        )
        return ModelOutput(
            state=state,
            predictions=Predictions(
                value=torch.stack(value_prediction, dim=1),
                support=torch.stack(support_prediction, dim=1),
            ),
            contexts=contexts,
            coverage=coverage,
            holonomy=holonomy,
            holonomy_token=holonomy_token,
        )
