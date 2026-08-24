"""Neural causal-path encoder for attention event graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from experiments.attention_holonomy_audit.graph import AttentionEventGraph

DEPTH_RELATION = 0
PROMPT_RELAY_RELATION = 1
RESPONSE_RELAY_RELATION = 2
NUM_RELATIONS = 3


def _segment_softmax(score: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if score.numel() == 0:
        return score
    maximum = score.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
    weight = torch.exp(score - maximum[index])
    normalizer = score.new_zeros(size)
    normalizer.index_add_(0, index, weight)
    return weight / normalizer[index].clamp_min(1e-12)


def _scatter_messages(
    message: torch.Tensor,
    target: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = message.new_zeros((size, message.shape[-1]))
    if message.numel():
        output.index_add_(0, target, message)
    return output


class HeadProfileEncoder(nn.Module):
    """Encode one event's complete layer-head attention profile."""

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        config,
    ) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.num_heads = int(num_heads)
        self.head_embedding = nn.Embedding(num_heads, hidden)
        self.layer_embedding = nn.Embedding(num_layers, hidden)
        self.role_embedding = nn.Embedding(2, hidden)
        self.lag_embedding = nn.Embedding(config.lag_buckets, hidden)
        self.value_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.head_encoder_heads,
            dim_feedforward=4 * hidden,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.head_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.head_encoder_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(hidden, 1)
        self.metadata = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, graph: AttentionEventGraph, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        events, heads = values.shape
        head_id = torch.arange(heads, device=values.device)
        token = self.value_encoder(
            torch.stack((torch.log1p(values), observed.float()), dim=-1)
        )
        token = token + self.head_embedding(head_id)[None]
        token = token + self.layer_embedding(graph.event_layer)[:, None]
        token = self.head_encoder(token)
        attention = torch.softmax(self.pool_score(token).squeeze(-1), dim=-1)
        pooled = (attention[..., None] * token).sum(dim=1)

        lag_bucket = torch.floor(torch.log2(graph.event_lag.float().clamp_min(1))).long()
        lag_bucket = lag_bucket.clamp_max(self.lag_embedding.num_embeddings - 1)
        query_position = graph.event_query.float() / max(graph.num_response_tokens - 1, 1)
        source_position = graph.event_source.float() / max(graph.num_tokens - 1, 1)
        metadata = torch.stack(
            (
                query_position,
                source_position,
                values.sum(dim=-1),
                observed.float().mean(dim=-1),
            ),
            dim=-1,
        )
        pooled = pooled + self.metadata(metadata)
        pooled = pooled + self.role_embedding(graph.event_role)
        pooled = pooled + self.lag_embedding(lag_bucket)
        return self.output_norm(pooled)


class LowRankTransport(nn.Module):
    """Relation-specific low-rank transport with event-conditioned gates."""

    def __init__(self, hidden_dim: int, rank: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.relation_embedding = nn.Embedding(NUM_RELATIONS, hidden_dim)
        self.left = nn.Parameter(torch.empty(NUM_RELATIONS, hidden_dim, rank))
        self.right = nn.Parameter(torch.empty(NUM_RELATIONS, hidden_dim, rank))
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        if source.numel() == 0 or not self.enabled:
            return source
        rel = self.relation_embedding(relation)
        right = self.right[relation]
        left = self.left[relation]
        coefficient = self.gate(torch.cat((source, target, rel), dim=-1))
        latent = torch.einsum("ed,edr->er", source, right) * coefficient
        update = torch.einsum("er,edr->ed", latent, left)
        return self.norm(source + update)


class RelationAggregator(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, transport_enabled: bool = True) -> None:
        super().__init__()
        self.transport = LowRankTransport(hidden_dim, rank, enabled=transport_enabled)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        relation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events = len(state)
        if edge_index.shape[1] == 0:
            return state.new_zeros(state.shape), state.new_zeros(events, dtype=torch.bool)
        source, target = edge_index
        message = self.transport(state[source], state[target], relation)
        score = self.score(torch.cat((message, state[target]), dim=-1)).squeeze(-1)
        weight = _segment_softmax(score, target, events)
        aggregate = _scatter_messages(message * weight[:, None], target, events)
        coverage = torch.zeros(events, dtype=torch.bool, device=state.device)
        coverage[target] = True
        return aggregate, coverage


class QuerySetMixer(nn.Module):
    """Inducing-point set attention over events entering the same query/layer."""

    def __init__(self, hidden_dim: int, inducing_points: int, dropout: float) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(inducing_points, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        state: torch.Tensor,
        graph: AttentionEventGraph,
        event_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = len(graph.query_ptr) - 1
        events = len(state)
        if groups <= 0 or events == 0:
            return state.new_zeros(state.shape), state.new_zeros(events, dtype=torch.bool)
        lengths = graph.query_ptr[1:] - graph.query_ptr[:-1]
        maximum = int(lengths.max().item()) if len(lengths) else 0
        padded = state.new_zeros((groups, maximum, state.shape[-1]))
        mask = torch.ones((groups, maximum), dtype=torch.bool, device=state.device)
        group_events: list[torch.Tensor] = []
        for group in range(groups):
            start = int(graph.query_ptr[group].item())
            stop = int(graph.query_ptr[group + 1].item())
            index = graph.query_event_index[start:stop]
            group_events.append(index)
            source_index = index if event_keep is None else index[event_keep[index]]
            if len(source_index):
                padded[group, : len(source_index)] = state[source_index]
                mask[group, : len(source_index)] = False
        valid_group = ~mask.all(dim=1)
        context = state.new_zeros((groups, state.shape[-1]))
        if bool(valid_group.any()):
            query = self.inducing[None].expand(int(valid_group.sum().item()), -1, -1)
            induced, _ = self.attention(
                query,
                padded[valid_group],
                padded[valid_group],
                key_padding_mask=mask[valid_group],
            )
            context[valid_group] = self.output(induced.mean(dim=1))
        event_context = state.new_zeros(state.shape)
        coverage = torch.zeros(events, dtype=torch.bool, device=state.device)
        for group, index in enumerate(group_events):
            if len(index):
                event_context[index] = context[group]
                kept = len(index) if event_keep is None else int(event_keep[index].sum().item())
                coverage[index] = kept > 0
        return event_context, coverage


class HoloRouteBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.use_depth = bool(config.use_depth)
        self.use_relay = bool(config.use_relay)
        self.use_query = bool(config.use_query)
        self.depth = RelationAggregator(hidden, config.transport_rank, config.use_transport)
        self.relay = RelationAggregator(hidden, config.transport_rank, config.use_transport)
        self.query = QuerySetMixer(
            hidden,
            config.query_inducing_points,
            config.dropout,
        )
        self.components = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(4))
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.feed_forward = nn.Sequential(
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
        graph: AttentionEventGraph,
        relay_keep: torch.Tensor | None = None,
        query_event_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        depth_relation = torch.full(
            (graph.depth_edge_index.shape[1],),
            DEPTH_RELATION,
            dtype=torch.long,
            device=state.device,
        )
        if self.use_depth:
            depth_context, depth_coverage = self.depth(
                state,
                graph.depth_edge_index,
                depth_relation,
            )
        else:
            depth_context = state.new_zeros(state.shape)
            depth_coverage = state.new_zeros(len(state), dtype=torch.bool)

        relay_edge = graph.relay_edge_index
        if relay_keep is not None:
            relay_edge = relay_edge[:, relay_keep]
        if relay_edge.shape[1]:
            predecessor = relay_edge[0]
            relay_relation = 1 + graph.event_role[predecessor]
        else:
            relay_relation = torch.empty(0, dtype=torch.long, device=state.device)
        if self.use_relay:
            relay_context, relay_coverage = self.relay(
                state,
                relay_edge,
                relay_relation,
            )
        else:
            relay_context = state.new_zeros(state.shape)
            relay_coverage = state.new_zeros(len(state), dtype=torch.bool)
        if self.use_query:
            query_context, query_coverage = self.query(
                state, graph, event_keep=query_event_keep
            )
        else:
            query_context = state.new_zeros(state.shape)
            query_coverage = state.new_zeros(len(state), dtype=torch.bool)

        raw = (state, depth_context, relay_context, query_context)
        component = [projection(value) for projection, value in zip(self.components, raw)]
        gate = torch.softmax(self.gate(torch.cat(raw, dim=-1)), dim=-1)
        updated = sum(gate[:, index : index + 1] * value for index, value in enumerate(component))
        updated = self.norm(updated + self.feed_forward(updated))
        return updated, {
            "depth": depth_context,
            "relay": relay_context,
            "query": query_context,
            "depth_coverage": depth_coverage,
            "relay_coverage": relay_coverage,
            "query_coverage": query_coverage,
            "gate": gate,
        }


@dataclass(frozen=True)
class HoloRouteOutput:
    state: torch.Tensor
    event_prediction: torch.Tensor
    depth_prediction: torch.Tensor
    path_prediction: torch.Tensor
    query_prediction: torch.Tensor
    depth_context: torch.Tensor
    relay_context: torch.Tensor
    query_context: torch.Tensor
    depth_coverage: torch.Tensor
    relay_coverage: torch.Tensor
    query_coverage: torch.Tensor
    context_disagreement: torch.Tensor
    holonomy_error: torch.Tensor
    holonomy_token: torch.Tensor
    gate: torch.Tensor


class HoloRouteEncoder(nn.Module):
    """Dual-axis neural encoder over attention events and causal paths."""

    def __init__(self, num_layers: int, num_heads: int, config) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.use_holonomy = bool(config.use_holonomy)
        hidden = config.hidden_dim
        self.event_encoder = HeadProfileEncoder(num_heads, num_layers, config)
        self.blocks = nn.ModuleList(HoloRouteBlock(config) for _ in range(config.message_blocks))
        self.event_decoder = nn.Linear(hidden, num_heads)
        self.depth_decoder = nn.Linear(hidden, num_heads)
        self.path_decoder = nn.Linear(hidden, num_heads)
        self.query_decoder = nn.Linear(hidden, num_heads)
        self.holonomy_predictor = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def _decode(self, state: torch.Tensor, decoder: nn.Linear) -> torch.Tensor:
        return torch.nn.functional.softplus(decoder(state))

    def _holonomy(
        self,
        state: torch.Tensor,
        graph: AttentionEventGraph,
        block: HoloRouteBlock,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_holonomy or graph.diamond_index.shape[1] == 0:
            return state.new_empty(0), torch.empty(0, dtype=torch.long, device=state.device)
        start, depth_middle, relay_middle, end = graph.diamond_index
        relay_a = 1 + graph.event_role[start]
        relay_b = 1 + graph.event_role[depth_middle]
        depth_type = torch.full_like(relay_a, DEPTH_RELATION)

        route_a = block.relay.transport(state[start], state[relay_middle], relay_a)
        route_a = block.depth.transport(route_a, state[end], depth_type)
        route_b = block.depth.transport(state[start], state[depth_middle], depth_type)
        route_b = block.relay.transport(route_b, state[end], relay_b)
        expected = self.holonomy_predictor(torch.cat((state[start], state[end]), dim=-1))
        residual = route_a - route_b - expected
        error = residual.square().mean(dim=-1)
        return error, graph.event_query[end]

    def forward(
        self,
        graph: AttentionEventGraph,
        *,
        event_head_value: torch.Tensor | None = None,
        event_head_observed: torch.Tensor | None = None,
        relay_keep: torch.Tensor | None = None,
        query_event_keep: torch.Tensor | None = None,
    ) -> HoloRouteOutput:
        value = graph.event_head_value if event_head_value is None else event_head_value
        observed = graph.event_head_observed if event_head_observed is None else event_head_observed
        state = self.event_encoder(graph, value, observed)
        context: dict[str, torch.Tensor] | None = None
        for block in self.blocks:
            state, context = block(
                state,
                graph,
                relay_keep=relay_keep,
                query_event_keep=query_event_keep,
            )
        assert context is not None

        both = context["depth_coverage"] & context["relay_coverage"]
        disagreement = state.new_zeros(len(state))
        if bool(both.any()):
            disagreement[both] = 1.0 - torch.nn.functional.cosine_similarity(
                context["depth"][both],
                context["relay"][both],
                dim=-1,
            )
        holonomy_error, holonomy_token = self._holonomy(state, graph, self.blocks[-1])
        return HoloRouteOutput(
            state=state,
            event_prediction=self._decode(state, self.event_decoder),
            depth_prediction=self._decode(context["depth"], self.depth_decoder),
            path_prediction=self._decode(context["relay"], self.path_decoder),
            query_prediction=self._decode(context["query"], self.query_decoder),
            depth_context=context["depth"],
            relay_context=context["relay"],
            query_context=context["query"],
            depth_coverage=context["depth_coverage"],
            relay_coverage=context["relay_coverage"],
            query_coverage=context["query_coverage"],
            context_disagreement=disagreement,
            holonomy_error=holonomy_error,
            holonomy_token=holonomy_token,
            gate=context["gate"],
        )
