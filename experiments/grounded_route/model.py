"""Layer-wise encoding of typed token attention graphs."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .aggregation import RouteAggregator, lag_bucket
from .config import ModelConfig
from .graph import TokenGraph
from .lineage import (
    HeadTransition,
    LINEAGE_STATES,
    lineage_layer,
    source_lineage,
    trace_lineage,
)


def sinusoidal_position(position: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        torch.arange(half, device=position.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angle = position.float()[:, None] * frequency[None]
    encoding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
    if dimension % 2:
        encoding = torch.nn.functional.pad(encoding, (0, 1))
    return encoding


@dataclass(frozen=True)
class EncoderOutput:
    node_embedding: torch.Tensor
    response_embedding: torch.Tensor
    prefix_state: torch.Tensor
    lineage: torch.Tensor


class GroundedRouteEncoder(nn.Module):
    """Aggregate exact typed neighbours into one reusable token embedding."""

    def __init__(
        self,
        layers: int,
        heads: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        hidden = self.config.hidden_dim

        self.layer_count = int(layers)
        self.head_count = int(heads)
        self.head_transition = HeadTransition(
            layers,
            heads,
            self.config.head_transition_identity_bias,
        )

        self.prompt_role = nn.Parameter(torch.empty(hidden))
        self.response_role = nn.Parameter(torch.empty(hidden))
        self.prefix_seed = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.prompt_role, std=0.02)
        nn.init.normal_(self.response_role, std=0.02)
        nn.init.normal_(self.prefix_seed, std=0.02)

        self.layer_id = nn.Embedding(layers, hidden)
        self.head_id = nn.Embedding(heads, hidden)
        self.source_role = nn.Embedding(2, hidden)
        self.lag_id = nn.Embedding(self.config.lag_buckets, hidden)
        self.lineage_value = nn.Linear(LINEAGE_STATES, hidden, bias=False)
        self.edge_message = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.config.edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.edge_hidden_dim, hidden),
        )
        self.aggregate = RouteAggregator(layers, heads, hidden)
        self.update = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(layers))
        self.output_norm = nn.LayerNorm(hidden)
        self.prefix_recurrence = nn.GRU(hidden, hidden, batch_first=True)

        self.route_query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.route_key = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden, bias=False),
        )

    def initial_nodes(self, graph: TokenGraph) -> torch.Tensor:
        hidden = self.config.hidden_dim
        prompt_distance = torch.arange(
            graph.response_start - 1,
            -1,
            -1,
            device=graph.device,
        )
        prompt = sinusoidal_position(prompt_distance, hidden) + self.prompt_role
        response = self.response_role[None].expand(graph.response_count, -1)
        return torch.cat((prompt, response), dim=0)

    def edge_messages(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        edges,
        provenance: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        target = edges.target - graph.response_start
        role = (edges.source >= graph.response_start).long()
        if self.config.message_mode == "neighbor":
            base = state[edges.source]
            lineage_value = provenance
            head_context = self.head_id(edges.head)
        else:
            base = state[graph.response_start + target]
            lineage_value = provenance.new_full(provenance.shape, 1.0 / LINEAGE_STATES)
            head_context = (transition @ self.head_id.weight)[edges.head]

        message = base
        message = message + self.layer_id(edges.layer)
        message = message + head_context
        message = message + self.source_role(role)
        message = message + self.lag_id(
            lag_bucket(edges.target - edges.source, self.config.lag_buckets)
        )
        message = message + self.lineage_value(lineage_value)
        return self.edge_message(message)

    def layer_step(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        previous_lineage: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = graph.layer_edges(layer, state.device)
        current_lineage, provenance = lineage_layer(
            graph,
            None if layer == 0 else previous_lineage,
            transition,
            layer,
            edges,
        )
        message = self.edge_messages(
            graph, state, edges, provenance, transition, layer
        )
        response_state = state[graph.response_start :]
        role = (edges.source >= graph.response_start).long()
        context = self.aggregate(
            response_state=response_state,
            message=message,
            weight=edges.weight,
            target=edges.target - graph.response_start,
            head=edges.head,
            role=role,
            diagonal=graph.diagonal[:, layer].to(state.device),
            unresolved=graph.unresolved[:, layer].to(state.device),
            head_identity=self.head_id.weight,
            layer=layer,
        )
        updated = self.update[layer](context, response_state)
        state = torch.cat((state[: graph.response_start], updated), dim=0)
        return state, current_lineage

    def apply_layer(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        previous_lineage: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def step(current_state, current_lineage, current_transition):
            return self.layer_step(
                graph,
                current_state,
                current_lineage,
                current_transition,
                layer,
            )

        if self.training and torch.is_grad_enabled():
            return checkpoint(
                step,
                state,
                previous_lineage,
                transition,
                use_reentrant=False,
            )
        return step(state, previous_lineage, transition)

    def shifted_prefix(self, response_embedding: torch.Tensor) -> torch.Tensor:
        if not len(response_embedding):
            return response_embedding
        sequence, _ = self.prefix_recurrence(
            response_embedding[None],
            self.prefix_seed.view(1, 1, -1),
        )
        return torch.cat((self.prefix_seed[None], sequence[0, :-1]), dim=0)

    def forward(self, graph: TokenGraph) -> EncoderOutput:
        graph = graph.canonicalize()
        transition = self.head_transition()
        state = self.initial_nodes(graph)
        previous_lineage = state.new_zeros(
            (graph.response_count, graph.head_count, LINEAGE_STATES)
        )
        lineage_history = []

        for layer in range(graph.layer_count):
            state, previous_lineage = self.apply_layer(
                graph,
                state,
                previous_lineage,
                transition[layer],
                layer,
            )
            lineage_history.append(previous_lineage)

        lineage = torch.stack(lineage_history, dim=1)
        node_embedding = self.output_norm(state)
        response_embedding = node_embedding[graph.response_start :]
        return EncoderOutput(
            node_embedding=node_embedding,
            response_embedding=response_embedding,
            prefix_state=self.shifted_prefix(response_embedding),
            lineage=lineage,
        )

    def encode(self, graph: TokenGraph) -> EncoderOutput:
        return self(graph)

    def endpoint_score(
        self,
        output: EncoderOutput,
        graph: TokenGraph,
        source: torch.Tensor,
        target: torch.Tensor,
        layer: torch.Tensor,
        head: torch.Tensor,
    ) -> torch.Tensor:
        source = source.to(output.node_embedding.device)
        target = target.to(output.node_embedding.device)
        layer = layer.to(output.node_embedding.device)
        head = head.to(output.node_embedding.device)
        prefix = output.prefix_state[target - graph.response_start]
        query = self.route_query(prefix + self.layer_id(layer) + self.head_id(head))
        key = self.route_key(output.node_embedding[source])
        return (query * key).sum(dim=-1) / math.sqrt(self.config.hidden_dim)


__all__ = [
    "EncoderOutput",
    "GroundedRouteEncoder",
    "HeadTransition",
    "lag_bucket",
    "source_lineage",
    "trace_lineage",
]
