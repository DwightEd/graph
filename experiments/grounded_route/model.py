"""Prefix-causal neural encoding on a typed token attention graph."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .graph import TokenGraph

PROMPT_ORIGIN = 0
RESPONSE_CLOSED = 1
UNRESOLVED = 2
LINEAGE_STATES = 3


def lag_bucket(lag: torch.Tensor, bucket_count: int) -> torch.Tensor:
    return torch.floor(torch.log2(lag.float().clamp_min(1))).long().clamp_max(
        bucket_count - 1
    )


class HeadTransition(nn.Module):
    """A layer-specific row-stochastic correspondence between attention heads."""

    def __init__(self, layers: int, heads: int, identity_bias: float) -> None:
        super().__init__()
        initial = torch.eye(heads).expand(layers, -1, -1) * identity_bias
        self.logit = nn.Parameter(initial.clone())

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logit, dim=-1)


def source_lineage(
    graph: TokenGraph,
    previous: torch.Tensor | None,
    transition: torch.Tensor,
    source: torch.Tensor,
    head: torch.Tensor,
) -> torch.Tensor:
    """Return provenance for prompt and response endpoints in original order."""

    reference = transition if previous is None else previous
    prompt_state = reference.new_tensor((1.0, 0.0, 0.0)).expand(len(source), -1)
    if previous is None:
        response_state = reference.new_tensor((0.0, 1.0, 0.0)).expand(len(source), -1)
    else:
        response_index = (source - graph.response_start).clamp_min(0)
        response_state = torch.einsum(
            "eh,ehk->ek",
            transition[head],
            previous[response_index],
        )
    return torch.where(
        (source < graph.response_start)[:, None],
        prompt_state,
        response_state,
    )


def trace_lineage(graph: TokenGraph, transition: torch.Tensor) -> torch.Tensor:
    """Propagate conserved prompt, closed-response and unresolved provenance."""

    graph = graph.canonicalize()
    history: list[torch.Tensor] = []
    for layer in range(graph.layer_count):
        edges = graph.layer_edges(layer, transition.device)
        current, _ = lineage_layer(
            graph,
            history[-1] if history else None,
            transition[layer],
            layer,
            edges,
        )
        history.append(current)

    return torch.stack(history, dim=1)


def lineage_layer(
    graph: TokenGraph,
    previous: torch.Tensor | None,
    transition: torch.Tensor,
    layer: int,
    edges,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate one layer and return target and retained-source lineage."""

    current = transition.new_zeros(
        (graph.response_count, graph.head_count, LINEAGE_STATES)
    )
    if edges.count:
        provenance = source_lineage(
            graph,
            previous,
            transition,
            edges.source,
            edges.head,
        )
        flat_target = (
            edges.target - graph.response_start
        ) * graph.head_count + edges.head
        current = current.view(-1, LINEAGE_STATES).index_add(
            0,
            flat_target,
            provenance * edges.weight[:, None],
        ).view(graph.response_count, graph.head_count, LINEAGE_STATES)
    else:
        provenance = current.new_empty((0, LINEAGE_STATES))

    if previous is None:
        previous_target = current.new_zeros(current.shape)
        previous_target[..., RESPONSE_CLOSED] = 1.0
    else:
        previous_target = torch.einsum("hj,rjk->rhk", transition, previous)
    diagonal = graph.diagonal[:, layer].to(device=current.device)
    unresolved = graph.unresolved[:, layer].to(device=current.device)
    current = current + diagonal[..., None] * previous_target
    unresolved_state = current.new_tensor((0.0, 0.0, 1.0))
    current = current + unresolved[..., None] * unresolved_state
    return current, provenance


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
    """Encode clean token graphs and predict endpoints from shifted prefixes."""

    def __init__(
        self,
        layers: int,
        heads: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        hidden = self.config.hidden_dim
        edge_hidden = self.config.edge_hidden_dim
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
        self.lineage = nn.Linear(LINEAGE_STATES, hidden, bias=False)
        self.edge_value = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_message = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, edge_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(edge_hidden, hidden),
        )
        self.self_message = nn.Linear(hidden, hidden)
        self.unresolved_message = nn.Parameter(torch.empty(layers, heads, hidden))
        nn.init.normal_(self.unresolved_message, std=0.02)

        self.head_query = nn.Linear(hidden, hidden, bias=False)
        self.head_key = nn.Linear(hidden, hidden, bias=False)
        self.head_score = nn.Linear(hidden, 1, bias=False)
        self.updates = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(layers))
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
        distance = torch.arange(
            graph.response_start - 1,
            -1,
            -1,
            device=graph.device,
        )
        prompt = sinusoidal_position(distance, hidden) + self.prompt_role
        response = self.response_role[None].expand(graph.response_count, -1)
        return torch.cat((prompt, response), dim=0)

    def edge_source_lineage(
        self,
        graph: TokenGraph,
        lineage: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
        source: torch.Tensor,
        head: torch.Tensor,
    ) -> torch.Tensor:
        return source_lineage(
            graph,
            lineage[:, layer - 1] if layer else None,
            transition[layer],
            source,
            head,
        )

    def propagate_layer(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        lineage: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        edges = graph.layer_edges(layer, state.device)
        provenance = source_lineage(
            graph,
            lineage[:, layer - 1] if layer else None,
            transition[layer],
            edges.source,
            edges.head,
        )
        return self.propagate_edges(graph, state, edges, provenance, layer)

    def propagate_edges(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        edges,
        provenance: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Update token states from one device-resident layer slice."""

        hidden = state.shape[-1]
        cells = state.new_zeros((graph.response_count, graph.head_count, hidden))
        if edges.count:
            source = edges.source
            target = edges.target - graph.response_start
            head = edges.head
            weight = edges.weight
            role = (source >= graph.response_start).long()
            message = state[source]
            message = message + self.layer_id(edges.layer)
            message = message + self.head_id(head)
            message = message + self.source_role(role)
            message = message + self.lag_id(
                lag_bucket(edges.target - source, self.config.lag_buckets)
            )
            message = message + self.lineage(provenance)
            message = message + self.edge_value(torch.log1p(weight)[:, None])
            message = self.edge_message(message) * weight[:, None]
            cells = cells.view(-1, hidden).index_add(
                0,
                target * graph.head_count + head,
                message,
            ).view(graph.response_count, graph.head_count, hidden)

        response_state = state[graph.response_start :]
        self_value = self.self_message(response_state)[:, None, :]
        diagonal = graph.diagonal[:, layer].to(device=state.device)
        unresolved = graph.unresolved[:, layer].to(device=state.device)
        cells = cells + diagonal[..., None] * self_value
        cells = cells + (
            unresolved[..., None]
            * self.unresolved_message[layer][None]
        )
        cells = cells + self.head_id.weight[None]

        query = self.head_query(response_state)[:, None]
        score = self.head_score(torch.tanh(query + self.head_key(cells))).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        context = (cells * weight[..., None]).sum(dim=1)
        updated = self.updates[layer](context, response_state)

        return torch.cat((state[: graph.response_start], updated), dim=0)

    def layer_step(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        previous_lineage: torch.Tensor,
        transition: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfer, consume, and release one sparse layer."""

        edges = graph.layer_edges(layer, state.device)
        current_lineage, provenance = lineage_layer(
            graph,
            None if layer == 0 else previous_lineage,
            transition,
            layer,
            edges,
        )
        state = self.propagate_edges(graph, state, edges, provenance, layer)
        return state, current_lineage

    def shifted_prefix(self, response_embedding: torch.Tensor) -> torch.Tensor:
        if not len(response_embedding):
            return response_embedding
        seed = self.prefix_seed.view(1, 1, -1)
        sequence, _ = self.prefix_recurrence(response_embedding[None], seed.transpose(0, 1))
        return torch.cat((self.prefix_seed[None], sequence[0, :-1]), dim=0)

    def forward(self, graph: TokenGraph) -> EncoderOutput:
        graph = graph.canonicalize()
        transition = self.head_transition()
        state = self.initial_nodes(graph)
        previous_lineage = state.new_zeros(
            (graph.response_count, graph.head_count, LINEAGE_STATES)
        )
        lineage_history: list[torch.Tensor] = []
        for layer in range(graph.layer_count):
            def current_step(state, previous, correspondence, current_layer=layer):
                return self.layer_step(
                    graph,
                    state,
                    previous,
                    correspondence,
                    current_layer,
                )

            if self.training and torch.is_grad_enabled():
                state, previous_lineage = checkpoint(
                    current_step,
                    state,
                    previous_lineage,
                    transition[layer],
                    use_reentrant=False,
                )
            else:
                state, previous_lineage = current_step(
                    state,
                    previous_lineage,
                    transition[layer],
                )
            lineage_history.append(previous_lineage)
        lineage = torch.stack(lineage_history, dim=1)
        node_embedding = self.output_norm(state)
        response_embedding = node_embedding[graph.response_start :]
        prefix_state = self.shifted_prefix(response_embedding)
        return EncoderOutput(
            node_embedding=node_embedding,
            response_embedding=response_embedding,
            prefix_state=prefix_state,
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
        """Score candidate endpoints without reading the current attention row."""

        device = output.node_embedding.device
        source = source.to(device=device)
        target = target.to(device=device)
        layer = layer.to(device=device)
        head = head.to(device=device)
        prefix = output.prefix_state[target - graph.response_start]
        query = prefix + self.layer_id(layer) + self.head_id(head)
        query = self.route_query(query)
        key = self.route_key(output.node_embedding[source])
        return (query * key).sum(dim=-1) / math.sqrt(self.config.hidden_dim)
