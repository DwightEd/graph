"""Directed node-to-row-hyperedge-to-target message passing."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.grounded_route.aggregation import lag_bucket
from experiments.grounded_route.graph import TokenGraph
from experiments.grounded_route.lineage import LINEAGE_STATES, UNRESOLVED
from experiments.grounded_route.model import sinusoidal_position

from .config import ModelConfig
from .flow import flow_step, initial_flow
from .hypergraph import DirectedLayerHypergraph, layer_hypergraph


@dataclass(frozen=True)
class EncoderOutput:
    node_embedding: torch.Tensor
    response_embedding: torch.Tensor
    lineage: torch.Tensor
    flow_logits: torch.Tensor
    layer_input: torch.Tensor | None = None


def segment_softmax(
    value: torch.Tensor,
    group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Softmax independently inside every incidence-slot group."""

    if not len(value):
        return value
    maximum = value.new_full((group_count,), -torch.inf)
    maximum.scatter_reduce_(0, group, value, reduce="amax", include_self=True)
    shifted = value - maximum[group]
    normalizer = value.new_zeros(group_count)
    normalizer.index_add_(0, group, shifted.exp())
    return shifted.exp() / normalizer[group].clamp_min(1e-12)


class SourceToHyperedge(nn.Module):
    """Pool prompt-rooted and response-closed path mass into paired slots."""

    def __init__(self, layers: int, heads: int, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        slots = config.slots_per_role
        slot_dim = config.slot_dim
        self.config = config

        self.layer_id = nn.Embedding(layers, hidden)
        self.head_id = nn.Embedding(heads, hidden)
        self.lag_id = nn.Embedding(config.lag_buckets, hidden)
        self.source_projection = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, config.edge_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.edge_hidden_dim, slots * slot_dim),
            )
            for _ in range(2)
        )
        self.source_role = nn.Embedding(2, hidden)
        self.lineage_value = nn.Linear(LINEAGE_STATES, hidden, bias=False)
        self.message_norm = nn.LayerNorm(slot_dim)
        self.slot_key = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_value = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_query = nn.Parameter(torch.empty(2, slots, slot_dim))
        self.self_message = nn.Linear(slot_dim, slot_dim)
        self.unresolved_message = nn.Parameter(
            torch.empty(layers, heads, config.slot_count, slot_dim)
        )
        nn.init.normal_(self.slot_query, std=0.02)
        nn.init.normal_(self.unresolved_message, std=0.02)

    def forward(
        self,
        node_state: torch.Tensor,
        view: DirectedLayerHypergraph,
        provenance: torch.Tensor,
        target_provenance: torch.Tensor,
        head_flow: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        edge_count = view.incidence_count
        device = node_state.device
        local_slot = torch.arange(config.slots_per_role, device=device)

        message = node_state.new_zeros(
            (edge_count, config.slot_count, config.slot_dim)
        )
        source = node_state[view.source].flatten(1)
        lineage = self.lineage_value(provenance).view(
            edge_count,
            config.slot_count,
            config.slot_dim,
        )
        source_role = self.source_role(view.role).view(
            edge_count,
            config.slot_count,
            config.slot_dim,
        )
        layer_slot = self.layer_id.weight[view.layer].view(
            config.slot_count,
            config.slot_dim,
        )
        head_slot = self.head_id(view.head[view.hyperedge]).view(
            edge_count,
            config.slot_count,
            config.slot_dim,
        )
        target = view.target[view.hyperedge]
        lag_slot = self.lag_id(
            lag_bucket(target - view.source, config.lag_buckets)
        ).view(edge_count, config.slot_count, config.slot_dim)

        for route in range(2):
            route_slot = route * config.slots_per_role + local_slot
            projected = self.source_projection[route](source).view(
                edge_count,
                config.slots_per_role,
                config.slot_dim,
            )
            message[:, route_slot] = (
                projected
                + lineage[:, route_slot]
                + source_role[:, route_slot]
                + layer_slot[route_slot][None]
                + head_slot[:, route_slot]
                + lag_slot[:, route_slot]
            )

        message = self.message_norm(message)
        query = self.slot_query.reshape(config.slot_count, config.slot_dim)[None]
        logit = (self.slot_key(message) * query).sum(dim=-1)
        logit = logit / math.sqrt(config.slot_dim)
        route_weight = view.weight[:, None] * provenance[:, :2]
        slot_weight = route_weight.repeat_interleave(config.slots_per_role, dim=1)
        logit = logit + slot_weight.clamp_min(1e-12).log()

        global_slot = torch.arange(config.slot_count, device=device)
        group = view.hyperedge[:, None] * config.slot_count + global_slot[None]
        attention = segment_softmax(
            logit.flatten(),
            group.flatten(),
            view.hyperedge_count * config.slot_count,
        ).view(edge_count, config.slot_count)

        hyperedge = node_state.new_zeros(
            (view.hyperedge_count * config.slot_count, config.slot_dim)
        )
        hyperedge.index_add_(
            0,
            group.flatten(),
            (self.slot_value(message) * attention[..., None]).flatten(0, 1),
        )
        hyperedge = hyperedge.view(
            view.hyperedge_count,
            config.slot_count,
            config.slot_dim,
        )

        route_mass = view.weight.new_zeros((view.hyperedge_count, 2))
        route_mass.index_add_(0, view.hyperedge, route_weight)
        slot_mass = route_mass.repeat_interleave(config.slots_per_role, dim=1)
        hyperedge = hyperedge * slot_mass[..., None]

        target_state = node_state[view.target]
        diagonal_route = view.diagonal[:, None] * target_provenance[:, :2]
        diagonal_slot = diagonal_route.repeat_interleave(
            config.slots_per_role,
            dim=1,
        )
        hyperedge = hyperedge + diagonal_slot[..., None] * self.self_message(
            target_state
        )
        hyperedge = hyperedge + head_flow[:, UNRESOLVED, None, None] * (
            self.unresolved_message[view.layer, view.head]
        )
        hyperedge = hyperedge + layer_slot[None] + self.head_id(view.head).view(
            view.hyperedge_count,
            config.slot_count,
            config.slot_dim,
        )
        return hyperedge


class HyperedgeToTarget(nn.Module):
    """Pool head-specific row hyperedges and update four target slots."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        slot_dim = config.slot_dim
        self.config = config
        self.query = nn.Linear(slot_dim, slot_dim, bias=False)
        self.key = nn.Linear(slot_dim, slot_dim, bias=False)
        self.score = nn.Linear(slot_dim, 1, bias=False)
        self.update = nn.ModuleList(
            nn.GRUCell(slot_dim, slot_dim) for _ in range(config.slot_count)
        )

    def forward(
        self,
        response_state: torch.Tensor,
        hyperedge_state: torch.Tensor,
        head_count: int,
    ) -> torch.Tensor:
        config = self.config
        rows = hyperedge_state.view(
            len(response_state),
            head_count,
            config.slot_count,
            config.slot_dim,
        )
        query = self.query(response_state)[:, None]
        score = self.score(torch.tanh(query + self.key(rows))).squeeze(-1)
        attention = torch.softmax(score, dim=1)
        context = (rows * attention[..., None]).sum(dim=1)
        return torch.stack(
            [
                self.update[slot](context[:, slot], response_state[:, slot])
                for slot in range(config.slot_count)
            ],
            dim=1,
        )


class DirectedRouteHypergraphEncoder(nn.Module):
    """Encode typed attention rows through explicit directed hyperedges."""

    def __init__(
        self,
        layers: int,
        heads: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.layer_count = int(layers)
        self.head_count = int(heads)
        hidden = self.config.hidden_dim

        self.prompt_role = nn.Parameter(torch.empty(hidden))
        self.response_role = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.prompt_role, std=0.02)
        nn.init.normal_(self.response_role, std=0.02)

        self.source_to_hyperedge = SourceToHyperedge(layers, heads, self.config)
        self.hyperedge_to_target = HyperedgeToTarget(self.config)
        self.output_norm = nn.LayerNorm(hidden)
        self.flow_readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, LINEAGE_STATES),
        )
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
        self.route_role = nn.Embedding(2, hidden)
        self.route_lag = nn.Embedding(self.config.lag_buckets, hidden)
        self.bucket_key = nn.Parameter(torch.empty(2, hidden))
        nn.init.normal_(self.bucket_key, std=0.02)

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
        return torch.cat((prompt, response), dim=0).view(
            graph.token_count,
            self.config.slot_count,
            self.config.slot_dim,
        )

    def layer_step(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        token_flow: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consume one Transformer layer without retaining duplicate edge copies."""

        edges = graph.layer_edges(layer, state.device)
        route = flow_step(
            graph,
            token_flow,
            layer,
            residual_weight=self.config.residual_weight,
            edges=edges,
        )
        view = layer_hypergraph(graph, layer, state.device, edges)
        hyperedge = self.source_to_hyperedge(
            state,
            view,
            route.provenance,
            token_flow[view.target],
            route.head_flow.reshape(view.hyperedge_count, LINEAGE_STATES),
        )
        updated = self.hyperedge_to_target(
            state[graph.response_start :],
            hyperedge,
            graph.head_count,
        )
        state = torch.cat((state[: graph.response_start], updated), dim=0)
        return state, route.token_state, route.head_flow

    def apply_layer(
        self,
        graph: TokenGraph,
        state: torch.Tensor,
        token_flow: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Checkpoint edge-sized activations while fitting full-depth graphs."""

        def step(current_state, current_flow):
            return self.layer_step(
                graph,
                current_state,
                current_flow,
                layer,
            )

        if self.training and torch.is_grad_enabled():
            return checkpoint(
                step,
                state,
                token_flow,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return step(state, token_flow)

    def forward(
        self,
        graph: TokenGraph,
        return_layer_input: bool = False,
    ) -> EncoderOutput:
        graph = graph.canonicalize()
        state = self.initial_nodes(graph)
        token_flow = initial_flow(graph).to(state.device, dtype=state.dtype)
        lineage_history = []
        flow_logits = []
        layer_inputs = []

        for layer in range(graph.layer_count):
            if return_layer_input:
                layer_inputs.append(state)
            state, token_flow, head_flow = self.apply_layer(
                graph,
                state,
                token_flow,
                layer,
            )
            lineage_history.append(head_flow)
            flow_logits.append(
                self.flow_readout(state[graph.response_start :].flatten(1))
            )

        flat_state = state.flatten(1)
        node_embedding = self.output_norm(flat_state)
        layer_input = torch.stack(layer_inputs) if return_layer_input else None
        return EncoderOutput(
            node_embedding=node_embedding,
            response_embedding=node_embedding[graph.response_start :],
            lineage=torch.stack(lineage_history, dim=1),
            flow_logits=torch.stack(flow_logits, dim=1),
            layer_input=layer_input,
        )

    def encode(self, graph: TokenGraph) -> EncoderOutput:
        return self(graph)

    def route_query_state(
        self,
        output: EncoderOutput,
        target: torch.Tensor,
        layer: torch.Tensor,
        head: torch.Tensor,
    ) -> torch.Tensor:
        if output.layer_input is None:
            raise ValueError("row scoring requires return_layer_input=True")
        target_state = output.layer_input[layer, target].flatten(1)
        layer_context = self.source_to_hyperedge.layer_id(layer)
        head_context = self.source_to_hyperedge.head_id(head)
        return self.route_query(target_state + layer_context + head_context)

    def endpoint_score(
        self,
        output: EncoderOutput,
        graph: TokenGraph,
        source: torch.Tensor,
        target: torch.Tensor,
        layer: torch.Tensor,
        head: torch.Tensor,
    ) -> torch.Tensor:
        device = output.node_embedding.device
        source = source.to(device)
        target = target.to(device)
        layer = layer.to(device)
        head = head.to(device)
        query = self.route_query_state(output, target, layer, head)
        source_state = output.layer_input[layer, source].flatten(1)
        role = (source >= graph.response_start).long()
        lag = lag_bucket(target - source, self.config.lag_buckets)
        key = self.route_key(
            source_state + self.route_role(role) + self.route_lag(lag)
        )
        return (query * key).sum(dim=-1) / math.sqrt(self.config.hidden_dim)

    def bucket_score(
        self,
        output: EncoderOutput,
        graph: TokenGraph,
        target: torch.Tensor,
        layer: torch.Tensor,
        head: torch.Tensor,
        bucket: str,
    ) -> torch.Tensor:
        if bucket not in {"self", "unresolved"}:
            raise ValueError("bucket must be 'self' or 'unresolved'")
        device = output.node_embedding.device
        target = target.to(device)
        layer = layer.to(device)
        head = head.to(device)
        query = self.route_query_state(output, target, layer, head)
        bucket_index = 0 if bucket == "self" else 1
        return (query * self.bucket_key[bucket_index]).sum(dim=-1) / math.sqrt(
            self.config.hidden_dim
        )
