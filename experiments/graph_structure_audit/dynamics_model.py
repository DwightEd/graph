"""Cross-layer routing dynamics with prompt/response-specific transport."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics_config import DynamicsConfig
from .graph_data import MultiplexGraph


@dataclass(frozen=True)
class DynamicsOutput:
    loss: torch.Tensor
    valid: torch.Tensor
    token_loss: torch.Tensor
    edge_loss: torch.Tensor
    prompt_edge_loss: torch.Tensor
    response_edge_loss: torch.Tensor
    diagonal_loss: torch.Tensor
    support_loss: torch.Tensor
    edge_error_map: torch.Tensor
    prompt_edge_error_map: torch.Tensor
    response_edge_error_map: torch.Tensor
    diagonal_error_map: torch.Tensor
    support_error_map: torch.Tensor
    edge_count_map: torch.Tensor
    prompt_edge_count_map: torch.Tensor
    response_edge_count_map: torch.Tensor
    self_gate: torch.Tensor
    prompt_gate: torch.Tensor
    response_gate: torch.Tensor
    embedding: torch.Tensor


class RoutingTransitionCell(nn.Module):
    """Update a token from self, prompt, and response routing components."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_path = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_path = nn.Linear(hidden_dim, hidden_dim)
        self.response_path = nn.Linear(hidden_dim, hidden_dim)
        self.gates = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        state: torch.Tensor,
        local: torch.Tensor,
        prompt: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = F.softmax(
            self.gates(torch.cat((state, local, prompt, response), dim=-1)),
            dim=-1,
        )
        candidates = torch.stack(
            (
                torch.tanh(self.self_path(state + local)),
                torch.tanh(self.prompt_path(prompt)),
                torch.tanh(self.response_path(response)),
            ),
            dim=1,
        )
        updated = (gate.unsqueeze(-1) * candidates).sum(dim=1)
        return updated, gate


class CrossOriginRoutingDynamics(nn.Module):
    """Predict the next layer's routing graph from the current layer graph."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: DynamicsConfig | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.config = DynamicsConfig() if config is None else config
        hidden = self.config.hidden_dim

        self.role = nn.Embedding(2, self.config.role_dim)
        self.position = nn.Sequential(
            nn.Linear(1, self.config.position_dim),
            nn.PReLU(),
        )
        self.lag = nn.Embedding(self.config.lag_bins, self.config.position_dim)
        self.node_init = nn.Sequential(
            nn.Linear(self.config.role_dim + self.config.position_dim, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        edge_input = (
            2 * num_heads + self.config.role_dim + self.config.position_dim
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input, hidden),
            nn.PReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        node_input = (
            2 * num_heads + self.config.role_dim + self.config.position_dim
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.prompt_message = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.response_message = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_weight = nn.Linear(hidden, 1)
        self.aggregate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.transition = RoutingTransitionCell(hidden)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.PReLU(),
            nn.Linear(hidden * 2, num_heads),
        )
        self.support_decoder = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.PReLU(),
            nn.Linear(hidden * 2, num_heads),
        )
        self.diagonal_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, num_heads),
        )

    def _lag_bin(self, graph: MultiplexGraph) -> torch.Tensor:
        source, target = graph.edge_index
        prompt = source < graph.response_idx
        prompt_bin = torch.floor(
            source.float()
            / float(max(graph.response_idx, 1))
            * self.config.lag_bins
        ).long().clamp_max(self.config.lag_bins - 1)
        response_bin = torch.floor(
            torch.log2((target - source).clamp_min(1).float())
        ).long().clamp_max(self.config.lag_bins - 1)
        return torch.where(prompt, prompt_bin, response_bin)

    def _relation_aggregate(
        self,
        graph: MultiplexGraph,
        edge_local: torch.Tensor,
        states: torch.Tensor,
        edge_value: torch.Tensor,
        relation: int,
    ) -> torch.Tensor:
        source, target = graph.edge_index
        selected = graph.node_role[source] == relation
        if not bool(selected.any()):
            return states.new_zeros(states.shape)
        local = edge_local[selected]
        source_state = states[source[selected]]
        encoder = (
            self.prompt_message if relation == 0 else self.response_message
        )
        message = encoder(torch.cat((source_state, local), dim=-1))
        observed_mass = edge_value[selected].sum(dim=-1)
        learned = F.softplus(self.message_weight(local).squeeze(-1))
        weight = learned * (observed_mass + 1e-4)

        total = states.new_zeros(states.shape)
        square = states.new_zeros(states.shape)
        normalizer = weight.new_zeros(graph.num_nodes)
        total.index_add_(0, target[selected], message * weight[:, None])
        square.index_add_(
            0,
            target[selected],
            message.square() * weight[:, None],
        )
        normalizer.index_add_(0, target[selected], weight)
        mean = total / normalizer.clamp_min(1e-8)[:, None]
        variance = (
            square / normalizer.clamp_min(1e-8)[:, None] - mean.square()
        )
        return self.aggregate(
            torch.cat(
                (mean, (variance.clamp_min(0) + 1e-6).sqrt()),
                dim=-1,
            )
        )

    @staticmethod
    def _token_map(
        graph: MultiplexGraph,
        edge_value: torch.Tensor,
        edge_mask: torch.Tensor,
        *,
        relation: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = graph.edge_index
        selected = torch.ones(
            graph.num_edges,
            dtype=torch.bool,
            device=graph.device,
        )
        if relation is not None:
            selected = graph.node_role[source] == relation
        edge_value = edge_value[selected]
        edge_mask = edge_mask[selected]
        token = target[selected] - graph.response_idx
        shape = (graph.num_response_tokens, edge_value.shape[-1])
        total = edge_value.new_zeros(shape)
        count = edge_value.new_zeros(shape)
        total.index_add_(0, token, edge_value * edge_mask)
        count.index_add_(0, token, edge_mask.float())
        return total / count.clamp_min(1.0), count

    def forward(
        self,
        graph: MultiplexGraph,
        *,
        message_mode: str = "full",
        input_dropout: bool | None = None,
    ) -> DynamicsOutput:
        source, target = graph.edge_index
        role = graph.node_role
        position = self.position(graph.node_position[:, None])
        states = self.node_init(
            torch.cat((self.role(role), position), dim=-1)
        )
        edge_role = self.role(role[source])
        lag = self.lag(self._lag_bin(graph))
        response = slice(graph.response_idx, graph.num_nodes)
        transitions = self.num_layers - 1
        use_dropout = self.training if input_dropout is None else input_dropout

        edge_maps = []
        prompt_maps = []
        response_maps = []
        edge_counts = []
        prompt_counts = []
        response_counts = []
        diagonal_maps = []
        support_maps = []
        gate_maps = []

        for layer in range(transitions):
            edge_value = graph.edge_attr[:, layer]
            edge_visible = graph.edge_observed[:, layer]
            diagonal_value = graph.diagonal[:, layer]
            diagonal_visible = graph.diagonal_observed[:, layer]
            if use_dropout and self.config.input_dropout > 0:
                keep_edge = (
                    torch.rand_like(edge_value)
                    >= self.config.input_dropout
                )
                keep_diagonal = (
                    torch.rand_like(diagonal_value)
                    >= self.config.input_dropout
                )
                edge_visible = edge_visible & keep_edge
                diagonal_visible = diagonal_visible & keep_diagonal
                edge_value = edge_value * edge_visible
                diagonal_value = diagonal_value * diagonal_visible

            edge_local = self.edge_encoder(
                torch.cat(
                    (edge_value, edge_visible.float(), edge_role, lag),
                    dim=-1,
                )
            )
            node_local = self.node_encoder(
                torch.cat(
                    (
                        diagonal_value,
                        diagonal_visible.float(),
                        self.role(role),
                        position,
                    ),
                    dim=-1,
                )
            )
            prompt = self._relation_aggregate(
                graph,
                edge_local,
                states,
                edge_value,
                0,
            )
            response_message = self._relation_aggregate(
                graph,
                edge_local,
                states,
                edge_value,
                1,
            )
            if message_mode == "none":
                prompt = torch.zeros_like(prompt)
                response_message = torch.zeros_like(response_message)
            elif message_mode == "prompt":
                response_message = torch.zeros_like(response_message)
            elif message_mode == "response":
                prompt = torch.zeros_like(prompt)

            states, gates = self.transition(
                states,
                node_local,
                prompt,
                response_message,
            )
            gate_maps.append(gates[response])

            target_context = torch.cat(
                (
                    states[source],
                    states[target],
                    edge_local,
                    prompt[target],
                    response_message[target],
                ),
                dim=-1,
            )
            edge_prediction = torch.sigmoid(
                self.edge_decoder(target_context)
            )
            support_logits = self.support_decoder(target_context)
            diagonal_prediction = torch.sigmoid(
                self.diagonal_decoder(
                    torch.cat((states, node_local), dim=-1)
                )
            )

            next_edge = graph.edge_attr[:, layer + 1]
            next_edge_observed = graph.edge_observed[:, layer + 1]
            edge_error = F.smooth_l1_loss(
                edge_prediction,
                next_edge,
                reduction="none",
            )
            edge_map, edge_count = self._token_map(
                graph,
                edge_error,
                next_edge_observed,
            )
            prompt_map, prompt_count = self._token_map(
                graph,
                edge_error,
                next_edge_observed,
                relation=0,
            )
            response_map, response_count = self._token_map(
                graph,
                edge_error,
                next_edge_observed,
                relation=1,
            )

            support_target = next_edge_observed.float()
            positive = support_target.sum().clamp_min(1.0)
            negative = (1.0 - support_target).sum().clamp_min(1.0)
            support_error = F.binary_cross_entropy_with_logits(
                support_logits,
                support_target,
                reduction="none",
                pos_weight=(negative / positive).detach(),
            )
            support_map, _ = self._token_map(
                graph,
                support_error,
                torch.ones_like(next_edge_observed),
            )

            next_diagonal = graph.diagonal[response, layer + 1]
            next_diagonal_observed = graph.diagonal_observed[
                response,
                layer + 1,
            ]
            diagonal_error = F.smooth_l1_loss(
                diagonal_prediction[response],
                next_diagonal,
                reduction="none",
            )
            diagonal_error = diagonal_error * next_diagonal_observed

            edge_maps.append(edge_map)
            prompt_maps.append(prompt_map)
            response_maps.append(response_map)
            edge_counts.append(edge_count)
            prompt_counts.append(prompt_count)
            response_counts.append(response_count)
            support_maps.append(support_map)
            diagonal_maps.append(diagonal_error)

        edge_error_map = torch.stack(edge_maps, dim=1)
        prompt_edge_error_map = torch.stack(prompt_maps, dim=1)
        response_edge_error_map = torch.stack(response_maps, dim=1)
        support_error_map = torch.stack(support_maps, dim=1)
        diagonal_error_map = torch.stack(diagonal_maps, dim=1)
        edge_count_map = torch.stack(edge_counts, dim=1)
        prompt_edge_count_map = torch.stack(prompt_counts, dim=1)
        response_edge_count_map = torch.stack(response_counts, dim=1)
        gates = torch.stack(gate_maps, dim=1)

        def map_mean(value, count):
            valid_map = count > 0
            return (
                (value * valid_map).sum(dim=(1, 2))
                / valid_map.sum(dim=(1, 2)).clamp_min(1)
            )

        edge_loss = map_mean(edge_error_map, edge_count_map)
        prompt_edge_loss = map_mean(
            prompt_edge_error_map,
            prompt_edge_count_map,
        )
        response_edge_loss = map_mean(
            response_edge_error_map,
            response_edge_count_map,
        )
        diagonal_loss = diagonal_error_map.mean(dim=(1, 2))
        support_loss = support_error_map.mean(dim=(1, 2))
        token_loss = (
            self.config.edge_loss_weight * edge_loss
            + self.config.diagonal_loss_weight * diagonal_loss
            + self.config.support_loss_weight * support_loss
        )
        valid = torch.ones(
            graph.num_response_tokens,
            dtype=torch.bool,
            device=graph.device,
        )
        return DynamicsOutput(
            loss=token_loss.mean(),
            valid=valid,
            token_loss=token_loss,
            edge_loss=edge_loss,
            prompt_edge_loss=prompt_edge_loss,
            response_edge_loss=response_edge_loss,
            diagonal_loss=diagonal_loss,
            support_loss=support_loss,
            edge_error_map=edge_error_map,
            prompt_edge_error_map=prompt_edge_error_map,
            response_edge_error_map=response_edge_error_map,
            diagonal_error_map=diagonal_error_map,
            support_error_map=support_error_map,
            edge_count_map=edge_count_map,
            prompt_edge_count_map=prompt_edge_count_map,
            response_edge_count_map=response_edge_count_map,
            self_gate=gates[..., 0],
            prompt_gate=gates[..., 1],
            response_gate=gates[..., 2],
            embedding=states[response],
        )
