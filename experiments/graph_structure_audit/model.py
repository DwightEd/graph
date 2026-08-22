"""Layered message passing over lossless layer-head attention edges."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RecoveryConfig
from .graph_data import MultiplexGraph
from .masking import MaskedGraph


@dataclass(frozen=True)
class RecoveryOutput:
    loss: torch.Tensor
    valid: torch.Tensor
    token_loss: torch.Tensor
    edge_loss: torch.Tensor
    diagonal_loss: torch.Tensor
    embedding: torch.Tensor
    edge_prediction: torch.Tensor
    diagonal_prediction: torch.Tensor


class LayeredGraphRecovery(nn.Module):
    """Recover masked channels while messages follow transformer depth."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: RecoveryConfig | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.config = RecoveryConfig() if config is None else config
        hidden = self.config.hidden_dim

        self.role = nn.Embedding(2, self.config.role_dim)
        self.lag = nn.Embedding(self.config.lag_bins, self.config.position_dim)
        self.position = nn.Sequential(
            nn.Linear(1, self.config.position_dim),
            nn.PReLU(),
        )
        self.node_init = nn.Sequential(
            nn.Linear(self.config.role_dim + self.config.position_dim, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_layer = nn.Sequential(
            nn.Linear(
                2 * num_heads + self.config.role_dim + self.config.position_dim,
                hidden,
            ),
            nn.PReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.node_layer = nn.Sequential(
            nn.Linear(
                2 * num_heads + self.config.role_dim + self.config.position_dim,
                hidden,
            ),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.update_input = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.GRUCell(hidden, hidden)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
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
            source.float() / float(max(graph.response_idx, 1)) * self.config.lag_bins
        ).long().clamp_max(self.config.lag_bins - 1)
        response_bin = torch.floor(
            torch.log2((target - source).clamp_min(1).float())
        ).long().clamp_max(self.config.lag_bins - 1)
        return torch.where(prompt, prompt_bin, response_bin)

    def forward(
        self,
        graph: MultiplexGraph,
        masked: MaskedGraph,
        *,
        message_passing: bool = True,
    ) -> RecoveryOutput:
        source, target = graph.edge_index
        role = graph.node_role
        position = self.position(graph.node_position[:, None])
        states = self.node_init(torch.cat((self.role(role), position), dim=-1))
        lag = self.lag(self._lag_bin(graph))
        edge_role = self.role(role[source])

        edge_predictions: list[torch.Tensor] = []
        diagonal_predictions: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            edge_value = masked.edge_value[:, layer]
            edge_visible = masked.edge_visible[:, layer].float()
            edge_local = self.edge_layer(
                torch.cat((edge_value, edge_visible, edge_role, lag), dim=-1)
            )

            diagonal_value = masked.diagonal_value[:, layer]
            diagonal_visible = masked.diagonal_visible[:, layer].float()
            node_local = self.node_layer(
                torch.cat(
                    (
                        diagonal_value,
                        diagonal_visible,
                        self.role(role),
                        position,
                    ),
                    dim=-1,
                )
            )

            edge_predictions.append(
                torch.sigmoid(
                    self.edge_decoder(
                        torch.cat((states[source], states[target], edge_local), dim=-1)
                    )
                )
            )
            diagonal_predictions.append(
                torch.sigmoid(self.diagonal_decoder(torch.cat((states, node_local), dim=-1)))
            )

            aggregate = states.new_zeros(states.shape)
            if message_passing and graph.num_edges:
                mass = edge_value.sum(dim=-1)
                message = self.message(torch.cat((states[source], edge_local), dim=-1))
                aggregate.index_add_(0, target, message * mass[:, None])
                normalizer = mass.new_zeros(graph.num_nodes)
                normalizer.index_add_(0, target, mass)
                aggregate = aggregate / normalizer.clamp_min(1e-8)[:, None]
            states = self.update(
                self.update_input(torch.cat((node_local, aggregate), dim=-1)),
                states,
            )

        edge_prediction = torch.stack(edge_predictions, dim=1)
        diagonal_prediction = torch.stack(diagonal_predictions, dim=1)
        token_edge_loss, token_edge_count = self._edge_loss(
            graph, masked, edge_prediction
        )
        token_diagonal_loss, token_diagonal_count = self._diagonal_loss(
            graph, masked, diagonal_prediction
        )
        total_count = token_edge_count + token_diagonal_count
        token_loss = (
            token_edge_loss * token_edge_count
            + token_diagonal_loss * token_diagonal_count
        ) / total_count.clamp_min(1.0)
        valid = total_count > 0
        loss = token_loss[valid].mean() if bool(valid.any()) else states.sum() * 0.0
        return RecoveryOutput(
            loss=loss,
            valid=valid,
            token_loss=token_loss,
            edge_loss=token_edge_loss,
            diagonal_loss=token_diagonal_loss,
            embedding=states[graph.response_idx :],
            edge_prediction=edge_prediction,
            diagonal_prediction=diagonal_prediction,
        )

    def _edge_loss(
        self,
        graph: MultiplexGraph,
        masked: MaskedGraph,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        error = F.smooth_l1_loss(prediction, graph.edge_attr, reduction="none")
        error = error * masked.edge_target
        per_edge = error.sum(dim=(1, 2))
        count_edge = masked.edge_target.sum(dim=(1, 2)).float()
        token = graph.edge_index[1] - graph.response_idx
        total = per_edge.new_zeros(graph.num_response_tokens)
        count = per_edge.new_zeros(graph.num_response_tokens)
        total.index_add_(0, token, per_edge)
        count.index_add_(0, token, count_edge)
        return total / count.clamp_min(1.0), count

    def _diagonal_loss(
        self,
        graph: MultiplexGraph,
        masked: MaskedGraph,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response = slice(graph.response_idx, graph.num_nodes)
        target = masked.diagonal_target[response]
        error = F.smooth_l1_loss(
            prediction[response], graph.diagonal[response], reduction="none"
        ) * target
        count = target.sum(dim=(1, 2)).float()
        total = error.sum(dim=(1, 2))
        return total / count.clamp_min(1.0), count
