"""Channel-aware dynamic source prediction for causal multiplex attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .controls import first_rewired_candidate, source_candidates
from .events import CausalEventSample, RP, RR


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    channel_embedding_dim: int = 8
    relation_embedding_dim: int = 4
    lag_frequencies: int = 4
    negatives_per_edge: int = 8
    dropout: float = 0.10
    weight_loss_weight: float = 0.10
    seed: int = 20260817

    def validate(self) -> None:
        if min(
            int(self.hidden_dim),
            int(self.channel_embedding_dim),
            int(self.relation_embedding_dim),
            int(self.lag_frequencies),
            int(self.negatives_per_edge),
        ) < 1:
            raise ValueError("CMRP model dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not np.isfinite(self.weight_loss_weight) or self.weight_loss_weight < 0:
            raise ValueError("weight_loss_weight must be finite and non-negative")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class RouterOutput:
    loss: torch.Tensor
    raw_route_surprise: torch.Tensor
    presence_nll: torch.Tensor
    source_nll: torch.Tensor
    weight_error: torch.Tensor
    rewired_source_nll: torch.Tensor
    rewire_gap: torch.Tensor
    selected_rr_edges: torch.Tensor
    state: torch.Tensor

    def detached_numpy(self) -> dict[str, np.ndarray]:
        return {
            "raw_route_surprise": self.raw_route_surprise.detach().cpu().numpy().astype(np.float32),
            "presence_nll": self.presence_nll.detach().cpu().numpy().astype(np.float32),
            "source_nll": self.source_nll.detach().cpu().numpy().astype(np.float32),
            "weight_error": self.weight_error.detach().cpu().numpy().astype(np.float32),
            "rewired_source_nll": self.rewired_source_nll.detach().cpu().numpy().astype(np.float32),
            "rewire_gap": self.rewire_gap.detach().cpu().numpy().astype(np.float32),
            "selected_rr_edges": self.selected_rr_edges.detach().cpu().numpy().astype(np.int32),
            "state": self.state.detach().cpu().numpy().astype(np.float32),
        }


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CausalMultiplexRouter(torch.nn.Module):
    """Predict retained RR source identities from the preceding graph state."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: ModelConfig | None = None,
    ):
        super().__init__()
        config = ModelConfig() if config is None else config
        config.validate()
        if int(num_layers) < 1 or int(num_heads) < 1:
            raise ValueError("attention geometry must be positive")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.config = config
        hidden = int(config.hidden_dim)
        channel_dim = int(config.channel_embedding_dim)
        relation_dim = int(config.relation_embedding_dim)
        lag_dim = 1 + 2 * int(config.lag_frequencies)

        self.layer_embedding = torch.nn.Embedding(self.num_layers, channel_dim)
        self.head_embedding = torch.nn.Embedding(self.num_heads, channel_dim)
        self.relation_embedding = torch.nn.Embedding(2, relation_dim)
        self.prompt_anchor = torch.nn.Parameter(torch.zeros(hidden))
        self.start_state = torch.nn.Parameter(torch.zeros(hidden))

        event_input = hidden + 2 * channel_dim + relation_dim + 2 + lag_dim
        self.event_mlp = MLP(event_input, 2 * hidden, hidden, config.dropout)
        token_input = 2 * hidden + 4 + 2
        self.token_mlp = MLP(token_input, 2 * hidden, hidden, config.dropout)
        self.state_update = torch.nn.GRUCell(hidden, hidden)

        query_input = hidden + 2 * channel_dim + 2
        self.query_mlp = MLP(query_input, 2 * hidden, hidden, config.dropout)
        self.source_mlp = MLP(hidden + lag_dim, 2 * hidden, hidden, config.dropout)
        self.presence_head = MLP(hidden + 2, hidden, 1, config.dropout)
        self.weight_head = MLP(2 * hidden, hidden, 1, config.dropout)
        self.reset_parameters()

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.prompt_anchor, std=0.02)
        torch.nn.init.normal_(self.start_state, std=0.02)
        for embedding in (
            self.layer_embedding,
            self.head_embedding,
            self.relation_embedding,
        ):
            torch.nn.init.normal_(embedding.weight, std=0.02)

    def _position_features(self, token: int, count: int, device) -> torch.Tensor:
        denominator = max(int(count) - 1, 1)
        relative = float(token) / float(denominator)
        log_progress = math.log1p(int(token)) / max(math.log1p(int(count)), 1.0)
        return torch.tensor((relative, log_progress), dtype=torch.float32, device=device)

    def _lag_features(self, lag: torch.Tensor) -> torch.Tensor:
        lag = lag.to(dtype=torch.float32)
        log_lag = torch.log1p(lag.clamp_min(0.0))
        frequencies = torch.arange(
            1,
            int(self.config.lag_frequencies) + 1,
            dtype=torch.float32,
            device=lag.device,
        )
        phase = log_lag[..., None] * frequencies
        scaled = log_lag / (1.0 + log_lag)
        return torch.cat(
            (scaled[..., None], torch.sin(phase), torch.cos(phase)), dim=-1
        )

    def _channel_embedding(self, channel: torch.Tensor) -> torch.Tensor:
        layer = torch.div(channel, self.num_heads, rounding_mode="floor")
        head = channel.remainder(self.num_heads)
        return torch.cat(
            (self.layer_embedding(layer), self.head_embedding(head)), dim=-1
        )

    def _source_scores(
        self,
        *,
        previous_state: torch.Tensor,
        states: list[torch.Tensor],
        token: int,
        channel: torch.Tensor,
        candidates: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channel_value = self._channel_embedding(channel.reshape(1)).reshape(-1)
        query = self.query_mlp(
            torch.cat((previous_state, channel_value, position), dim=0)
        )
        source_state = torch.stack([states[int(value)] for value in candidates], dim=0)
        candidate_lag = int(token) - candidates
        source_value = self.source_mlp(
            torch.cat((source_state, self._lag_features(candidate_lag)), dim=1)
        )
        logits = source_value @ query / math.sqrt(float(self.config.hidden_dim))
        return logits, query, source_value

    def _event_messages(
        self,
        events: CausalEventSample,
        token: int,
        states: list[torch.Tensor],
    ) -> torch.Tensor:
        current = events.target_slice(token)
        relation = events.relation[current]
        if relation.numel() == 0:
            return torch.empty(
                (0, self.config.hidden_dim),
                dtype=torch.float32,
                device=events.weight.device,
            )
        source_states = []
        for relation_value, source_value in zip(
            relation.tolist(), events.source[current].tolist(), strict=True
        ):
            if int(relation_value) == RP:
                source_states.append(self.prompt_anchor)
            else:
                source_states.append(states[int(source_value)])
        source_state = torch.stack(source_states, dim=0)
        channel_value = self._channel_embedding(events.channel[current])
        relation_value = self.relation_embedding(relation)
        weight = events.weight[current].clamp_min(0.0)
        floor = max(float(events.attention_floor), float(self.config.hidden_dim) * 0.0 + 1e-12)
        weight_feature = torch.stack(
            (weight, torch.log1p(weight / floor)), dim=1
        )
        lag_feature = self._lag_features(events.lag[current])
        return self.event_mlp(
            torch.cat(
                (
                    source_state,
                    channel_value,
                    relation_value,
                    weight_feature,
                    lag_feature,
                ),
                dim=1,
            )
        )

    def forward(self, events: CausalEventSample) -> RouterOutput:
        events.validate()
        if events.num_layers != self.num_layers or events.num_heads != self.num_heads:
            raise ValueError("event geometry differs from model geometry")
        device = self.start_state.device
        if events.weight.device != device:
            events = events.to(device)

        previous_state = self.start_state
        states: list[torch.Tensor] = []
        raw_rows = []
        presence_rows = []
        source_rows = []
        weight_rows = []
        rewired_rows = []
        gap_rows = []
        count_rows = []

        for token in range(events.response_count):
            position = self._position_features(token, events.response_count, device)
            presence_logit = self.presence_head(
                torch.cat((previous_state, position), dim=0)
            ).reshape(())
            has_rr = (events.role_summary[token, 3] > 0).to(torch.float32)
            presence_nll = F.binary_cross_entropy_with_logits(
                presence_logit, has_rr, reduction="none"
            )

            current = events.target_slice(token)
            local_indices = torch.arange(
                current.start, current.stop, dtype=torch.long, device=device
            )
            rr_indices = local_indices[events.relation[current] == RR]
            edge_source_losses = []
            edge_weight_losses = []
            edge_rewired_losses = []
            for edge_index_tensor in rr_indices:
                edge_index = int(edge_index_tensor.item())
                candidates = source_candidates(
                    events,
                    edge_index,
                    negatives=self.config.negatives_per_edge,
                    seed=self.config.seed,
                )
                logits, query, source_value = self._source_scores(
                    previous_state=previous_state,
                    states=states,
                    token=token,
                    channel=events.channel[edge_index],
                    candidates=candidates,
                    position=position,
                )
                log_probability = F.log_softmax(logits, dim=0)
                true_nll = -log_probability[0]
                edge_source_losses.append(true_nll)
                rewired_index, available = first_rewired_candidate(candidates)
                if available:
                    edge_rewired_losses.append(-log_probability[rewired_index])

                predicted_log_weight = self.weight_head(
                    torch.cat((query, source_value[0]), dim=0)
                ).reshape(())
                target_log_weight = torch.log(
                    events.weight[edge_index].clamp_min(1e-12)
                )
                edge_weight_losses.append(
                    F.smooth_l1_loss(
                        predicted_log_weight,
                        target_log_weight,
                        reduction="none",
                    )
                )

            zero = torch.zeros((), dtype=torch.float32, device=device)
            source_nll = (
                torch.stack(edge_source_losses).mean()
                if edge_source_losses
                else zero
            )
            weight_error = (
                torch.stack(edge_weight_losses).mean()
                if edge_weight_losses
                else zero
            )
            rewired_nll = (
                torch.stack(edge_rewired_losses).mean()
                if edge_rewired_losses
                else torch.full((), float("nan"), device=device)
            )
            rewire_gap = rewired_nll - source_nll
            raw_surprise = presence_nll + source_nll

            messages = self._event_messages(events, token, states)
            if len(messages):
                mean_message = messages.mean(dim=0)
                max_message = messages.max(dim=0).values
            else:
                mean_message = torch.zeros_like(previous_state)
                max_message = torch.zeros_like(previous_state)
            role_summary = torch.log1p(events.role_summary[token])
            token_input = self.token_mlp(
                torch.cat(
                    (mean_message, max_message, role_summary, position), dim=0
                )
            )
            current_state = self.state_update(token_input, previous_state)
            states.append(current_state)
            previous_state = current_state

            raw_rows.append(raw_surprise)
            presence_rows.append(presence_nll)
            source_rows.append(source_nll)
            weight_rows.append(weight_error)
            rewired_rows.append(rewired_nll)
            gap_rows.append(rewire_gap)
            count_rows.append(
                torch.as_tensor(len(rr_indices), dtype=torch.int64, device=device)
            )

        raw = torch.stack(raw_rows)
        presence = torch.stack(presence_rows)
        source = torch.stack(source_rows)
        weight = torch.stack(weight_rows)
        rewired = torch.stack(rewired_rows)
        gap = torch.stack(gap_rows)
        count = torch.stack(count_rows)
        state = torch.stack(states)
        loss = raw.mean() + float(self.config.weight_loss_weight) * weight.mean()
        return RouterOutput(
            loss=loss,
            raw_route_surprise=raw,
            presence_nll=presence,
            source_nll=source,
            weight_error=weight,
            rewired_source_nll=rewired,
            rewire_gap=gap,
            selected_rr_edges=count,
            state=state,
        )
