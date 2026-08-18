"""Mechanism-Guided Causal Attention Set-Flow architecture.

The hierarchy follows the attention object:
source members -> weighted source sets -> heads -> Transformer depth -> token
time.  An online encoder is paired with a stop-gradient EMA teacher.  A learned
energy head is trained from mechanism-guided causal source-set corruptions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import CORRUPTION_NAMES, SetFlowModelConfig, SourceSetConfig
from .corruptions import CorruptionConfig, CorruptionPlan, apply_corruption
from .data import CausalSourceSetGraph, LayerSourceSets
from .set_layers import SetEncoder


@dataclass(frozen=True)
class EncoderOutput:
    token_embedding: torch.Tensor
    depth_state: torch.Tensor
    channel_state: torch.Tensor
    channel_active: torch.Tensor
    channel_corruption_mask: torch.Tensor


@dataclass(frozen=True)
class EnergyOutput:
    general: torch.Tensor
    token_general: torch.Tensor
    channel_general: torch.Tensor
    channel_logmeanexp: torch.Tensor
    type_energy: torch.Tensor
    token_type: torch.Tensor
    channel_type: torch.Tensor


@dataclass(frozen=True)
class ProjectedOutput:
    token: torch.Tensor
    channel: torch.Tensor


class FourierScalarEncoder(nn.Module):
    def __init__(self, output_dim: int, fourier_dim: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(32.0), int(fourier_dim))
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.project = nn.Sequential(
            nn.Linear(1 + 2 * int(fourier_dim), output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().unsqueeze(-1)
        angle = value * self.frequencies
        return self.project(
            torch.cat((value, torch.sin(angle), torch.cos(angle)), dim=-1)
        )


class SourceFieldEncoder(nn.Module):
    """Encode typed source fields without flattening raw heterogeneous inputs."""

    def __init__(self, config: SetFlowModelConfig, set_types: int = 2) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        fourier = int(config.scalar_fourier_dim)
        self.lag = FourierScalarEncoder(dim, fourier)
        self.weight = FourierScalarEncoder(dim, fourier)
        self.received = FourierScalarEncoder(dim, fourier)
        self.delta = FourierScalarEncoder(dim, fourier)
        self.ancestry = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.ancestry_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.type_embedding = nn.Embedding(int(set_types), dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        *,
        source_state: torch.Tensor,
        lag: torch.Tensor,
        weight: torch.Tensor,
        received: torch.Tensor,
        received_delta: torch.Tensor,
        valid: torch.Tensor,
        set_type: int,
    ) -> torch.Tensor:
        if source_state.shape[:-1] != lag.shape:
            raise ValueError("source ancestry and scalar geometry differ")
        for value in (weight, received, received_delta, valid):
            if value.shape != lag.shape:
                raise ValueError("source-set scalar fields are not aligned")
        lag_state = self.lag(torch.log1p(lag.clamp_min(0.0)))
        weight_state = self.weight(torch.log1p(weight.clamp_min(0.0)))
        received_state = self.received(torch.log1p(received.clamp_min(0.0)))
        delta_state = self.delta(torch.asinh(received_delta))
        ancestry = self.ancestry(source_state)
        gate = self.ancestry_gate(
            torch.cat((weight_state, received_state), dim=-1)
        )
        type_state = self.type_embedding(
            torch.full_like(lag, int(set_type), dtype=torch.long)
        )
        output = self.norm(
            self.dropout(
                lag_state
                + weight_state
                + received_state
                + delta_state
                + gate * ancestry
                + type_state
            )
        )
        return output * valid.unsqueeze(-1).to(output.dtype)


class WeightedSourceSetEncoder(nn.Module):
    """Encode complete bounded source sets with permutation-invariant pooling."""

    def __init__(self, config: SetFlowModelConfig, *, set_type: int) -> None:
        super().__init__()
        self.set_type = int(set_type)
        self.row_chunk_size = int(config.set_row_chunk_size)
        dim = int(config.hidden_dim)
        self.fields = SourceFieldEncoder(config)
        self.empty_member = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.empty_member, std=1.0 / math.sqrt(dim))
        self.set_encoder = SetEncoder(
            dim,
            config.set_heads,
            config.induced_points,
            config.set_blocks,
            config.dropout,
        )

    def forward(
        self,
        *,
        source_state_table: torch.Tensor,
        source_index: torch.Tensor,
        lag: torch.Tensor,
        weight: torch.Tensor,
        received: torch.Tensor,
        received_delta: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        tokens, heads, members = valid.shape
        rows = tokens * heads
        flat_source = source_index.reshape(rows, members).long()
        flat_lag = lag.reshape(rows, members)
        flat_weight = weight.reshape(rows, members)
        flat_received = received.reshape(rows, members)
        flat_delta = received_delta.reshape(rows, members)
        flat_valid = valid.reshape(rows, members)
        pooled_parts: list[torch.Tensor] = []

        for start in range(0, rows, self.row_chunk_size):
            end = min(rows, start + self.row_chunk_size)
            current_valid = flat_valid[start:end]
            source_state = source_state_table[flat_source[start:end]]
            members_state = self.fields(
                source_state=source_state,
                lag=flat_lag[start:end],
                weight=flat_weight[start:end],
                received=flat_received[start:end],
                received_delta=flat_delta[start:end],
                valid=current_valid,
                set_type=self.set_type,
            )
            chunk_rows = end - start
            members_state = torch.cat(
                (
                    members_state,
                    self.empty_member.expand(chunk_rows, -1, -1),
                ),
                dim=1,
            )
            empty = ~current_valid.any(dim=1, keepdim=True)
            member_mask = torch.cat((current_valid, empty), dim=1)
            _, pooled = self.set_encoder(members_state, member_mask)
            pooled_parts.append(pooled)
        return torch.cat(pooled_parts, dim=0).reshape(tokens, heads, -1)


class DualSourceSetHeadEncoder(nn.Module):
    """Fuse current routing and accumulated received-support source sets."""

    def __init__(self, config: SetFlowModelConfig) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        self.route = WeightedSourceSetEncoder(config, set_type=0)
        self.memory = WeightedSourceSetEncoder(config, set_type=1)
        self.mass = FourierScalarEncoder(dim, config.scalar_fourier_dim)
        self.tail = FourierScalarEncoder(dim, config.scalar_fourier_dim)
        self.degree = FourierScalarEncoder(dim, config.scalar_fourier_dim)
        self.route_gate = nn.Linear(dim, dim, bias=False)
        self.memory_gate = nn.Linear(dim, dim, bias=False)
        self.context_gate = nn.Linear(dim, dim)
        self.interaction = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        source_sets: LayerSourceSets,
        source_state: torch.Tensor,
        *,
        attention_floor: float,
    ) -> torch.Tensor:
        scale = max(float(attention_floor), 1e-8)
        route_state = self.route(
            source_state_table=source_state,
            source_index=source_sets.route_source,
            lag=_source_lag(source_sets.route_source),
            weight=source_sets.route_weight / scale,
            received=source_sets.route_received / scale,
            received_delta=source_sets.route_received_delta / scale,
            valid=source_sets.route_mask,
        )
        memory_state = self.memory(
            source_state_table=source_state,
            source_index=source_sets.memory_source,
            lag=_source_lag(source_sets.memory_source),
            weight=source_sets.memory_current_weight / scale,
            received=source_sets.memory_received / scale,
            received_delta=source_sets.memory_received_delta / scale,
            valid=source_sets.memory_mask,
        )
        context = (
            self.mass(torch.log1p(source_sets.total_mass / scale))
            + self.tail(torch.log1p(source_sets.tail_mass / scale))
            + self.degree(torch.log1p(source_sets.edge_count))
        )
        gate = torch.sigmoid(
            self.route_gate(route_state)
            + self.memory_gate(memory_state)
            + self.context_gate(context)
        )
        return self.norm(
            gate * route_state
            + (1.0 - gate) * memory_state
            + self.interaction(route_state * memory_state)
            + context
        )


class HeadMixer(nn.Module):
    def __init__(self, num_heads: int, num_layers: int, config: SetFlowModelConfig):
        super().__init__()
        dim = int(config.hidden_dim)
        self.token_chunk_size = int(config.mixer_token_chunk_size)
        self.head_identity = nn.Parameter(torch.empty(num_heads, dim))
        self.layer_identity = nn.Parameter(torch.empty(num_layers, dim))
        nn.init.normal_(self.head_identity, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.layer_identity, std=1.0 / math.sqrt(dim))
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.set_heads,
            dim_feedforward=dim * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=config.head_mixer_layers
        )
        self.pool_score = nn.Linear(dim, 1)

    def forward(
        self,
        head_state: torch.Tensor,
        *,
        active: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_parts: list[torch.Tensor] = []
        channel_parts: list[torch.Tensor] = []
        for start in range(0, len(head_state), self.token_chunk_size):
            end = min(len(head_state), start + self.token_chunk_size)
            token, channel = self._forward_chunk(
                head_state[start:end], active[start:end], layer_index=layer_index
            )
            token_parts.append(token)
            channel_parts.append(channel)
        return torch.cat(token_parts), torch.cat(channel_parts)

    def _forward_chunk(self, head_state, active, *, layer_index):
        visible = (
            head_state
            + self.head_identity.unsqueeze(0)
            + self.layer_identity[int(layer_index)].view(1, 1, -1)
        )
        safe_active = active.clone()
        empty = ~safe_active.any(dim=1)
        if bool(empty.any()):
            safe_active[empty, 0] = True
        mixed = self.encoder(visible, src_key_padding_mask=~safe_active)
        score = self.pool_score(mixed).squeeze(-1)
        score = score.masked_fill(~safe_active, float("-inf"))
        weight = torch.softmax(score, dim=1)
        token_state = (mixed * weight.unsqueeze(-1)).sum(dim=1)
        token_state = torch.where(
            empty.unsqueeze(-1), torch.zeros_like(token_state), token_state
        )
        mixed = torch.where(
            active.unsqueeze(-1), mixed, torch.zeros_like(mixed)
        )
        return token_state, mixed


class DepthMixer(nn.Module):
    def __init__(self, num_layers: int, config: SetFlowModelConfig) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        self.token_chunk_size = int(config.mixer_token_chunk_size)
        self.layer_position = nn.Parameter(torch.empty(num_layers, dim))
        nn.init.normal_(self.layer_position, std=1.0 / math.sqrt(dim))
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.set_heads,
            dim_feedforward=dim * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=config.depth_mixer_layers
        )
        self.pool_score = nn.Linear(dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final_parts: list[torch.Tensor] = []
        encoded_parts: list[torch.Tensor] = []
        for start in range(0, len(states), self.token_chunk_size):
            end = min(len(states), start + self.token_chunk_size)
            encoded = self.encoder(
                states[start:end] + self.layer_position.unsqueeze(0)
            )
            weight = torch.softmax(
                self.pool_score(encoded).squeeze(-1), dim=1
            )
            final_parts.append((encoded * weight.unsqueeze(-1)).sum(dim=1))
            encoded_parts.append(encoded)
        return torch.cat(final_parts), torch.cat(encoded_parts)


class SetFlowEncoder(nn.Module):
    """Hierarchical source-set encoder with exact-source depth ancestry."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        *,
        source_config: SourceSetConfig,
        model_config: SetFlowModelConfig,
    ) -> None:
        super().__init__()
        source_config.validate()
        model_config.validate()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.source_config = source_config
        self.config = model_config
        dim = int(model_config.hidden_dim)
        self.source_set = DualSourceSetHeadEncoder(model_config)
        self.head_mixer = HeadMixer(num_heads, num_layers, model_config)
        self.depth_recurrence = nn.GRUCell(dim, dim)
        self.depth_mixer = DepthMixer(num_layers, model_config)
        self.time_encoder = nn.GRU(dim, dim, batch_first=True)

    def forward(
        self,
        graph: CausalSourceSetGraph,
        *,
        corruption_plan: CorruptionPlan | None = None,
        corruption_config: CorruptionConfig | None = None,
        device: str | torch.device | None = None,
    ) -> EncoderOutput:
        graph.validate()
        if graph.num_layers != self.num_layers or graph.num_heads != self.num_heads:
            raise ValueError("graph geometry differs from Set-Flow encoder")
        device = next(self.parameters()).device if device is None else torch.device(device)
        tokens = graph.response_count
        previous_depth = torch.zeros(
            (tokens, self.config.hidden_dim), device=device
        )
        depth_rows: list[torch.Tensor] = []
        channel_rows: list[torch.Tensor] = []
        active_rows: list[torch.Tensor] = []
        corruption_rows: list[torch.Tensor] = []
        use_checkpoint = bool(
            self.config.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

        for layer_index in range(self.num_layers):
            source_sets = graph.materialize_layer(
                layer_index, self.source_config, device=device
            )
            if corruption_plan is not None:
                if corruption_config is None:
                    raise ValueError("corruption plan requires corruption config")
                source_sets, changed = apply_corruption(
                    source_sets,
                    corruption_plan,
                    layer_index=layer_index,
                    config=corruption_config,
                )
            else:
                changed = torch.zeros(
                    (tokens, self.num_heads), dtype=torch.bool, device=device
                )
            active = source_sets.route_mask.any(dim=-1) | source_sets.memory_mask.any(
                dim=-1
            )
            if use_checkpoint:
                tensors = source_sets.tensor_tuple()

                def run_layer(previous, *values, current_layer=layer_index):
                    materialized = LayerSourceSets.from_tensor_tuple(tuple(values))
                    return self._forward_layer(
                        materialized,
                        previous,
                        active=(
                            materialized.route_mask.any(dim=-1)
                            | materialized.memory_mask.any(dim=-1)
                        ),
                        layer_index=current_layer,
                        attention_floor=graph.attention_floor,
                    )

                previous_depth, channel_state = checkpoint(
                    run_layer,
                    previous_depth,
                    *tensors,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    determinism_check="default",
                )
            else:
                previous_depth, channel_state = self._forward_layer(
                    source_sets,
                    previous_depth,
                    active=active,
                    layer_index=layer_index,
                    attention_floor=graph.attention_floor,
                )
            depth_rows.append(previous_depth)
            channel_rows.append(channel_state)
            active_rows.append(active)
            corruption_rows.append(changed)

        depth_state = torch.stack(depth_rows, dim=1)
        channel_state = torch.stack(channel_rows, dim=1)
        channel_active = torch.stack(active_rows, dim=1)
        channel_corruption = torch.stack(corruption_rows, dim=1)
        if use_checkpoint:
            final_depth, encoded_depth = checkpoint(
                self.depth_mixer,
                depth_state,
                use_reentrant=False,
                preserve_rng_state=True,
                determinism_check="default",
            )
        else:
            final_depth, encoded_depth = self.depth_mixer(depth_state)
        temporal, _ = self.time_encoder(final_depth.unsqueeze(0))
        return EncoderOutput(
            token_embedding=temporal.squeeze(0),
            depth_state=encoded_depth,
            channel_state=channel_state,
            channel_active=channel_active,
            channel_corruption_mask=channel_corruption,
        )

    def _forward_layer(
        self,
        source_sets: LayerSourceSets,
        previous_depth: torch.Tensor,
        *,
        active: torch.Tensor,
        layer_index: int,
        attention_floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_head = self.source_set(
            source_sets, previous_depth, attention_floor=attention_floor
        )
        token_layer, channel_state = self.head_mixer(
            raw_head, active=active, layer_index=layer_index
        )
        return self.depth_recurrence(token_layer, previous_depth), channel_state


class StateDriftFusion(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None:
        super().__init__()
        hidden = int(dim) * int(multiplier)
        self.state = _mlp(dim, hidden, dim)
        self.velocity = _mlp(dim, hidden, dim)
        self.curvature = _mlp(dim, hidden, dim)
        self.gate = nn.Linear(dim * 3, 3)
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(values)
        curvature = torch.zeros_like(values)
        if len(values) > 1:
            velocity[1:] = values[1:] - values[:-1]
        if len(values) > 2:
            curvature[2:] = velocity[2:] - velocity[1:-1]
        branches = (
            self.state(values),
            self.velocity(velocity),
            self.curvature(curvature),
        )
        weight = torch.softmax(self.gate(torch.cat(branches, dim=-1)), dim=-1)
        fused = sum(
            weight[:, index : index + 1] * branch
            for index, branch in enumerate(branches)
        )
        return self.norm(fused)


class MechanismEnergyHead(nn.Module):
    def __init__(self, config: SetFlowModelConfig) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        hidden = dim * int(config.energy_hidden_multiplier)
        types = len(CORRUPTION_NAMES)
        self.token_fusion = StateDriftFusion(
            dim, config.energy_hidden_multiplier
        )
        self.token_general = _mlp(dim, hidden, 1)
        self.token_type = _mlp(dim, hidden, types)
        self.channel_general = _mlp(dim, hidden, 1)
        self.channel_type = _mlp(dim, hidden, types)
        self.combine_gate = nn.Linear(dim, 1)

    def forward(self, encoded: EncoderOutput) -> EnergyOutput:
        token_state = self.token_fusion(encoded.token_embedding)
        token_general = self.token_general(token_state).squeeze(-1)
        token_type = self.token_type(token_state)
        channel_general = self.channel_general(encoded.channel_state).squeeze(-1)
        channel_type = self.channel_type(encoded.channel_state)
        channel_logmean = _masked_logmeanexp(
            channel_general, encoded.channel_active, dims=(1, 2)
        )
        channel_type_mean = _masked_logmeanexp(
            channel_type,
            encoded.channel_active.unsqueeze(-1).expand_as(channel_type),
            dims=(1, 2),
        )
        gate = torch.sigmoid(self.combine_gate(token_state)).squeeze(-1)
        general = gate * token_general + (1.0 - gate) * channel_logmean
        type_energy = gate[:, None] * token_type + (
            1.0 - gate[:, None]
        ) * channel_type_mean
        return EnergyOutput(
            general=general,
            token_general=token_general,
            channel_general=channel_general,
            channel_logmeanexp=channel_logmean,
            type_energy=type_energy,
            token_type=token_type,
            channel_type=channel_type,
        )


class CausalSetFlowModel(nn.Module):
    """Online Set-Flow encoder, EMA teacher, projectors, and anomaly energy."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        source_config: SourceSetConfig | None = None,
        model_config: SetFlowModelConfig | None = None,
    ) -> None:
        super().__init__()
        source_config = SourceSetConfig() if source_config is None else source_config
        model_config = (
            SetFlowModelConfig() if model_config is None else model_config
        )
        self.source_config = source_config
        self.config = model_config
        self.online_encoder = SetFlowEncoder(
            num_layers,
            num_heads,
            source_config=source_config,
            model_config=model_config,
        )
        self.teacher_encoder = deepcopy(self.online_encoder)
        for parameter in self.teacher_encoder.parameters():
            parameter.requires_grad_(False)
        self.teacher_encoder.eval()
        dim = int(model_config.hidden_dim)
        hidden = dim * int(model_config.projector_hidden_multiplier)
        self.token_predictor = _mlp(dim, hidden, dim)
        self.channel_predictor = _mlp(dim, hidden, dim)
        self.energy_head = MechanismEnergyHead(model_config)

    @property
    def num_layers(self) -> int:
        return self.online_encoder.num_layers

    @property
    def num_heads(self) -> int:
        return self.online_encoder.num_heads

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    def encode_online(
        self,
        graph: CausalSourceSetGraph,
        *,
        corruption_plan: CorruptionPlan | None = None,
        corruption_config: CorruptionConfig | None = None,
        device: str | torch.device | None = None,
    ) -> EncoderOutput:
        return self.online_encoder(
            graph,
            corruption_plan=corruption_plan,
            corruption_config=corruption_config,
            device=device,
        )

    @torch.no_grad()
    def encode_teacher(
        self,
        graph: CausalSourceSetGraph,
        *,
        device: str | torch.device | None = None,
    ) -> EncoderOutput:
        self.teacher_encoder.eval()
        return self.teacher_encoder(graph, device=device)

    def project(self, encoded: EncoderOutput) -> ProjectedOutput:
        return ProjectedOutput(
            token=self.token_predictor(encoded.token_embedding),
            channel=self.channel_predictor(encoded.channel_state),
        )

    def energy(self, encoded: EncoderOutput) -> EnergyOutput:
        return self.energy_head(encoded)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        momentum = float(momentum)
        for online, teacher in zip(
            self.online_encoder.parameters(),
            self.teacher_encoder.parameters(),
            strict=True,
        ):
            teacher.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)
        for online, teacher in zip(
            self.online_encoder.buffers(),
            self.teacher_encoder.buffers(),
            strict=True,
        ):
            teacher.copy_(online)

    @torch.inference_mode()
    def score_graph(
        self,
        graph: CausalSourceSetGraph,
        *,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        previous = self.training
        self.eval()
        encoded = self.encode_online(graph, device=device)
        energy = self.energy(encoded)
        result = {
            "embedding": encoded.token_embedding.float(),
            "general_energy": energy.general.float(),
            "token_energy": energy.token_general.float(),
            "channel_energy": energy.channel_logmeanexp.float(),
            "type_energy": energy.type_energy.float(),
            "channel_energy_max": torch.where(
                encoded.channel_active,
                energy.channel_general.float(),
                torch.full_like(energy.channel_general.float(), float("-inf")),
            ).flatten(1).amax(dim=1),
        }
        self.train(previous)
        return result


def _source_lag(source: torch.Tensor) -> torch.Tensor:
    token = torch.arange(source.shape[0], device=source.device)[:, None, None]
    return (token - source.long()).clamp_min(1).float()


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(output_dim)),
    )


def _masked_logmeanexp(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dims: tuple[int, ...],
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("masked log-mean-exp geometry differs")
    masked = values.masked_fill(~mask, float("-inf"))
    count = mask.sum(dim=dims).clamp_min(1).to(values.dtype)
    result = torch.logsumexp(masked, dim=dims) - torch.log(count)
    empty = ~mask.any(dim=dims)
    return torch.where(empty, torch.zeros_like(result), result)