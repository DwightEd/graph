"""Learnable Causal Attention Set-Flow architecture.

The hierarchy mirrors the data rather than flattening it:
source fields -> route/memory sets -> heads -> Transformer depth -> token time.
All training targets are derived from attention itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import SetFlowModelConfig, SourceSetConfig
from .data import CausalSourceSetGraph, LayerSourceSets
from .losses import LossBreakdown, variance_floor_loss
from .masking import (
    deterministic_generator,
    sample_layer_mask_plan,
    sample_sequence_mask_plan,
)
from .set_layers import SetEncoder


@dataclass(frozen=True)
class SetFlowOutput:
    token_embedding: torch.Tensor
    depth_state: torch.Tensor
    route_element_error: torch.Tensor
    memory_element_error: torch.Tensor
    head_reconstruction_error: torch.Tensor
    layer_reconstruction_error: torch.Tensor
    temporal_prediction_error: torch.Tensor
    loss: LossBreakdown

    def score_components(self) -> dict[str, torch.Tensor]:
        return {
            "route_element": self.route_element_error,
            "memory_element": self.memory_element_error,
            "head_reconstruction": self.head_reconstruction_error,
            "layer_reconstruction": self.layer_reconstruction_error,
            "temporal_prediction": self.temporal_prediction_error,
        }


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
    """Encode typed source fields separately, then fuse them additively."""

    def __init__(self, config: SetFlowModelConfig, set_types: int = 2) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        fourier = int(config.scalar_fourier_dim)
        self.lag = FourierScalarEncoder(dim, fourier)
        self.weight = FourierScalarEncoder(dim, fourier)
        self.received = FourierScalarEncoder(dim, fourier)
        self.delta = FourierScalarEncoder(dim, fourier)
        self.source = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.type_embedding = nn.Embedding(int(set_types), dim)
        self.scalar_mask = nn.Parameter(torch.empty(3, dim))
        nn.init.normal_(self.scalar_mask, std=1.0 / math.sqrt(dim))
        self.ancestry_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
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
        field_mask: torch.Tensor,
        set_type: int,
    ) -> torch.Tensor:
        if source_state.shape[:-1] != lag.shape:
            raise ValueError("source-state and scalar geometry differ")
        if any(
            value.shape != lag.shape
            for value in (weight, received, received_delta, valid, field_mask)
        ):
            raise ValueError("source-set scalar fields are not aligned")
        lag_embed = self.lag(torch.log1p(lag.clamp_min(0.0)))
        weight_embed = self.weight(torch.log1p(weight.clamp_min(0.0)))
        received_embed = self.received(torch.log1p(received.clamp_min(0.0)))
        delta_embed = self.delta(torch.asinh(received_delta))
        masked = field_mask.unsqueeze(-1)
        weight_embed = torch.where(masked, self.scalar_mask[0], weight_embed)
        received_embed = torch.where(masked, self.scalar_mask[1], received_embed)
        delta_embed = torch.where(masked, self.scalar_mask[2], delta_embed)
        ancestry = self.source(source_state)
        # This small concatenation operates on already typed embeddings; raw
        # source/head/layer features are never flattened into one vector.
        gate = self.ancestry_gate(
            torch.cat((weight_embed, received_embed), dim=-1)
        )
        type_embed = self.type_embedding(
            torch.full_like(lag, int(set_type), dtype=torch.long)
        )
        output = self.norm(
            self.dropout(
                lag_embed
                + weight_embed
                + received_embed
                + delta_embed
                + gate * ancestry
                + type_embed
            )
        )
        return output * valid.unsqueeze(-1).to(output.dtype)


class WeightedSourceSetEncoder(nn.Module):
    def __init__(self, config: SetFlowModelConfig, *, set_type: int) -> None:
        super().__init__()
        self.set_type = int(set_type)
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
        self.scalar_decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3)
        )

    def forward(
        self,
        *,
        source_state: torch.Tensor,
        lag: torch.Tensor,
        weight: torch.Tensor,
        received: torch.Tensor,
        received_delta: torch.Tensor,
        valid: torch.Tensor,
        field_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, heads, members = valid.shape

        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(tokens * heads, members, *value.shape[3:])

        member_state = self.fields(
            source_state=flatten(source_state),
            lag=flatten(lag),
            weight=flatten(weight),
            received=flatten(received),
            received_delta=flatten(received_delta),
            valid=flatten(valid),
            field_mask=flatten(field_mask),
            set_type=self.set_type,
        )
        flat_valid = valid.reshape(tokens * heads, members)
        member_state = torch.cat(
            (member_state, self.empty_member.expand(tokens * heads, -1, -1)),
            dim=1,
        )
        member_mask = torch.cat(
            (
                flat_valid,
                torch.ones(
                    (tokens * heads, 1),
                    device=valid.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        contextual, pooled = self.set_encoder(member_state, member_mask)
        prediction = self.scalar_decoder(contextual[:, :members])
        return (
            pooled.reshape(tokens, heads, -1),
            prediction.reshape(tokens, heads, members, 3),
        )


class DualSourceSetHeadEncoder(nn.Module):
    """Fuse current routing and received-support memory as interacting sets."""

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
        layer: LayerSourceSets,
        source_state: torch.Tensor,
        *,
        route_field_mask: torch.Tensor,
        memory_field_mask: torch.Tensor,
        attention_floor: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route_lag = _source_lag(layer.route_source)
        memory_lag = _source_lag(layer.memory_source)
        route_source_state = source_state[layer.route_source]
        memory_source_state = source_state[layer.memory_source]
        scale = max(float(attention_floor), 1e-8)
        route_state, route_prediction = self.route(
            source_state=route_source_state,
            lag=route_lag,
            weight=layer.route_weight / scale,
            received=layer.route_received / scale,
            received_delta=layer.route_received_delta / scale,
            valid=layer.route_mask,
            field_mask=route_field_mask,
        )
        memory_state, memory_prediction = self.memory(
            source_state=memory_source_state,
            lag=memory_lag,
            weight=layer.memory_current_weight / scale,
            received=layer.memory_received / scale,
            received_delta=layer.memory_received_delta / scale,
            valid=layer.memory_mask,
            field_mask=memory_field_mask,
        )
        context = (
            self.mass(torch.log1p(layer.total_mass / scale))
            + self.tail(torch.log1p(layer.tail_mass / scale))
            + self.degree(torch.log1p(layer.edge_count))
        )
        gate = torch.sigmoid(
            self.route_gate(route_state)
            + self.memory_gate(memory_state)
            + self.context_gate(context)
        )
        fused = self.norm(
            gate * route_state
            + (1.0 - gate) * memory_state
            + self.interaction(route_state * memory_state)
            + context
        )
        return fused, {
            "route_prediction": route_prediction,
            "route_target": _scalar_targets(
                layer.route_weight / scale,
                layer.route_received / scale,
                layer.route_received_delta / scale,
            ),
            "memory_prediction": memory_prediction,
            "memory_target": _scalar_targets(
                layer.memory_current_weight / scale,
                layer.memory_received / scale,
                layer.memory_received_delta / scale,
            ),
        }


class HeadMixer(nn.Module):
    def __init__(
        self, num_heads: int, num_layers: int, config: SetFlowModelConfig
    ) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        self.head_identity = nn.Parameter(torch.empty(num_heads, dim))
        self.layer_identity = nn.Parameter(torch.empty(num_layers, dim))
        self.mask_token = nn.Parameter(torch.empty(dim))
        for parameter in (self.head_identity, self.layer_identity):
            nn.init.normal_(parameter, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.mask_token, std=1.0 / math.sqrt(dim))
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
        self.reconstruct = nn.Linear(dim, dim)
        self.pool_score = nn.Linear(dim, 1)

    def forward(
        self,
        head_state: torch.Tensor,
        *,
        active: torch.Tensor,
        masked: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = head_state
        visible = torch.where(
            masked.unsqueeze(-1),
            self.mask_token.view(1, 1, -1),
            head_state,
        )
        visible = (
            visible
            + self.head_identity.unsqueeze(0)
            + self.layer_identity[int(layer_index)].view(1, 1, -1)
        )
        safe_active = active.clone()
        empty = ~safe_active.any(dim=1)
        if bool(empty.any()):
            safe_active[empty, 0] = True
        mixed = self.encoder(visible, src_key_padding_mask=~safe_active)
        prediction = self.reconstruct(mixed)
        score = self.pool_score(mixed).squeeze(-1)
        score = score.masked_fill(~safe_active, float("-inf"))
        weight = torch.softmax(score, dim=1)
        token_state = (mixed * weight.unsqueeze(-1)).sum(dim=1)
        if bool(empty.any()):
            token_state[empty] = 0.0
        return token_state, prediction, target


class DepthMixer(nn.Module):
    def __init__(self, num_layers: int, config: SetFlowModelConfig) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        self.layer_position = nn.Parameter(torch.empty(num_layers, dim))
        self.mask_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.layer_position, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.mask_token, std=1.0 / math.sqrt(dim))
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
        self.reconstruct = nn.Linear(dim, dim)
        self.pool_score = nn.Linear(dim, 1)

    def forward(
        self, states: torch.Tensor, layer_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible = torch.where(
            layer_mask.view(1, -1, 1),
            self.mask_token.view(1, 1, -1),
            states,
        )
        encoded = self.encoder(visible + self.layer_position.unsqueeze(0))
        prediction = self.reconstruct(encoded)
        weight = torch.softmax(self.pool_score(encoded).squeeze(-1), dim=1)
        return (encoded * weight.unsqueeze(-1)).sum(dim=1), prediction


class CausalSetFlowModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        source_config: SourceSetConfig | None = None,
        model_config: SetFlowModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.source_config = (
            SourceSetConfig() if source_config is None else source_config
        )
        self.config = (
            SetFlowModelConfig() if model_config is None else model_config
        )
        self.source_config.validate()
        self.config.validate()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        dim = int(self.config.hidden_dim)
        self.head_set = DualSourceSetHeadEncoder(self.config)
        self.head_mixer = HeadMixer(num_heads, num_layers, self.config)
        self.depth_recurrence = nn.GRUCell(dim, dim)
        self.depth_mixer = DepthMixer(num_layers, self.config)
        self.time_encoder = nn.GRU(dim, dim, batch_first=True)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(
        self,
        graph: CausalSourceSetGraph,
        *,
        mask_seed: int | None = None,
        apply_masks: bool = True,
        device: str | torch.device | None = None,
    ) -> SetFlowOutput:
        graph.validate()
        if graph.num_layers != self.num_layers or graph.num_heads != self.num_heads:
            raise ValueError("graph geometry differs from model geometry")
        device = (
            next(self.parameters()).device
            if device is None
            else torch.device(device)
        )
        generator = (
            deterministic_generator(mask_seed, device=device)
            if mask_seed is not None
            else None
        )
        sequence_mask = sample_sequence_mask_plan(
            self.num_layers,
            self.config.layer_mask_probability if apply_masks else 0.0,
            device=device,
            generator=generator,
        )
        tokens = graph.response_count
        previous_depth = torch.zeros(
            (tokens, self.config.hidden_dim), device=device
        )
        depth_rows = []
        route_errors = []
        memory_errors = []
        head_errors = []

        for layer_index in range(self.num_layers):
            source_sets = graph.materialize_layer(
                layer_index, self.source_config, device=device
            )
            active = source_sets.route_mask.any(dim=-1) | source_sets.memory_mask.any(
                dim=-1
            )
            plan = sample_layer_mask_plan(
                source_sets.route_mask,
                source_sets.memory_mask,
                element_probability=(
                    self.config.element_mask_probability if apply_masks else 0.0
                ),
                head_probability=(
                    self.config.head_mask_probability if apply_masks else 0.0
                ),
                generator=generator,
            )
            raw_head, targets = self.head_set(
                source_sets,
                previous_depth,
                route_field_mask=plan.route_element,
                memory_field_mask=plan.memory_element,
                attention_floor=graph.attention_floor,
            )
            token_layer, head_prediction, head_target = self.head_mixer(
                raw_head,
                active=active,
                masked=plan.head,
                layer_index=layer_index,
            )
            previous_depth = self.depth_recurrence(token_layer, previous_depth)
            depth_rows.append(previous_depth)
            route_errors.append(
                _scalar_error_per_token(
                    targets["route_prediction"],
                    targets["route_target"],
                    plan.route_element,
                )
            )
            memory_errors.append(
                _scalar_error_per_token(
                    targets["memory_prediction"],
                    targets["memory_target"],
                    plan.memory_element,
                )
            )
            head_errors.append(
                _vector_error_per_token(
                    head_prediction, head_target, plan.head
                )
            )

        depth_state = torch.stack(depth_rows, dim=1)
        final_depth, layer_prediction = self.depth_mixer(
            depth_state, sequence_mask.layer
        )
        layer_error = _layer_error_per_token(
            layer_prediction, depth_state, sequence_mask.layer
        )
        temporal_sequence, _ = self.time_encoder(final_depth.unsqueeze(0))
        token_embedding = temporal_sequence.squeeze(0)
        temporal_error = torch.zeros(tokens, device=device)
        if tokens > 1:
            predicted = self.temporal_predictor(token_embedding[:-1])
            temporal_error[1:] = 1.0 - F.cosine_similarity(
                predicted,
                final_depth[1:],
                dim=-1,
                eps=self.config.epsilon,
            )

        route_error = torch.stack(route_errors, dim=1).mean(dim=1)
        memory_error = torch.stack(memory_errors, dim=1).mean(dim=1)
        head_error = torch.stack(head_errors, dim=1).mean(dim=1)
        route_loss = route_error.mean()
        memory_loss = memory_error.mean()
        head_loss = head_error.mean()
        layer_loss = layer_error.mean()
        temporal_loss = (
            temporal_error[1:].mean() if tokens > 1 else temporal_error.sum()
        )
        variance_loss = variance_floor_loss(token_embedding)
        total = (
            self.config.element_loss_weight * (route_loss + memory_loss)
            + self.config.head_loss_weight * head_loss
            + self.config.layer_loss_weight * layer_loss
            + self.config.temporal_loss_weight * temporal_loss
            + self.config.variance_loss_weight * variance_loss
        )
        return SetFlowOutput(
            token_embedding=token_embedding,
            depth_state=depth_state,
            route_element_error=route_error,
            memory_element_error=memory_error,
            head_reconstruction_error=head_error,
            layer_reconstruction_error=layer_error,
            temporal_prediction_error=temporal_error,
            loss=LossBreakdown(
                total=total,
                route_element=route_loss,
                memory_element=memory_loss,
                head=head_loss,
                layer=layer_loss,
                temporal=temporal_loss,
                variance=variance_loss,
            ),
        )

    @torch.no_grad()
    def deterministic_scores(
        self,
        graph: CausalSourceSetGraph,
        *,
        masks: int,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        previous_training = self.training
        self.eval()
        components: dict[str, list[torch.Tensor]] = {}
        embeddings = []
        for index in range(int(masks)):
            output = self(
                graph,
                mask_seed=int(seed) + index,
                apply_masks=True,
            )
            embeddings.append(output.token_embedding)
            for name, value in output.score_components().items():
                components.setdefault(name, []).append(value)
        result = {
            name: torch.stack(values).mean(dim=0)
            for name, values in components.items()
        }
        result["embedding"] = torch.stack(embeddings).mean(dim=0)
        self.train(previous_training)
        return result


def _source_lag(source: torch.Tensor) -> torch.Tensor:
    token = torch.arange(source.shape[0], device=source.device)[:, None, None]
    return (token - source).clamp_min(1).float()


def _scalar_targets(
    weight: torch.Tensor,
    received: torch.Tensor,
    received_delta: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        (
            torch.log1p(weight.clamp_min(0.0)),
            torch.log1p(received.clamp_min(0.0)),
            torch.asinh(received_delta),
        ),
        dim=-1,
    )


def _scalar_error_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    selected = mask.to(error.dtype)
    return (error * selected).sum(dim=(1, 2)) / selected.sum(dim=(1, 2)).clamp_min(
        1.0
    )


def _vector_error_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = 1.0 - F.cosine_similarity(
        prediction, target, dim=-1, eps=1e-8
    )
    selected = mask.to(error.dtype)
    return (error * selected).sum(dim=1) / selected.sum(dim=1).clamp_min(1.0)


def _layer_error_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layer_mask: torch.Tensor,
) -> torch.Tensor:
    error = 1.0 - F.cosine_similarity(
        prediction, target, dim=-1, eps=1e-8
    )
    selected = layer_mask.to(error.dtype).view(1, -1)
    return (error * selected).sum(dim=1) / selected.sum(dim=1).clamp_min(1.0)
