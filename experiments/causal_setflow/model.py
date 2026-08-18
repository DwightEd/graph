"""Learnable Causal Attention Set-Flow architecture.

The hierarchy mirrors the data rather than flattening it:
source fields -> route/memory sets -> heads -> Transformer depth -> token time.
All training targets are derived from attention itself.

Execution chunking is exact because source sets and head/depth mixer batches are
independent along their leading row dimension. Activation checkpointing wraps
only the neural computation after one exact source-set layer has been
materialized, so backward does not repeat sparse graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    """Encode every weighted source set without flattening its members."""

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
        self.scalar_decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3)
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
        field_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, heads, members = valid.shape
        rows = tokens * heads
        flat_source = source_index.reshape(rows, members)
        flat_lag = lag.reshape(rows, members)
        flat_weight = weight.reshape(rows, members)
        flat_received = received.reshape(rows, members)
        flat_delta = received_delta.reshape(rows, members)
        flat_valid = valid.reshape(rows, members)
        flat_field_mask = field_mask.reshape(rows, members)
        pooled_parts: list[torch.Tensor] = []
        prediction_parts: list[torch.Tensor] = []

        for start in range(0, rows, self.row_chunk_size):
            end = min(rows, start + self.row_chunk_size)
            current_valid = flat_valid[start:end]
            current_source_state = source_state_table[
                flat_source[start:end].long()
            ]
            member_state = self.fields(
                source_state=current_source_state,
                lag=flat_lag[start:end],
                weight=flat_weight[start:end],
                received=flat_received[start:end],
                received_delta=flat_delta[start:end],
                valid=current_valid,
                field_mask=flat_field_mask[start:end],
                set_type=self.set_type,
            )
            chunk_rows = end - start
            member_state = torch.cat(
                (
                    member_state,
                    self.empty_member.expand(chunk_rows, -1, -1),
                ),
                dim=1,
            )
            empty_active = ~current_valid.any(dim=1, keepdim=True)
            member_mask = torch.cat((current_valid, empty_active), dim=1)
            contextual, pooled = self.set_encoder(member_state, member_mask)
            pooled_parts.append(pooled)
            prediction_parts.append(
                self.scalar_decoder(contextual[:, :members])
            )

        pooled = torch.cat(pooled_parts, dim=0)
        prediction = torch.cat(prediction_parts, dim=0)
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
        scale = max(float(attention_floor), 1e-8)
        route_state, route_prediction = self.route(
            source_state_table=source_state,
            source_index=layer.route_source,
            lag=route_lag,
            weight=layer.route_weight / scale,
            received=layer.route_received / scale,
            received_delta=layer.route_received_delta / scale,
            valid=layer.route_mask,
            field_mask=route_field_mask,
        )
        memory_state, memory_prediction = self.memory(
            source_state_table=source_state,
            source_index=layer.memory_source,
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
        self.token_chunk_size = int(config.mixer_token_chunk_size)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_parts: list[torch.Tensor] = []
        prediction_parts: list[torch.Tensor] = []
        for start in range(0, len(head_state), self.token_chunk_size):
            end = min(len(head_state), start + self.token_chunk_size)
            token_state, prediction = self._forward_chunk(
                head_state[start:end],
                active[start:end],
                masked[start:end],
                layer_index=layer_index,
            )
            token_parts.append(token_state)
            prediction_parts.append(prediction)
        return torch.cat(token_parts, dim=0), torch.cat(prediction_parts, dim=0)

    def _forward_chunk(
        self,
        head_state: torch.Tensor,
        active: torch.Tensor,
        masked: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        token_state = torch.where(
            empty.unsqueeze(-1), torch.zeros_like(token_state), token_state
        )
        return token_state, prediction


class DepthMixer(nn.Module):
    def __init__(self, num_layers: int, config: SetFlowModelConfig) -> None:
        super().__init__()
        dim = int(config.hidden_dim)
        self.token_chunk_size = int(config.mixer_token_chunk_size)
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
        final_parts: list[torch.Tensor] = []
        prediction_parts: list[torch.Tensor] = []
        for start in range(0, len(states), self.token_chunk_size):
            end = min(len(states), start + self.token_chunk_size)
            final, prediction = self._forward_chunk(
                states[start:end], layer_mask
            )
            final_parts.append(final)
            prediction_parts.append(prediction)
        return torch.cat(final_parts, dim=0), torch.cat(
            prediction_parts, dim=0
        )

    def _forward_chunk(
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
        # This recurrence is not redundant with the depth mixer: its state is
        # gathered by exact source indices at the next layer and is therefore
        # the SetWalk ancestry state.
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
        depth_rows: list[torch.Tensor] = []
        route_errors: list[torch.Tensor] = []
        memory_errors: list[torch.Tensor] = []
        head_errors: list[torch.Tensor] = []
        use_checkpoint = bool(
            self.config.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

        for layer_index in range(self.num_layers):
            source_sets = graph.materialize_layer(
                layer_index, self.source_config, device=device
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
            if use_checkpoint:
                layer_tensors = source_sets.tensor_tuple()

                def run_layer(
                    previous: torch.Tensor,
                    *values: torch.Tensor,
                    current_layer: int = layer_index,
                ) -> tuple[torch.Tensor, ...]:
                    materialized = LayerSourceSets.from_tensor_tuple(
                        tuple(values[:13])
                    )
                    return self._forward_layer(
                        materialized,
                        previous,
                        route_field_mask=values[13],
                        memory_field_mask=values[14],
                        head_mask=values[15],
                        layer_index=current_layer,
                        attention_floor=graph.attention_floor,
                    )

                (
                    previous_depth,
                    route_error,
                    memory_error,
                    head_error,
                ) = checkpoint(
                    run_layer,
                    previous_depth,
                    *layer_tensors,
                    plan.route_element,
                    plan.memory_element,
                    plan.head,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    determinism_check="default",
                )
            else:
                (
                    previous_depth,
                    route_error,
                    memory_error,
                    head_error,
                ) = self._forward_layer(
                    source_sets,
                    previous_depth,
                    route_field_mask=plan.route_element,
                    memory_field_mask=plan.memory_element,
                    head_mask=plan.head,
                    layer_index=layer_index,
                    attention_floor=graph.attention_floor,
                )
            depth_rows.append(previous_depth)
            route_errors.append(route_error)
            memory_errors.append(memory_error)
            head_errors.append(head_error)

        depth_state = torch.stack(depth_rows, dim=1)
        if use_checkpoint:
            final_depth, layer_prediction = checkpoint(
                self.depth_mixer,
                depth_state,
                sequence_mask.layer,
                use_reentrant=False,
                preserve_rng_state=True,
                determinism_check="default",
            )
        else:
            final_depth, layer_prediction = self.depth_mixer(
                depth_state, sequence_mask.layer
            )
        layer_error = _layer_error_per_token(
            layer_prediction, depth_state.detach(), sequence_mask.layer
        )
        temporal_sequence, _ = self.time_encoder(final_depth.unsqueeze(0))
        token_embedding = temporal_sequence.squeeze(0)
        temporal_error = torch.zeros(tokens, device=device, dtype=torch.float32)
        if tokens > 1:
            predicted = self.temporal_predictor(token_embedding[:-1])
            temporal_error[1:] = 1.0 - F.cosine_similarity(
                predicted.float(),
                final_depth[1:].detach().float(),
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
        variance_loss = variance_floor_loss(token_embedding.float())
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

    def _forward_layer(
        self,
        source_sets: LayerSourceSets,
        previous_depth: torch.Tensor,
        *,
        route_field_mask: torch.Tensor,
        memory_field_mask: torch.Tensor,
        head_mask: torch.Tensor,
        layer_index: int,
        attention_floor: float,
    ) -> tuple[torch.Tensor, ...]:
        active = source_sets.route_mask.any(dim=-1) | source_sets.memory_mask.any(
            dim=-1
        )
        raw_head, targets = self.head_set(
            source_sets,
            previous_depth,
            route_field_mask=route_field_mask,
            memory_field_mask=memory_field_mask,
            attention_floor=attention_floor,
        )
        token_layer, head_prediction = self.head_mixer(
            raw_head,
            active=active,
            masked=head_mask,
            layer_index=layer_index,
        )
        next_depth = self.depth_recurrence(token_layer, previous_depth)
        route_error = _scalar_error_per_token(
            targets["route_prediction"],
            targets["route_target"],
            route_field_mask,
        )
        memory_error = _scalar_error_per_token(
            targets["memory_prediction"],
            targets["memory_target"],
            memory_field_mask,
        )
        head_error = _vector_error_per_token(
            head_prediction, raw_head.detach(), head_mask
        )
        return next_depth, route_error, memory_error, head_error

    @torch.inference_mode()
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
        embeddings: list[torch.Tensor] = []
        for index in range(int(masks)):
            output = self(
                graph,
                mask_seed=int(seed) + index,
                apply_masks=True,
            )
            embeddings.append(output.token_embedding.float())
            for name, value in output.score_components().items():
                components.setdefault(name, []).append(value.float())
        result = {
            name: torch.stack(values).mean(dim=0)
            for name, values in components.items()
        }
        result["embedding"] = torch.stack(embeddings).mean(dim=0)
        self.train(previous_training)
        return result


def _source_lag(source: torch.Tensor) -> torch.Tensor:
    token = torch.arange(source.shape[0], device=source.device)[:, None, None]
    return (token - source.to(torch.long)).clamp_min(1).float()


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
    ).detach()


def _scalar_error_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = F.smooth_l1_loss(
        prediction.float(), target.detach().float(), reduction="none"
    ).mean(dim=-1)
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
        prediction.float(), target.detach().float(), dim=-1, eps=1e-8
    )
    selected = mask.to(error.dtype)
    return (error * selected).sum(dim=1) / selected.sum(dim=1).clamp_min(1.0)


def _layer_error_per_token(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layer_mask: torch.Tensor,
) -> torch.Tensor:
    error = 1.0 - F.cosine_similarity(
        prediction.float(), target.detach().float(), dim=-1, eps=1e-8
    )
    selected = layer_mask.to(error.dtype).view(1, -1)
    return (error * selected).sum(dim=1) / selected.sum(dim=1).clamp_min(1.0)