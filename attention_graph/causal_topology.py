"""Topology-preserving, label-free features from retained causal attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .evidence_flow import _lag_bin_rewired_sources

if TYPE_CHECKING:
    from cache import AttentionSample


@dataclass(frozen=True)
class CausalTopologyConfig:
    fourier_frequencies: int = 4
    rewire_seed: int = 0
    epsilon: float = 1e-8


@dataclass(frozen=True)
class TopologyEncoding:
    """Per-response-token tensors, all shaped ``[T, L, H, feature]``."""

    balance_log_scale: torch.Tensor
    attention_marginals: torch.Tensor
    retained_support: torch.Tensor
    prompt_provenance: torch.Tensor
    rr_one_hop: torch.Tensor
    rr_two_hop: torch.Tensor
    rewired_rr_one_hop: torch.Tensor
    rewired_rr_two_hop: torch.Tensor


def _as_token_channels(values: torch.Tensor, layers: int, heads: int) -> torch.Tensor:
    _, tokens, features = values.shape
    return values.permute(1, 0, 2).reshape(tokens, layers, heads, features)


def _rr_features(
    base_state: torch.Tensor,
    target_rows: torch.Tensor,
    source_rows: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one and two RR hops without averaging channels or signed deltas."""
    rows, _ = base_state.shape
    mass = torch.zeros(rows, dtype=weights.dtype, device=weights.device)
    mass.index_add_(0, target_rows, weights)
    normalized = weights / mass[target_rows]

    neighbor_mean = torch.zeros_like(base_state)
    neighbor_mean.index_add_(
        0, target_rows, normalized[:, None] * base_state[source_rows]
    )
    absolute_difference = torch.zeros_like(base_state)
    absolute_difference.index_add_(
        0,
        target_rows,
        normalized[:, None] * (base_state[target_rows] - base_state[source_rows]).abs(),
    )
    neighborhood_variance = torch.zeros_like(base_state)
    neighborhood_variance.index_add_(
        0,
        target_rows,
        normalized[:, None] * (base_state[source_rows] - neighbor_mean[target_rows]).square(),
    )
    two_hop = torch.zeros_like(base_state)
    two_hop.index_add_(
        0, target_rows, normalized[:, None] * neighbor_mean[source_rows]
    )
    one_hop = torch.cat(
        (neighbor_mean, absolute_difference, neighborhood_variance), dim=1
    )
    return one_hop, two_hop


class CausalTopologyEncoder:
    """Turn one sparse attention sample into unpooled causal topology features."""

    def __init__(self, config: CausalTopologyConfig):
        if config.fourier_frequencies < 1:
            raise ValueError("fourier_frequencies must be positive")
        if config.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.config = config

    def encode(self, attention: AttentionSample) -> TopologyEncoding:
        """Use only retained CSR edges and exact stored diagonal attention values."""
        with torch.no_grad():
            return self._encode(attention)

    def _encode(self, attention: AttentionSample) -> TopologyEncoding:
        prompt_count = int(attention.response_idx)
        response_count = int(attention.num_response_tokens)
        layers = int(attention.num_layers)
        heads = int(attention.num_heads)
        channels = layers * heads
        device = attention.response_values.device
        row_count = channels * response_count

        row_ptr = attention.response_row_ptr.to(device=device, dtype=torch.long)
        row_lengths = row_ptr[1:] - row_ptr[:-1]
        rows = torch.repeat_interleave(
            torch.arange(row_count, device=device), row_lengths,
            output_size=int(attention.response_values.numel()),
        )
        sources = attention.response_column_indices.to(device=device, dtype=torch.long)
        weights = attention.response_values.to(device=device, dtype=torch.float32)
        floor = torch.as_tensor(
            attention.attention_floor,
            dtype=attention.response_values.dtype,
            device=device,
        ).float()
        excess = (weights - floor).clamp_min(0)

        prompt_mask = sources < prompt_count
        prompt_rows = rows[prompt_mask]
        prompt_weights = weights[prompt_mask]
        prompt_excess = excess[prompt_mask]
        prompt_mass = torch.full(
            (row_count,), prompt_count * floor, dtype=weights.dtype, device=device
        )
        prompt_mass.index_add_(0, prompt_rows, prompt_excess)
        prompt_support = torch.zeros_like(prompt_mass)
        prompt_support.index_add_(0, prompt_rows, torch.ones_like(prompt_weights))

        rr_mask = ~prompt_mask
        rr_rows = rows[rr_mask]
        rr_sources = sources[rr_mask]
        rr_weights = weights[rr_mask]
        rr_excess = excess[rr_mask]
        target_position = torch.arange(response_count, device=device).repeat(channels)
        rr_mass = target_position.to(weights.dtype) * floor
        rr_mass.index_add_(0, rr_rows, rr_excess)
        rr_support = torch.zeros_like(prompt_mass)
        rr_support.index_add_(0, rr_rows, torch.ones_like(rr_weights))

        diagonal = attention.attention_diagonal.to(device=device, dtype=torch.float32)
        diagonal = diagonal.reshape(channels, -1)[:, prompt_count:].reshape(-1)
        prompt_mean = prompt_mass / float(prompt_count)
        generated_mean = (rr_mass + diagonal) / (target_position + 1).to(weights.dtype)
        total_mean = prompt_mean + generated_mean
        balance = torch.where(
            total_mean > 0,
            prompt_mean / total_mean,
            torch.full_like(total_mean, float(attention.attention_floor)),
        )
        base_state = torch.stack((
            balance,
            torch.log(total_mean.clamp_min(self.config.epsilon)),
        ), dim=1)
        retained_support = torch.stack((
            prompt_support / float(prompt_count),
            torch.where(
                target_position > 0,
                rr_support / target_position.to(weights.dtype),
                torch.zeros_like(rr_support),
            ),
        ), dim=1)

        frequencies = torch.arange(
            1, self.config.fourier_frequencies + 1, device=device, dtype=weights.dtype
        )
        prompt_positions = torch.arange(prompt_count, device=device, dtype=weights.dtype)
        phases = 2 * torch.pi * prompt_positions[:, None] * frequencies / prompt_count
        prompt_code = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)
        provenance = (
            floor * prompt_code.sum(dim=0, keepdim=True)
        ).expand(row_count, -1).clone()
        provenance.index_add_(
            0, prompt_rows, prompt_excess[:, None] * prompt_code[sources[prompt_mask]]
        )
        provenance = torch.where(
            prompt_mass[:, None] > 0,
            provenance / prompt_mass[:, None].clamp_min(self.config.epsilon),
            torch.zeros_like(provenance),
        )

        informative_rr = rr_excess > 0
        rr_rows = rr_rows[informative_rr]
        rr_sources = rr_sources[informative_rr]
        rr_excess = rr_excess[informative_rr]
        channel = torch.div(rr_rows, response_count, rounding_mode="floor")
        rr_source_relative = rr_sources - prompt_count
        rr_source_rows = channel * response_count + rr_source_relative
        rr_one_hop, rr_two_hop = _rr_features(
            base_state, rr_rows, rr_source_rows, rr_excess
        )

        rr_target_relative = rr_rows.remainder(response_count)
        rewired_sources = _lag_bin_rewired_sources(
            rr_source_relative,
            rr_target_relative,
            prompt_count=prompt_count,
            seed=self.config.rewire_seed,
            sample_id=attention.sample_id,
            channel=channel,
        )
        rewired_source_rows = channel * response_count + (rewired_sources - prompt_count)
        rewired_one_hop, rewired_two_hop = _rr_features(
            base_state, rr_rows, rewired_source_rows, rr_excess
        )

        return TopologyEncoding(
            balance_log_scale=_as_token_channels(
                base_state.reshape(channels, response_count, -1), layers, heads
            ),
            attention_marginals=_as_token_channels(
                torch.stack((prompt_mass, rr_mass, diagonal), dim=1).reshape(
                    channels, response_count, -1
                ),
                layers,
                heads,
            ),
            retained_support=_as_token_channels(
                retained_support.reshape(channels, response_count, -1), layers, heads
            ),
            prompt_provenance=_as_token_channels(
                provenance.reshape(channels, response_count, -1), layers, heads
            ),
            rr_one_hop=_as_token_channels(
                rr_one_hop.reshape(channels, response_count, -1), layers, heads
            ),
            rr_two_hop=_as_token_channels(
                rr_two_hop.reshape(channels, response_count, -1), layers, heads
            ),
            rewired_rr_one_hop=_as_token_channels(
                rewired_one_hop.reshape(channels, response_count, -1), layers, heads
            ),
            rewired_rr_two_hop=_as_token_channels(
                rewired_two_hop.reshape(channels, response_count, -1), layers, heads
            ),
        )
