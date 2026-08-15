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
    row_block_size: int = 4096


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


def _csr_blocks(attention, row_ptr: torch.Tensor, row_count: int, block_size: int):
    """Yield bounded COO views of channel-major CSR rows."""
    device = attention.response_values.device
    for row_start in range(0, row_count, block_size):
        row_end = min(row_start + block_size, row_count)
        lengths = row_ptr[row_start + 1:row_end + 1] - row_ptr[row_start:row_end]
        entry_start = int(row_ptr[row_start])
        entry_end = int(row_ptr[row_end])
        rows = torch.repeat_interleave(
            torch.arange(row_start, row_end, device=device),
            lengths,
            output_size=entry_end - entry_start,
        )
        yield (
            rows,
            attention.response_column_indices[entry_start:entry_end].to(
                device=device, dtype=torch.long
            ),
            attention.response_values[entry_start:entry_end].to(
                device=device, dtype=torch.float32
            ),
        )


def _rr_routes(
    rows: torch.Tensor,
    sources: torch.Tensor,
    weights: torch.Tensor,
    *,
    prompt_count: int,
    response_count: int,
    sample_id: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rr_mask = sources >= prompt_count
    target_rows = rows[rr_mask]
    rr_sources = sources[rr_mask]
    rr_weights = weights[rr_mask]
    channel = torch.div(target_rows, response_count, rounding_mode="floor")
    target_relative = target_rows.remainder(response_count)
    source_relative = rr_sources - prompt_count
    source_rows = channel * response_count + source_relative
    rewired_sources = _lag_bin_rewired_sources(
        source_relative,
        target_relative,
        prompt_count=prompt_count,
        seed=seed,
        sample_id=sample_id,
        channel=channel,
    )
    rewired_source_rows = channel * response_count + rewired_sources - prompt_count
    return target_rows, source_rows, rewired_source_rows, rr_weights


class CausalTopologyEncoder:
    """Turn one sparse attention sample into unpooled causal topology features."""

    def __init__(self, config: CausalTopologyConfig):
        if config.fourier_frequencies < 1:
            raise ValueError("fourier_frequencies must be positive")
        if config.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if config.row_block_size < 1:
            raise ValueError("row_block_size must be positive")
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
        dtype = torch.float32
        prompt_mass = torch.zeros(row_count, dtype=dtype, device=device)
        rr_mass = torch.zeros_like(prompt_mass)
        prompt_support = torch.zeros_like(prompt_mass)
        rr_support = torch.zeros_like(prompt_mass)
        target_position = torch.arange(response_count, device=device).repeat(channels)

        frequencies = torch.arange(
            1, self.config.fourier_frequencies + 1, device=device, dtype=dtype
        )
        prompt_positions = torch.arange(prompt_count, device=device, dtype=dtype)
        phases = 2 * torch.pi * prompt_positions[:, None] * frequencies / prompt_count
        prompt_code = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)
        provenance = torch.zeros(
            row_count, prompt_code.shape[1], dtype=dtype, device=device
        )

        def blocks():
            return _csr_blocks(
                attention, row_ptr, row_count, self.config.row_block_size
            )
        for rows, sources, weights in blocks():
            prompt_mask = sources < prompt_count
            prompt_rows = rows[prompt_mask]
            prompt_weights = weights[prompt_mask]
            prompt_mass.index_add_(0, prompt_rows, prompt_weights)
            prompt_support.index_add_(
                0, prompt_rows, torch.ones_like(prompt_weights)
            )
            provenance.index_add_(
                0,
                prompt_rows,
                prompt_weights[:, None] * prompt_code[sources[prompt_mask]],
            )

            rr_mask = ~prompt_mask
            rr_rows = rows[rr_mask]
            rr_weights = weights[rr_mask]
            rr_mass.index_add_(0, rr_rows, rr_weights)
            rr_support.index_add_(0, rr_rows, torch.ones_like(rr_weights))

        diagonal = attention.attention_diagonal.to(device=device, dtype=torch.float32)
        diagonal = diagonal.reshape(channels, -1)[:, prompt_count:].reshape(-1)
        prompt_mean = prompt_mass / float(prompt_count)
        generated_mean = (rr_mass + diagonal) / (target_position + 1).to(dtype)
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
                rr_support / target_position.to(dtype),
                torch.zeros_like(rr_support),
            ),
        ), dim=1)
        provenance = torch.where(
            prompt_mass[:, None] > 0,
            provenance / prompt_mass[:, None].clamp_min(self.config.epsilon),
            torch.zeros_like(provenance),
        )

        neighbor_mean = torch.zeros_like(base_state)
        rewired_neighbor_mean = torch.zeros_like(base_state)
        for rows, sources, weights in blocks():
            target_rows, source_rows, rewired_rows, rr_weights = _rr_routes(
                rows,
                sources,
                weights,
                prompt_count=prompt_count,
                response_count=response_count,
                sample_id=str(attention.sample_id),
                seed=self.config.rewire_seed,
            )
            normalized = rr_weights / rr_mass[target_rows]
            neighbor_mean.index_add_(
                0, target_rows, normalized[:, None] * base_state[source_rows]
            )
            rewired_neighbor_mean.index_add_(
                0, target_rows, normalized[:, None] * base_state[rewired_rows]
            )

        absolute_difference = torch.zeros_like(base_state)
        neighborhood_variance = torch.zeros_like(base_state)
        two_hop = torch.zeros_like(base_state)
        rewired_absolute_difference = torch.zeros_like(base_state)
        rewired_neighborhood_variance = torch.zeros_like(base_state)
        rewired_two_hop = torch.zeros_like(base_state)
        for rows, sources, weights in blocks():
            target_rows, source_rows, rewired_rows, rr_weights = _rr_routes(
                rows,
                sources,
                weights,
                prompt_count=prompt_count,
                response_count=response_count,
                sample_id=str(attention.sample_id),
                seed=self.config.rewire_seed,
            )
            normalized = rr_weights / rr_mass[target_rows]
            scaled = normalized[:, None]
            absolute_difference.index_add_(
                0,
                target_rows,
                scaled * (base_state[target_rows] - base_state[source_rows]).abs(),
            )
            neighborhood_variance.index_add_(
                0,
                target_rows,
                scaled * (base_state[source_rows] - neighbor_mean[target_rows]).square(),
            )
            two_hop.index_add_(
                0, target_rows, scaled * neighbor_mean[source_rows]
            )
            rewired_absolute_difference.index_add_(
                0,
                target_rows,
                scaled * (base_state[target_rows] - base_state[rewired_rows]).abs(),
            )
            rewired_neighborhood_variance.index_add_(
                0,
                target_rows,
                scaled * (
                    base_state[rewired_rows] - rewired_neighbor_mean[target_rows]
                ).square(),
            )
            rewired_two_hop.index_add_(
                0, target_rows, scaled * rewired_neighbor_mean[rewired_rows]
            )

        rr_one_hop = torch.cat((
            neighbor_mean, absolute_difference, neighborhood_variance,
        ), dim=1)
        rr_two_hop = two_hop
        rewired_one_hop = torch.cat((
            rewired_neighbor_mean,
            rewired_absolute_difference,
            rewired_neighborhood_variance,
        ), dim=1)

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
