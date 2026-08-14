"""Label-free sparse attention mechanisms and compact token features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

MECHANISM_NAMES = (
    "retained_length_normalized_lookback", "retained_prompt_share",
    "prompt_share_lower_bound", "prompt_share_upper_bound",
    "retained_prompt_mass", "retained_history_mass", "diagonal_mass", "retained_mass", "unresolved_mass",
    "coarsened_entropy_lower_bound", "coarsened_top1_upper_bound",
    "coarsened_hhi_upper_bound", "coarsened_effective_support_count_lower_bound",
    "retained_inverse_history_lag", "cache_diagonal_share_upper_bound", "cache_diagonal_share_lower_bound",
)
MECHANISM_FAMILY_SLICES = {
    "routing": slice(0, 4),
    "mass": slice(4, 9),
    "concentration": slice(9, 13),
    "locality": slice(13, 16),
}


@dataclass(frozen=True)
class MechanismResult:
    """Per-response-token mechanisms, ordered [token, layer, head, mechanism]."""

    values: torch.Tensor
    valid: torch.Tensor
    names: tuple[str, ...] = MECHANISM_NAMES
    family_slices: dict[str, slice] | None = None

    def __post_init__(self) -> None:
        if self.family_slices is None:
            object.__setattr__(self, "family_slices", MECHANISM_FAMILY_SLICES)


@dataclass(frozen=True)
class CompactFeatures:
    values: torch.Tensor
    valid: torch.Tensor
    names: tuple[str, ...]
    family_slices: dict[str, slice]


def extract_token_mechanisms(sample, *, channel_block_size: int = 256) -> MechanismResult:
    """Extract mechanisms from canonical CSR without averaging layers or heads.

    The CSR is processed in channel blocks.  Missing attention below the stored
    floor remains in the explicit OTHER bucket; it is never redistributed.
    """
    device = sample.response_values.device
    response_count, channels = sample.num_response_tokens, sample.num_channels
    rows = response_count * channels
    result = torch.zeros((rows, len(MECHANISM_NAMES)), device=device, dtype=torch.float32)
    valid = torch.ones_like(result, dtype=torch.bool)
    diagonal = sample.attention_diagonal.float().reshape(channels, sample.num_tokens)
    diagonal = diagonal[:, sample.response_idx:].reshape(-1)
    row_ptr = sample.response_row_ptr.long()
    columns = sample.response_column_indices.long()
    values = sample.response_values.float()
    response_positions = torch.arange(response_count, device=device)
    target_positions = sample.response_idx + response_positions

    for channel_start in range(0, channels, channel_block_size):
        channel_end = min(channel_start + channel_block_size, channels)
        first_row, last_row = channel_start * response_count, channel_end * response_count
        start, end = int(row_ptr[first_row]), int(row_ptr[last_row])
        local_rows = last_row - first_row
        local = torch.zeros((local_rows, len(MECHANISM_NAMES)), device=device, dtype=torch.float32)
        local_diagonal = diagonal[first_row:last_row]
        local[:, 6] = local_diagonal

        if start != end:
            edge_counts = row_ptr[first_row + 1:last_row + 1] - row_ptr[first_row:last_row]
            edge_rows = torch.repeat_interleave(torch.arange(local_rows, device=device), edge_counts)
            edge_columns = columns[start:end]
            edge_values = values[start:end]
            token_index = (edge_rows + first_row).remainder(response_count)
            targets = target_positions[token_index]
            prompt = edge_columns < sample.response_idx
            history = ~prompt
            local[:, 4].index_add_(0, edge_rows[prompt], edge_values[prompt])
            local[:, 5].index_add_(0, edge_rows[history], edge_values[history])
            lags = (targets - edge_columns).float()
            entropy_terms = -(edge_values * edge_values.clamp_min(1e-12).log())
            local[:, 9].index_add_(0, edge_rows, entropy_terms)
            local[:, 11].index_add_(0, edge_rows, edge_values.square())
            history_weighted_lag = torch.zeros(local_rows, device=device)
            history_weighted_lag.index_add_(0, edge_rows[history], edge_values[history] * lags[history])
            local[:, 13] = torch.where(local[:, 5] > 0, local[:, 5] / history_weighted_lag.clamp_min(1e-12), 0)
            edge_max = torch.full((local_rows,), -torch.inf, device=device)
            edge_max.scatter_reduce_(0, edge_rows, edge_values, reduce="amax", include_self=True)
        else:
            edge_counts = torch.zeros(local_rows, device=device, dtype=torch.long)
            edge_max = torch.full((local_rows,), -torch.inf, device=device)

        local[:, 7] = local[:, 4] + local[:, 5] + local_diagonal
        local[:, 8] = (1 - local[:, 7]).clamp_min(0)
        token_index = torch.arange(first_row, last_row, device=device).remainder(response_count)
        prompt_mean = local[:, 4] / float(sample.response_idx)
        response_mean = (local[:, 5] + local_diagonal) / (token_index + 1).float()
        routing_total = prompt_mean + response_mean
        local[:, 0] = prompt_mean / routing_total.clamp_min(1e-12)
        local[:, 1] = local[:, 4] / local[:, 7].clamp_min(1e-12)
        total_mass = local[:, 7] + local[:, 8]
        local[:, 2] = local[:, 4] / total_mass.clamp_min(1e-12)
        local[:, 3] = (local[:, 4] + local[:, 8]) / total_mass.clamp_min(1e-12)
        local[:, 9] += -(local_diagonal * local_diagonal.clamp_min(1e-12).log())
        local[:, 9] += -(local[:, 8] * local[:, 8].clamp_min(1e-12).log())
        local[:, 11] += local_diagonal.square() + local[:, 8].square()
        category_count = (
            edge_counts
            + (local_diagonal > 0).long()
            + (local[:, 8] > 0).long()
        ).float()
        local[:, 9] = local[:, 9] / total_mass.clamp_min(1e-12) + total_mass.clamp_min(1e-12).log()
        local[:, 10] = (
            torch.maximum(torch.maximum(edge_max, local_diagonal), local[:, 8])
            / total_mass.clamp_min(1e-12)
        )
        local[:, 11] /= total_mass.square().clamp_min(1e-12)
        local[:, 12] = 1 / local[:, 11].clamp_min(1e-12)
        response_mass = local[:, 5] + local_diagonal
        local[:, 14] = local_diagonal / response_mass.clamp_min(1e-12)
        local[:, 15] = local_diagonal / (local[:, 5] + local_diagonal + local[:, 8]).clamp_min(1e-12)
        has_retained = local[:, 7] > 0
        valid_block = torch.ones_like(local, dtype=torch.bool)
        valid_block[:, 0] = routing_total > 0
        valid_block[:, 1] = has_retained
        valid_block[:, 2:4] = (total_mass > 0)[:, None]
        valid_block[:, 9] = category_count > 1
        valid_block[:, 10:13] = (category_count > 0)[:, None]
        valid_block[:, 13] = local[:, 5] > 0
        valid_block[:, 14] = response_mass > 0
        valid_block[:, 15] = (local[:, 5] + local_diagonal + local[:, 8]) > 0
        bounds_valid = local[:, 7] <= 1
        valid_block[:, 2:4] &= bounds_valid[:, None]
        valid_block[:, 9:13] &= bounds_valid[:, None]
        valid_block[:, 14:16] &= bounds_valid[:, None]
        result[first_row:last_row] = local
        valid[first_row:last_row] = valid_block

    values_tlhk = result.reshape(channels, response_count, -1).permute(1, 0, 2)
    valid_tlhk = valid.reshape(channels, response_count, -1).permute(1, 0, 2)
    return MechanismResult(
        values_tlhk.reshape(response_count, sample.num_layers, sample.num_heads, -1),
        valid_tlhk.reshape(response_count, sample.num_layers, sample.num_heads, -1),
    )


def compact_token_features(result: MechanismResult, *, ema_decay: float = 0.9) -> CompactFeatures:
    """Compress layer/head mechanisms while retaining fixed bin and temporal views."""
    values, valid = result.values, result.valid
    tokens, layers, heads, mechanisms = values.shape
    blocks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    names: list[str] = []

    def pooled(value: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]):
        count = mask.sum(dim=dimensions)
        return (value * mask).sum(dim=dimensions) / count.clamp_min(1), count > 0

    global_mean, global_valid = pooled(values, valid, (1, 2))
    global_std = torch.sqrt(pooled((values - global_mean[:, None, None]).square(), valid, (1, 2))[0])
    blocks.extend((global_mean, global_std))
    masks.extend((global_valid, global_valid))
    names.extend(f"{name}:{view}" for view in ("global_mean", "global_std") for name in result.names)

    for axis, size in (("layer", layers), ("head", heads)):
        for bin_index in range(4):
            lower, upper = bin_index * size // 4, (bin_index + 1) * size // 4
            if lower == upper:
                block = torch.zeros((tokens, mechanisms), device=values.device)
                mask = torch.zeros_like(block, dtype=torch.bool)
            elif axis == "layer":
                block, mask = pooled(values[:, lower:upper], valid[:, lower:upper], (1, 2))
            else:
                block, mask = pooled(values[:, :, lower:upper], valid[:, :, lower:upper], (1, 2))
            blocks.append(block)
            masks.append(mask)
            names.extend(f"{name}:{axis}_bin_{bin_index}" for name in result.names)

    delta = torch.zeros_like(global_mean)
    delta[1:] = global_mean[1:] - global_mean[:-1]
    delta_mask = torch.zeros_like(global_valid)
    delta_mask[1:] = global_valid[1:] & global_valid[:-1]
    blocks.append(delta)
    masks.append(delta_mask)
    names.extend(f"{name}:token_delta" for name in result.names)

    ema = torch.zeros_like(global_mean)
    innovation = torch.zeros_like(global_mean)
    innovation_mask = torch.zeros_like(global_valid)
    state = torch.zeros(mechanisms, device=values.device)
    has_history = torch.zeros(mechanisms, dtype=torch.bool, device=values.device)
    for token in range(tokens):
        current = global_valid[token]
        innovation[token, current] = global_mean[token, current] - state[current]
        innovation_mask[token] = current & has_history
        state[current] = torch.where(
            has_history[current],
            ema_decay * state[current] + (1 - ema_decay) * global_mean[token, current],
            global_mean[token, current],
        )
        has_history |= current
        ema[token] = state
    blocks.append(innovation)
    masks.append(innovation_mask)
    names.extend(f"{name}:ema_innovation" for name in result.names)

    if layers < 2:
        drift = torch.zeros_like(global_mean)
        drift_mask = torch.zeros_like(global_valid)
    else:
        early, early_mask = pooled(values[:, :layers // 2], valid[:, :layers // 2], (1, 2))
        late, late_mask = pooled(values[:, layers // 2:], valid[:, layers // 2:], (1, 2))
        drift, drift_mask = late - early, late_mask & early_mask
    blocks.append(drift)
    masks.append(drift_mask)
    names.extend(f"{name}:early_late_layer_drift" for name in result.names)

    view_count = len(blocks)
    compact_values = torch.stack(blocks, dim=1).permute(0, 2, 1).reshape(tokens, -1)
    compact_valid = torch.stack(masks, dim=1).permute(0, 2, 1).reshape(tokens, -1)
    views = tuple(name.split(":", 1)[1] for name in names[::mechanisms])
    compact_names = tuple(
        f"{mechanism}:{view}" for mechanism in result.names for view in views
    )
    compact_slices = {
        family: slice(start.start * view_count, start.stop * view_count)
        for family, start in result.family_slices.items()
    }
    return CompactFeatures(compact_values, compact_valid, compact_names, compact_slices)
