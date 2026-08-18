"""Build deterministic causal SetWalk embeddings from sparse attention.

Each ``(response token, layer, head)`` row is one weighted hyperedge whose
members are its retained prompt and response-history sources.  A walk step
does not collapse that hyperedge to a pairwise edge: it follows one causal
response source to the previous layer while carrying a permutation-invariant
encoding of the *entire* source set at every visited hyperedge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


VIEW_NAMES = (
    "setwalk",
    "no_walk",
    "pairwise_walk",
    "layer_shuffled",
)

DIAGNOSTIC_NAMES = (
    "premature_local_collapse",
    "local_collapse_mean",
    "head_set_disagreement",
    "response_chain_survival_hop1",
    "response_chain_survival_hop2",
    "layer_order_dependence",
)

DIAGNOSTIC_DIRECTIONS = {
    "premature_local_collapse": "higher",
    "local_collapse_mean": "higher",
    "head_set_disagreement": "higher",
    "response_chain_survival_hop1": "higher",
    "response_chain_survival_hop2": "higher",
    "layer_order_dependence": "lower",
}

LAYER_PROFILE_NAMES = (
    "prompt_fraction",
    "response_fraction",
    "recent_response_fraction",
    "source_concentration",
    "local_response_collapse",
    "head_set_disagreement",
    "hop1_survival",
    "hop2_survival",
)


@dataclass(frozen=True)
class SetWalkConfig:
    """Configuration with no label-selected or trainable representation weights."""

    fourier_features: int = 8
    dct_components: int = 3
    recent_lag_max: int = 4
    block_rows: int = 8192
    seed: int = 20260818
    epsilon: float = 1e-8

    def validate(self) -> None:
        if min(
            int(self.fourier_features),
            int(self.dct_components),
            int(self.recent_lag_max),
            int(self.block_rows),
        ) < 1:
            raise ValueError("SetWalk integer settings must be positive")
        if not 0 < float(self.epsilon) < 1:
            raise ValueError("epsilon must be in (0, 1)")


def _fourier_parameters(count: int, seed: int, *, device, dtype):
    generator = np.random.default_rng(int(seed))
    frequency = generator.normal(size=(int(count), 3)).astype(np.float32)
    frequency /= np.linalg.norm(frequency, axis=1, keepdims=True).clip(1e-6)
    frequency *= generator.uniform(0.5, 4.0, size=(int(count), 1)).astype(
        np.float32
    )
    phase = generator.uniform(0.0, 2.0 * np.pi, size=int(count)).astype(
        np.float32
    )
    return (
        torch.as_tensor(frequency, device=device, dtype=dtype),
        torch.as_tensor(phase, device=device, dtype=dtype),
    )


def _dct_basis(components: int, length: int, *, device, dtype):
    keep = min(int(components), int(length))
    position = torch.arange(length, device=device, dtype=dtype)
    frequency = torch.arange(keep, device=device, dtype=dtype)[:, None]
    basis = torch.cos(torch.pi * (position[None, :] + 0.5) * frequency / length)
    basis[0] *= np.sqrt(1.0 / length)
    if keep > 1:
        basis[1:] *= np.sqrt(2.0 / length)
    return basis


def _head_pool(values, weights, epsilon):
    """Permutation-invariant weighted head mean and coordinate-wise spread."""

    denominator = weights.sum(dim=0)
    mean = (values * weights[:, :, None]).sum(dim=0) / denominator[:, None].clamp_min(
        epsilon
    )
    difference = values - mean[None, :, :]
    dispersion = (
        (difference.square() * weights[:, :, None]).sum(dim=0)
        / denominator[:, None].clamp_min(epsilon)
    ).sqrt()
    valid = denominator > epsilon
    mean = torch.where(valid[:, None], mean, torch.zeros_like(mean))
    dispersion = torch.where(
        valid[:, None], dispersion, torch.zeros_like(dispersion)
    )
    return mean, dispersion, weights > epsilon


def _dct_embed(profile, config):
    # profile: [response, ordered layer, feature]
    basis = _dct_basis(
        config.dct_components,
        profile.shape[1],
        device=profile.device,
        dtype=profile.dtype,
    )
    transformed = torch.einsum("kl,rlp->rkp", basis, profile)
    return transformed.reshape(len(profile), -1)


def _propagate_view(base, row_mass, rr_edges, order, config, *, include_walk=True):
    """Compute exact first moments of one- and two-step causal SetWalks."""

    layers, heads, response_count, feature_dim = base.shape
    epsilon = float(config.epsilon)
    ordered_base = []
    ordered_base_dispersion = []
    ordered_hop1 = []
    ordered_hop1_dispersion = []
    ordered_hop1_survival = []
    ordered_hop2 = []
    ordered_hop2_dispersion = []
    ordered_hop2_survival = []

    base_mean = {}
    hop1_mean = {}
    hop1_survival = {}

    for sequence_index, layer in enumerate(map(int, order)):
        current_mass = row_mass[layer]
        current_base, current_dispersion, current_active = _head_pool(
            base[layer], current_mass, epsilon
        )
        base_mean[sequence_index] = current_base
        ordered_base.append(current_base)
        ordered_base_dispersion.append(current_dispersion)

        message1 = torch.zeros(
            (heads, response_count, feature_dim),
            dtype=base.dtype,
            device=base.device,
        )
        mass1 = torch.zeros(
            (heads, response_count), dtype=base.dtype, device=base.device
        )
        message2 = torch.zeros_like(message1)
        mass2 = torch.zeros_like(mass1)

        if include_walk and sequence_index > 0:
            edge_head, edge_query, edge_source, edge_weight = rr_edges[layer]
            if edge_weight.numel():
                probability = edge_weight / current_mass[
                    edge_head, edge_query
                ].clamp_min(epsilon)
                flat_row = edge_head * response_count + edge_query
                message1.view(-1, feature_dim).index_add_(
                    0,
                    flat_row,
                    probability[:, None] * base_mean[sequence_index - 1][edge_source],
                )
                mass1.view(-1).index_add_(0, flat_row, probability)

                if sequence_index > 1:
                    inherited = hop1_survival[sequence_index - 1][edge_source]
                    probability2 = probability * inherited
                    message2.view(-1, feature_dim).index_add_(
                        0,
                        flat_row,
                        probability2[:, None]
                        * hop1_mean[sequence_index - 1][edge_source],
                    )
                    mass2.view(-1).index_add_(0, flat_row, probability2)

        conditional1 = message1 / mass1[:, :, None].clamp_min(epsilon)
        conditional2 = message2 / mass2[:, :, None].clamp_min(epsilon)
        pooled1, dispersion1, _ = _head_pool(conditional1, mass1, epsilon)
        pooled2, dispersion2, _ = _head_pool(conditional2, mass2, epsilon)
        active_count = current_active.to(base.dtype).sum(dim=0).clamp_min(1.0)
        survival1 = (mass1 * current_active.to(base.dtype)).sum(dim=0) / active_count
        survival2 = (mass2 * current_active.to(base.dtype)).sum(dim=0) / active_count

        hop1_mean[sequence_index] = pooled1
        hop1_survival[sequence_index] = survival1
        ordered_hop1.append(pooled1)
        ordered_hop1_dispersion.append(dispersion1)
        ordered_hop1_survival.append(survival1)
        ordered_hop2.append(pooled2)
        ordered_hop2_dispersion.append(dispersion2)
        ordered_hop2_survival.append(survival2)

    base_stack = torch.stack(ordered_base, dim=1)
    base_dispersion = torch.stack(ordered_base_dispersion, dim=1)
    if not include_walk:
        profile = torch.cat((base_stack, base_dispersion), dim=2)
        return _dct_embed(profile, config), {
            "base_disagreement": base_dispersion,
        }

    hop1_stack = torch.stack(ordered_hop1, dim=1)
    hop2_stack = torch.stack(ordered_hop2, dim=1)
    hop1_dispersion = torch.stack(ordered_hop1_dispersion, dim=1)
    hop2_dispersion = torch.stack(ordered_hop2_dispersion, dim=1)
    survival1 = torch.stack(ordered_hop1_survival, dim=1)
    survival2 = torch.stack(ordered_hop2_survival, dim=1)
    profile = torch.cat(
        (
            base_stack,
            hop1_stack,
            hop2_stack,
            base_dispersion,
            hop1_dispersion,
            hop2_dispersion,
            survival1[:, :, None],
            survival2[:, :, None],
        ),
        dim=2,
    )
    return _dct_embed(profile, config), {
        "base_disagreement": base_dispersion,
        "hop1_survival": survival1,
        "hop2_survival": survival2,
    }


def _append_layer_edges(storage, layer, head, query, source, weight):
    for current_layer in torch.unique(layer).tolist():
        selected = layer == int(current_layer)
        storage[int(current_layer)][0].append(head[selected])
        storage[int(current_layer)][1].append(query[selected])
        storage[int(current_layer)][2].append(source[selected])
        storage[int(current_layer)][3].append(weight[selected])


def _finalize_layer_edges(storage, *, device, dtype):
    result = []
    for head, query, source, weight in storage:
        if weight:
            result.append(
                tuple(
                    torch.cat(values)
                    for values in (head, query, source, weight)
                )
            )
        else:
            empty_index = torch.empty(0, dtype=torch.long, device=device)
            empty_weight = torch.empty(0, dtype=dtype, device=device)
            result.append((empty_index, empty_index, empty_index, empty_weight))
    return result


def extract_setwalk_representations(sample, config: SetWalkConfig | None = None):
    """Return label-free token embeddings and interpretable layer profiles."""

    config = SetWalkConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("SetWalk requires non-empty prompt and response")
    device = attention.response_values.device
    dtype = torch.float32
    frequency, phase = _fourier_parameters(
        config.fourier_features,
        config.seed,
        device=device,
        dtype=dtype,
    )

    rows = layers * heads * response_count
    total_mass = torch.zeros(rows, dtype=dtype, device=device)
    prompt_mass = torch.zeros_like(total_mass)
    rr_mass = torch.zeros_like(total_mass)
    recent_rr_mass = torch.zeros_like(total_mass)
    edge_count = torch.zeros_like(total_mass)
    squared_mass = torch.zeros_like(total_mass)
    coordinate_sum = torch.zeros((rows, 3), dtype=dtype, device=device)
    cosine_sum = torch.zeros(
        (rows, config.fourier_features), dtype=dtype, device=device
    )
    sine_sum = torch.zeros_like(cosine_sum)
    edge_storage = [[[], [], [], []] for _ in range(layers)]

    response_scale = max(1.0, float(response_count - 1))
    lag_scale = max(1.0, float(np.log1p(response_count)))
    prompt_scale = max(1.0, float(prompt_count - 1))

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        causal = block.source < block.target
        if not bool(causal.any()):
            continue
        layer = block.layer[causal].long()
        head = block.head[causal].long()
        query = block.query[causal].long()
        source = block.source[causal].long()
        weight = block.weight[causal].to(dtype=dtype)
        row = (layer * heads + head) * response_count + query
        is_prompt = source < prompt_count
        is_history = ~is_prompt
        lag = (query - (source - prompt_count)).clamp_min(1)

        role = torch.where(
            is_prompt,
            torch.full_like(weight, -1.0),
            torch.ones_like(weight),
        )
        local_coordinate = torch.where(
            is_prompt,
            source.to(dtype) / prompt_scale,
            torch.log1p(lag.to(dtype)) / lag_scale,
        )
        source_coordinate = torch.where(
            is_prompt,
            source.to(dtype) / prompt_scale,
            (source - prompt_count).to(dtype) / response_scale,
        )
        coordinate = torch.stack((role, local_coordinate, source_coordinate), dim=1)
        angle = 2.0 * torch.pi * (coordinate @ frequency.T) + phase

        total_mass.index_add_(0, row, weight)
        edge_count.index_add_(0, row, torch.ones_like(weight))
        squared_mass.index_add_(0, row, weight.square())
        coordinate_sum.index_add_(0, row, weight[:, None] * coordinate)
        cosine_sum.index_add_(0, row, weight[:, None] * torch.cos(angle))
        sine_sum.index_add_(0, row, weight[:, None] * torch.sin(angle))
        if bool(is_prompt.any()):
            prompt_mass.index_add_(0, row[is_prompt], weight[is_prompt])
        if bool(is_history.any()):
            history_row = row[is_history]
            history_weight = weight[is_history]
            history_lag = lag[is_history]
            rr_mass.index_add_(0, history_row, history_weight)
            recent = history_lag <= int(config.recent_lag_max)
            if bool(recent.any()):
                recent_rr_mass.index_add_(
                    0, history_row[recent], history_weight[recent]
                )
            _append_layer_edges(
                edge_storage,
                layer[is_history],
                head[is_history],
                query[is_history],
                source[is_history] - prompt_count,
                history_weight,
            )

    epsilon = float(config.epsilon)
    denominator = total_mass.clamp_min(epsilon)
    coordinate_mean = coordinate_sum / denominator[:, None]
    cosine_mean = cosine_sum / denominator[:, None]
    sine_mean = sine_sum / denominator[:, None]
    power = cosine_mean.square() + sine_mean.square()
    prompt_fraction = prompt_mass / denominator
    response_fraction = rr_mass / denominator
    recent_fraction = recent_rr_mass / rr_mass.clamp_min(epsilon)
    concentration = squared_mass / denominator.square()
    scalar = torch.stack(
        (
            torch.log1p(total_mass),
            torch.log1p(edge_count),
            prompt_fraction,
            response_fraction,
            recent_fraction,
            concentration,
        ),
        dim=1,
    )
    set_base = torch.cat((cosine_mean, sine_mean, power, scalar), dim=1)
    pairwise_base = torch.cat((coordinate_mean, scalar), dim=1)
    inactive = total_mass <= epsilon
    set_base[inactive] = 0
    pairwise_base[inactive] = 0

    set_base = set_base.reshape(layers, heads, response_count, -1)
    pairwise_base = pairwise_base.reshape(layers, heads, response_count, -1)
    row_mass = total_mass.reshape(layers, heads, response_count)
    prompt_fraction = prompt_fraction.reshape(layers, heads, response_count)
    response_fraction = response_fraction.reshape(layers, heads, response_count)
    recent_fraction = recent_fraction.reshape(layers, heads, response_count)
    concentration = concentration.reshape(layers, heads, response_count)
    rr_edges = _finalize_layer_edges(edge_storage, device=device, dtype=dtype)

    true_order = np.arange(layers, dtype=np.int16)
    shuffled_order = np.random.default_rng(config.seed).permutation(layers).astype(
        np.int16
    )
    if np.array_equal(shuffled_order, true_order):
        shuffled_order = np.roll(shuffled_order, 1)

    setwalk, true_state = _propagate_view(
        set_base, row_mass, rr_edges, true_order, config
    )
    no_walk, _ = _propagate_view(
        set_base, row_mass, rr_edges, true_order, config, include_walk=False
    )
    pairwise, _ = _propagate_view(
        pairwise_base, row_mass, rr_edges, true_order, config
    )
    shuffled, _ = _propagate_view(
        set_base, row_mass, rr_edges, shuffled_order, config
    )

    active = (row_mass > epsilon).to(dtype)
    head_denominator = active.sum(dim=1).clamp_min(1.0)
    layer_prompt = (prompt_fraction * active).sum(dim=1) / head_denominator
    layer_response = (response_fraction * active).sum(dim=1) / head_denominator
    layer_recent = (recent_fraction * active).sum(dim=1) / head_denominator
    layer_concentration = (concentration * active).sum(dim=1) / head_denominator
    # Convert [layer,response] to [response,layer].
    layer_prompt = layer_prompt.T
    layer_response = layer_response.T
    layer_recent = layer_recent.T
    layer_concentration = layer_concentration.T
    local_collapse = layer_response * layer_recent * layer_concentration
    profiles = torch.stack(
        (
            layer_prompt,
            layer_response,
            layer_recent,
            layer_concentration,
            local_collapse,
            true_state["base_disagreement"].square().mean(dim=2).sqrt(),
            true_state["hop1_survival"],
            true_state["hop2_survival"],
        ),
        dim=2,
    )

    third = max(1, int(np.ceil(layers / 3)))
    order_dependence = (
        (setwalk - shuffled).square().mean(dim=1).clamp_min(0).sqrt()
    )
    diagnostics = torch.stack(
        (
            local_collapse[:, :third].mean(dim=1)
            - local_collapse[:, -third:].mean(dim=1),
            local_collapse.mean(dim=1),
            true_state["base_disagreement"].square().mean(dim=(1, 2)).sqrt(),
            true_state["hop1_survival"].mean(dim=1),
            true_state["hop2_survival"].mean(dim=1),
            order_dependence,
        ),
        dim=1,
    )
    embeddings = {
        "setwalk": setwalk,
        "no_walk": no_walk,
        "pairwise_walk": pairwise,
        "layer_shuffled": shuffled,
    }
    if any(not bool(torch.isfinite(value).all()) for value in embeddings.values()):
        raise FloatingPointError("SetWalk embedding contains non-finite values")
    if not bool(torch.isfinite(diagnostics).all()):
        raise FloatingPointError("SetWalk diagnostics contain non-finite values")

    return {
        "embeddings": {
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in embeddings.items()
        },
        "diagnostics": diagnostics.detach().cpu().numpy().astype(np.float32),
        "diagnostic_names": np.asarray(DIAGNOSTIC_NAMES, dtype=str),
        "layer_profiles": profiles.detach().cpu().numpy().astype(np.float32),
        "layer_profile_names": np.asarray(LAYER_PROFILE_NAMES, dtype=str),
        "true_layer_order": true_order,
        "shuffled_layer_order": shuffled_order,
    }
