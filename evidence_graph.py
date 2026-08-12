"""Evidence-aligned causal graph construction from canonical sparse attention.

The graph is designed around empirically observed hallucination signals:
weaker prompt grounding, stronger response-history reliance, sparser and more
local support, greater concentration, and layer-dependent routing. It keeps
concrete source identity so topology can be tested with source-shuffle controls.

No hallucination labels are read or stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import torch


EDGE_FEATURE_NAMES = (
    "pair_mass",
    "within_relation_share",
    "relation_mass",
    "normalized_lag",
    "channel_fraction",
    "max_channel_weight",
    "early_layer_mass",
    "middle_layer_mass",
    "late_layer_mass",
)

RESPONSE_STATE_NAMES = (
    "incoming_mass",
    "prompt_mass",
    "history_mass",
    "prompt_mass_share",
    "retained_mass_fraction",
    "prompt_degree",
    "history_degree",
    "support_density",
    "normalized_entropy",
    "hhi",
    "history_lag_mean",
    "history_lag_std",
    "early_prompt_mass",
    "middle_prompt_mass",
    "late_prompt_mass",
    "early_history_mass",
    "middle_history_mass",
    "late_history_mass",
    "prompt_ancestry",
    "grounded_history_relay",
    "unsupported_history_feedback",
    "expected_prompt_hops",
)

NODE_CONTEXT_NAMES = (
    "log_position",
    "prompt_role",
    "response_role",
) + RESPONSE_STATE_NAMES


@dataclass(frozen=True)
class EvidenceGraphConfig:
    """Configuration for typed adaptive support and prompt provenance."""

    mass_cover: float = 0.80
    relay_discount: float = 0.85

    def validate(self) -> None:
        if (
            isinstance(self.mass_cover, bool)
            or not isinstance(self.mass_cover, Real)
            or not isfinite(float(self.mass_cover))
            or not 0.0 < float(self.mass_cover) <= 1.0
        ):
            raise ValueError("mass_cover must be finite and in (0, 1]")
        if (
            isinstance(self.relay_discount, bool)
            or not isinstance(self.relay_discount, Real)
            or not isfinite(float(self.relay_discount))
            or not 0.0 < float(self.relay_discount) < 1.0
        ):
            raise ValueError("relay_discount must be finite and in (0, 1)")


@dataclass
class EvidenceGraph:
    """Sparse typed graph plus label-free response structural states."""

    num_nodes: int
    response_idx: int
    num_channels: int
    node_attr: torch.Tensor
    node_context: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    edge_attr: torch.Tensor
    edge_ptr: torch.Tensor
    edge_channel: torch.Tensor
    edge_value: torch.Tensor
    response_state: torch.Tensor
    mass_cover: float
    relay_discount: float
    edge_feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    node_context_names: tuple[str, ...] = NODE_CONTEXT_NAMES
    response_state_names: tuple[str, ...] = RESPONSE_STATE_NAMES

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _decode_traces(sample):
    response_count = sample.num_response_tokens
    counts = sample.response_row_ptr[1:].long() - sample.response_row_ptr[:-1].long()
    rows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
    channel = rows // response_count
    target = sample.response_idx + rows.remainder(response_count)
    source = sample.response_column_indices.long()
    value = sample.response_values.float()
    return source, target, channel, value


def _layer_groups(num_layers: int, num_heads: int, device):
    layer = torch.arange(num_layers, device=device)
    group = torch.minimum(
        torch.full_like(layer, 2),
        torch.div(layer * 3, num_layers, rounding_mode="floor"),
    )
    layer_count = torch.bincount(group, minlength=3)
    channel_count = layer_count * int(num_heads)
    return group, channel_count.clamp_min(1)


def _aggregate_pairs(sample):
    source, target, channel, value = _decode_traces(sample)
    device = value.device
    if value.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "source": empty,
            "target": empty,
            "weight": value,
            "max_weight": value,
            "channel_count": empty,
            "layer_profile": value.new_empty((0, 3)),
            "inverse": empty,
            "trace_channel": channel,
            "trace_value": value,
        }

    pair_key = target * sample.num_tokens + source
    unique_pair, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)
    pair_count = unique_pair.numel()

    weight = torch.zeros(pair_count, dtype=torch.float32, device=device)
    weight.index_add_(0, inverse, value)
    weight /= float(sample.num_channels)

    max_weight = torch.full(
        (pair_count,), -torch.inf, dtype=torch.float32, device=device
    )
    max_weight.scatter_reduce_(0, inverse, value, reduce="amax", include_self=True)

    channel_count = torch.bincount(inverse, minlength=pair_count).long()
    layer = torch.div(channel, sample.num_heads, rounding_mode="floor")
    layer_group, group_channels = _layer_groups(
        sample.num_layers, sample.num_heads, device
    )
    group = layer_group[layer]
    profile_flat = torch.zeros(pair_count * 3, dtype=torch.float32, device=device)
    profile_flat.index_add_(0, inverse * 3 + group, value)
    layer_profile = profile_flat.reshape(pair_count, 3)
    layer_profile /= group_channels.to(torch.float32).unsqueeze(0)

    return {
        "source": unique_pair.remainder(sample.num_tokens).long(),
        "target": torch.div(
            unique_pair, sample.num_tokens, rounding_mode="floor"
        ).long(),
        "weight": weight,
        "max_weight": max_weight,
        "channel_count": channel_count,
        "layer_profile": layer_profile,
        "inverse": inverse,
        "trace_channel": channel,
        "trace_value": value,
    }


def _typed_mass_cover_ids(sample, pairs, mass_cover: float) -> torch.Tensor:
    """Keep the fewest pair edges covering rho mass inside each target/relation."""
    if pairs["weight"].numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pairs["weight"].device)

    relation = (pairs["source"] >= sample.response_idx).long()
    group = pairs["target"] * 2 + relation
    selected: list[torch.Tensor] = []
    for group_id in torch.unique(group, sorted=True):
        ids = torch.nonzero(group == group_id, as_tuple=False).flatten()
        ranked = ids[
            torch.argsort(pairs["weight"][ids], descending=True, stable=True)
        ]
        mass = pairs["weight"][ranked]
        total = mass.sum()
        if not bool(total > 0):
            continue
        reached = torch.nonzero(
            mass.cumsum(0) >= float(mass_cover) * total, as_tuple=False
        )
        count = int(reached[0]) + 1 if reached.numel() else len(ranked)
        selected.append(ranked[:count])

    if not selected:
        return torch.empty(0, dtype=torch.long, device=pairs["weight"].device)
    chosen = torch.cat(selected)
    relation = (pairs["source"][chosen] >= sample.response_idx).long()
    order_key = pairs["target"][chosen] * 2 + relation
    return chosen[torch.argsort(order_key, stable=True)]


def _selected_traces(pairs, chosen):
    value = pairs["trace_value"]
    device = value.device
    if chosen.numel() == 0:
        return (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            value[:0],
        )

    pair_to_edge = torch.full(
        (pairs["weight"].numel(),), -1, dtype=torch.long, device=device
    )
    pair_to_edge[chosen] = torch.arange(chosen.numel(), device=device)
    trace_edge = pair_to_edge[pairs["inverse"]]
    keep = trace_edge >= 0
    order = torch.argsort(trace_edge[keep], stable=True)
    trace_edge = trace_edge[keep][order]
    counts = torch.bincount(trace_edge, minlength=chosen.numel())
    edge_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0))
    )
    edge_channel = pairs["trace_channel"][keep][order].to(torch.int32)
    edge_value = pairs["trace_value"][keep][order]
    return edge_ptr, edge_channel, edge_value


def _entropy_hhi(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.numel() == 0:
        zero = weights.new_tensor(0.0)
        return zero, zero
    probabilities = weights / weights.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    if weights.numel() > 1:
        entropy = entropy / weights.new_tensor(float(weights.numel())).log()
    else:
        entropy = entropy * 0.0
    return entropy, probabilities.square().sum()


def _response_states(sample, pairs, chosen, config: EvidenceGraphConfig):
    response_count = sample.num_response_tokens
    response_idx = sample.response_idx
    device = pairs["weight"].device
    dtype = torch.float32
    state = torch.zeros(
        (response_count, len(RESPONSE_STATE_NAMES)), dtype=dtype, device=device
    )

    all_relation_mass = torch.zeros((response_count, 2), dtype=dtype, device=device)
    if pairs["weight"].numel():
        all_row = pairs["target"] - response_idx
        all_type = (pairs["source"] >= response_idx).long()
        all_relation_mass.view(-1).index_add_(
            0, all_row * 2 + all_type, pairs["weight"]
        )

    selected_source = pairs["source"][chosen]
    selected_target = pairs["target"][chosen]
    selected_weight = pairs["weight"][chosen]
    selected_profile = pairs["layer_profile"][chosen]

    ancestry = torch.zeros(response_count, dtype=dtype, device=device)
    expected_hops = torch.zeros_like(ancestry)

    for row in range(response_count):
        target = response_idx + row
        ids = torch.nonzero(selected_target == target, as_tuple=False).flatten()
        sources = selected_source[ids]
        weights = selected_weight[ids]
        profiles = selected_profile[ids]
        prompt = sources < response_idx
        history = ~prompt

        all_prompt = all_relation_mass[row, 0]
        all_history = all_relation_mass[row, 1]
        all_total = all_prompt + all_history
        selected_total = weights.sum()
        zero = weights.new_tensor(0.0)
        prompt_weight = weights[prompt].sum() if bool(prompt.any()) else zero
        history_weight = weights[history].sum() if bool(history.any()) else zero
        prompt_share = all_prompt / all_total.clamp_min(1e-12)
        retained_fraction = selected_total / all_total.clamp_min(1e-12)
        entropy, hhi = _entropy_hhi(weights)

        prompt_degree = prompt.sum().to(dtype)
        history_degree = history.sum().to(dtype)
        support_density = (prompt_degree + history_degree) / max(target, 1)

        lag_mean = zero
        lag_std = zero
        if bool(history.any()) and bool(history_weight > 0):
            lag = (target - sources[history]).to(dtype) / max(response_count - 1, 1)
            probability = weights[history] / history_weight
            lag_mean = (probability * lag).sum()
            lag_std = (probability * (lag - lag_mean).square()).sum().sqrt()

        layer_relation = torch.zeros((3, 2), dtype=dtype, device=device)
        if ids.numel():
            if bool(prompt.any()):
                layer_relation[:, 0] = profiles[prompt].sum(0)
            if bool(history.any()):
                layer_relation[:, 1] = profiles[history].sum(0)

        direct = prompt_weight / selected_total.clamp_min(1e-12)
        grounded_relay = zero
        unsupported = zero
        hop_numerator = direct
        if bool(history.any()) and bool(selected_total > 0):
            source_local = sources[history] - response_idx
            alpha = weights[history] / selected_total
            source_ancestry = ancestry[source_local]
            grounded_relay = (alpha * source_ancestry).sum()
            unsupported = (alpha * (1.0 - source_ancestry)).sum()
            hop_numerator = direct + float(config.relay_discount) * (
                alpha * source_ancestry * (expected_hops[source_local] + 1.0)
            ).sum()

        ancestry[row] = (
            direct + float(config.relay_discount) * grounded_relay
        ).clamp(0.0, 1.0)
        if bool(ancestry[row] > 0):
            expected_hops[row] = hop_numerator / ancestry[row].clamp_min(1e-12)

        state[row] = torch.stack(
            (
                all_total,
                all_prompt,
                all_history,
                prompt_share,
                retained_fraction,
                prompt_degree,
                history_degree,
                support_density,
                entropy,
                hhi,
                lag_mean,
                lag_std,
                layer_relation[0, 0],
                layer_relation[1, 0],
                layer_relation[2, 0],
                layer_relation[0, 1],
                layer_relation[1, 1],
                layer_relation[2, 1],
                ancestry[row],
                grounded_relay,
                unsupported,
                expected_hops[row],
            )
        )

    return state


def build_evidence_graph(
    sample,
    config: EvidenceGraphConfig | None = None,
) -> EvidenceGraph:
    """Build a typed adaptive support graph and graph-derived response states."""
    config = config or EvidenceGraphConfig()
    config.validate()

    pairs = _aggregate_pairs(sample)
    chosen = _typed_mass_cover_ids(sample, pairs, float(config.mass_cover))
    edge_ptr, edge_channel, edge_value = _selected_traces(pairs, chosen)

    source = pairs["source"][chosen]
    target = pairs["target"][chosen]
    edge_type = (source >= sample.response_idx).long()
    edge_weight = pairs["weight"][chosen]
    row = target - sample.response_idx

    relation_mass = torch.zeros(
        (sample.num_response_tokens, 2),
        dtype=torch.float32,
        device=edge_weight.device,
    )
    if pairs["weight"].numel():
        all_row = pairs["target"] - sample.response_idx
        all_type = (pairs["source"] >= sample.response_idx).long()
        relation_mass.view(-1).index_add_(
            0, all_row * 2 + all_type, pairs["weight"]
        )
    selected_relation_mass = (
        relation_mass[row, edge_type] if chosen.numel() else edge_weight
    )
    within_share = edge_weight / selected_relation_mass.clamp_min(1e-12)
    normalized_lag = (target - source).to(torch.float32) / target.clamp_min(1).to(
        torch.float32
    )
    channel_fraction = pairs["channel_count"][chosen].to(torch.float32) / float(
        sample.num_channels
    )
    edge_attr = torch.cat(
        (
            edge_weight.unsqueeze(1),
            within_share.unsqueeze(1),
            selected_relation_mass.unsqueeze(1),
            normalized_lag.unsqueeze(1),
            channel_fraction.unsqueeze(1),
            pairs["max_weight"][chosen].unsqueeze(1),
            pairs["layer_profile"][chosen],
        ),
        dim=1,
    )

    diagonal = sample.attention_diagonal.permute(2, 0, 1).reshape(
        sample.num_tokens, -1
    )
    position = torch.arange(
        sample.num_tokens, dtype=torch.float32, device=diagonal.device
    )
    response_role = position >= sample.response_idx
    node_context = torch.zeros(
        (sample.num_tokens, len(NODE_CONTEXT_NAMES)),
        dtype=torch.float32,
        device=diagonal.device,
    )
    node_context[:, 0] = torch.log1p(position)
    node_context[:, 1] = (~response_role).to(torch.float32)
    node_context[:, 2] = response_role.to(torch.float32)
    response_state = _response_states(sample, pairs, chosen, config)
    node_context[sample.response_idx :, 3:] = response_state

    if not bool(torch.isfinite(edge_attr).all() and torch.isfinite(node_context).all()):
        raise ValueError("evidence graph contains non-finite features")

    return EvidenceGraph(
        num_nodes=sample.num_tokens,
        response_idx=sample.response_idx,
        num_channels=sample.num_channels,
        node_attr=diagonal,
        node_context=node_context,
        edge_index=torch.stack((source, target)),
        edge_type=edge_type.to(torch.int8),
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        edge_ptr=edge_ptr,
        edge_channel=edge_channel,
        edge_value=edge_value,
        response_state=response_state,
        mass_cover=float(config.mass_cover),
        relay_discount=float(config.relay_discount),
    )
