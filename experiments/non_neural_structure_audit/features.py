"""Interpretable structure coordinates derived without labels or learning."""

from __future__ import annotations

import numpy as np
import torch

from experiments.attention_phenomenology.routing import RoutingState
from experiments.attention_phenomenology.sources import (
    response_lag_statistics,
    summarize_exact_sources,
)

from .lineage import LINEAGE_INDEX, LineageTrace

FEATURE_NAMES = (
    "prompt_mass",
    "history_mass",
    "self_mass",
    "unresolved_mass",
    "response_takeover",
    "prompt_head_std",
    "history_head_std",
    "head_role_disagreement",
    "response_effective_sources",
    "response_top1_share",
    "recent_response_share",
    "response_mean_lag",
    "prompt_connected_total",
    "prompt_connected_relay",
    "response_base_local",
    "inherited_response_base",
    "multihop_response_base",
    "lineage_unresolved",
    "response_to_prompt_log_ratio",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

LINEAGE_FEATURE_NAMES = (
    "prompt_connected_total",
    "prompt_connected_relay",
    "response_base_local",
    "inherited_response_base",
    "multihop_response_base",
    "lineage_unresolved",
    "response_to_prompt_log_ratio",
)

RELATION_SPECS = (
    ("direct_role", "response_takeover", 1.0),
    ("endpoint_concentration", "response_top1_share", 1.0),
    ("head_fracture", "head_role_disagreement", 1.0),
    ("prompt_connected_lineage", "prompt_connected_total", -1.0),
    ("inherited_response_base", "inherited_response_base", 1.0),
    ("multihop_response_base", "multihop_response_base", 1.0),
    ("lineage_margin", "response_to_prompt_log_ratio", 1.0),
    ("censoring_control", "unresolved_mass", 1.0),
)
RELATION_NAMES = tuple(spec[0] for spec in RELATION_SPECS)
LINEAGE_RELATION_NAMES = tuple(
    relation
    for relation, feature, _ in RELATION_SPECS
    if feature in LINEAGE_FEATURE_NAMES
)


def _valid_head_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=-1)
    total = torch.where(valid, values, torch.zeros_like(values)).sum(dim=-1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))


def _head_disagreement(routing: RoutingState, epsilon: float) -> torch.Tensor:
    mass = torch.stack((routing.prompt_mass, routing.response_mass), dim=-1)
    total = mass.sum(dim=-1)
    valid = total > epsilon
    root = (mass / total[..., None].clamp_min(epsilon)).sqrt()
    affinity = (root.unsqueeze(-2) * root.unsqueeze(-3)).sum(dim=-1)
    distance = (1.0 - affinity.clamp(0.0, 1.0)).sqrt()
    heads = routing.edges.num_heads
    pairs = torch.triu(
        torch.ones((heads, heads), dtype=torch.bool, device=distance.device),
        diagonal=1,
    )
    selected = valid.unsqueeze(-1) & valid.unsqueeze(-2) & pairs
    count = selected.sum(dim=(-2, -1))
    total_distance = torch.where(selected, distance, 0.0).sum(dim=(-2, -1))
    return torch.where(
        count > 0, total_distance / count.clamp_min(1), torch.zeros_like(total_distance)
    )


def build_layer_features(
    routing: RoutingState,
    lineage: LineageTrace,
    *,
    recent_tokens: int = 4,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return ``[response token, layer step, feature]`` coordinates."""

    response_sources = summarize_exact_sources(
        routing, role="response", epsilon=epsilon
    )
    recent_share, mean_lag = response_lag_statistics(
        routing,
        response_sources,
        recent_tokens=recent_tokens,
        epsilon=epsilon,
    )
    off_diagonal = routing.prompt_mass + routing.response_mass
    response_takeover = torch.where(
        off_diagonal > epsilon,
        routing.response_mass / off_diagonal.clamp_min(epsilon),
        torch.zeros_like(off_diagonal),
    )

    layer_index = torch.as_tensor(
        lineage.layer_order, dtype=torch.long, device=routing.edges.device
    )
    prompt_mass = routing.prompt_mass[:, layer_index]
    response_mass = routing.response_mass[:, layer_index]
    self_mass = routing.self_mass[:, layer_index]
    unresolved_mass = routing.unresolved_mass[:, layer_index]
    response_takeover = response_takeover[:, layer_index]
    recent_share = recent_share[:, layer_index]
    mean_lag = mean_lag[:, layer_index]
    response_valid = response_sources.valid[:, layer_index]
    response_effective = response_sources.effective_sources[:, layer_index]
    response_top1 = response_sources.top1_share[:, layer_index]

    state = lineage.state
    prompt_connected = (
        state[..., LINEAGE_INDEX["prompt_direct"]]
        + state[..., LINEAGE_INDEX["prompt_relay"]]
    )
    inherited_response = (
        state[..., LINEAGE_INDEX["response_relay_one_hop"]]
        + state[..., LINEAGE_INDEX["response_relay_multihop"]]
    )
    known_lineage = prompt_connected + inherited_response
    fields = {
        "prompt_mass": prompt_mass.mean(dim=-1),
        "history_mass": response_mass.mean(dim=-1),
        "self_mass": self_mass.mean(dim=-1),
        "unresolved_mass": unresolved_mass.mean(dim=-1),
        "response_takeover": _valid_head_mean(
            response_takeover, off_diagonal[:, layer_index] > epsilon
        ),
        "prompt_head_std": prompt_mass.std(dim=-1, unbiased=False),
        "history_head_std": response_mass.std(dim=-1, unbiased=False),
        "head_role_disagreement": _head_disagreement(routing, epsilon)[:, layer_index],
        "response_effective_sources": _valid_head_mean(
            response_effective, response_valid
        ),
        "response_top1_share": _valid_head_mean(response_top1, response_valid),
        "recent_response_share": _valid_head_mean(recent_share, response_valid),
        "response_mean_lag": _valid_head_mean(mean_lag, response_valid),
        "prompt_connected_total": prompt_connected,
        "prompt_connected_relay": state[..., LINEAGE_INDEX["prompt_relay"]],
        "response_base_local": state[..., LINEAGE_INDEX["response_base"]],
        "inherited_response_base": inherited_response,
        "multihop_response_base": state[..., LINEAGE_INDEX["response_relay_multihop"]],
        "lineage_unresolved": state[..., LINEAGE_INDEX["unresolved"]],
        "response_to_prompt_log_ratio": known_lineage
        * torch.log((inherited_response + epsilon) / (prompt_connected + epsilon)),
    }
    return torch.stack([fields[name] for name in FEATURE_NAMES], dim=-1)


def relation_scores(standardized_features: np.ndarray) -> np.ndarray:
    """Orient one transparent coordinate per relation so higher means riskier."""

    standardized_features = np.asarray(standardized_features, dtype=np.float32)
    scores = [
        direction * standardized_features[:, :, FEATURE_INDEX[feature]].mean(axis=1)
        for _, feature, direction in RELATION_SPECS
    ]
    return np.stack(scores, axis=1).astype(np.float32)
