"""Deterministic response-token encodings from exact frozen operator graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .graph import GraphTensors
from .math_utils import cosine, safe_norm
from .schema import ROLE_NAMES


TEMPORAL_FEATURE_NAMES = (
    "response_position_fraction",
    "final_hidden_norm",
    "final_hidden_delta_norm",
    "final_hidden_previous_cosine",
    "prompt_route_mean",
    "history_route_mean",
    "self_route_mean",
    "prompt_route_late_mean",
    "history_route_late_mean",
    "prompt_route_token_delta",
    "history_route_token_delta",
    "prompt_route_layer_slope",
    "history_route_layer_slope",
    "prompt_residual_mean",
    "prompt_residual_late_mean",
    "prompt_residual_token_delta",
    "prompt_residual_layer_slope",
    "grounding_mismatch_mean",
    "grounding_mismatch_late_mean",
    "grounding_mismatch_token_delta",
    "mlp_to_prompt_message_ratio_mean",
    "mlp_to_prompt_message_ratio_late_mean",
    "attention_mlp_conflict_mean",
    "prompt_history_conflict_mean",
    "history_operator_dispersion_mean",
    "history_operator_dispersion_late_mean",
    "prompt_operator_dispersion_mean",
    "prompt_operator_dispersion_late_mean",
    "history_operator_code_previous_cosine",
    "prompt_operator_code_previous_cosine",
    "layer_update_norm_mean",
    "layer_update_norm_std",
    "pre_post_cosine_mean",
    "response_lock_in_index",
)


@dataclass(frozen=True)
class NodeEncoding:
    temporal_features: torch.Tensor
    temporal_feature_names: Sequence[str]
    node_embedding: torch.Tensor
    node_feature_names: Sequence[str]


def _index(names: Sequence[str], name: str) -> int:
    try:
        return list(names).index(name)
    except ValueError as error:
        raise ValueError(f"required graph feature is missing: {name}") from error


def _previous_difference(value: torch.Tensor) -> torch.Tensor:
    """Response-aligned first difference with an explicit zero at token zero."""

    if value.ndim != 1:
        raise ValueError("previous difference expects a one-dimensional vector")
    output = torch.zeros_like(value)
    if len(value) > 1:
        output[1:] = value[1:] - value[:-1]
    return output


def _previous_cosine(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("previous cosine expects [token,feature]")
    output = torch.zeros(value.shape[0], dtype=torch.float32)
    if value.shape[0] > 1:
        output[1:] = cosine(value[1:], value[:-1], eps=eps)
    return output


def _layer_slope(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Least-squares slope over the complete layer axis for every token."""

    if value.ndim != 2:
        raise ValueError("layer slope expects [token,layer]")
    layers = int(value.shape[1])
    if layers == 1:
        return torch.zeros(value.shape[0], dtype=torch.float32)
    coordinate = torch.linspace(-1.0, 1.0, layers, dtype=torch.float32)
    denominator = coordinate.pow(2).sum().clamp_min(eps)
    centered = value.float() - value.float().mean(dim=1, keepdim=True)
    return (centered * coordinate[None]).sum(dim=1) / denominator


def _late_slice(layer_count: int) -> slice:
    width = max(1, int(layer_count) // 4)
    return slice(int(layer_count) - width, int(layer_count))


def _role_route_mass(
    route_features: torch.Tensor,
    route_feature_names: Sequence[str],
    role: str,
) -> torch.Tensor:
    """Mean head route mass for ``[token,layer]``."""

    index = _index(route_feature_names, f"{role}_mass")
    return route_features[..., index].float().mean(dim=2)


def _layer_feature(
    layer_features: torch.Tensor,
    layer_feature_names: Sequence[str],
    name: str,
) -> torch.Tensor:
    return layer_features[..., _index(layer_feature_names, name)].float()


def _operator_code(
    layer_features: torch.Tensor,
    layer_feature_names: Sequence[str],
    role: str,
) -> torch.Tensor:
    indices = [
        index
        for index, name in enumerate(layer_feature_names)
        if name.startswith(f"{role}_operator_mean_unit_code_head_")
    ]
    if not indices:
        raise ValueError(f"operator-code block is missing for role {role}")
    return layer_features[..., indices].float()


def temporal_encoding(
    graph: GraphTensors,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute label-free route, operator, and hidden-state trajectory features."""

    route = graph.route_features.float()
    layer = graph.layer_features.float()
    hidden = graph.final_hidden.float()
    if route.ndim != 4:
        raise ValueError("route_features must be [token,layer,head,feature]")
    if layer.ndim != 3 or hidden.ndim != 2:
        raise ValueError("layer/final hidden tensors have invalid dimensions")
    response, layers = int(route.shape[0]), int(route.shape[1])
    if layer.shape[:2] != (response, layers) or hidden.shape[0] != response:
        raise ValueError("graph tensors are not response/layer aligned")
    late = _late_slice(layers)

    prompt_route = _role_route_mass(
        route, graph.route_feature_names, ROLE_NAMES[0]
    )
    history_route = _role_route_mass(
        route, graph.route_feature_names, ROLE_NAMES[1]
    )
    self_route = _role_route_mass(
        route, graph.route_feature_names, ROLE_NAMES[2]
    )
    prompt_residual = _layer_feature(
        layer, graph.layer_feature_names, "prompt_residual_fraction"
    )
    mismatch = _layer_feature(
        layer, graph.layer_feature_names, "grounding_mismatch_absolute"
    )
    mlp_norm = _layer_feature(layer, graph.layer_feature_names, "mlp_output_norm")
    prompt_message_norm = _layer_feature(
        layer, graph.layer_feature_names, "prompt_residual_message_norm"
    )
    mlp_to_prompt = mlp_norm / prompt_message_norm.clamp_min(eps)
    attention_mlp_cosine = _layer_feature(
        layer, graph.layer_feature_names, "attention_mlp_cosine"
    )
    prompt_history_cosine = _layer_feature(
        layer, graph.layer_feature_names, "prompt_history_cosine"
    )
    history_dispersion = _layer_feature(
        layer, graph.layer_feature_names, "history_operator_dispersion"
    )
    prompt_dispersion = _layer_feature(
        layer, graph.layer_feature_names, "prompt_operator_dispersion"
    )
    update_norm = _layer_feature(
        layer, graph.layer_feature_names, "layer_update_norm"
    )
    pre_post = _layer_feature(layer, graph.layer_feature_names, "pre_post_cosine")

    prompt_code = _operator_code(layer, graph.layer_feature_names, "prompt")
    history_code = _operator_code(layer, graph.layer_feature_names, "history")
    # Compare token t with token t-1 at every layer, then average over layers.
    prompt_code_previous = torch.zeros(response, dtype=torch.float32)
    history_code_previous = torch.zeros(response, dtype=torch.float32)
    if response > 1:
        prompt_code_previous[1:] = cosine(
            prompt_code[1:].reshape((response - 1) * layers, -1),
            prompt_code[:-1].reshape((response - 1) * layers, -1),
            eps=eps,
        ).reshape(response - 1, layers).mean(dim=1)
        history_code_previous[1:] = cosine(
            history_code[1:].reshape((response - 1) * layers, -1),
            history_code[:-1].reshape((response - 1) * layers, -1),
            eps=eps,
        ).reshape(response - 1, layers).mean(dim=1)

    hidden_delta = torch.zeros_like(hidden)
    if response > 1:
        hidden_delta[1:] = hidden[1:] - hidden[:-1]
    hidden_previous_cosine = _previous_cosine(hidden, eps=eps)

    prompt_route_mean = prompt_route.mean(dim=1)
    history_route_mean = history_route.mean(dim=1)
    prompt_residual_mean = prompt_residual.mean(dim=1)
    history_late = history_route[:, late].mean(dim=1)
    prompt_late = prompt_route[:, late].mean(dim=1)
    prompt_residual_late = prompt_residual[:, late].mean(dim=1)
    history_code_stability = (history_code_previous + 1.0) * 0.5
    response_lock_in = (
        history_late
        * (1.0 - prompt_residual_late).clamp(0.0, 1.0)
        * history_code_stability.clamp(0.0, 1.0)
    )

    position = torch.arange(response, dtype=torch.float32)
    position = position / max(response - 1, 1)
    columns = (
        position,
        safe_norm(hidden, eps=eps),
        safe_norm(hidden_delta, eps=eps),
        hidden_previous_cosine,
        prompt_route_mean,
        history_route_mean,
        self_route.mean(dim=1),
        prompt_late,
        history_late,
        _previous_difference(prompt_route_mean),
        _previous_difference(history_route_mean),
        _layer_slope(prompt_route, eps=eps),
        _layer_slope(history_route, eps=eps),
        prompt_residual_mean,
        prompt_residual_late,
        _previous_difference(prompt_residual_mean),
        _layer_slope(prompt_residual, eps=eps),
        mismatch.mean(dim=1),
        mismatch[:, late].mean(dim=1),
        _previous_difference(mismatch.mean(dim=1)),
        mlp_to_prompt.mean(dim=1),
        mlp_to_prompt[:, late].mean(dim=1),
        1.0 - attention_mlp_cosine.mean(dim=1),
        1.0 - prompt_history_cosine.mean(dim=1),
        history_dispersion.mean(dim=1),
        history_dispersion[:, late].mean(dim=1),
        prompt_dispersion.mean(dim=1),
        prompt_dispersion[:, late].mean(dim=1),
        history_code_previous,
        prompt_code_previous,
        update_norm.mean(dim=1),
        update_norm.std(dim=1, unbiased=False),
        pre_post.mean(dim=1),
        response_lock_in,
    )
    output = torch.stack(columns, dim=-1).float()
    if output.shape != (response, len(TEMPORAL_FEATURE_NAMES)):
        raise RuntimeError("temporal feature names and tensor width diverged")
    if not torch.isfinite(output).all():
        raise ValueError("temporal encoding contains non-finite values")
    return output


def _flatten_names(prefix: str, shape: Sequence[int], terminal_names: Sequence[str]) -> list[str]:
    """Name all leading coordinates and the final semantic feature coordinate."""

    if not shape:
        return [f"{prefix}_{name}" for name in terminal_names]
    if len(shape) == 1:
        return [
            f"{prefix}_{index}_{name}"
            for index in range(int(shape[0]))
            for name in terminal_names
        ]
    if len(shape) == 2:
        return [
            f"{prefix}_layer_{layer}_head_{head}_{name}"
            for layer in range(int(shape[0]))
            for head in range(int(shape[1]))
            for name in terminal_names
        ]
    raise ValueError("only one or two leading feature axes are supported")


def build_node_encoding(
    graph: GraphTensors,
    *,
    eps: float = 1e-8,
) -> NodeEncoding:
    """Concatenate all exact graph channels without learned compression."""

    temporal = temporal_encoding(graph, eps=eps)
    response = int(graph.final_hidden.shape[0])
    route_flat = graph.route_features.float().reshape(response, -1)
    layer_flat = graph.layer_features.float().reshape(response, -1)
    hidden = graph.final_hidden.float()
    node = torch.cat((hidden, route_flat, layer_flat, temporal), dim=-1)

    hidden_names = [f"final_hidden_{index}" for index in range(hidden.shape[1])]
    route_names = _flatten_names(
        "route",
        graph.route_features.shape[1:3],
        graph.route_feature_names,
    )
    layer_names = _flatten_names(
        "layer",
        graph.layer_features.shape[1:2],
        graph.layer_feature_names,
    )
    names = tuple(hidden_names + route_names + layer_names + list(TEMPORAL_FEATURE_NAMES))
    if node.shape != (response, len(names)):
        raise RuntimeError("node feature names and concatenated encoding diverged")
    if not torch.isfinite(node).all():
        raise ValueError("node encoding contains non-finite values")
    return NodeEncoding(
        temporal_features=temporal,
        temporal_feature_names=TEMPORAL_FEATURE_NAMES,
        node_embedding=node,
        node_feature_names=names,
    )


__all__ = [
    "NodeEncoding",
    "TEMPORAL_FEATURE_NAMES",
    "build_node_encoding",
    "temporal_encoding",
]
