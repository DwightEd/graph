"""Compact head-resolved route graphs and temporal route contraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ROLE_INDEX = {
    "evidence": 0,
    "other_prompt": 1,
    "response_history": 2,
    "predictor_self": 3,
}
ROLE_NAMES = tuple(ROLE_INDEX)
ROUTE_ROLE_NAMES = ("evidence", "response_history")
_EPS = 1e-12


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class RouteGraph:
    """Sparse source-to-predictor incidences plus omitted adaptive-cover tail.

    The edge arrays have length ``E``. The row arrays have length ``L*T*H*2``
    in layer, response-token, head, route-role order. Remainder mass has no
    invented source endpoint and therefore remains separate from explicit
    edges.
    """

    source: np.ndarray
    target: np.ndarray
    layer: np.ndarray
    head: np.ndarray
    role: np.ndarray
    magnitude: np.ndarray
    row_target: np.ndarray
    row_layer: np.ndarray
    row_head: np.ndarray
    row_role: np.ndarray
    remainder: np.ndarray
    cover_size: np.ndarray


def build_graph(artifact: Mapping[str, Any]) -> RouteGraph:
    """Build the saved adaptive-cover graph without merging layers or heads."""

    trace = artifact["trace"]
    source = _array(trace["route_source_index"]).astype(np.int64)
    magnitude = _array(trace["route_source_magnitude"]).astype(np.float64)
    remainder = _array(trace["route_source_remainder"]).astype(np.float64)
    cover_size = _array(trace["route_source_cover_size"]).astype(np.int64)
    if source.ndim != 5 or magnitude.shape != source.shape:
        raise ValueError(
            "route sources and magnitudes must be [layer, token, head, role, slot]"
        )
    layers, tokens, heads, route_roles, slots = source.shape
    if route_roles != len(ROUTE_ROLE_NAMES):
        raise ValueError("route graph must contain evidence and history covers")
    row_shape = (layers, tokens, heads, route_roles)
    if remainder.shape != row_shape or cover_size.shape != row_shape:
        raise ValueError(
            "route remainder and cover size must be [layer, token, head, role]"
        )
    if np.any(cover_size < 0):
        raise ValueError("route cover size must be nonnegative")
    if not np.isfinite(magnitude).all() or np.any(magnitude < 0):
        raise ValueError("route magnitudes must be finite and nonnegative")
    if not np.isfinite(remainder).all() or np.any(remainder < 0):
        raise ValueError("route remainder must be finite and nonnegative")

    layer, token, head, route_role = np.indices(row_shape)
    target = int(artifact["response_start"]) - 1 + token
    saved_count = np.minimum(cover_size, slots)
    selected = np.arange(slots) < saved_count[..., None]
    edge_target = np.broadcast_to(target[..., None], source.shape)
    if np.any(source[selected] < 0):
        raise ValueError("covered route slots must have explicit source endpoints")
    if np.any(source[selected] > edge_target[selected]):
        raise ValueError("route graph contains a non-causal source endpoint")

    role_mass = _array(trace["edge_role_mass"]).astype(np.float64)
    if role_mass.shape != (layers, tokens, heads, len(ROLE_NAMES)):
        raise ValueError("edge role mass must align with graph rows")
    if not np.isfinite(role_mass).all() or np.any(role_mass < 0):
        raise ValueError("edge role mass must be finite and nonnegative")
    explicit_mass = (magnitude * selected).sum(axis=-1)
    total_mass = role_mass[..., [ROLE_INDEX[name] for name in ROUTE_ROLE_NAMES]]
    if not np.allclose(explicit_mass + remainder, total_mass, rtol=5e-3, atol=5e-3):
        raise ValueError("explicit route mass plus remainder does not conserve mass")

    evidence_mask = _array(artifact["evidence_mask"]).astype(bool)
    response_start = int(artifact["response_start"])
    if evidence_mask.shape != (response_start,):
        raise ValueError("evidence_mask must align exactly with the prompt")
    edge_source = source[selected]
    edge_target = edge_target[selected]
    edge_role_index = np.broadcast_to(route_role[..., None], source.shape)[selected]
    edge_role = np.asarray(ROUTE_ROLE_NAMES)[edge_role_index]
    evidence_edge = edge_role_index == 0
    evidence_source = edge_source[evidence_edge]
    prompt_source = evidence_source < response_start
    wrong_evidence = ~prompt_source
    wrong_evidence[prompt_source] |= ~evidence_mask[evidence_source[prompt_source]] | (
        evidence_source[prompt_source] == edge_target[evidence_edge][prompt_source]
    )
    if np.any(wrong_evidence):
        raise ValueError("evidence cover contains a non-evidence source")
    history_edge = edge_role_index == 1
    if np.any(edge_source[history_edge] < response_start) or np.any(
        edge_source[history_edge] >= edge_target[history_edge]
    ):
        raise ValueError("history cover contains a non-history source")

    return RouteGraph(
        source=edge_source,
        target=edge_target,
        layer=np.broadcast_to(layer[..., None], source.shape)[selected],
        head=np.broadcast_to(head[..., None], source.shape)[selected],
        role=edge_role,
        magnitude=magnitude[selected],
        row_target=target.ravel(),
        row_layer=layer.ravel(),
        row_head=head.ravel(),
        row_role=np.asarray(ROUTE_ROLE_NAMES)[route_role.ravel()],
        remainder=remainder.ravel(),
        cover_size=cover_size.ravel(),
    )


def route_contraction(
    artifact: Mapping[str, Any],
    role: str,
    *,
    family: str = "edge",
    window: int = 4,
    return_valid: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Measure shrinking head-source incidence support against recent history.

    Effective routes are computed upstream from the joint ``(head, source)``
    incidence distribution, so heads are not averaged before this statistic.
    Positive values mean that the current route support is narrower than the
    preceding four-token geometric mean. Layers are weighted by current role
    mass after summing their head-resolved masses.
    """

    if role not in ROLE_INDEX:
        raise ValueError(f"unknown source role: {role}")
    if window <= 0:
        raise ValueError("route contraction window must be positive")
    trace = artifact["trace"]
    support = _array(trace[f"{family}_role_effective_routes"]).astype(np.float64)
    mass = _array(trace[f"{family}_role_mass"]).astype(np.float64)
    if support.ndim != 3:
        raise ValueError("role effective routes must be [layer, token, role]")
    if mass.ndim != 4 or mass.shape[:2] != support.shape[:2]:
        raise ValueError("role mass must be [layer, token, head, role]")
    if mass.shape[-1] != support.shape[-1] or ROLE_INDEX[role] >= mass.shape[-1]:
        raise ValueError("route role axes are not aligned")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("route mass must be finite and nonnegative")

    role_index = ROLE_INDEX[role]
    layer_mass = mass[..., role_index].sum(axis=2)
    role_support = support[..., role_index]
    valid = layer_mass > _EPS
    if not np.isfinite(role_support).all() or np.any(role_support[valid] <= 0):
        raise ValueError(
            "effective routes with observed mass must be finite and positive"
        )
    log_support = np.zeros_like(role_support)
    log_support[valid] = np.log(role_support[valid])
    layer_contraction = np.zeros_like(log_support)
    comparable = np.zeros_like(valid)
    for layer in range(log_support.shape[0]):
        for token in range(1, log_support.shape[1]):
            prior = np.arange(max(0, token - window), token)
            prior = prior[valid[layer, prior]]
            if valid[layer, token] and len(prior):
                comparable[layer, token] = True
                layer_contraction[layer, token] = (
                    log_support[layer, prior].mean() - log_support[layer, token]
                )

    comparable_mass = layer_mass * comparable
    contraction = (layer_contraction * comparable_mass).sum(axis=0) / np.maximum(
        comparable_mass.sum(axis=0), _EPS
    )
    token_valid = comparable_mass.sum(axis=0) > _EPS
    return (contraction, token_valid) if return_valid else contraction


def route_mass_contraction(
    artifact: Mapping[str, Any],
    role: str,
    *,
    family: str = "edge",
    window: int = 4,
    return_valid: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Measure disappearance of total role mass independently of route support."""

    if role not in ROLE_INDEX:
        raise ValueError(f"unknown source role: {role}")
    if window <= 0:
        raise ValueError("route mass contraction window must be positive")
    mass = _array(artifact["trace"][f"{family}_role_mass"]).astype(np.float64)
    if mass.ndim != 4 or ROLE_INDEX[role] >= mass.shape[-1]:
        raise ValueError("role mass must be [layer, token, head, role]")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("route mass must be finite and nonnegative")
    token_mass = mass[..., ROLE_INDEX[role]].sum(axis=(0, 2))
    contraction = np.zeros(len(token_mass), dtype=np.float64)
    valid = np.zeros(len(token_mass), dtype=bool)
    positive = token_mass[token_mass > _EPS]
    if not len(positive):
        return (contraction, valid) if return_valid else contraction
    normalized = token_mass / np.median(positive)
    maximum = -np.log(_EPS)
    for token in range(1, len(token_mass)):
        prior = normalized[max(0, token - window) : token]
        prior = prior[prior > _EPS]
        if len(prior):
            valid[token] = True
            contraction[token] = np.log(prior + _EPS).mean() - np.log(
                normalized[token] + _EPS
            )
    contraction = np.clip(contraction, -maximum, maximum)
    return (contraction, valid) if return_valid else contraction
