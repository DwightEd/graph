"""Sparse, head-resolved view of the two provenance registers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

REGISTER_NAMES = ("evidence_adoption", "autonomous_history")
STAGE_NAMES = ("input_state", "attention_write", "mlp_write", "output_state")
ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _route_arrays(trace: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    source = _array(trace["register_route_source_index"]).astype(np.int64)
    head = _array(trace["register_route_head_index"]).astype(np.int64)
    magnitude = _array(trace["register_route_magnitude"]).astype(np.float64)
    contribution = _array(trace["register_route_contribution"]).astype(np.float64)
    root = _array(trace["register_route_root_contribution"]).astype(np.float64)
    carrier = _array(trace["register_route_carrier_contribution"]).astype(np.float64)
    gate = _array(trace["register_route_gate_contribution"]).astype(np.float64)
    remainder_magnitude = _array(trace["register_route_remainder_magnitude"]).astype(
        np.float64
    )
    remainder_contribution = _array(
        trace["register_route_remainder_contribution"]
    ).astype(np.float64)
    remainder_root = _array(trace["register_route_remainder_root_contribution"]).astype(
        np.float64
    )
    remainder_carrier = _array(
        trace["register_route_remainder_carrier_contribution"]
    ).astype(np.float64)
    remainder_gate = _array(trace["register_route_remainder_gate_contribution"]).astype(
        np.float64
    )
    cover_size = _array(trace["register_route_cover_size"]).astype(np.int64)

    if source.ndim != 4 or source.shape[2] != len(REGISTER_NAMES):
        raise ValueError("register routes must be [layer, token, register, slot]")
    if (
        head.shape != source.shape
        or magnitude.shape != source.shape
        or contribution.shape != source.shape
        or root.shape != source.shape
        or carrier.shape != source.shape
        or gate.shape != source.shape
    ):
        raise ValueError(
            "route endpoints, heads, magnitude, and contribution must align"
        )
    row_shape = source.shape[:-1]
    if (
        remainder_magnitude.shape != row_shape
        or remainder_contribution.shape != row_shape
        or remainder_root.shape != row_shape
        or remainder_carrier.shape != row_shape
        or remainder_gate.shape != row_shape
        or cover_size.shape != row_shape
    ):
        raise ValueError("route tails must be [layer, token, register]")
    if np.any(cover_size < 0):
        raise ValueError("route cover size must be nonnegative")
    if not np.isfinite(magnitude).all() or np.any(magnitude < 0):
        raise ValueError("route magnitudes must be finite and nonnegative")
    if not np.isfinite(remainder_magnitude).all() or np.any(remainder_magnitude < 0):
        raise ValueError("route remainder magnitudes must be finite and nonnegative")
    if not all(
        np.isfinite(value).all()
        for value in (
            contribution,
            root,
            carrier,
            gate,
            remainder_contribution,
            remainder_root,
            remainder_carrier,
            remainder_gate,
        )
    ):
        raise ValueError("route contributions must be finite")
    if not np.allclose(contribution, root + carrier + gate, atol=2e-5):
        raise ValueError("explicit route components do not reconstruct contribution")
    if not np.allclose(
        remainder_contribution,
        remainder_root + remainder_carrier + remainder_gate,
        atol=2e-5,
    ):
        raise ValueError("route tail components do not reconstruct contribution")
    return (
        source,
        head,
        magnitude,
        contribution,
        root,
        carrier,
        gate,
        remainder_magnitude,
        remainder_contribution,
        remainder_root,
        remainder_carrier,
        remainder_gate,
        cover_size,
    )


def sparse_contribution_sum(artifact: Mapping[str, Any]) -> np.ndarray:
    """Sum saved signed edge contributions and the endpoint-free tail.

    The result is ``[layer, response token, register]``. It is one for a
    nonzero exactly decomposed attention write and zero for a zero write.
    """

    arrays = _route_arrays(artifact["trace"])
    source, contribution, remainder, cover_size = (
        arrays[0],
        arrays[3],
        arrays[8],
        arrays[12],
    )
    slots = source.shape[-1]
    selected = np.arange(slots) < np.minimum(cover_size, slots)[..., None]
    return np.where(selected, contribution, 0.0).sum(axis=-1) + remainder


@dataclass(frozen=True)
class RegisterRouteGraph:
    """Sparse register routes plus their aligned additive-stage DAG."""

    source: np.ndarray
    target: np.ndarray
    layer: np.ndarray
    head: np.ndarray
    register: np.ndarray
    role: np.ndarray
    magnitude: np.ndarray
    contribution: np.ndarray
    root_contribution: np.ndarray
    carrier_contribution: np.ndarray
    gate_contribution: np.ndarray
    route_source_node: np.ndarray
    route_target_node: np.ndarray

    row_target: np.ndarray
    row_layer: np.ndarray
    row_register: np.ndarray
    remainder_magnitude: np.ndarray
    remainder_contribution: np.ndarray
    remainder_root_contribution: np.ndarray
    remainder_carrier_contribution: np.ndarray
    remainder_gate_contribution: np.ndarray
    cover_size: np.ndarray
    register_role_mass: np.ndarray
    register_role_contribution: np.ndarray
    register_role_root_contribution: np.ndarray
    register_role_carrier_contribution: np.ndarray
    register_role_gate_contribution: np.ndarray

    node_id: np.ndarray
    node_target: np.ndarray
    node_layer: np.ndarray
    node_register: np.ndarray
    node_stage: np.ndarray
    node_norm: np.ndarray
    vertical_source: np.ndarray
    vertical_target: np.ndarray

    register_conservation_error: np.ndarray
    register_attention_edge_error: np.ndarray


def _source_roles(
    source: np.ndarray,
    target: np.ndarray,
    response_start: int,
    evidence_mask: np.ndarray,
) -> np.ndarray:
    if np.any(source > target):
        raise ValueError("provenance graph contains a future source endpoint")
    role = np.full(source.shape, "response_history", dtype="<U16")
    prompt = source < response_start
    role[prompt] = "other_prompt"
    prompt_index = source.clip(0, response_start - 1)
    role[prompt & evidence_mask[prompt_index]] = "evidence"
    role[source == target] = "predictor_self"
    return role


def build_graph(artifact: Mapping[str, Any]) -> RegisterRouteGraph:
    """Expose captured residual-message routes without averaging heads.

    Explicit edges retain their true token and head endpoints. The global
    adaptive-cover tail stays a row statistic because assigning it an endpoint
    or head would invent a route. Register norms form stage nodes and vertical
    additive component transitions.
    """

    trace = artifact["trace"]
    (
        source,
        head,
        magnitude,
        contribution,
        root,
        carrier,
        gate,
        remainder_magnitude,
        remainder_contribution,
        remainder_root,
        remainder_carrier,
        remainder_gate,
        cover_size,
    ) = _route_arrays(trace)
    layers, tokens, registers, slots = source.shape
    response_start = int(artifact["response_start"])
    evidence_mask = _array(artifact["evidence_mask"]).astype(bool)
    if response_start <= 0 or evidence_mask.shape != (response_start,):
        raise ValueError("evidence_mask must align exactly with the prompt")

    row_shape = (layers, tokens, registers)
    layer, token, register = np.indices(row_shape)
    target = response_start - 1 + token
    selected = np.arange(slots) < np.minimum(cover_size, slots)[..., None]
    if np.any(source[selected] < 0) or np.any(head[selected] < 0):
        raise ValueError(
            "saved route slots must have explicit source and head endpoints"
        )
    edge_source = source[selected]
    edge_target = np.broadcast_to(target[..., None], source.shape)[selected]
    edge_role = _source_roles(edge_source, edge_target, response_start, evidence_mask)

    role_mass = _array(trace["register_role_mass"]).astype(np.float64)
    role_contribution = _array(trace["register_role_contribution"]).astype(np.float64)
    role_root = _array(trace["register_role_root_contribution"]).astype(np.float64)
    role_carrier = _array(trace["register_role_carrier_contribution"]).astype(
        np.float64
    )
    role_gate = _array(trace["register_role_gate_contribution"]).astype(np.float64)
    if role_mass.ndim != 5 or role_mass.shape[:2] != (layers, tokens):
        raise ValueError(
            "register role mass must be [layer, token, head, register, role]"
        )
    if role_mass.shape[3:] != (registers, len(ROLE_NAMES)):
        raise ValueError("register role axes do not align")
    if any(
        value.shape != role_mass.shape
        for value in (role_contribution, role_root, role_carrier, role_gate)
    ):
        raise ValueError("register role contribution must align with role mass")
    if (
        not np.isfinite(role_mass).all()
        or np.any(role_mass < 0)
        or not all(
            np.isfinite(value).all()
            for value in (role_contribution, role_root, role_carrier, role_gate)
        )
    ):
        raise ValueError("register role measurements must be finite")
    if not np.allclose(
        role_contribution, role_root + role_carrier + role_gate, atol=2e-5
    ):
        raise ValueError("register role components do not reconstruct contribution")
    sparse_magnitude = np.where(selected, magnitude, 0.0).sum(-1)
    sparse_contribution = np.where(selected, contribution, 0.0).sum(-1)
    dense_magnitude = role_mass.sum(axis=(2, 4))
    dense_contribution = role_contribution.sum(axis=(2, 4))
    if not np.allclose(
        sparse_magnitude + remainder_magnitude,
        dense_magnitude,
        atol=5e-3,
        rtol=5e-3,
    ):
        raise ValueError("sparse magnitude and tail do not match dense role mass")
    if not np.allclose(
        sparse_contribution + remainder_contribution,
        dense_contribution,
        atol=2e-4,
        rtol=2e-4,
    ):
        raise ValueError("sparse contribution and tail do not match dense role total")
    for component, explicit, remainder, dense in (
        ("root", root, remainder_root, role_root),
        ("carrier", carrier, remainder_carrier, role_carrier),
        ("gate", gate, remainder_gate, role_gate),
    ):
        sparse = np.where(selected, explicit, 0.0).sum(-1) + remainder
        if not np.allclose(
            sparse,
            dense.sum(axis=(2, 4)),
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError(
                f"sparse {component} contribution does not match dense role total"
            )
    heads = role_mass.shape[2]
    if np.any(head[selected] >= heads):
        raise ValueError("saved route head endpoint is out of range")

    register_norm = _array(trace["register_norm"]).astype(np.float64)
    node_shape = (layers, tokens, registers, len(STAGE_NAMES))
    if register_norm.shape != node_shape:
        raise ValueError("register norms must be [layer, token, register, stage]")
    if not np.isfinite(register_norm).all() or np.any(register_norm < 0):
        raise ValueError("register norms must be finite and nonnegative")
    conservation_error = _array(trace["register_conservation_error"]).astype(np.float64)
    attention_edge_error = _array(trace["register_attention_edge_error"]).astype(
        np.float64
    )
    error_shape = (layers, tokens, registers)
    if (
        conservation_error.shape != error_shape
        or attention_edge_error.shape != error_shape
    ):
        raise ValueError("register errors must be [layer, token, register]")
    if (
        not np.isfinite(conservation_error).all()
        or not np.isfinite(attention_edge_error).all()
        or np.any(conservation_error < 0)
        or np.any(attention_edge_error < 0)
    ):
        raise ValueError("register errors must be finite and nonnegative")

    node_layer, node_token, node_register, node_stage = np.indices(node_shape)
    node_id = np.arange(register_norm.size).reshape(node_shape)
    within_source = node_id[..., :3].ravel()
    within_target = np.broadcast_to(
        node_id[..., 3, None], node_id[..., :3].shape
    ).ravel()
    across_source = node_id[:-1, ..., -1].ravel()
    across_target = node_id[1:, ..., 0].ravel()

    edge_layer = np.broadcast_to(layer[..., None], source.shape)[selected]
    edge_token = np.broadcast_to(token[..., None], source.shape)[selected]
    edge_register_index = np.broadcast_to(register[..., None], source.shape)[selected]
    route_target_node = node_id[
        edge_layer,
        edge_token,
        edge_register_index,
        STAGE_NAMES.index("attention_write"),
    ]
    source_token = edge_source - (response_start - 1)
    captured_source = source_token >= 0
    route_source_node = np.full(len(edge_source), -1, dtype=np.int64)
    route_source_node[captured_source] = node_id[
        edge_layer[captured_source],
        source_token[captured_source],
        edge_register_index[captured_source],
        STAGE_NAMES.index("input_state"),
    ]

    return RegisterRouteGraph(
        source=edge_source,
        target=edge_target,
        layer=np.broadcast_to(layer[..., None], source.shape)[selected],
        head=head[selected],
        register=np.asarray(REGISTER_NAMES)[
            np.broadcast_to(register[..., None], source.shape)[selected]
        ],
        role=edge_role,
        magnitude=magnitude[selected],
        contribution=contribution[selected],
        root_contribution=root[selected],
        carrier_contribution=carrier[selected],
        gate_contribution=gate[selected],
        route_source_node=route_source_node,
        route_target_node=route_target_node,
        row_target=target.ravel(),
        row_layer=layer.ravel(),
        row_register=np.asarray(REGISTER_NAMES)[register.ravel()],
        remainder_magnitude=remainder_magnitude.ravel(),
        remainder_contribution=remainder_contribution.ravel(),
        remainder_root_contribution=remainder_root.ravel(),
        remainder_carrier_contribution=remainder_carrier.ravel(),
        remainder_gate_contribution=remainder_gate.ravel(),
        cover_size=cover_size.ravel(),
        register_role_mass=role_mass,
        register_role_contribution=role_contribution,
        register_role_root_contribution=role_root,
        register_role_carrier_contribution=role_carrier,
        register_role_gate_contribution=role_gate,
        node_id=node_id.ravel(),
        node_target=(response_start - 1 + node_token).ravel(),
        node_layer=node_layer.ravel(),
        node_register=np.asarray(REGISTER_NAMES)[node_register.ravel()],
        node_stage=np.asarray(STAGE_NAMES)[node_stage.ravel()],
        node_norm=register_norm.ravel(),
        vertical_source=np.concatenate((within_source, across_source)),
        vertical_target=np.concatenate((within_target, across_target)),
        register_conservation_error=conservation_error,
        register_attention_edge_error=attention_edge_error,
    )
