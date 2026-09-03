"""Assemble one shortcut-route artifact from captured native operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .route_artifact import RouteArtifact, RouteReadout
from .route_shortcut import (
    NUMERIC,
    LayerRoutes,
    PredictionEvents,
    measure_layer_routes,
    route_axes,
    route_axes_from_sparse,
    sparsify_routes,
    token_carriers,
)
from .route_suffix import (
    ObservedSuffix,
    injection_contribution,
    reverse_observed_suffix,
)

ROOT_CLOSURE_RELATIVE_LIMIT = 1e-3
ROOT_CLOSURE_ABSOLUTE_LIMIT = 1e-6


@dataclass(frozen=True)
class CapturedRouteOperators:
    """Label-free tensors captured from one native teacher-forced forward."""

    response_start: int
    events: PredictionEvents
    source_token_id: Tensor  # [source], canonical model input token sequence
    competitor_token_id: Tensor  # [event]
    target_logprob: Tensor  # [event]
    source_position: Tensor  # [source]
    evidence_mask: Tensor  # [source]
    root_values: Sequence[Tensor]  # layer: [source, root, kv_head, head_dim]
    input_roots: Tensor  # [event, root, hidden] at predictor q
    suffix: ObservedSuffix
    self_value_numeric_write: Tensor  # [event, layer, head, hidden]
    post_attention_numeric_write: Tensor  # [event, layer, hidden]
    layer_numeric_write: Tensor  # [event, layer, hidden], decoder-layer output
    final_rms_numeric_write: Tensor  # [event, hidden], normalized output
    terminal_root_margin: Tensor  # [event, root]
    native_margin: Tensor  # [event]
    operator_error: Tensor  # [event], outward absolute bound
    operator_valid: Tensor  # [event]


def build_route_artifact(
    captured: CapturedRouteOperators,
    *,
    top_k: int = 64,
    cover_mass: float = 0.95,
) -> RouteArtifact:
    """Compute head-resolved atoms, three axes, exact tails, and closure."""

    if len(captured.root_values) != len(captured.suffix.attention):
        raise ValueError("one rooted value table is required per decoder layer")
    suffix = reverse_observed_suffix(captured.suffix)
    events = len(captured.events.query_position)
    layer_count = len(captured.suffix.attention)
    hidden = captured.suffix.readout_direction.shape[1]
    heads = captured.suffix.attention[0].shape[1]
    if captured.self_value_numeric_write.shape != (
        events,
        layer_count,
        heads,
        hidden,
    ):
        raise ValueError(
            "self_value_numeric_write must be [event, layer, head, hidden]"
        )
    if captured.post_attention_numeric_write.shape != (events, layer_count, hidden):
        raise ValueError("post_attention_numeric_write must be [event, layer, hidden]")
    if captured.layer_numeric_write.shape != (events, layer_count, hidden):
        raise ValueError("layer_numeric_write must be [event, layer, hidden]")
    if captured.final_rms_numeric_write.shape != (events, hidden):
        raise ValueError("final_rms_numeric_write must be [event, hidden]")
    if len(suffix.layer_output) != layer_count or any(
        value.shape != (events, hidden) for value in suffix.layer_output
    ):
        raise ValueError("one layer-output adjoint is required per event and layer")
    carrier = token_carriers(
        captured.source_position,
        captured.response_start,
        captured.evidence_mask,
    )
    layers: list[LayerRoutes] = []
    for layer, (attention, root_values, output_weight, adjoint) in enumerate(
        zip(
            captured.suffix.attention,
            captured.root_values,
            captured.suffix.output_weight,
            suffix.attention_write,
            strict=True,
        )
    ):
        layers.append(
            measure_layer_routes(
                layer=layer,
                attention=attention,
                root_values=root_values,
                output_weight=output_weight,
                suffix_adjoint=adjoint,
                query_position=captured.events.query_position,
                source_position=captured.source_position,
                carrier=carrier,
            )
        )

    injection = injection_contribution(captured.input_roots, suffix.input)
    initial_numeric_phi = injection[:, NUMERIC].clone()
    self_value_numeric_phi = torch.einsum(
        "elhd,eld->elh",
        captured.self_value_numeric_write.float(),
        torch.stack(suffix.attention_write, dim=1),
    )
    post_attention_numeric_phi = torch.einsum(
        "eld,eld->el",
        captured.post_attention_numeric_write.float(),
        torch.stack(suffix.attention_write, dim=1),
    )
    layer_numeric_phi = torch.einsum(
        "eld,eld->el",
        captured.layer_numeric_write.float(),
        torch.stack(suffix.layer_output, dim=1),
    )
    final_numeric_phi = torch.einsum(
        "ed,ed->e",
        captured.final_rms_numeric_write.float(),
        captured.suffix.readout_direction.float(),
    )
    local_numeric_phi = torch.cat(
        (
            self_value_numeric_phi.flatten(1),
            post_attention_numeric_phi,
            layer_numeric_phi,
            final_numeric_phi[:, None],
        ),
        dim=1,
    )
    injection[:, NUMERIC] += local_numeric_phi.sum(dim=1)
    route_numeric_variation = torch.stack(
        [route.root_phi[..., NUMERIC].abs().sum(dim=(1, 2)) for route in layers],
        dim=1,
    ).sum(dim=1)
    numeric_total_variation = (
        initial_numeric_phi.abs()
        + local_numeric_phi.abs().sum(dim=1)
        + route_numeric_variation
    )
    arrived = injection + torch.stack(
        [route.root_phi.sum(dim=(1, 2)) for route in layers], dim=1
    ).sum(dim=1)
    terminal = captured.terminal_root_margin.float()
    root_error = terminal - arrived
    root_scale = terminal.abs() + arrived.abs()
    native_error = (terminal.sum(dim=1) - captured.native_margin.float()).abs()
    root_tolerance = (
        ROOT_CLOSURE_ABSOLUTE_LIMIT + ROOT_CLOSURE_RELATIVE_LIMIT * root_scale
    )
    native_tolerance = ROOT_CLOSURE_ABSOLUTE_LIMIT + (
        ROOT_CLOSURE_RELATIVE_LIMIT * captured.native_margin.float().abs()
    )
    closure_valid = (root_error.abs() <= root_tolerance).all(dim=1)
    closure_valid &= native_error <= (
        native_tolerance + captured.operator_error.float()
    )
    event_valid = captured.operator_valid.bool() & closure_valid
    outward_error = captured.operator_error.float() + root_error.abs().sum(dim=1)
    dense_axes = route_axes(
        layers,
        injection,
        event_valid=event_valid,
        resolution=outward_error,
        numeric_total_variation=numeric_total_variation,
    )
    sparse = sparsify_routes(layers, top_k=top_k, cover_mass=cover_mass)
    axes = route_axes_from_sparse(
        sparse,
        injection,
        event_valid=event_valid,
        resolution=outward_error,
        numeric_total_variation=numeric_total_variation,
    )
    if not torch.allclose(
        dense_axes.root_carrier_mass,
        axes.root_carrier_mass,
        rtol=5e-5,
        atol=5e-6,
    ):
        raise RuntimeError("sparse route moments do not reproduce dense root masses")
    readout = RouteReadout(
        competitor_token_id=captured.competitor_token_id,
        target_logprob=captured.target_logprob.float(),
        injection_phi=injection,
        terminal_root_margin=terminal,
        native_margin=captured.native_margin.float(),
        root_closure_error=root_error,
        numeric_self_v_phi=self_value_numeric_phi,
        numeric_post_attention_phi=post_attention_numeric_phi,
        numeric_layer_phi=layer_numeric_phi,
        numeric_final_phi=final_numeric_phi,
        numeric_total_variation=numeric_total_variation,
        operator_error=captured.operator_error.float(),
        operator_valid=event_valid,
    )
    return RouteArtifact(
        response_start=captured.response_start,
        source_token_id=captured.source_token_id,
        evidence_mask=captured.evidence_mask,
        top_k=top_k,
        cover_mass=cover_mass,
        events=captured.events,
        routes=sparse,
        axes=axes,
        readout=readout,
    )
