"""Compact per-sample NPZ artifacts for the shortcut-route audit."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .route_shortcut import (
    CARRIER_NAMES,
    NUMERIC,
    PredictionEvents,
    RouteAxes,
    SparseRoutes,
    moments_from_sparse,
    route_axes_from_sparse,
    token_carriers,
)

SCHEMA = "shortcut-route-avwo-v1"


@dataclass(frozen=True)
class RouteReadout:
    """Terminal closure quantities for one sample."""

    competitor_token_id: Tensor  # [event]
    target_logprob: Tensor  # [event]
    injection_phi: Tensor  # [event, root]
    terminal_root_margin: Tensor  # [event, root]
    native_margin: Tensor  # [event]
    root_closure_error: Tensor  # [event, root]
    numeric_self_v_phi: Tensor  # [event, layer, head]
    numeric_post_attention_phi: Tensor  # [event, layer]
    numeric_layer_phi: Tensor  # [event, layer]
    numeric_final_phi: Tensor  # [event]
    numeric_total_variation: Tensor  # [event]
    operator_error: Tensor  # [event], unabsorbed outward margin bound
    operator_valid: Tensor  # [event]


@dataclass(frozen=True)
class RouteArtifact:
    """The saved graph view, three mechanism axes, and readout closure."""

    response_start: int
    source_token_id: Tensor  # [source], canonical model input token sequence
    evidence_mask: Tensor  # [source], declared evidence carrier membership
    top_k: int
    cover_mass: float
    events: PredictionEvents
    routes: SparseRoutes
    axes: RouteAxes
    readout: RouteReadout


def save_route_artifact(path: str | Path, artifact: RouteArtifact) -> None:
    """Save one label-free sample without serializing Python objects."""

    validate_artifact(artifact)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "response_start": np.asarray(artifact.response_start, dtype=np.int32),
        "source_token_id": artifact.source_token_id.detach().cpu().numpy(),
        "evidence_mask": artifact.evidence_mask.detach().cpu().numpy(),
        "top_k": np.asarray(artifact.top_k, dtype=np.int32),
        "cover_mass": np.asarray(artifact.cover_mass, dtype=np.float64),
    }
    add_fields(arrays, "event", artifact.events)
    add_fields(arrays, "route", artifact.routes)
    add_fields(arrays, "axis", artifact.axes)
    add_fields(arrays, "readout", artifact.readout)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def load_route_artifact(path: str | Path) -> RouteArtifact:
    """Load the fixed label-free schema and reconstruct its data objects."""

    with np.load(Path(path), allow_pickle=False) as stored:
        if str(stored["schema"].item()) != SCHEMA:
            raise ValueError("route artifact schema does not match")
        artifact = RouteArtifact(
            response_start=int(stored["response_start"].item()),
            source_token_id=torch.from_numpy(
                np.array(stored["source_token_id"], copy=True)
            ),
            evidence_mask=torch.from_numpy(
                np.array(stored["evidence_mask"], copy=True)
            ),
            top_k=int(stored["top_k"].item()),
            cover_mass=float(stored["cover_mass"].item()),
            events=load_fields(stored, "event", PredictionEvents),
            routes=load_fields(stored, "route", SparseRoutes),
            axes=load_fields(stored, "axis", RouteAxes),
            readout=load_fields(stored, "readout", RouteReadout),
        )
    validate_artifact(artifact)
    return artifact


def add_fields(arrays: dict[str, np.ndarray], prefix: str, value: object) -> None:
    """Encode one dataclass using stable field names and numeric arrays."""

    for field in fields(value):
        tensor = torch.as_tensor(getattr(value, field.name)).detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        arrays[f"{prefix}_{field.name}"] = tensor.numpy()


def load_fields(stored: np.lib.npyio.NpzFile, prefix: str, cls: type):
    """Decode one fixed dataclass from numeric NPZ arrays."""

    return cls(
        **{
            field.name: torch.from_numpy(
                np.array(stored[f"{prefix}_{field.name}"], copy=True)
            )
            for field in fields(cls)
        }
    )


def validate_artifact(artifact: RouteArtifact) -> None:
    """Check causal alignment and the small set of cross-object invariants."""

    events = artifact.events
    if isinstance(artifact.response_start, bool) or not isinstance(
        artifact.response_start, int
    ):
        raise TypeError("response_start must be an integer")
    if artifact.response_start <= 0:
        raise ValueError("response_start must be positive")
    if events.query_position.ndim != 1:
        raise ValueError("event coordinates must be aligned vectors")
    count = len(events.query_position)
    if events.prediction_position.shape != (count,):
        raise ValueError("event coordinates must be aligned vectors")
    if (
        events.query_position.dtype != torch.int64
        or events.prediction_position.dtype != torch.int64
    ):
        raise ValueError("event coordinates must use int64")
    if not torch.equal(events.prediction_position, events.query_position + 1):
        raise ValueError("every saved event must predict q + 1")
    if events.target_token_id.shape != (count,):
        raise ValueError("target token ids must align to events")
    if events.target_token_id.dtype != torch.int64:
        raise ValueError("target token ids must use int64")
    source_token_id = artifact.source_token_id
    if source_token_id.ndim != 1 or source_token_id.dtype != torch.int64:
        raise ValueError("source_token_id must be an int64 sequence vector")
    if artifact.response_start > len(source_token_id):
        raise ValueError("response_start must index the canonical source sequence")
    if artifact.evidence_mask.shape != source_token_id.shape:
        raise ValueError("evidence_mask must align to the canonical source sequence")
    if artifact.evidence_mask.dtype != torch.bool:
        raise ValueError("evidence_mask must have boolean dtype")
    if artifact.evidence_mask[artifact.response_start :].any():
        raise ValueError("response tokens cannot use the evidence-prompt carrier")
    if isinstance(artifact.top_k, bool) or not isinstance(artifact.top_k, int):
        raise TypeError("top_k must be an integer")
    if artifact.top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if artifact.top_k > np.iinfo(np.int32).max:
        raise ValueError("top_k exceeds the serialized int32 range")
    if not math.isfinite(artifact.cover_mass) or not 0 < artifact.cover_mass <= 1:
        raise ValueError("cover_mass must be in (0, 1]")
    expected_prediction = torch.arange(
        artifact.response_start,
        len(source_token_id) + 1,
        dtype=events.prediction_position.dtype,
        device=events.prediction_position.device,
    )
    if not torch.equal(events.prediction_position, expected_prediction):
        raise ValueError("prediction positions must cover the complete response")
    if count > 1 and not torch.equal(
        events.target_token_id[:-1].to(device=source_token_id.device),
        source_token_id[artifact.response_start :],
    ):
        raise ValueError("response targets must bind to the canonical source tokens")
    for name, token_ids in (
        ("source_token_id", source_token_id),
        ("target_token_id", events.target_token_id),
    ):
        if (token_ids < 0).any():
            raise ValueError(f"{name} must be nonnegative")
    readout = artifact.readout
    if readout.competitor_token_id.shape != (count,):
        raise ValueError("competitor token ids must align to events")
    if readout.competitor_token_id.dtype != torch.int64:
        raise ValueError("competitor token ids must use int64")
    if (readout.competitor_token_id < 0).any():
        raise ValueError("competitor token ids must be nonnegative")
    if readout.target_logprob.shape != (count,):
        raise ValueError("target log probabilities must align to events")
    if not readout.target_logprob.is_floating_point():
        raise ValueError("target log probabilities must have a floating dtype")
    if readout.injection_phi.shape != (count, 4):
        raise ValueError("injection_phi must be [event, root]")
    if readout.terminal_root_margin.shape != (count, 4):
        raise ValueError("terminal_root_margin must be [event, root]")
    for value in (
        readout.native_margin,
        readout.numeric_total_variation,
        readout.operator_error,
        readout.operator_valid,
    ):
        if value.shape != (count,):
            raise ValueError("readout vectors must contain one value per event")
    if readout.operator_valid.dtype != torch.bool:
        raise ValueError("operator_valid must have boolean dtype")
    if readout.root_closure_error.shape != (count, 4):
        raise ValueError("root_closure_error must be [event, root]")
    if (
        readout.numeric_self_v_phi.ndim != 3
        or readout.numeric_self_v_phi.shape[0] != count
    ):
        raise ValueError("numeric_self_v_phi must be [event, layer, head]")
    numeric_layer_shape = readout.numeric_self_v_phi.shape[:2]
    if readout.numeric_post_attention_phi.shape != numeric_layer_shape:
        raise ValueError("numeric_post_attention_phi must be [event, layer]")
    if readout.numeric_layer_phi.shape != numeric_layer_shape:
        raise ValueError("numeric_layer_phi must be [event, layer]")
    if readout.numeric_final_phi.shape != (count,):
        raise ValueError("numeric_final_phi must contain one value per event")
    scientific_readout = tuple(
        getattr(readout, field.name)
        for field in fields(RouteReadout)
        if field.name not in {"competitor_token_id", "operator_valid"}
    )
    if any(not value.is_floating_point() for value in scientific_readout):
        raise ValueError("readout scientific quantities must be floating point")
    if any(not torch.isfinite(value).all() for value in scientific_readout):
        raise ValueError("readout scientific quantities must be finite")
    routes = artifact.routes
    for name, coordinate in (
        ("row_event", routes.row_event),
        ("row_layer", routes.row_layer),
        ("row_head", routes.row_head),
        ("row_ptr", routes.row_ptr),
        ("carrier", routes.carrier),
    ):
        if not is_integer_tensor(coordinate):
            raise ValueError(f"route {name} must have an integer dtype")
    if (
        routes.row_ptr.ndim != 1
        or routes.row_ptr.numel() == 0
        or int(routes.row_ptr[0]) != 0
    ):
        raise ValueError("route row_ptr must start at zero")
    if (routes.row_ptr.diff() < 0).any():
        raise ValueError("route row_ptr must be monotone")
    if (routes.row_ptr.diff() > artifact.top_k).any():
        raise ValueError("a sparse row exceeds the saved top_k")
    if routes.source_position.ndim != 1:
        raise ValueError("saved source positions must be a vector")
    edge_count = len(routes.source_position)
    if int(routes.row_ptr[-1]) != edge_count:
        raise ValueError("route row_ptr does not close the selected edges")
    edge_fields = (
        routes.source_position,
        routes.carrier,
        routes.attention,
        routes.value_energy,
        routes.physical_message_norm,
    )
    if any(value.shape != (edge_count,) for value in edge_fields):
        raise ValueError("selected physical-edge arrays must align")
    if routes.root_phi.shape != (edge_count, 4):
        raise ValueError("selected root contributions must be [edge, root]")
    if routes.source_position.dtype != torch.int64:
        raise ValueError("saved source positions must use int64")
    if not is_integer_tensor(routes.tail_count):
        raise ValueError("tail_count must have an integer dtype")
    tail_shape = (len(routes.row_event), len(CARRIER_NAMES))
    scalar_tail_fields = (
        routes.tail_count,
        routes.tail_attention_sum,
        routes.tail_value_energy_sum,
        routes.tail_message_norm_sum,
        routes.tail_message_norm_max,
        routes.tail_physical_positive,
        routes.tail_physical_negative,
        routes.tail_physical_pos_xlogx,
        routes.tail_physical_neg_xlogx,
    )
    if any(value.shape != tail_shape for value in scalar_tail_fields):
        raise ValueError("scalar tail moments must align to row and carrier")
    root_tail_shape = (*tail_shape, 4)
    if (
        routes.tail_root_positive.shape != root_tail_shape
        or routes.tail_root_negative.shape != root_tail_shape
    ):
        raise ValueError("root tail moments must align to row, carrier, and root")
    floating_route_fields = tuple(
        getattr(routes, field.name)
        for field in fields(SparseRoutes)
        if field.name
        not in {
            "row_event",
            "row_layer",
            "row_head",
            "row_ptr",
            "source_position",
            "carrier",
            "tail_count",
        }
    )
    if any(not value.is_floating_point() for value in floating_route_fields):
        raise ValueError("route scientific quantities must be floating point")
    nonnegative_fields = (
        routes.attention,
        routes.value_energy,
        routes.physical_message_norm,
        routes.tail_count,
        routes.tail_attention_sum,
        routes.tail_value_energy_sum,
        routes.tail_message_norm_sum,
        routes.tail_message_norm_max,
        routes.tail_root_positive,
        routes.tail_root_negative,
        routes.tail_physical_positive,
        routes.tail_physical_negative,
    )
    if any(not torch.isfinite(value.float()).all() for value in nonnegative_fields):
        raise ValueError("saved route magnitudes must be finite")
    if any((value < 0).any() for value in nonnegative_fields):
        raise ValueError("saved route magnitudes must be nonnegative")
    signed_fields = (
        routes.root_phi,
        routes.tail_physical_pos_xlogx,
        routes.tail_physical_neg_xlogx,
    )
    if any(not torch.isfinite(value).all() for value in signed_fields):
        raise ValueError("saved route contributions must be finite")
    empty_tail = routes.tail_count == 0
    empty_scalar_moments = scalar_tail_fields[1:]
    if any((value[empty_tail] != 0).any() for value in empty_scalar_moments):
        raise ValueError("an empty carrier tail must have zero scalar moments")
    empty_root = empty_tail[..., None].expand_as(routes.tail_root_positive)
    if (routes.tail_root_positive[empty_root] != 0).any() or (
        routes.tail_root_negative[empty_root] != 0
    ).any():
        raise ValueError("an empty carrier tail must have zero root moments")
    if (routes.tail_message_norm_max > routes.tail_message_norm_sum).any():
        raise ValueError("tail maximum message norm cannot exceed its sum")
    moments = moments_from_sparse(routes)
    event_count, layer_count, head_count = moments.physical_mass.shape[:3]
    if event_count != count:
        raise ValueError("route row grid must contain every prediction event")
    if readout.numeric_self_v_phi.shape != (count, layer_count, head_count):
        raise ValueError("numeric_self_v_phi must align to route layers and heads")
    if readout.numeric_layer_phi.shape != (count, layer_count):
        raise ValueError("numeric local layer fields must align to route layers")
    source_event = torch.repeat_interleave(
        routes.row_event.long(), routes.row_ptr.diff().long()
    )
    selected_row = torch.repeat_interleave(
        torch.arange(len(routes.row_event), device=routes.source_position.device),
        routes.row_ptr.diff().to(
            device=routes.source_position.device, dtype=torch.long
        ),
    )
    if edge_count:
        if not bool((routes.source_position >= 0).all()):
            raise ValueError("saved source positions must be nonnegative")
        if not bool((routes.source_position < len(source_token_id)).all()):
            raise ValueError("saved source positions must index source_token_id")
        query = events.query_position[source_event]
        if not bool((routes.source_position < query).all()):
            raise ValueError("saved routes must be strict causal arrivals")
        source_key = selected_row * len(source_token_id) + routes.source_position
        if len(torch.unique(source_key)) != edge_count:
            raise ValueError("a sparse row cannot repeat a physical source")

    source_position = torch.arange(
        len(source_token_id),
        dtype=torch.long,
        device=routes.source_position.device,
    )
    expected_carrier = token_carriers(
        source_position,
        artifact.response_start,
        artifact.evidence_mask.to(device=source_position.device),
    )
    saved_carrier = routes.carrier.to(device=source_position.device, dtype=torch.long)
    if edge_count and not torch.equal(
        saved_carrier,
        expected_carrier[routes.source_position.long()],
    ):
        raise ValueError("selected route carrier disagrees with evidence membership")
    selected_count = torch.zeros(
        len(routes.row_event) * len(CARRIER_NAMES),
        dtype=torch.long,
        device=source_position.device,
    )
    if edge_count:
        selected_count.scatter_add_(
            0,
            selected_row * len(CARRIER_NAMES) + saved_carrier,
            torch.ones(edge_count, dtype=torch.long, device=source_position.device),
        )
    selected_count = selected_count.reshape(len(routes.row_event), len(CARRIER_NAMES))
    carrier_prefix = torch.stack(
        [
            (expected_carrier == role).long().cumsum(dim=0)
            for role in range(len(CARRIER_NAMES))
        ],
        dim=1,
    )
    event_query = events.query_position.to(device=source_position.device)
    event_carrier_count = torch.zeros(
        count,
        len(CARRIER_NAMES),
        dtype=torch.long,
        device=source_position.device,
    )
    positive_query = event_query > 0
    event_carrier_count[positive_query] = carrier_prefix[
        event_query[positive_query] - 1
    ]
    expected_row_count = event_carrier_count[
        routes.row_event.to(device=source_position.device, dtype=torch.long)
    ]
    observed_row_count = selected_count + routes.tail_count.to(
        device=source_position.device, dtype=torch.long
    )
    if not torch.equal(observed_row_count, expected_row_count):
        raise ValueError("selected and tail carriers do not cover every causal source")

    selected_norm = routes.physical_message_norm
    if edge_count and (selected_norm <= 0).any():
        raise ValueError("selected routes must have positive message norm")
    if edge_count > 1:
        adjacent = selected_row[:-1] == selected_row[1:]
        stronger_after = selected_norm[:-1] < selected_norm[1:]
        tied_out_of_order = (selected_norm[:-1] == selected_norm[1:]) & (
            routes.source_position[:-1] > routes.source_position[1:]
        )
        if ((stronger_after | tied_out_of_order) & adjacent).any():
            raise ValueError("selected routes do not follow the saved ranking")
    row_selected_count = routes.row_ptr.diff().to(
        device=selected_norm.device, dtype=torch.long
    )
    retained_norm = torch.zeros(
        len(routes.row_event),
        dtype=selected_norm.dtype,
        device=selected_norm.device,
    )
    if edge_count:
        retained_norm.scatter_add_(
            0,
            selected_row.to(device=selected_norm.device),
            selected_norm,
        )
    tail_norm = routes.tail_message_norm_sum.sum(dim=1).to(
        device=selected_norm.device, dtype=selected_norm.dtype
    )
    total_norm = retained_norm + tail_norm
    positive_total = total_norm > 0
    if artifact.top_k > 0 and (positive_total & (row_selected_count == 0)).any():
        raise ValueError("a positive row must retain its strongest route")
    last_norm = torch.zeros_like(retained_norm)
    has_selected = row_selected_count > 0
    if has_selected.any():
        last_index = (
            routes.row_ptr[1:]
            .to(device=selected_norm.device, dtype=torch.long)[has_selected]
            .sub(1)
        )
        last_norm[has_selected] = selected_norm[last_index]
    coverage_target = artifact.cover_mass * total_norm
    accumulation_depth = (
        expected_row_count.sum(dim=1)
        .to(device=selected_norm.device, dtype=selected_norm.dtype)
        .clamp_min(1)
        .log2()
        .ceil()
        + 4
    )
    coverage_tolerance = (
        accumulation_depth * torch.finfo(selected_norm.dtype).eps * total_norm.abs()
    )
    not_capped = row_selected_count < artifact.top_k
    if (
        positive_total
        & not_capped
        & (retained_norm + coverage_tolerance < coverage_target)
    ).any():
        raise ValueError("an uncapped sparse row does not reach cover_mass")
    previous_norm = retained_norm - last_norm
    if (has_selected & (previous_norm > coverage_target + coverage_tolerance)).any():
        raise ValueError("a sparse row is not the minimal coverage prefix")
    tail_max = routes.tail_message_norm_max.max(dim=1).values.to(
        device=selected_norm.device, dtype=selected_norm.dtype
    )
    if (has_selected & (tail_max > last_norm)).any():
        raise ValueError("a stronger physical route was moved into the tail")

    device = readout.numeric_total_variation.device
    selected_event = source_event.to(device=device)
    selected_numeric = routes.root_phi[:, NUMERIC].to(device=device).abs()
    edge_numeric = torch.zeros(count, device=device, dtype=selected_numeric.dtype)
    edge_numeric.scatter_add_(0, selected_event, selected_numeric)
    tail_numeric = (
        routes.tail_root_positive[..., NUMERIC]
        + routes.tail_root_negative[..., NUMERIC]
    ).sum(dim=1)
    edge_numeric.scatter_add_(
        0,
        routes.row_event.to(device=device, dtype=torch.long),
        tail_numeric.to(device=device, dtype=edge_numeric.dtype),
    )
    self_value_numeric = readout.numeric_self_v_phi.to(device=device)
    post_attention_numeric = readout.numeric_post_attention_phi.to(device=device)
    layer_numeric = readout.numeric_layer_phi.to(device=device)
    final_numeric = readout.numeric_final_phi.to(device=device)
    local_numeric_net = (
        self_value_numeric.sum(dim=(1, 2))
        + post_attention_numeric.sum(dim=1)
        + layer_numeric.sum(dim=1)
        + final_numeric
    )
    initial_numeric = (
        readout.injection_phi[:, NUMERIC].to(device=device) - local_numeric_net
    )
    expected_numeric_variation = (
        initial_numeric.abs()
        + edge_numeric
        + self_value_numeric.abs().sum(dim=(1, 2))
        + post_attention_numeric.abs().sum(dim=1)
        + layer_numeric.abs().sum(dim=1)
        + final_numeric.abs()
    )
    if not torch.allclose(
        readout.numeric_total_variation,
        expected_numeric_variation,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("numeric_total_variation does not close saved N terms")
    if (
        not torch.isfinite(readout.operator_error).all()
        or (readout.operator_error < 0).any()
    ):
        raise ValueError("operator_error must be finite and nonnegative")
    route_root_net = (
        (moments.root_mass[..., 0] - moments.root_mass[..., 1])
        .sum(dim=(1, 2, 3))
        .to(
            device=readout.injection_phi.device,
            dtype=readout.injection_phi.dtype,
        )
    )
    expected_root_error = readout.terminal_root_margin - (
        readout.injection_phi + route_root_net
    )
    root_total_variation = moments.root_mass.sum(dim=(1, 2, 3, 5)).to(
        device=readout.root_closure_error.device,
        dtype=readout.root_closure_error.dtype,
    )
    reduction_terms = (
        events.query_position.to(
            device=root_total_variation.device,
            dtype=root_total_variation.dtype,
        ).clamp_min(1)
        * layer_count
        * head_count
    )
    reduction_depth = reduction_terms.log2().ceil() + 8
    accumulation_tolerance = 1e-6 + (
        reduction_depth[:, None]
        * torch.finfo(root_total_variation.dtype).eps
        * root_total_variation
    )
    if (
        (readout.root_closure_error - expected_root_error).abs()
        > accumulation_tolerance
    ).any():
        raise ValueError("root_closure_error does not close saved route terms")
    native_error = (
        readout.terminal_root_margin.sum(dim=1) - readout.native_margin
    ).abs()
    operator_tolerance = 1e-6 + 1e-5 * readout.native_margin.abs()
    if (native_error > readout.operator_error + operator_tolerance).any():
        raise ValueError("operator_error does not bound the native margin error")
    expected_resolution = (
        readout.numeric_total_variation
        + readout.operator_error
        + readout.root_closure_error.abs().sum(dim=1)
    )
    if not torch.allclose(
        artifact.axes.resolution,
        expected_resolution,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("axis resolution does not close saved error terms")
    reconstructed_axes = route_axes_from_sparse(
        routes,
        readout.injection_phi,
        event_valid=readout.operator_valid,
        resolution=readout.operator_error + readout.root_closure_error.abs().sum(dim=1),
        numeric_total_variation=readout.numeric_total_variation,
    )
    for field in fields(RouteAxes):
        saved = getattr(artifact.axes, field.name)
        reconstructed = getattr(reconstructed_axes, field.name).to(device=saved.device)
        if saved.shape != reconstructed.shape:
            raise ValueError(f"axis {field.name} has an incompatible shape")
        if saved.dtype != reconstructed.dtype:
            raise ValueError(f"axis {field.name} has an incompatible dtype")
        if saved.dtype == torch.bool:
            agrees = torch.equal(saved, reconstructed)
        else:
            agrees = torch.allclose(
                saved,
                reconstructed.to(dtype=saved.dtype),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )
        if not agrees:
            raise ValueError(f"axis {field.name} does not match sparse route moments")


def is_integer_tensor(value: Tensor) -> bool:
    """Return whether a tensor uses one of PyTorch's integer dtypes."""

    return value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
