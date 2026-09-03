"""Head-resolved true-message measurements for response-route shortcuts.

The module measures the observed teacher-forced computation.  It does not read
hallucination labels and does not combine its three mechanism axes into a
detector.  A route keeps two independent identities throughout:

* ``root`` says where its transported value originated (E/Q/R/N);
* ``carrier`` says which physical token emitted the message.

Consequently an evidence-rooted message emitted by an earlier response token
is a grounded relay, not response-born content.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

ROOT_NAMES = ("evidence", "question", "response", "numeric")
EVIDENCE, QUESTION, RESPONSE, NUMERIC = range(4)

CARRIER_NAMES = ("evidence_prompt", "other_prompt", "response_history")
EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY = range(3)

SUPPORT, VETO = range(2)


@dataclass(frozen=True)
class PredictionEvents:
    """Causal query positions and the response tokens they predict."""

    query_position: Tensor  # [E]
    prediction_position: Tensor  # [E], exactly query + 1
    target_token_id: Tensor  # [E]


@dataclass(frozen=True)
class LayerRoutes:
    """Dense true-message measurements for one decoder layer.

    Shapes are ``E`` events, ``H`` query heads, ``S`` physical sources, and
    four roots in ``(E, Q, R, N)`` order.  Only strict first arrivals
    ``source_position < query_position`` are present; predictor self belongs
    to the downstream suffix rather than this edge table.
    """

    layer: int
    query_position: Tensor  # [E]
    source_position: Tensor  # [S]
    carrier: Tensor  # [S]
    causal: Tensor  # [E, S]
    attention: Tensor  # [E, H, S]
    value_energy: Tensor  # [E, H, S], before attention and W_O
    physical_message_norm: Tensor  # [E, H, S], ||A W_O V||
    root_phi: Tensor  # [E, H, S, 4], signed target-vs-runner contribution

    @property
    def physical_phi(self) -> Tensor:
        """Signed contribution of the complete scientific physical edge."""

        return self.root_phi[..., :NUMERIC].sum(dim=-1)

    @property
    def support(self) -> Tensor:
        return self.physical_phi.clamp_min(0)

    @property
    def veto(self) -> Tensor:
        return (-self.physical_phi).clamp_min(0)

    @property
    def root_support(self) -> Tensor:
        return self.root_phi.clamp_min(0)

    @property
    def root_veto(self) -> Tensor:
        return (-self.root_phi).clamp_min(0)


@dataclass(frozen=True)
class RouteMoments:
    """Lossless axis statistics for every event, layer, head, and carrier."""

    physical_mass: Tensor  # [E, L, H, 3, 2]
    root_mass: Tensor  # [E, L, H, 3, 4, 2]
    physical_xlogx: Tensor  # [E, L, H, 3, 2]
    eligible_source_count: Tensor  # [E, L, H, 3]


def prediction_events(token_ids: Tensor, response_start: int) -> PredictionEvents:
    """Build the only valid teacher-forcing alignment for response tokens."""

    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if ids.ndim != 1 or not 0 < response_start < len(ids):
        raise ValueError("token_ids and response_start must define a response")
    prediction = torch.arange(
        response_start,
        len(ids),
        dtype=torch.long,
        device=ids.device,
    )
    query = prediction - 1
    return PredictionEvents(query, prediction, ids[prediction])


@dataclass(frozen=True)
class RouteAxes:
    """Three separate route measurements and their definedness masks."""

    carrier_drift: Tensor  # [E, 2], support then veto
    carrier_drift_map: Tensor  # [E, L, H, 2], response-carrier fraction
    carrier_drift_map_defined: Tensor  # [E, L, H, 2]
    carrier_drift_by_head: Tensor  # [E, H, 2]
    carrier_drift_defined: Tensor  # [E, 2]
    carrier_drift_head_defined: Tensor  # [E, H, 2]
    prompt_source_dispersion: Tensor  # [E, 2]
    prompt_source_dispersion_by_layer_head: Tensor  # [E, L, H, 2]
    prompt_source_dispersion_defined: Tensor  # [E, 2]
    prompt_source_dispersion_row_defined: Tensor  # [E, L, H, 2]
    response_born_takeover: Tensor  # [E, 2]
    response_born_takeover_by_layer_head: Tensor  # [E, L, H, 2]
    response_born_takeover_defined: Tensor  # [E, 2]
    response_born_takeover_row_defined: Tensor  # [E, L, H, 2]
    root_carrier_mass: Tensor  # [E, 3, 4, 2]
    injection_mass: Tensor  # [E, 4, 2]
    resolution: Tensor  # [E], numeric variation plus external error bound

    @property
    def direct_evidence(self) -> Tensor:
        """Evidence-rooted mass emitted by declared evidence tokens."""

        return self.root_carrier_mass[:, EVIDENCE_PROMPT, EVIDENCE]

    @property
    def grounded_response_relay(self) -> Tensor:
        """Evidence-rooted mass emitted by earlier response tokens."""

        return self.root_carrier_mass[:, RESPONSE_HISTORY, EVIDENCE]

    @property
    def response_born_history(self) -> Tensor:
        """Response-rooted mass emitted by earlier response tokens."""

        return self.root_carrier_mass[:, RESPONSE_HISTORY, RESPONSE]


@dataclass(frozen=True)
class SparseRoutes:
    """Deterministic per-row route prefix plus lossless score moments.

    One selected row is one physical edge.  Root contributions are columns of
    that edge, so message-norm coverage is never counted once per root.  The
    anonymous tail retains its known carrier and root; it is never reassigned
    to the numerical root.
    """

    row_event: Tensor  # [rows]
    row_layer: Tensor  # [rows]
    row_head: Tensor  # [rows]
    row_ptr: Tensor  # [rows + 1]
    source_position: Tensor  # [M]
    carrier: Tensor  # [M]
    attention: Tensor  # [M]
    value_energy: Tensor  # [M]
    physical_message_norm: Tensor  # [M]
    root_phi: Tensor  # [M, 4]
    tail_count: Tensor  # [rows, 3]
    tail_attention_sum: Tensor  # [rows, 3]
    tail_value_energy_sum: Tensor  # [rows, 3]
    tail_message_norm_sum: Tensor  # [rows, 3]
    tail_message_norm_max: Tensor  # [rows, 3]
    tail_root_positive: Tensor  # [rows, 3, 4]
    tail_root_negative: Tensor  # [rows, 3, 4]
    tail_physical_positive: Tensor  # [rows, 3]
    tail_physical_negative: Tensor  # [rows, 3]
    tail_physical_pos_xlogx: Tensor  # [rows, 3]
    tail_physical_neg_xlogx: Tensor  # [rows, 3]


def token_carriers(
    source_position: Tensor,
    response_start: int,
    evidence_mask: Tensor,
) -> Tensor:
    """Return physical source roles without changing semantic ancestry."""

    if source_position.ndim != 1 or evidence_mask.shape != source_position.shape:
        raise ValueError("source positions and evidence mask must be aligned vectors")
    prompt = source_position < response_start
    carrier = torch.full_like(source_position, RESPONSE_HISTORY, dtype=torch.long)
    carrier[prompt] = OTHER_PROMPT
    carrier[prompt & evidence_mask.bool()] = EVIDENCE_PROMPT
    return carrier


def measure_layer_routes(
    *,
    layer: int,
    attention: Tensor,
    root_values: Tensor,
    output_weight: Tensor,
    suffix_adjoint: Tensor,
    query_position: Tensor,
    source_position: Tensor,
    carrier: Tensor,
) -> LayerRoutes:
    """Measure all strict first-arrival AVWO messages in one layer.

    ``root_values[s, r, k]`` is the dynamic rooted value
    ``W_V[l,k] D[l,s] x_root[l,s,r]``.  ``output_weight`` is the native
    ``o_proj.weight`` with shape ``[hidden, query_heads * head_dim]``.
    The returned norm is the norm after the matching query-head block of
    ``W_O``; it is not an attention proxy.
    """

    if attention.ndim != 3 or root_values.ndim != 4:
        raise ValueError("attention and rooted values must have ranks 3 and 4")
    events, heads, sources = attention.shape
    if root_values.shape[0] != sources or root_values.shape[1] != len(ROOT_NAMES):
        raise ValueError("rooted values must be [source, 4, kv_head, head_dim]")
    kv_heads, head_dim = root_values.shape[2:]
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if output_weight.shape[1] != heads * head_dim:
        raise ValueError("W_O input width does not match query heads")
    hidden = output_weight.shape[0]
    if suffix_adjoint.shape != (events, hidden):
        raise ValueError("one hidden-space suffix adjoint is required per event")
    if query_position.shape != (events,) or source_position.shape != (sources,):
        raise ValueError("query and source positions do not align to routes")
    if carrier.shape != (sources,):
        raise ValueError("one carrier role is required per physical source")
    if ((carrier < 0) | (carrier >= len(CARRIER_NAMES))).any():
        raise ValueError("carrier contains an unknown physical source role")

    device = attention.device
    dtype = torch.float32
    query = query_position.to(device=device, dtype=torch.long)
    source = source_position.to(device=device, dtype=torch.long)
    source_carrier = carrier.to(device=device, dtype=torch.long)
    causal = source.unsqueeze(0) < query.unsqueeze(1)

    repeats = heads // kv_heads
    head_to_kv = torch.arange(heads, device=device) // repeats
    values = root_values.to(device=device, dtype=dtype)
    # [H, S, R, D_h]; contiguous query-head groups follow native GQA.
    head_values = values[:, :, head_to_kv].permute(2, 0, 1, 3)
    total_values = head_values.sum(dim=2)

    weight = output_weight.to(device=device, dtype=dtype)
    weight = weight.reshape(hidden, heads, head_dim).permute(1, 0, 2)
    adjoint = suffix_adjoint.to(device=device, dtype=dtype)
    head_adjoint = torch.einsum("ed,hdk->ehk", adjoint, weight)

    observed_attention = attention.to(dtype=dtype) * causal[:, None]
    root_phi = observed_attention[..., None] * torch.einsum(
        "ehd,hsrd->ehsr", head_adjoint, head_values
    )

    # ||W_O^h v|| is evaluated through its exact Gram form to avoid creating
    # an [event, head, source, hidden] tensor.
    output_gram = torch.einsum("hid,hie->hde", weight, weight)
    projected_square = torch.einsum(
        "hsd,hde,hse->hs", total_values, output_gram, total_values
    ).clamp_min(0)
    base_message_norm = projected_square.sqrt()
    physical_message_norm = observed_attention.abs() * base_message_norm.unsqueeze(0)
    value_energy = total_values.norm(dim=-1).unsqueeze(0).expand(events, -1, -1)
    value_energy = value_energy * causal[:, None]

    return LayerRoutes(
        layer=int(layer),
        query_position=query,
        source_position=source,
        carrier=source_carrier,
        causal=causal,
        attention=observed_attention,
        value_energy=value_energy,
        physical_message_norm=physical_message_norm,
        root_phi=root_phi,
    )


def moments_from_layers(layers: Sequence[LayerRoutes]) -> RouteMoments:
    """Accumulate dense edge atoms without losing endpoint-level signs."""

    phi, carrier, causal = stack_route_layers(layers)
    events, layer_count, heads, _, roots = phi.shape
    physical = phi[..., :NUMERIC].sum(dim=-1)
    physical_signed = torch.stack(
        (physical.clamp_min(0), (-physical).clamp_min(0)), dim=-1
    )
    root_signed = torch.stack((phi.clamp_min(0), (-phi).clamp_min(0)), dim=-1)
    shape = (events, layer_count, heads, len(CARRIER_NAMES), 2)
    physical_mass = torch.zeros(shape, dtype=phi.dtype, device=phi.device)
    root_mass = torch.zeros(*shape[:-1], roots, 2, dtype=phi.dtype, device=phi.device)
    physical_xlogx = torch.zeros_like(physical_mass)
    eligible_count = torch.zeros(shape[:-1], dtype=torch.int64, device=phi.device)

    for role in range(len(CARRIER_NAMES)):
        role_mask = causal & (carrier == role)[None]
        physical_role = physical_signed * role_mask[:, None, None, :, None]
        root_role = root_signed * role_mask[:, None, None, :, None, None]
        physical_mass[..., role, :] = physical_role.sum(dim=3)
        root_mass[..., role, :, :] = root_role.sum(dim=3)
        physical_xlogx[..., role, :] = xlogx(physical_role).sum(dim=3)
        count = role_mask.sum(dim=1)
        eligible_count[..., role] = count[:, None, None]

    return RouteMoments(
        physical_mass=physical_mass,
        root_mass=root_mass,
        physical_xlogx=physical_xlogx,
        eligible_source_count=eligible_count,
    )


def moments_from_sparse(routes: SparseRoutes) -> RouteMoments:
    """Recover the same moments from selected edges and anonymous tails."""

    rows = len(routes.row_event)
    if rows == 0:
        raise ValueError("sparse routes must contain at least one complete row grid")
    if routes.row_layer.shape != (rows,) or routes.row_head.shape != (rows,):
        raise ValueError("sparse route row coordinates must align")
    if routes.row_ptr.shape != (rows + 1,) or int(routes.row_ptr[0]) != 0:
        raise ValueError("sparse row_ptr must delimit every row")
    edge_count = len(routes.source_position)
    if int(routes.row_ptr[-1]) != edge_count or routes.root_phi.shape != (
        edge_count,
        len(ROOT_NAMES),
    ):
        raise ValueError("sparse selected edges and root columns must align")

    row_event = routes.row_event.long()
    row_layer = routes.row_layer.long()
    row_head = routes.row_head.long()
    if (row_event < 0).any() or (row_layer < 0).any() or (row_head < 0).any():
        raise ValueError("sparse row coordinates must be nonnegative")
    events = int(row_event.max().item()) + 1
    layer_count = int(row_layer.max().item()) + 1
    heads = int(row_head.max().item()) + 1
    flat_row = (row_event * layer_count + row_layer) * heads + row_head
    expected_rows = events * layer_count * heads
    if rows != expected_rows or not torch.equal(
        flat_row.sort().values,
        torch.arange(rows, device=flat_row.device),
    ):
        raise ValueError("sparse rows must cover each event, layer, and head once")

    carrier_count = len(CARRIER_NAMES)
    tail_shape = (rows, carrier_count)
    if routes.tail_count.shape != tail_shape:
        raise ValueError("sparse tail counts must retain every carrier")
    for value in (
        routes.tail_attention_sum,
        routes.tail_value_energy_sum,
        routes.tail_message_norm_sum,
        routes.tail_message_norm_max,
        routes.tail_physical_positive,
        routes.tail_physical_negative,
        routes.tail_physical_pos_xlogx,
        routes.tail_physical_neg_xlogx,
    ):
        if value.shape != tail_shape:
            raise ValueError("sparse physical tail moments must align")
    root_tail_shape = (*tail_shape, len(ROOT_NAMES))
    if (
        routes.tail_root_positive.shape != root_tail_shape
        or routes.tail_root_negative.shape != root_tail_shape
    ):
        raise ValueError("sparse root tail moments must retain carrier and root")

    dtype = routes.root_phi.dtype
    device = routes.root_phi.device
    physical_flat = torch.zeros(rows, carrier_count, 2, dtype=dtype, device=device)
    root_flat = torch.zeros(
        rows,
        carrier_count,
        len(ROOT_NAMES),
        2,
        dtype=dtype,
        device=device,
    )
    xlogx_flat = torch.zeros_like(physical_flat)
    count_flat = torch.zeros(rows, carrier_count, dtype=torch.int64, device=device)
    coordinate = flat_row.to(device=device)
    physical_flat[coordinate] = torch.stack(
        (
            routes.tail_physical_positive.to(device=device, dtype=dtype),
            routes.tail_physical_negative.to(device=device, dtype=dtype),
        ),
        dim=-1,
    )
    root_flat[coordinate] = torch.stack(
        (
            routes.tail_root_positive.to(device=device, dtype=dtype),
            routes.tail_root_negative.to(device=device, dtype=dtype),
        ),
        dim=-1,
    )
    xlogx_flat[coordinate] = torch.stack(
        (
            routes.tail_physical_pos_xlogx.to(device=device, dtype=dtype),
            routes.tail_physical_neg_xlogx.to(device=device, dtype=dtype),
        ),
        dim=-1,
    )
    count_flat[coordinate] = routes.tail_count.to(device=device, dtype=torch.int64)

    selected_carrier = routes.carrier.to(device=device, dtype=torch.long)
    selected_phi = routes.root_phi.to(device=device, dtype=dtype)
    if ((selected_carrier < 0) | (selected_carrier >= carrier_count)).any():
        raise ValueError("selected edge has an unknown carrier")
    for row in range(rows):
        start, stop = int(routes.row_ptr[row]), int(routes.row_ptr[row + 1])
        destination = int(flat_row[row])
        edge_roots = selected_phi[start:stop]
        edge_carriers = selected_carrier[start:stop]
        for role in range(carrier_count):
            selected = edge_carriers == role
            role_roots = edge_roots[selected]
            role_physical = role_roots[:, :NUMERIC].sum(dim=-1)
            role_signed = torch.stack(
                (role_physical.clamp_min(0), (-role_physical).clamp_min(0)),
                dim=-1,
            )
            physical_flat[destination, role] += role_signed.sum(dim=0)
            root_flat[destination, role] += torch.stack(
                (role_roots.clamp_min(0), (-role_roots).clamp_min(0)),
                dim=-1,
            ).sum(dim=0)
            xlogx_flat[destination, role] += xlogx(role_signed).sum(dim=0)
            count_flat[destination, role] += selected.sum()

    return RouteMoments(
        physical_mass=physical_flat.reshape(
            events, layer_count, heads, carrier_count, 2
        ),
        root_mass=root_flat.reshape(
            events,
            layer_count,
            heads,
            carrier_count,
            len(ROOT_NAMES),
            2,
        ),
        physical_xlogx=xlogx_flat.reshape(events, layer_count, heads, carrier_count, 2),
        eligible_source_count=count_flat.reshape(
            events, layer_count, heads, carrier_count
        ),
    )


def route_axes(
    layers: Sequence[LayerRoutes],
    injection_phi: Tensor,
    *,
    event_valid: Tensor | None = None,
    resolution: Tensor | float = 0.0,
    numeric_total_variation: Tensor | None = None,
) -> RouteAxes:
    """Compute the three axes from dense routes through shared moments."""

    return reduce_route_moments(
        moments_from_layers(layers),
        injection_phi,
        event_valid=event_valid,
        resolution=resolution,
        numeric_total_variation=numeric_total_variation,
    )


def route_axes_from_sparse(
    routes: SparseRoutes,
    injection_phi: Tensor,
    *,
    event_valid: Tensor | None = None,
    resolution: Tensor | float = 0.0,
    numeric_total_variation: Tensor | None = None,
) -> RouteAxes:
    """Compute the identical axes from selected edges plus exact tails."""

    return reduce_route_moments(
        moments_from_sparse(routes),
        injection_phi,
        event_valid=event_valid,
        resolution=resolution,
        numeric_total_variation=numeric_total_variation,
    )


def reduce_route_moments(
    moments: RouteMoments,
    injection_phi: Tensor,
    *,
    event_valid: Tensor | None = None,
    resolution: Tensor | float = 0.0,
    numeric_total_variation: Tensor | None = None,
) -> RouteAxes:
    """Reduce shared dense-or-sparse moments into the three mechanism axes.

    Support and veto are measured independently.  ``resolution`` is any
    additional outward error bound.  When supplied, ``numeric_total_variation``
    is the complete precomputed N variation; otherwise it is reconstructed
    from route moments and the net numeric injection.  Undefined quantities
    are returned as NaN together with explicit masks.  No direction is flipped
    and the three axes are not fused.
    """

    physical_mass = moments.physical_mass
    if physical_mass.ndim != 5:
        raise ValueError("physical moments must be [event, layer, head, carrier, sign]")
    events, layer_count, heads, carriers, signs = physical_mass.shape
    if (carriers, signs) != (len(CARRIER_NAMES), 2):
        raise ValueError("physical moments have incompatible carrier or sign axes")
    expected_root_shape = (
        events,
        layer_count,
        heads,
        carriers,
        len(ROOT_NAMES),
        signs,
    )
    if moments.root_mass.shape != expected_root_shape:
        raise ValueError("root moments must preserve event, layer, head, and carrier")
    if moments.physical_xlogx.shape != physical_mass.shape:
        raise ValueError("xlogx moments must align with physical masses")
    if moments.eligible_source_count.shape != physical_mass.shape[:-1]:
        raise ValueError("eligible counts must align with route rows and carriers")
    if injection_phi.shape != (events, len(ROOT_NAMES)):
        raise ValueError("injection must provide E/Q/R/N contribution per event")
    device = physical_mass.device
    dtype = physical_mass.dtype
    valid = (
        torch.ones(events, dtype=torch.bool, device=device)
        if event_valid is None
        else event_valid.to(device=device, dtype=torch.bool)
    )
    if valid.shape != (events,):
        raise ValueError("event_valid must contain one flag per event")
    threshold = torch.as_tensor(resolution, dtype=dtype, device=device)
    if threshold.ndim == 0:
        threshold = threshold.expand(events)
    if (
        threshold.shape != (events,)
        or not torch.isfinite(threshold).all()
        or (threshold < 0).any()
    ):
        raise ValueError("resolution must be nonnegative and scalar or [event]")
    finite = torch.isfinite(physical_mass).flatten(1).all(dim=1)
    finite &= torch.isfinite(moments.root_mass).flatten(1).all(dim=1)
    finite &= torch.isfinite(moments.physical_xlogx).flatten(1).all(dim=1)
    finite &= torch.isfinite(injection_phi).flatten(1).all(dim=1).to(device)
    valid &= finite
    edge_numeric_variation = moments.root_mass[..., NUMERIC, :].sum(dim=(1, 2, 3, 4))
    observed_numeric_lower_bound = (
        edge_numeric_variation
        + injection_phi.to(device=device, dtype=dtype)[:, NUMERIC].abs()
    )
    if numeric_total_variation is None:
        numeric_variation = observed_numeric_lower_bound
    else:
        numeric_variation = numeric_total_variation.to(device=device, dtype=dtype)
        if (
            numeric_variation.shape != (events,)
            or not torch.isfinite(numeric_variation).all()
            or (numeric_variation < 0).any()
        ):
            raise ValueError(
                "numeric_total_variation must be a nonnegative finite [event] vector"
            )
        lower_bound_tolerance = 1e-6 + 1e-5 * observed_numeric_lower_bound
        if (
            numeric_variation < observed_numeric_lower_bound - lower_bound_tolerance
        ).any():
            raise ValueError(
                "numeric_total_variation is below observed edge and injection N"
            )
    threshold = threshold + numeric_variation

    layer_depth = (
        torch.linspace(-1, 1, layer_count, device=device, dtype=dtype)
        if layer_count > 1
        else torch.zeros(1, device=device, dtype=dtype)
    )
    prompt_mass = physical_mass[..., :RESPONSE_HISTORY, :].sum(dim=3)
    response_mass = physical_mass[..., RESPONSE_HISTORY, :]
    carrier_row_mass = prompt_mass + response_mass
    carrier_map_defined = (carrier_row_mass > threshold[:, None, None, None]) & valid[
        :, None, None, None
    ]
    carrier_map = divide_or_nan(
        response_mass,
        carrier_row_mass,
        carrier_map_defined,
    )

    prompt_head_total = prompt_mass.sum(dim=1)
    response_head_total = response_mass.sum(dim=1)
    head_defined = (
        (prompt_head_total > threshold[:, None, None])
        & (response_head_total > threshold[:, None, None])
        & valid[:, None, None]
        & (layer_count > 1)
    )
    prompt_head_depth = divide_or_nan(
        (prompt_mass * layer_depth[None, :, None, None]).sum(dim=1),
        prompt_head_total,
        head_defined,
    )
    response_head_depth = divide_or_nan(
        (response_mass * layer_depth[None, :, None, None]).sum(dim=1),
        response_head_total,
        head_defined,
    )
    drift_by_head = torch.where(
        head_defined,
        (response_head_depth - prompt_head_depth) / 2,
        torch.nan,
    )

    prompt_total = prompt_head_total.sum(dim=1)
    response_total = response_head_total.sum(dim=1)
    drift_defined = (
        (prompt_total > threshold[:, None])
        & (response_total > threshold[:, None])
        & valid[:, None]
        & (layer_count > 1)
    )
    prompt_depth = divide_or_nan(
        (prompt_mass * layer_depth[None, :, None, None]).sum(dim=(1, 2)),
        prompt_total,
        drift_defined,
    )
    response_depth = divide_or_nan(
        (response_mass * layer_depth[None, :, None, None]).sum(dim=(1, 2)),
        response_total,
        drift_defined,
    )
    carrier_drift = torch.where(
        drift_defined, (response_depth - prompt_depth) / 2, torch.nan
    )

    row_mass = prompt_mass
    prompt_xlogx = moments.physical_xlogx[..., :RESPONSE_HISTORY, :].sum(dim=3)
    eligible_prompt = moments.eligible_source_count[..., :RESPONSE_HISTORY].sum(dim=3)
    row_defined = (
        (row_mass > threshold[:, None, None, None])
        & (eligible_prompt[..., None] >= 2)
        & valid[:, None, None, None]
    )
    safe_mass = row_mass.clamp_min(torch.finfo(dtype).tiny)
    entropy = safe_mass.log() - prompt_xlogx / safe_mass
    normalizer = eligible_prompt.clamp_min(2).to(dtype).log()
    entropy = entropy / normalizer[..., None]
    entropy = torch.where(row_defined, entropy, torch.nan)

    dispersion_mass = torch.where(row_defined, row_mass, 0)
    dispersion_denominator = dispersion_mass.sum(dim=(1, 2))
    dispersion_defined = (dispersion_denominator > threshold[:, None]) & valid[:, None]
    dispersion = divide_or_nan(
        (torch.nan_to_num(entropy) * dispersion_mass).sum(dim=(1, 2)),
        dispersion_denominator,
        dispersion_defined,
    )

    response_root_mass_by_row = moments.root_mass[..., RESPONSE_HISTORY, :, :]
    response_row_denominator = response_root_mass_by_row[..., :NUMERIC, :].sum(dim=3)
    takeover_row_defined = (
        response_row_denominator > threshold[:, None, None, None]
    ) & valid[:, None, None, None]
    takeover_by_row = divide_or_nan(
        response_root_mass_by_row[..., RESPONSE, :],
        response_row_denominator,
        takeover_row_defined,
    )
    valid_response_root_mass = torch.where(
        takeover_row_defined[..., None, :],
        response_root_mass_by_row,
        0,
    )
    response_root_mass = valid_response_root_mass.sum(dim=(1, 2))
    takeover_denominator = response_root_mass[:, :NUMERIC].sum(dim=1)
    takeover_defined = (takeover_denominator > threshold[:, None]) & valid[:, None]
    takeover = divide_or_nan(
        response_root_mass[:, RESPONSE],
        takeover_denominator,
        takeover_defined,
    )

    root_carrier_mass = moments.root_mass.sum(dim=(1, 2))

    injection = injection_phi.to(device=device, dtype=dtype)
    injection_mass = torch.stack(
        (injection.clamp_min(0), (-injection).clamp_min(0)), dim=-1
    )
    return RouteAxes(
        carrier_drift=carrier_drift,
        carrier_drift_map=carrier_map,
        carrier_drift_map_defined=carrier_map_defined,
        carrier_drift_by_head=drift_by_head,
        carrier_drift_defined=drift_defined,
        carrier_drift_head_defined=head_defined,
        prompt_source_dispersion=dispersion,
        prompt_source_dispersion_by_layer_head=entropy,
        prompt_source_dispersion_defined=dispersion_defined,
        prompt_source_dispersion_row_defined=row_defined,
        response_born_takeover=takeover,
        response_born_takeover_by_layer_head=takeover_by_row,
        response_born_takeover_defined=takeover_defined,
        response_born_takeover_row_defined=takeover_row_defined,
        root_carrier_mass=root_carrier_mass,
        injection_mass=injection_mass,
        resolution=threshold,
    )


def sparsify_routes(
    layers: Sequence[LayerRoutes],
    top_k: int = 64,
    cover_mass: float = 0.95,
) -> SparseRoutes:
    """Keep the strongest true messages in each ``(event, layer, head)`` row.

    Ranking is target-readout independent: descending physical AVWO norm with
    ascending source position as the deterministic tie break.  Each row keeps
    the smallest prefix reaching ``cover_mass``, capped by ``top_k``.  Tail
    moments are sufficient to reproduce all three axes exactly.
    """

    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not 0 < cover_mass <= 1:
        raise ValueError("cover_mass must be in (0, 1]")
    phi, carriers, causal = stack_route_layers(layers)
    norms = torch.stack([route.physical_message_norm for route in layers], dim=1)
    attention = torch.stack([route.attention for route in layers], dim=1)
    energy = torch.stack([route.value_energy for route in layers], dim=1)
    events, layer_count, heads, _, _ = phi.shape
    device, dtype = phi.device, phi.dtype
    source = layers[0].source_position

    row_event: list[int] = []
    row_layer: list[int] = []
    row_head: list[int] = []
    row_ptr = [0]
    selected_source: list[Tensor] = []
    selected_carrier: list[Tensor] = []
    selected_attention: list[Tensor] = []
    selected_energy: list[Tensor] = []
    selected_norm: list[Tensor] = []
    selected_phi: list[Tensor] = []

    row_count = events * layer_count * heads
    tail_count = torch.zeros(
        row_count, len(CARRIER_NAMES), dtype=torch.int32, device=device
    )
    tail_attention = torch.zeros(
        row_count, len(CARRIER_NAMES), dtype=dtype, device=device
    )
    tail_energy = torch.zeros_like(tail_attention)
    tail_norm = torch.zeros_like(tail_attention)
    tail_norm_max = torch.zeros_like(tail_attention)
    tail_root_positive = torch.zeros(
        row_count,
        len(CARRIER_NAMES),
        len(ROOT_NAMES),
        dtype=dtype,
        device=device,
    )
    tail_root_negative = torch.zeros_like(tail_root_positive)
    tail_physical_positive = torch.zeros_like(tail_attention)
    tail_physical_negative = torch.zeros_like(tail_attention)
    tail_pos_xlogx = torch.zeros_like(tail_attention)
    tail_neg_xlogx = torch.zeros_like(tail_attention)

    row = 0
    for event in range(events):
        eligible = torch.nonzero(causal[event], as_tuple=True)[0]
        # Stable two-pass sorting implements (-message_norm, source_position).
        source_order = torch.argsort(source[eligible], stable=True)
        source_sorted = eligible[source_order]
        for layer in range(layer_count):
            for head in range(heads):
                row_event.append(event)
                row_layer.append(layers[layer].layer)
                row_head.append(head)
                magnitude = norms[event, layer, head, source_sorted]
                magnitude_order = torch.argsort(magnitude, descending=True, stable=True)
                ranked_source = source_sorted[magnitude_order]
                ranked_magnitude = magnitude[magnitude_order]
                positive_count = int((ranked_magnitude > 0).sum().item())
                if positive_count:
                    cumulative = ranked_magnitude[:positive_count].cumsum(dim=0)
                    required = (
                        int((cumulative < cover_mass * cumulative[-1]).sum().item()) + 1
                    )
                    retained = min(required, top_k)
                else:
                    retained = 0
                keep = ranked_source[:retained]
                omitted = torch.ones_like(causal[event])
                omitted[keep] = False
                omitted &= causal[event]

                selected_source.append(source[keep])
                selected_carrier.append(carriers[keep])
                selected_attention.append(attention[event, layer, head, keep])
                selected_energy.append(energy[event, layer, head, keep])
                selected_norm.append(norms[event, layer, head, keep])
                selected_phi.append(phi[event, layer, head, keep])
                row_ptr.append(row_ptr[-1] + len(keep))

                for role in range(len(CARRIER_NAMES)):
                    tail = omitted & (carriers == role)
                    tail_count[row, role] = tail.sum()
                    tail_attention[row, role] = attention[
                        event, layer, head, tail
                    ].sum()
                    tail_energy[row, role] = energy[event, layer, head, tail].sum()
                    tail_norm[row, role] = norms[event, layer, head, tail].sum()
                    if tail.any():
                        tail_norm_max[row, role] = norms[event, layer, head, tail].max()
                    root_tail = phi[event, layer, head, tail]
                    tail_root_positive[row, role] = root_tail.clamp_min(0).sum(dim=0)
                    tail_root_negative[row, role] = (-root_tail).clamp_min(0).sum(dim=0)
                    physical_tail = root_tail[:, :NUMERIC].sum(dim=-1)
                    positive = physical_tail.clamp_min(0)
                    negative = (-physical_tail).clamp_min(0)
                    tail_physical_positive[row, role] = positive.sum()
                    tail_physical_negative[row, role] = negative.sum()
                    tail_pos_xlogx[row, role] = xlogx(positive).sum()
                    tail_neg_xlogx[row, role] = xlogx(negative).sum()
                row += 1

    def concatenate(
        values: list[Tensor], shape: tuple[int, ...], *, kind: torch.dtype
    ) -> Tensor:
        return (
            torch.cat(values)
            if values
            else torch.empty(shape, dtype=kind, device=device)
        )

    return SparseRoutes(
        row_event=torch.tensor(row_event, dtype=torch.int32, device=device),
        row_layer=torch.tensor(row_layer, dtype=torch.int16, device=device),
        row_head=torch.tensor(row_head, dtype=torch.int16, device=device),
        row_ptr=torch.tensor(row_ptr, dtype=torch.int64, device=device),
        source_position=concatenate(selected_source, (0,), kind=source.dtype),
        carrier=concatenate(selected_carrier, (0,), kind=carriers.dtype),
        attention=concatenate(selected_attention, (0,), kind=dtype),
        value_energy=concatenate(selected_energy, (0,), kind=dtype),
        physical_message_norm=concatenate(selected_norm, (0,), kind=dtype),
        root_phi=concatenate(selected_phi, (0, len(ROOT_NAMES)), kind=dtype),
        tail_count=tail_count,
        tail_attention_sum=tail_attention,
        tail_value_energy_sum=tail_energy,
        tail_message_norm_sum=tail_norm,
        tail_message_norm_max=tail_norm_max,
        tail_root_positive=tail_root_positive,
        tail_root_negative=tail_root_negative,
        tail_physical_positive=tail_physical_positive,
        tail_physical_negative=tail_physical_negative,
        tail_physical_pos_xlogx=tail_pos_xlogx,
        tail_physical_neg_xlogx=tail_neg_xlogx,
    )


def stack_route_layers(
    layers: Sequence[LayerRoutes],
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack aligned layer routes as ``[event, layer, head, source, root]``."""

    if not layers:
        raise ValueError("at least one layer of routes is required")
    if tuple(route.layer for route in layers) != tuple(range(len(layers))):
        raise ValueError("route layers must be complete and ordered from zero")
    reference = layers[0]
    for route in layers[1:]:
        if (
            route.query_position.shape != reference.query_position.shape
            or route.attention.shape != reference.attention.shape
            or not torch.equal(route.query_position, reference.query_position)
            or not torch.equal(route.source_position, reference.source_position)
            or not torch.equal(route.carrier, reference.carrier)
            or not torch.equal(route.causal, reference.causal)
        ):
            raise ValueError(
                "route layers must have aligned events, heads, and sources"
            )
    return (
        torch.stack([route.root_phi for route in layers], dim=1),
        reference.carrier,
        reference.causal,
    )


def divide_or_nan(numerator: Tensor, denominator: Tensor, defined: Tensor) -> Tensor:
    """Divide only where a mechanism is scientifically defined."""

    safe = torch.where(defined, denominator, 1)
    value = numerator / safe
    return torch.where(defined, value, torch.nan)


def xlogx(value: Tensor) -> Tensor:
    """Return ``x log x`` with the continuous value zero at ``x=0``."""

    return torch.where(
        value > 0, value * value.clamp_min(torch.finfo(value.dtype).tiny).log(), 0
    )
