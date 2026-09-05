"""Connected ETCC extraction and exact cut/patch/block confirmation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from experiments.common.llama_message_intervention import (
    ForwardCache,
    MessageGate,
    first_changed_layer,
    forward_layers,
    gate_to,
)

from .attribution import contrast_direction
from .flow import FlowEdges, PairedFlow
from .throughput import FlowThroughput
from .worlds import PairedWorld, TargetContrast


@dataclass(frozen=True)
class CorridorEffect:
    edge_count: int
    pair_effect: float
    necessity: float
    sufficiency: float
    blocked_sufficiency: float
    mediated_sufficiency: float
    clean_restoration_error: float
    corrupt_restoration_error: float
    restoration_error: float
    restoration_tolerance: float
    restoration_valid: bool


@dataclass(frozen=True)
class CarrierEffect:
    layer: int
    position: int
    route_throughput: float
    state_delta_norm: float
    target_score: float
    necessity: float
    rescue: float
    block_effect: float
    blocked_rescue: float
    mediated_rescue: float
    block_tolerance: float
    confirmed: bool


@dataclass(frozen=True)
class RootEffect:
    """Target-specific routing screen and exact paired root intervention."""

    unit_id: int
    route_mass: float
    gradient_score: float
    necessity: float
    sufficiency: float
    causal_score: float
    evaluated: bool


def replace_edges_gate(edges: FlowEdges, replacement: Tensor) -> MessageGate:
    """Delete named native edges and add their paired head codes in place."""

    sparse: dict[int, list[Tensor]] = {}
    patches: dict[int, dict[int, dict[int, Tensor]]] = {}
    for index in range(edges.count):
        layer = int(edges.layer[index])
        head = int(edges.head[index])
        query = int(edges.target[index])
        source = int(edges.source[index])
        sparse.setdefault(layer, []).append(
            torch.tensor([head, query, source], dtype=torch.long)
        )
        by_head = patches.setdefault(layer, {}).setdefault(query, {})
        by_head[head] = by_head.get(
            head, torch.zeros_like(replacement[index].float())
        ) + replacement[index].float()
    return MessageGate(
        split_layer=0,
        sparse_layer_edges={
            layer: torch.stack(rows) for layer, rows in sparse.items()
        },
        head_output_patch=patches,
    )


def replace_root_gate(
    replacement: ForwardCache,
    positions: Tensor,
) -> MessageGate:
    """Replace candidate token embeddings at the input to decoder layer zero."""

    return MessageGate(
        split_layer=0,
        residual_replace={
            0: {
                int(position): replacement.layer_input[0][position].float()
                for position in positions.tolist()
            }
        },
    )


def rerun_margin(
    model,
    cache: ForwardCache,
    gate: MessageGate,
    target: TargetContrast,
) -> float:
    """Rerun the changed suffix and read one fixed positive-negative margin."""

    start = first_changed_layer(gate, cache.layer_count)
    direction, bias = contrast_direction(model, target)
    if start is None:
        state = cache.final_hidden[target.query_position].to(direction.device)
        return float(torch.dot(state.float(), direction) + bias)
    device = model.get_input_embeddings().weight.device
    checkpoint = start if start in cache.layer_input else 0
    with torch.inference_mode():
        final = forward_layers(
            model,
            cache.layer_input[checkpoint].to(device)[None],
            checkpoint,
            gate=gate_to(gate, device),
            attention_query_chunk=cache.attention_query_chunk,
        )[0]
    return float(torch.dot(final[target.query_position].float(), direction) + bias)


def confirm_corridor(
    model,
    flow: PairedFlow,
    corridor: FlowEdges,
) -> CorridorEffect:
    """Measure corridor necessity, rescue, final-path blocking, and restoration."""

    target = flow.target
    restoration = rerun_margin(
        model,
        flow.clean_cache,
        replace_edges_gate(corridor, corridor.clean_code),
        target,
    )
    corrupt_restoration = rerun_margin(
        model,
        flow.corrupt_cache,
        replace_edges_gate(corridor, corridor.corrupt_code),
        target,
    )
    clean_with_corrupt = rerun_margin(
        model,
        flow.clean_cache,
        replace_edges_gate(corridor, corridor.corrupt_code),
        target,
    )
    corrupt_with_clean = rerun_margin(
        model,
        flow.corrupt_cache,
        replace_edges_gate(corridor, corridor.clean_code),
        target,
    )
    terminal = corridor.target == target.query_position
    blocked_code = corridor.clean_code.clone()
    blocked_code[terminal] = corridor.corrupt_code[terminal]
    blocked = rerun_margin(
        model,
        flow.corrupt_cache,
        replace_edges_gate(corridor, blocked_code),
        target,
    )
    necessity = flow.clean_margin - clean_with_corrupt
    sufficiency = corrupt_with_clean - flow.corrupt_margin
    blocked_sufficiency = blocked - flow.corrupt_margin
    clean_restoration_error = abs(restoration - flow.clean_margin)
    corrupt_restoration_error = abs(
        corrupt_restoration - flow.corrupt_margin
    )
    restoration_error = max(
        clean_restoration_error,
        corrupt_restoration_error,
    )
    tolerance = {
        torch.float32: 1e-5,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }.get(model.dtype, 2e-2)
    return CorridorEffect(
        corridor.count,
        flow.pair_effect,
        necessity,
        sufficiency,
        blocked_sufficiency,
        sufficiency - blocked_sufficiency,
        clean_restoration_error,
        corrupt_restoration_error,
        restoration_error,
        tolerance,
        restoration_error <= tolerance,
    )


def confirm_roots(
    model,
    flow: PairedFlow,
    world: PairedWorld,
    throughput: FlowThroughput,
    *,
    limit: int,
) -> tuple[RootEffect, ...]:
    """Screen candidate units, then test exact clean/corrupt embedding patches."""

    candidates = list(world.candidate_unit_id)
    candidates.sort(
        key=lambda unit_id: float(throughput.unit_mass[unit_id]),
        reverse=True,
    )
    evaluated = set(candidates if limit == 0 else candidates[:limit])
    direction = 1.0 if flow.pair_effect >= 0 else -1.0
    gradient_by_position: dict[int, float] = {}
    if flow.stages is not None:
        gradient_by_position = {
            int(position): float(flow.stages.state_score[0, index])
            for index, position in enumerate(flow.stages.position)
        }

    effects = []
    for unit_id in candidates:
        positions = world.units.positions((unit_id,))
        gradient_score = (
            sum(
                gradient_by_position.get(int(position), 0.0)
                for position in positions
            )
            if flow.stages is not None
            else float("nan")
        )
        if unit_id not in evaluated:
            effects.append(
                RootEffect(
                    unit_id,
                    float(throughput.unit_mass[unit_id]),
                    gradient_score,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                )
            )
            continue
        clean_with_corrupt = rerun_margin(
            model,
            flow.clean_cache,
            replace_root_gate(flow.corrupt_cache, positions),
            flow.target,
        )
        corrupt_with_clean = rerun_margin(
            model,
            flow.corrupt_cache,
            replace_root_gate(flow.clean_cache, positions),
            flow.target,
        )
        necessity = flow.clean_margin - clean_with_corrupt
        sufficiency = corrupt_with_clean - flow.corrupt_margin
        causal_score = min(direction * necessity, direction * sufficiency)
        effects.append(
            RootEffect(
                unit_id,
                float(throughput.unit_mass[unit_id]),
                gradient_score,
                necessity,
                sufficiency,
                causal_score,
                True,
            )
        )
    return tuple(effects)


def select_root(effects: tuple[RootEffect, ...]) -> int:
    """Choose a causally bidirectional root, falling back to route mass."""

    if not effects:
        raise ValueError("root selection requires candidate effects")
    routed = [effect for effect in effects if effect.route_mass > 0]
    confirmed = [
        effect for effect in routed if effect.evaluated and effect.causal_score > 0
    ]
    pool = confirmed or [effect for effect in routed if effect.evaluated]
    if not pool:
        pool = [effect for effect in effects if effect.evaluated] or list(effects)
    return max(
        pool,
        key=lambda effect: (
            effect.causal_score
            if math.isfinite(effect.causal_score)
            else -math.inf,
            effect.route_mass,
        ),
    ).unit_id


def carrier_gate(
    patch: ForwardCache,
    reset: ForwardCache,
    layer: int,
    position: int,
    *,
    block: bool,
    heads: int,
) -> MessageGate:
    """Patch one clean carrier state, optionally preventing every downstream relay."""

    replacements = {layer: {position: patch.layer_input[layer][position].float()}}
    sparse: dict[int, Tensor] | None = None
    if block:
        source_count = patch.layer_input[0].shape[0]
        sparse = {}
        query = torch.arange(position, source_count, dtype=torch.long)
        head = torch.arange(heads, dtype=torch.long)
        for current in range(layer, patch.layer_count):
            grid_head, grid_query = torch.meshgrid(head, query, indexing="ij")
            sparse[current] = torch.stack(
                (
                    grid_head.flatten(),
                    grid_query.flatten(),
                    torch.full(
                        (heads * len(query),), position, dtype=torch.long
                    ),
                ),
                dim=1,
            )
            if current > layer:
                replacements[current] = {
                    position: reset.layer_input[current][position].float()
                }
    return MessageGate(
        split_layer=0,
        sparse_layer_edges=sparse,
        residual_replace=replacements,
    )


def confirm_carriers(
    model,
    flow: PairedFlow,
    throughput: FlowThroughput,
    excluded_position: Tensor,
    *,
    limit: int = 3,
) -> tuple[CarrierEffect, ...]:
    """Confirm routed carrier nodes with state patch and downstream Value cut."""

    if limit == 0:
        return ()
    direction = 1.0 if flow.pair_effect >= 0 else -1.0
    excluded = {int(position) for position in excluded_position.tolist()}
    stage_slot = (
        {}
        if flow.stages is None
        else {
            int(position): slot
            for slot, position in enumerate(flow.stages.position.tolist())
        }
    )
    candidates = []
    for layer in range(flow.clean_cache.layer_count):
        for position in flow.row_position.tolist():
            position = int(position)
            if position >= flow.target.query_position or position in excluded:
                continue
            route = float(throughput.node[layer, position])
            if not math.isfinite(route) or route <= 0:
                continue
            delta = (
                flow.clean_cache.layer_input[layer][position].float()
                - flow.corrupt_cache.layer_input[layer][position].float()
            )
            delta_norm = float(delta.norm())
            if flow.stages is None:
                target_score = float("nan")
                rank_score = route
            else:
                slot = stage_slot.get(position)
                if slot is None:
                    raise ValueError("routed carrier lacks a stage-gradient position")
                target_score = float(flow.stages.state_score[layer, slot])
                aligned_score = direction * target_score
                if not math.isfinite(aligned_score) or aligned_score <= 0:
                    continue
                rank_score = route * aligned_score
            candidates.append(
                (rank_score, route, delta_norm, target_score, layer, position)
            )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    heads = int(model.config.num_attention_heads)
    block_tolerance = {
        torch.float32: 1e-5,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }.get(model.dtype, 2e-2)
    effects = []
    for _, route, delta_norm, target_score, current_layer, position in candidates[
        :limit
    ]:
        removed = rerun_margin(
            model,
            flow.clean_cache,
            carrier_gate(
                flow.corrupt_cache,
                flow.clean_cache,
                current_layer,
                position,
                block=False,
                heads=heads,
            ),
            flow.target,
        )
        patched = rerun_margin(
            model,
            flow.corrupt_cache,
            carrier_gate(
                flow.clean_cache,
                flow.corrupt_cache,
                current_layer,
                position,
                block=False,
                heads=heads,
            ),
            flow.target,
        )
        block_control = rerun_margin(
            model,
            flow.corrupt_cache,
            carrier_gate(
                flow.corrupt_cache,
                flow.corrupt_cache,
                current_layer,
                position,
                block=True,
                heads=heads,
            ),
            flow.target,
        )
        blocked = rerun_margin(
            model,
            flow.corrupt_cache,
            carrier_gate(
                flow.clean_cache,
                flow.corrupt_cache,
                current_layer,
                position,
                block=True,
                heads=heads,
            ),
            flow.target,
        )
        rescue = patched - flow.corrupt_margin
        block_effect = block_control - flow.corrupt_margin
        blocked_rescue = blocked - block_control
        necessity = flow.clean_margin - removed
        effects.append(
            CarrierEffect(
                current_layer,
                position,
                route,
                delta_norm,
                target_score,
                necessity,
                rescue,
                block_effect,
                blocked_rescue,
                rescue - blocked_rescue,
                block_tolerance,
                bool(
                    direction * necessity > 0
                    and direction * rescue > 0
                    and direction * (rescue - blocked_rescue) > 0
                    and direction * blocked_rescue <= block_tolerance
                ),
            )
        )
    return tuple(effects)
