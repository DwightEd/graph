"""One ETCC audit: paired flow, connected corridor, and causal confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import save_result
from .corridor import (
    CarrierEffect,
    CorridorEffect,
    RootEffect,
    confirm_carriers,
    confirm_corridor,
    confirm_roots,
    select_root,
)
from .flow import FlowEdges, FlowSignal, PairedFlow, capture_paired_flow
from .throughput import FlowThroughput, compute_throughput
from .worlds import PAIR_SCHEMA, PairedWorld, TargetContrast

ETCC_SCHEMA = 1


@dataclass(frozen=True)
class TargetAudit:
    world: PairedWorld
    flow: PairedFlow
    throughput: FlowThroughput
    corridor: FlowEdges
    effect: CorridorEffect
    corridor_confirmed: bool
    roots: tuple[RootEffect, ...]
    screen_pair_effect: float
    selected_root_unit_id: int
    selected_root_effect: RootEffect
    selected_root_confirmed: bool
    carriers: tuple[CarrierEffect, ...]


def audit_target(
    model,
    world: PairedWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    *,
    carrier_scope: str,
    coverage: float,
    gradient_steps: int,
    query_chunk: int,
    root_screen_limit: int,
    carrier_limit: int,
    materialize_messages: bool,
) -> TargetAudit:
    if root_screen_limit < 0 or carrier_limit < 0:
        raise ValueError("root and carrier limits must be non-negative")
    multiple_candidates = len(world.candidate_unit_id) > 1
    screen_flow = capture_paired_flow(
        model,
        world,
        target,
        signal,
        carrier_scope=carrier_scope,
        coverage=coverage,
        gradient_steps=gradient_steps,
        query_chunk=query_chunk,
        materialize_messages=(
            materialize_messages if not multiple_candidates else False
        ),
    )
    screen_prefix = world.prefix(target)
    candidate_throughput = compute_throughput(
        screen_flow,
        screen_prefix.units.token_unit_id,
        screen_prefix.units.count,
        screen_prefix.candidate_unit_id,
    )
    roots = confirm_roots(
        model,
        screen_flow,
        screen_prefix,
        candidate_throughput,
        limit=root_screen_limit,
    )
    selected_root = select_root(roots)
    screen_pair_effect = screen_flow.pair_effect
    if multiple_candidates:
        isolated_world = world.isolate(selected_root)
        del screen_flow, candidate_throughput
        flow = capture_paired_flow(
            model,
            isolated_world,
            target,
            signal,
            carrier_scope=carrier_scope,
            coverage=coverage,
            gradient_steps=gradient_steps,
            query_chunk=query_chunk,
            materialize_messages=materialize_messages,
        )
        prefix = isolated_world.prefix(target)
        throughput = compute_throughput(
            flow,
            prefix.units.token_unit_id,
            prefix.units.count,
            (selected_root,),
        )
        selected_effect = confirm_roots(
            model,
            flow,
            prefix,
            throughput,
            limit=0,
        )[0]
    else:
        isolated_world = world
        flow = screen_flow
        prefix = screen_prefix
        throughput = candidate_throughput
        selected_effect = roots[0]
    selected_root_confirmed = bool(
        selected_effect.evaluated
        and throughput.root_mass > 0
        and selected_effect.causal_score > 0
    )
    corridor = flow.edges.select(throughput.edge > 0)
    effect = confirm_corridor(model, flow, corridor)
    direction = 1.0 if flow.pair_effect >= 0 else -1.0
    corridor_confirmed = bool(
        selected_root_confirmed
        and effect.restoration_valid
        and direction * effect.necessity > 0
        and direction * effect.sufficiency > 0
        and direction * effect.mediated_sufficiency > 0
    )
    carriers = confirm_carriers(
        model,
        flow,
        throughput,
        prefix.units.positions(prefix.candidate_unit_id),
        limit=carrier_limit,
    )
    return TargetAudit(
        isolated_world,
        flow,
        throughput,
        corridor,
        effect,
        corridor_confirmed,
        roots,
        screen_pair_effect,
        selected_root,
        selected_effect,
        selected_root_confirmed,
        carriers,
    )


def save_audit(
    path: str | Path,
    world: PairedWorld,
    audit: TargetAudit,
    *,
    model_id: str,
    model_dtype: str,
    coverage: float,
    gradient_steps: int,
    carrier_scope: str,
    query_chunk: int,
    root_screen_limit: int,
    carrier_limit: int,
    materialize_messages: bool,
) -> None:
    """Save canonical coordinates, edge codes, stage ledger, and rerun effects."""

    flow = audit.flow
    isolated_world = audit.world
    edges = flow.edges
    effect = audit.effect
    carriers = audit.carriers
    arrays: dict[str, object] = {
        "etcc_schema": ETCC_SCHEMA,
        "pair_schema": PAIR_SCHEMA,
        "method": "ETCC",
        "sample_id": world.sample_id,
        "tokenizer_id": world.tokenizer_id,
        "corruption": isolated_world.corruption,
        "screen_corruption": world.corruption,
        "model_id": model_id,
        "model_dtype": model_dtype,
        "layer_count": flow.clean_cache.layer_count,
        "head_count": flow.row_total.shape[1],
        "head_dim": edges.clean_code.shape[1],
        "hidden_size": flow.clean_cache.final_hidden.shape[1],
        "flow_signal": flow.signal.value,
        "edge_score_semantics": (
            "raw_clean_attention"
            if flow.signal is FlowSignal.ATTENTION
            else "signed_path_gradient_of_clean_minus_corrupt_AV"
        ),
        "edge_payload_semantics": "pre_WO_clean_and_corrupt_AV_code",
        "edge_coverage": coverage,
        "gradient_steps": gradient_steps if flow.signal is FlowSignal.MESSAGE else 0,
        "carrier_scope": carrier_scope,
        "query_chunk": query_chunk,
        "root_screen_limit": root_screen_limit,
        "carrier_limit": carrier_limit,
        "message_vector_materialized": materialize_messages,
        "response_start": world.response_start,
        "query_position": flow.target.query_position,
        "prediction_position": flow.target.query_position + 1,
        "positive_token_id": flow.target.positive_token_id,
        "negative_token_id": flow.target.negative_token_id,
        "contrast_origin": flow.target.origin,
        "causal_source_count": flow.target.query_position + 1,
        "clean_token_ids": isolated_world.clean_token_ids,
        "corrupt_token_ids": isolated_world.corrupt_token_ids,
        "screen_corrupt_token_ids": world.corrupt_token_ids,
        "token_unit_id": world.units.token_unit_id,
        "unit_name": np.asarray(world.units.name),
        "unit_kind": np.asarray(world.units.kind),
        "candidate_unit_id": np.asarray(
            world.candidate_unit_id, dtype=np.int32
        ),
        "selected_root_unit_id": audit.selected_root_unit_id,
        "selected_root_confirmed": audit.selected_root_confirmed,
        "selected_root_necessity": audit.selected_root_effect.necessity,
        "selected_root_sufficiency": audit.selected_root_effect.sufficiency,
        "selected_root_causal_score": audit.selected_root_effect.causal_score,
        "source_unit_route_mass": audit.throughput.unit_mass,
        "row_position": flow.row_position,
        "row_total": flow.row_total,
        "row_retained": flow.row_retained,
        "row_message_budget": flow.aggregation.message_budget,
        "row_net_message_norm": flow.aggregation.net_message_norm,
        "row_message_coherence": flow.aggregation.coherence,
        "row_signed_target_score": flow.aggregation.signed_score,
        "row_positive_target_score": flow.aggregation.positive_score,
        "row_negative_target_score": flow.aggregation.negative_score,
        "row_selector_score": flow.aggregation.selector_score,
        "row_content_score": flow.aggregation.content_score,
        "edge_layer": edges.layer,
        "edge_head": edges.head,
        "edge_source": edges.source,
        "edge_target": edges.target,
        "edge_source_unit": edges.source_unit,
        "edge_attention_clean": edges.attention_clean,
        "edge_attention_corrupt": edges.attention_corrupt,
        "edge_score": edges.score,
        "edge_clean_target_score": edges.clean_target_score,
        "edge_corrupt_target_score": edges.corrupt_target_score,
        "edge_selector_score": edges.selector_score,
        "edge_content_score": edges.content_score,
        "edge_clean_message_norm": edges.clean_message_norm,
        "edge_corrupt_message_norm": edges.corrupt_message_norm,
        "edge_delta_message_norm": edges.delta_message_norm,
        "edge_clean_code": edges.clean_code,
        "edge_corrupt_code": edges.corrupt_code,
        "edge_clean_message_vector": edges.clean_message_vector,
        "edge_corrupt_message_vector": edges.corrupt_message_vector,
        "edge_delta_message_vector": edges.delta_message_vector,
        "edge_transition_probability": audit.throughput.edge_probability,
        "edge_root_throughput": audit.throughput.edge,
        "residual_transition_probability": (
            audit.throughput.residual_probability
        ),
        "reverse_node_visit": audit.throughput.reverse_visit,
        "root_conditioned_node_throughput": audit.throughput.node,
        "selected_root_route_mass": audit.throughput.root_mass,
        "clean_margin": flow.clean_margin,
        "corrupt_margin": flow.corrupt_margin,
        "pair_effect": effect.pair_effect,
        "screen_pair_effect": audit.screen_pair_effect,
        "corridor_edge_count": effect.edge_count,
        "corridor_necessity": effect.necessity,
        "corridor_sufficiency": effect.sufficiency,
        "corridor_blocked_sufficiency": effect.blocked_sufficiency,
        "corridor_mediated_sufficiency": effect.mediated_sufficiency,
        "corridor_clean_restoration_error": effect.clean_restoration_error,
        "corridor_corrupt_restoration_error": effect.corrupt_restoration_error,
        "corridor_restoration_error": effect.restoration_error,
        "corridor_restoration_tolerance": effect.restoration_tolerance,
        "corridor_restoration_valid": effect.restoration_valid,
        "corridor_confirmed": audit.corridor_confirmed,
        "root_unit_id": np.asarray(
            [item.unit_id for item in audit.roots], dtype=np.int32
        ),
        "root_route_mass": np.asarray(
            [item.route_mass for item in audit.roots], dtype=np.float32
        ),
        "root_gradient_score": np.asarray(
            [item.gradient_score for item in audit.roots], dtype=np.float32
        ),
        "root_necessity": np.asarray(
            [item.necessity for item in audit.roots], dtype=np.float32
        ),
        "root_sufficiency": np.asarray(
            [item.sufficiency for item in audit.roots], dtype=np.float32
        ),
        "root_causal_score": np.asarray(
            [item.causal_score for item in audit.roots], dtype=np.float32
        ),
        "root_evaluated": np.asarray(
            [item.evaluated for item in audit.roots], dtype=bool
        ),
        "carrier_layer": np.asarray([item.layer for item in carriers], dtype=np.int16),
        "carrier_position": np.asarray(
            [item.position for item in carriers], dtype=np.int32
        ),
        "carrier_source_unit": np.asarray(
            [
                int(world.units.token_unit_id[item.position])
                for item in carriers
            ],
            dtype=np.int32,
        ),
        "carrier_route_throughput": np.asarray(
            [item.route_throughput for item in carriers], dtype=np.float32
        ),
        "carrier_state_delta_norm": np.asarray(
            [item.state_delta_norm for item in carriers], dtype=np.float32
        ),
        "carrier_target_score": np.asarray(
            [item.target_score for item in carriers], dtype=np.float32
        ),
        "carrier_necessity": np.asarray(
            [item.necessity for item in carriers], dtype=np.float32
        ),
        "carrier_rescue": np.asarray(
            [item.rescue for item in carriers], dtype=np.float32
        ),
        "carrier_block_effect": np.asarray(
            [item.block_effect for item in carriers], dtype=np.float32
        ),
        "carrier_blocked_rescue": np.asarray(
            [item.blocked_rescue for item in carriers], dtype=np.float32
        ),
        "carrier_mediated_rescue": np.asarray(
            [item.mediated_rescue for item in carriers], dtype=np.float32
        ),
        "carrier_block_tolerance": np.asarray(
            [item.block_tolerance for item in carriers], dtype=np.float32
        ),
        "carrier_confirmed": np.asarray(
            [item.confirmed for item in carriers], dtype=bool
        ),
    }
    if flow.stages is None:
        arrays.update(
            stage_position=np.empty(0, dtype=np.int32),
            state_delta_norm=np.empty((0, 0), dtype=np.float32),
            state_target_score=np.empty((0, 0), dtype=np.float32),
            attention_write_delta_norm=np.empty((0, 0), dtype=np.float32),
            attention_write_target_score=np.empty((0, 0), dtype=np.float32),
            mlp_write_delta_norm=np.empty((0, 0), dtype=np.float32),
            mlp_write_target_score=np.empty((0, 0), dtype=np.float32),
        )
    else:
        arrays.update(
            stage_position=flow.stages.position,
            state_delta_norm=flow.stages.state_delta_norm,
            state_target_score=flow.stages.state_score,
            attention_write_delta_norm=flow.stages.attention_delta_norm,
            attention_write_target_score=flow.stages.attention_score,
            mlp_write_delta_norm=flow.stages.mlp_delta_norm,
            mlp_write_target_score=flow.stages.mlp_score,
        )
    save_result(path, arrays)
