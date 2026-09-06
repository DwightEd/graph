"""Build and persist one head-resolved native mechanism-audit payload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifact_schema import (
    AUDIT_SCHEMA,
    METHOD_VERSION,
    ROUTE_EVENT_LIMIT,
    NativeAuditMetadata,
)
from .artifacts import save_result
from .flow import SOURCE_LOCATION_BUCKET_NAMES, SourceLocationBuckets
from .native import NativeTargetAudit
from .native_world import NativeWorld
from .reanchor_timeline import (
    SOURCE_KIND_NAMES,
    ReanchorTimeline,
    ReanchorTimelineAuditor,
)
from .route_model import CHANNEL_NAMES
from .worlds import TargetContrast


@dataclass(frozen=True)
class _RouteBackbone:
    """One widest connected root-to-query path through the unrolled graph."""

    edge_index: torch.Tensor
    node_layer: torch.Tensor
    node_position: torch.Tensor
    node_throughput: torch.Tensor
    node_origin: torch.Tensor
    step_throughput: torch.Tensor
    step_is_residual: torch.Tensor


def _residual_throughput(audit: NativeTargetAudit) -> torch.Tensor:
    """Return selected-root flow carried by implicit residual transitions."""

    throughput = audit.throughput
    visit = throughput.reverse_visit.float()
    node = throughput.node.float()
    root_reach = torch.where(visit > 0, node / visit, torch.zeros_like(node))
    return visit[1:] * throughput.residual_probability.float() * root_reach[:-1]


def _route_backbone(
    _world: NativeWorld,
    audit: NativeTargetAudit,
) -> _RouteBackbone:
    """Attach saved metrics to the backbone frozen by :class:`AuditPlan`."""

    edge_index = audit.plan.backbone_edge_index.long().cpu()
    position = audit.plan.backbone_position.long().cpu()
    throughput = audit.throughput
    node = throughput.node.float().cpu()
    layer_count = node.shape[0] - 1
    query = int(audit.flow.target.query_position)
    if len(position) == 1:
        if int(position[0]) != query:
            raise ValueError("empty backbone does not end at the audited query")
        empty = torch.empty(0, dtype=torch.long)
        return _RouteBackbone(
            empty,
            torch.tensor([layer_count], dtype=torch.int16),
            torch.tensor([query], dtype=torch.int32),
            node[layer_count, query].reshape(1),
            audit.dynamics.node_register[layer_count, query].reshape(1, -1),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
        )
    if len(position) != layer_count + 1 or int(position[-1]) != query:
        raise ValueError("frozen backbone has the wrong layer path")

    residual = _residual_throughput(audit).cpu()
    layer = torch.arange(layer_count, dtype=torch.long)
    step_flow = residual[layer, position[:-1]].clone()
    residual_step = torch.ones(layer_count, dtype=torch.bool)
    edges = audit.flow.edges
    for index in edge_index.tolist():
        edge_layer = int(edges.layer[index])
        if int(edges.source[index]) != int(position[edge_layer]) or int(
            edges.target[index]
        ) != int(position[edge_layer + 1]):
            raise ValueError("frozen backbone edge does not match its node path")
        step_flow[edge_layer] = throughput.edge[index]
        residual_step[edge_layer] = False

    layer = torch.arange(layer_count + 1, dtype=torch.long)
    return _RouteBackbone(
        edge_index,
        layer.to(torch.int16),
        position.to(torch.int32),
        node[layer, position],
        audit.dynamics.node_register[layer, position],
        step_flow,
        residual_step,
    )


def _saved_route_edges(
    audit: NativeTargetAudit,
    backbone: _RouteBackbone,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the frozen plan, then fill the display budget by throughput."""

    backbone_edge = backbone.edge_index.long()
    on_backbone = torch.zeros(audit.flow.edges.count, dtype=torch.bool)
    on_backbone[backbone_edge] = True
    corridor_edge = audit.plan.corridor_edge_index.long()
    required = torch.unique(torch.cat((backbone_edge, corridor_edge)), sorted=True)
    required_mask = torch.zeros(audit.flow.edges.count, dtype=torch.bool)
    required_mask[required] = True
    supported = torch.nonzero(
        (audit.throughput.edge > 0) & ~required_mask, as_tuple=False
    ).flatten()
    remaining = max(0, limit - len(required))
    count = min(remaining, len(supported))
    if count:
        score = audit.throughput.edge.index_select(0, supported)
        order = torch.topk(score, k=count, sorted=True).indices
        marginal = supported.index_select(0, order)
        selected = torch.cat((required, marginal))
    else:
        selected = required
    return selected, on_backbone.index_select(0, selected)


def _frozen_corridor_arrays(audit: NativeTargetAudit) -> dict[str, object]:
    """Serialize the complete frozen corridor independently of plot edge limits."""

    edge_index = audit.plan.corridor_edge_index.long()
    edges = audit.flow.edges.select(edge_index)
    layer = edges.layer.long()
    source = edges.source.long()
    source_origin = audit.dynamics.node_register[layer, source].float()
    source_total = source_origin.sum(dim=-1)
    evidence_share = torch.where(
        source_total > 0,
        source_origin[:, 0] / source_total,
        torch.zeros_like(source_total),
    )
    return {
        "frozen_corridor_edge_index": edge_index,
        "frozen_corridor_layer": edges.layer,
        "frozen_corridor_head": edges.head,
        "frozen_corridor_source": edges.source,
        "frozen_corridor_target": edges.target,
        "frozen_corridor_source_unit": edges.source_unit,
        "frozen_corridor_attention_native": edges.attention_clean,
        "frozen_corridor_attention_root_cut": edges.attention_corrupt,
        "frozen_corridor_native_functional_score": edges.clean_target_score,
        "frozen_corridor_root_cut_functional_score": edges.corrupt_target_score,
        "frozen_corridor_source_evidence_lineage_fraction": evidence_share,
        "frozen_corridor_evidence_lineage_action": (
            evidence_share * edges.clean_target_score.float()
        ),
        "frozen_corridor_native_message_norm": edges.clean_message_norm,
        "frozen_corridor_root_cut_message_norm": edges.corrupt_message_norm,
        "frozen_corridor_delta_message_norm": edges.delta_message_norm,
        "frozen_corridor_root_throughput": audit.throughput.edge.index_select(
            0, edge_index
        ),
    }


def _root_arrays(audit: NativeTargetAudit) -> dict[str, object]:
    planned = audit.plan.roots
    effects = {root.unit_id: root for root in audit.roots}
    return {
        "root_unit_id": np.asarray([root.unit_id for root in planned], dtype=np.int32),
        "root_route_mass": np.asarray(
            [root.route_mass for root in planned], dtype=np.float32
        ),
        "root_signed_action": np.asarray(
            [root.signed_action for root in planned], dtype=np.float32
        ),
        "root_absolute_action": np.asarray(
            [root.absolute_action for root in planned], dtype=np.float32
        ),
        "root_functional_agreement": np.asarray(
            [root.functional_agreement for root in planned], dtype=np.float32
        ),
        "root_selection_score": np.asarray(
            [root.score for root in planned], dtype=np.float32
        ),
        "root_value_necessity": np.asarray(
            [effects[root.unit_id].necessity for root in planned], dtype=np.float32
        ),
        "root_conditional_sufficiency": np.asarray(
            [effects[root.unit_id].sufficiency for root in planned], dtype=np.float32
        ),
        "root_causal_score": np.asarray(
            [effects[root.unit_id].causal_score for root in planned], dtype=np.float32
        ),
        "root_evaluated": np.asarray(
            [effects[root.unit_id].evaluated for root in planned], dtype=bool
        ),
    }


def _hub_arrays(audit: NativeTargetAudit) -> dict[str, object]:
    hubs = audit.plan.hubs
    return {
        "hub_layer": np.asarray([hub.layer for hub in hubs], dtype=np.int16),
        "hub_position": np.asarray([hub.position for hub in hubs], dtype=np.int32),
        "hub_route_mass": np.asarray(
            [hub.route_mass for hub in hubs], dtype=np.float32
        ),
        "hub_signed_action": np.asarray(
            [hub.signed_action for hub in hubs], dtype=np.float32
        ),
        "hub_graph_score": np.asarray([hub.score for hub in hubs], dtype=np.float32),
    }


def _carrier_arrays(
    world: NativeWorld,
    audit: NativeTargetAudit,
) -> dict[str, object]:
    carriers = audit.carriers
    return {
        "carrier_layer": np.asarray(
            [carrier.layer for carrier in carriers], dtype=np.int16
        ),
        "carrier_position": np.asarray(
            [carrier.position for carrier in carriers], dtype=np.int32
        ),
        "carrier_source_unit": np.asarray(
            [int(world.units.token_unit_id[carrier.position]) for carrier in carriers],
            dtype=np.int32,
        ),
        "carrier_route_throughput": np.asarray(
            [carrier.route_throughput for carrier in carriers], dtype=np.float32
        ),
        "carrier_state_delta_norm": np.asarray(
            [carrier.state_delta_norm for carrier in carriers], dtype=np.float32
        ),
        "carrier_target_score": np.asarray(
            [carrier.target_score for carrier in carriers], dtype=np.float32
        ),
        "carrier_necessity": np.asarray(
            [carrier.necessity for carrier in carriers], dtype=np.float32
        ),
        "carrier_rescue": np.asarray(
            [carrier.rescue for carrier in carriers], dtype=np.float32
        ),
        "carrier_block_effect": np.asarray(
            [carrier.block_effect for carrier in carriers], dtype=np.float32
        ),
        "carrier_blocked_rescue": np.asarray(
            [carrier.blocked_rescue for carrier in carriers], dtype=np.float32
        ),
        "carrier_mediated_rescue": np.asarray(
            [carrier.mediated_rescue for carrier in carriers], dtype=np.float32
        ),
        "carrier_block_tolerance": np.asarray(
            [carrier.block_tolerance for carrier in carriers], dtype=np.float32
        ),
        "carrier_confirmed": np.asarray(
            [carrier.confirmed for carrier in carriers], dtype=bool
        ),
    }


def _target_reanchor_arrays(
    world: NativeWorld,
    target: TargetContrast,
) -> dict[str, object]:
    """Serialize the clean full-response event that selected this target."""

    selection_table = getattr(world, "target_selection", ())
    if not selection_table:
        return {
            "target_reanchor_selection_recorded": False,
            "target_reanchor_policy": "none",
            "target_reanchor_has_event": False,
            "target_reanchor_is_center": False,
            "target_reanchor_fallback": False,
            "target_reanchor_center_position": -1,
            "target_reanchor_window_offset": 0,
            "target_reanchor_layer": -1,
            "target_reanchor_head": -1,
            "target_reanchor_source_kind": "none",
            "target_reanchor_source_position": -1,
            "target_reanchor_source_unit_id": -1,
            "target_reanchor_score": 0.0,
            "target_reanchor_support": 0,
            "target_reanchor_previous_anchor_fraction": 0.0,
            "target_reanchor_anchor_fraction": 0.0,
            "target_reanchor_relative_anchor_rise": 0.0,
            "target_reanchor_relative_local_fall": 0.0,
            "target_reanchor_scan_signal": "none",
        }
    selection = selection_table[world.targets.index(target)]
    selection.check(target)
    return {
        "target_reanchor_selection_recorded": True,
        "target_reanchor_policy": selection.policy,
        "target_reanchor_has_event": selection.has_event,
        "target_reanchor_is_center": selection.is_center,
        "target_reanchor_fallback": selection.fallback,
        "target_reanchor_center_position": selection.center_position,
        "target_reanchor_window_offset": selection.window_offset,
        "target_reanchor_layer": selection.layer,
        "target_reanchor_head": selection.head,
        "target_reanchor_source_kind": selection.source_kind,
        "target_reanchor_source_position": selection.source_position,
        "target_reanchor_source_unit_id": selection.source_unit_id,
        "target_reanchor_score": selection.score,
        "target_reanchor_support": selection.support,
        "target_reanchor_previous_anchor_fraction": (
            selection.previous_anchor_fraction
        ),
        "target_reanchor_anchor_fraction": selection.anchor_fraction,
        "target_reanchor_relative_anchor_rise": selection.relative_anchor_rise,
        "target_reanchor_relative_local_fall": selection.relative_local_fall,
        "target_reanchor_scan_signal": selection.scan_signal,
    }


def _reanchor_timeline_arrays(
    timeline: ReanchorTimeline,
    source_location: SourceLocationBuckets,
) -> dict[str, object]:
    """Serialize the direct, head-specific generation-time switch audit."""

    candidates = timeline.candidates
    return {
        "reanchor_source_kind_name": np.asarray(SOURCE_KIND_NAMES),
        "reanchor_bucket_name": np.asarray(SOURCE_LOCATION_BUCKET_NAMES),
        "reanchor_bucket_attention": source_location.attention,
        "reanchor_bucket_transport": source_location.transport,
        "reanchor_bucket_downstream_action": source_location.downstream_action,
        "reanchor_source_position": source_location.source_position,
        "reanchor_source_unit": source_location.source_unit_id,
        "reanchor_source_attention": source_location.source_attention,
        "reanchor_source_transport": source_location.source_transport,
        "reanchor_source_downstream_action": (source_location.source_downstream_action),
        "reanchor_anchor_fraction": timeline.anchor_fraction,
        "reanchor_switch_delta": timeline.switch_delta,
        "reanchor_relative_anchor_rise": timeline.relative_anchor_rise,
        "reanchor_relative_local_fall": timeline.relative_local_fall,
        "reanchor_score": timeline.score,
        "reanchor_dominance_flip": timeline.dominance_flip,
        "reanchor_anchor_source_kind": timeline.anchor_source_kind,
        "reanchor_anchor_source_position": timeline.anchor_source_position,
        "reanchor_anchor_source_unit": timeline.anchor_source_unit_id,
        "reanchor_anchor_source_transport": timeline.anchor_source_transport,
        "reanchor_anchor_source_evidence_fraction": (
            timeline.anchor_source_evidence_fraction
        ),
        "reanchor_local_source_position": timeline.local_source_position,
        "reanchor_local_source_unit": timeline.local_source_unit_id,
        "reanchor_candidate_layer": np.asarray(
            [item.layer for item in candidates], dtype=np.int16
        ),
        "reanchor_candidate_head": np.asarray(
            [item.head for item in candidates], dtype=np.int16
        ),
        "reanchor_candidate_position": np.asarray(
            [item.position for item in candidates], dtype=np.int32
        ),
        "reanchor_candidate_source_kind": np.asarray(
            [item.source_kind for item in candidates]
        ),
        "reanchor_candidate_source_position": np.asarray(
            [item.source_position for item in candidates], dtype=np.int32
        ),
        "reanchor_candidate_source_unit": np.asarray(
            [item.source_unit_id for item in candidates], dtype=np.int32
        ),
        "reanchor_candidate_source_evidence_fraction": np.asarray(
            [item.source_evidence_fraction for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_previous_local_source_position": np.asarray(
            [item.previous_local_source_position for item in candidates],
            dtype=np.int32,
        ),
        "reanchor_candidate_previous_local_source_unit": np.asarray(
            [item.previous_local_source_unit_id for item in candidates],
            dtype=np.int32,
        ),
        "reanchor_candidate_score": np.asarray(
            [item.score for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_switch_delta": np.asarray(
            [item.switch_delta for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_previous_anchor_fraction": np.asarray(
            [item.previous_anchor_fraction for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_anchor_fraction": np.asarray(
            [item.anchor_fraction for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_prompt_transport": np.asarray(
            [item.prompt_transport for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_other_prompt_transport": np.asarray(
            [item.other_prompt_transport for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_remote_response_transport": np.asarray(
            [item.remote_response_transport for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_previous_local_transport": np.asarray(
            [item.previous_local_transport for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_local_transport": np.asarray(
            [item.local_transport for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_long_range_downstream_action": np.asarray(
            [item.long_range_downstream_action for item in candidates],
            dtype=np.float32,
        ),
        "reanchor_candidate_bucket_downstream_action": np.asarray(
            [item.bucket_downstream_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_source_downstream_action": np.asarray(
            [item.source_downstream_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_local_downstream_action": np.asarray(
            [item.local_downstream_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_current_target_match": np.asarray(
            [item.current_target_match for item in candidates], dtype=bool
        ),
        "reanchor_candidate_long_range_immediate_action": np.asarray(
            [item.long_range_immediate_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_bucket_immediate_action": np.asarray(
            [item.bucket_immediate_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_source_immediate_action": np.asarray(
            [item.source_immediate_action for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_selected_root_integration_budget": np.asarray(
            [item.selected_root_integration_budget for item in candidates],
            dtype=np.float32,
        ),
        "reanchor_candidate_selected_root_integration_coherence": np.asarray(
            [item.selected_root_integration_coherence for item in candidates],
            dtype=np.float32,
        ),
        "reanchor_candidate_selected_root_integration_action": np.asarray(
            [item.selected_root_integration_action for item in candidates],
            dtype=np.float32,
        ),
        "reanchor_candidate_dominance_flip": np.asarray(
            [item.dominance_flip for item in candidates], dtype=bool
        ),
        "reanchor_candidate_relative_anchor_rise": np.asarray(
            [item.relative_anchor_rise for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_relative_local_fall": np.asarray(
            [item.relative_local_fall for item in candidates], dtype=np.float32
        ),
        "reanchor_candidate_support": np.asarray(
            [item.support for item in candidates], dtype=np.int32
        ),
    }


def native_audit_arrays(
    world: NativeWorld,
    audit: NativeTargetAudit,
    metadata: NativeAuditMetadata,
) -> dict[str, object]:
    """Return the complete route ledgers and exact intervention outcomes."""

    flow = audit.flow
    dynamics = audit.dynamics
    if dynamics.local_window != metadata.local_window:
        raise ValueError("audit local_window does not match run metadata")
    if audit.plan.budget != metadata.route_budget:
        raise ValueError("audit route_budget does not match run metadata")
    backbone = _route_backbone(world, audit)
    edge_index, edge_on_backbone = _saved_route_edges(
        audit, backbone, metadata.saved_edges
    )
    in_frozen_corridor = torch.zeros(flow.edges.count, dtype=torch.bool)
    in_frozen_corridor[audit.plan.corridor_edge_index.long()] = True
    edges = flow.edges.select(edge_index)
    edge_layer = edges.layer.long()
    edge_source = edges.source.long()
    source_origin = dynamics.node_register[edge_layer, edge_source].float()
    source_total = source_origin.sum(dim=-1)
    source_evidence_share = torch.where(
        source_total > 0,
        source_origin[:, 0] / source_total,
        torch.zeros_like(source_total),
    )
    evidence_lineage_action = source_evidence_share * edges.clean_target_score.float()
    reanchor_timeline = ReanchorTimelineAuditor.audit(
        flow, dynamics, world, limit=ROUTE_EVENT_LIMIT
    )
    effect = audit.effect
    selected_root_evaluated = bool(audit.selected_root_effect.evaluated)
    any_exact_evaluated = bool(
        selected_root_evaluated
        or audit.corridor_evaluated
        or audit.carrier_evaluated_count
    )
    full_chain_evaluated = bool(
        selected_root_evaluated
        and audit.corridor_evaluated
        and audit.carrier_evaluated_count > 0
    )
    carrier_any_confirmed = any(carrier.confirmed for carrier in audit.carriers)
    full_chain_confirmed = bool(
        full_chain_evaluated
        and audit.selected_root_confirmed
        and audit.corridor_confirmed
        and carrier_any_confirmed
    )
    budget = metadata.route_budget
    arrays: dict[str, object] = {
        "subset_audit_schema": AUDIT_SCHEMA,
        "method_version": METHOD_VERSION,
        "method": "native head-resolved all-evidence provenance route",
        "dataset_sample_id": metadata.dataset_sample_id,
        "sample_id": world.sample_id,
        "source_id": metadata.source_id,
        "split": metadata.split,
        "task_type": metadata.task_type,
        "generator_model": metadata.generator_model,
        "tokenizer_id": world.tokenizer_id,
        "model_id": metadata.model_id,
        "model_dtype": metadata.model_dtype,
        "target_selection_policy": metadata.target_policy,
        "target_selection_rank": metadata.target_rank,
        "response_start": world.response_start,
        "layer_count": dynamics.node_register.shape[0] - 1,
        "query_position": flow.target.query_position,
        "prediction_position": flow.target.query_position + 1,
        "positive_token_id": flow.target.positive_token_id,
        "negative_token_id": flow.target.negative_token_id,
        "contrast_origin": flow.target.origin,
        "flow_signal": flow.signal.value,
        "analysis_stage": "confirmed" if any_exact_evaluated else "discovery",
        "edge_coverage": metadata.coverage,
        "carrier_scope": metadata.carrier_scope,
        "query_chunk": metadata.query_chunk,
        "route_budget_edges_per_head": budget.edges_per_head,
        "route_budget_max_rows": budget.max_rows,
        "route_budget_root_candidates": budget.root_candidates,
        "route_budget_hub_candidates": budget.hub_candidates,
        "route_budget_corridor_edges": budget.corridor_edges,
        "route_budget_confirm": budget.confirm,
        "local_window": metadata.local_window,
        "edge_save_limit": metadata.saved_edges,
        "edge_saved_count": len(edge_index),
        "token_ids": world.token_ids,
        "token_unit_id": world.units.token_unit_id,
        "unit_name": np.asarray(world.units.name),
        "unit_kind": np.asarray(world.units.kind),
        "evidence_unit_id": np.asarray(world.evidence_unit_id, dtype=np.int32),
        "selected_root_unit_id": audit.selected_root_unit_id,
        "selected_root_selection_fallback": audit.plan.root_selection_fallback,
        "selected_root_evaluated": selected_root_evaluated,
        "selected_root_confirmed": audit.selected_root_confirmed,
        "selected_root_route_mass": audit.throughput.root_mass,
        "selected_root_value_necessity": audit.selected_root_effect.necessity,
        "selected_root_conditional_sufficiency": (
            audit.selected_root_effect.sufficiency
        ),
        "selected_root_causal_score": audit.selected_root_effect.causal_score,
        "native_margin": flow.clean_margin,
        "root_cut_margin": flow.corrupt_margin,
        "root_value_effect": flow.clean_margin - flow.corrupt_margin,
        "all_evidence_cut_margin": audit.all_evidence_cut_margin,
        "corridor_edge_count": audit.corridor.count,
        "corridor_necessity": effect.necessity,
        "corridor_conditional_rescue": effect.sufficiency,
        "corridor_blocked_rescue": effect.blocked_sufficiency,
        "corridor_mediated_rescue": effect.mediated_sufficiency,
        "corridor_native_restoration_error": effect.clean_restoration_error,
        "corridor_cut_restoration_error": effect.corrupt_restoration_error,
        "corridor_restoration_error": effect.restoration_error,
        "corridor_restoration_tolerance": effect.restoration_tolerance,
        "corridor_restoration_valid": effect.restoration_valid,
        "corridor_evaluated": audit.corridor_evaluated,
        "corridor_confirmed": audit.corridor_confirmed,
        "carrier_evaluated_count": audit.carrier_evaluated_count,
        "carrier_any_confirmed": carrier_any_confirmed,
        "full_chain_evaluated": full_chain_evaluated,
        "full_chain_confirmed": full_chain_confirmed,
        "edge_layer": edges.layer,
        "edge_head": edges.head,
        "edge_source": edges.source,
        "edge_target": edges.target,
        "edge_source_unit": edges.source_unit,
        "edge_attention_native": edges.attention_clean,
        "edge_attention_root_cut": edges.attention_corrupt,
        "edge_native_functional_score": edges.clean_target_score,
        "edge_root_cut_functional_score": edges.corrupt_target_score,
        "edge_source_evidence_lineage_fraction": source_evidence_share,
        "edge_evidence_lineage_action": evidence_lineage_action,
        "edge_native_message_norm": edges.clean_message_norm,
        "edge_root_cut_message_norm": edges.corrupt_message_norm,
        "edge_delta_message_norm": edges.delta_message_norm,
        "edge_root_throughput": audit.throughput.edge.index_select(0, edge_index),
        "edge_on_backbone": edge_on_backbone,
        "edge_in_frozen_corridor": in_frozen_corridor.index_select(0, edge_index),
        "backbone_node_layer": backbone.node_layer,
        "backbone_node_position": backbone.node_position,
        "backbone_node_throughput": backbone.node_throughput,
        "backbone_node_origin": backbone.node_origin,
        "backbone_step_throughput": backbone.step_throughput,
        "backbone_step_is_residual": backbone.step_is_residual,
        "route_origin_name": np.asarray(CHANNEL_NAMES),
        "route_edge_origin": dynamics.edge_register.index_select(0, edge_index),
        "route_node_origin": dynamics.node_register,
        "route_node_throughput": audit.throughput.node,
        "route_row_position": dynamics.row_position,
        "route_row_total": flow.row_total,
        "route_row_retained": flow.row_retained,
        "route_head_transport": dynamics.head_transport,
        "route_head_action": dynamics.head_gradient_action,
        "route_head_direct_evidence": dynamics.head_direct_evidence,
        "route_head_integration": dynamics.head_integration,
        "route_layer_integration": dynamics.layer_integration,
        "route_cross_head_vector_coherence": (dynamics.cross_head_vector_coherence),
        "route_cross_head_functional_agreement": (
            dynamics.cross_head_functional_agreement
        ),
        "route_evidence_source_reuse": dynamics.evidence_source_reuse,
        "route_response_source_reuse": dynamics.response_source_reuse,
        "route_stage_position": dynamics.stage_position,
        "route_stage_displacement": dynamics.stage_displacement,
        "route_stage_action": dynamics.stage_gradient_action,
        "route_module_vector_cosine": dynamics.attention_mlp_vector_cosine,
        "route_module_functional_agreement": (
            dynamics.attention_mlp_functional_agreement
        ),
        "route_state_continuity": dynamics.state_continuity,
    }
    arrays.update(_root_arrays(audit))
    arrays.update(_hub_arrays(audit))
    arrays.update(_carrier_arrays(world, audit))
    arrays.update(_target_reanchor_arrays(world, flow.target))
    if flow.source_location is None:
        raise ValueError("native audit lacks full-row source-location buckets")
    arrays.update(_reanchor_timeline_arrays(reanchor_timeline, flow.source_location))
    arrays.update(_frozen_corridor_arrays(audit))
    return arrays


def save_native_audit(
    path: str | Path,
    world: NativeWorld,
    audit: NativeTargetAudit,
    metadata: NativeAuditMetadata,
) -> None:
    """Atomically save one schema-3 native mechanism artifact."""

    save_result(path, native_audit_arrays(world, audit, metadata))
