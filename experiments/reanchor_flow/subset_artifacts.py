"""Plot-ready persistence for one head-resolved native mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import save_result
from .flow import FlowSignal
from .native import NativeTargetAudit
from .native_world import NativeWorld
from .route_model import CHANNEL_NAMES, HeadResolvedRouteModel, RouteEvent
from .worlds import TargetContrast

AUDIT_SCHEMA = 2
METHOD_VERSION = "head-resolved-native-route/2"
ROUTE_EVENT_LIMIT = 32


@dataclass(frozen=True)
class NativeAuditMetadata:
    """Frozen run coordinates that are not part of the mechanism tensors."""

    dataset_sample_id: str
    source_id: str
    split: str
    task_type: str
    generator_model: str
    model_id: str
    model_dtype: str
    target_policy: str
    target_rank: int
    coverage: float
    carrier_scope: str
    query_chunk: int
    root_screen_limit: int
    carrier_limit: int
    local_window: int
    saved_edges: int


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
    world: NativeWorld,
    audit: NativeTargetAudit,
) -> _RouteBackbone:
    """Find a widest path, retaining residual steps as explicit path nodes.

    The dynamic program maximizes the minimum selected-root throughput along
    the path.  Unlike independent edge ranking, the result is guaranteed to
    start in the selected source unit and end at the audited query node.
    """

    throughput = audit.throughput
    node = throughput.node.float().cpu()
    layer_count = node.shape[0] - 1
    token_count = node.shape[1]
    query = int(audit.flow.target.query_position)
    token_unit = world.units.token_unit_id.long().cpu()
    root_position = torch.nonzero(
        (token_unit == audit.selected_root_unit_id)
        & (torch.arange(len(token_unit)) < world.response_start),
        as_tuple=False,
    ).flatten()

    best = torch.full((token_count,), -torch.inf)
    best[root_position] = torch.inf
    predecessor_position = torch.full((layer_count, token_count), -1, dtype=torch.long)
    predecessor_edge = torch.full((layer_count, token_count), -2, dtype=torch.long)
    predecessor_flow = torch.zeros(layer_count, token_count)
    residual_flow = _residual_throughput(audit).cpu()
    edge_flow = throughput.edge.float().cpu()
    edges = audit.flow.edges

    for layer_index in range(layer_count):
        following = torch.full_like(best, -torch.inf)
        positive_residual = torch.nonzero(
            residual_flow[layer_index] > 0, as_tuple=False
        ).flatten()
        for position in positive_residual.tolist():
            candidate = min(
                float(best[position]),
                float(residual_flow[layer_index, position]),
            )
            if candidate > float(following[position]):
                following[position] = candidate
                predecessor_position[layer_index, position] = position
                predecessor_edge[layer_index, position] = -1
                predecessor_flow[layer_index, position] = residual_flow[
                    layer_index, position
                ]

        layer_edge = torch.nonzero(
            (edges.layer == layer_index) & (edge_flow > 0), as_tuple=False
        ).flatten()
        for edge_index in layer_edge.tolist():
            source = int(edges.source[edge_index])
            target = int(edges.target[edge_index])
            candidate = min(float(best[source]), float(edge_flow[edge_index]))
            if candidate > float(following[target]):
                following[target] = candidate
                predecessor_position[layer_index, target] = source
                predecessor_edge[layer_index, target] = edge_index
                predecessor_flow[layer_index, target] = edge_flow[edge_index]
        best = following

    if not torch.isfinite(best[query]):
        if throughput.root_mass > 0:
            raise ValueError("positive selected-root flow has no connected backbone")
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

    positions = torch.empty(layer_count + 1, dtype=torch.long)
    path_edges: list[int] = []
    step_flow = torch.empty(layer_count)
    residual_step = torch.empty(layer_count, dtype=torch.bool)
    positions[layer_count] = query
    for layer_index in range(layer_count - 1, -1, -1):
        target = int(positions[layer_index + 1])
        source = int(predecessor_position[layer_index, target])
        edge_index = int(predecessor_edge[layer_index, target])
        positions[layer_index] = source
        step_flow[layer_index] = predecessor_flow[layer_index, target]
        residual_step[layer_index] = edge_index == -1
        if edge_index >= 0:
            path_edges.append(edge_index)

    path_edges.reverse()
    layer = torch.arange(layer_count + 1, dtype=torch.long)
    return _RouteBackbone(
        torch.tensor(path_edges, dtype=torch.long),
        layer.to(torch.int16),
        positions.to(torch.int32),
        node[layer, positions],
        audit.dynamics.node_register[layer, positions],
        step_flow,
        residual_step,
    )


def _saved_route_edges(
    audit: NativeTargetAudit,
    backbone: _RouteBackbone,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the backbone first, then fill the edge budget by throughput."""

    backbone_edge = backbone.edge_index.long()
    on_backbone = torch.zeros(audit.flow.edges.count, dtype=torch.bool)
    on_backbone[backbone_edge] = True
    supported = torch.nonzero(
        (audit.throughput.edge > 0) & ~on_backbone, as_tuple=False
    ).flatten()
    remaining = max(0, limit - len(backbone_edge))
    count = min(remaining, len(supported))
    if count:
        score = audit.throughput.edge.index_select(0, supported)
        order = torch.topk(score, k=count, sorted=True).indices
        marginal = supported.index_select(0, order)
        selected = torch.cat((backbone_edge, marginal))
    else:
        selected = backbone_edge
    return selected, on_backbone.index_select(0, selected)


def _root_arrays(audit: NativeTargetAudit) -> dict[str, object]:
    roots = audit.roots
    return {
        "root_unit_id": np.asarray([root.unit_id for root in roots], dtype=np.int32),
        "root_route_mass": np.asarray(
            [root.route_mass for root in roots], dtype=np.float32
        ),
        "root_functional_score": np.asarray(
            [root.gradient_score for root in roots], dtype=np.float32
        ),
        "root_value_necessity": np.asarray(
            [root.necessity for root in roots], dtype=np.float32
        ),
        "root_conditional_sufficiency": np.asarray(
            [root.sufficiency for root in roots], dtype=np.float32
        ),
        "root_causal_score": np.asarray(
            [root.causal_score for root in roots], dtype=np.float32
        ),
        "root_evaluated": np.asarray([root.evaluated for root in roots], dtype=bool),
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


def _event_arrays(events: tuple[RouteEvent, ...]) -> dict[str, object]:
    return {
        "route_event_layer": np.asarray(
            [event.layer for event in events], dtype=np.int16
        ),
        "route_event_head": np.asarray(
            [event.head for event in events], dtype=np.int16
        ),
        "route_event_position": np.asarray(
            [event.position for event in events], dtype=np.int32
        ),
        "route_event_score": np.asarray(
            [event.score for event in events], dtype=np.float32
        ),
        "route_event_evidence_transport": np.asarray(
            [event.evidence_transport for event in events], dtype=np.float32
        ),
        "route_event_evidence_action": np.asarray(
            [event.evidence_gradient_action for event in events], dtype=np.float32
        ),
        "route_event_direct_fraction": np.asarray(
            [event.direct_fraction for event in events], dtype=np.float32
        ),
        "route_event_local_response_transport": np.asarray(
            [event.local_response_transport for event in events], dtype=np.float32
        ),
        "route_event_local_response_action": np.asarray(
            [event.local_response_gradient_action for event in events],
            dtype=np.float32,
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
    backbone = _route_backbone(world, audit)
    edge_index, edge_on_backbone = _saved_route_edges(
        audit, backbone, metadata.saved_edges
    )
    edges = flow.edges.select(edge_index)
    edge_layer = edges.layer.long()
    edge_source = edges.source.long()
    source_origin = dynamics.node_register[edge_layer, edge_source].float()
    source_total = source_origin.sum(dim=-1)
    source_root_share = torch.where(
        source_total > 0,
        source_origin[:, 0] / source_total,
        torch.zeros_like(source_total),
    )
    root_lineage_action = source_root_share * edges.clean_target_score.float()
    events = HeadResolvedRouteModel.reanchor_events(dynamics, limit=ROUTE_EVENT_LIMIT)
    effect = audit.effect
    carrier_any_confirmed = any(carrier.confirmed for carrier in audit.carriers)
    full_chain_confirmed = audit.corridor_confirmed and carrier_any_confirmed

    arrays: dict[str, object] = {
        "subset_audit_schema": AUDIT_SCHEMA,
        "method_version": METHOD_VERSION,
        "method": "native head-resolved selected-source route",
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
        "target_token_ids": np.asarray(
            [flow.target.positive_token_id, flow.target.negative_token_id],
            dtype=np.int32,
        ),
        "contrast_origin": flow.target.origin,
        "flow_signal": flow.signal.value,
        "edge_coverage": metadata.coverage,
        "carrier_scope": metadata.carrier_scope,
        "query_chunk": metadata.query_chunk,
        "root_screen_limit": metadata.root_screen_limit,
        "carrier_limit": metadata.carrier_limit,
        "local_window": metadata.local_window,
        "edge_save_limit": metadata.saved_edges,
        "edge_saved_count": len(edge_index),
        "token_ids": world.token_ids,
        "token_unit_id": world.units.token_unit_id,
        "unit_name": np.asarray(world.units.name),
        "unit_kind": np.asarray(world.units.kind),
        "evidence_unit_id": np.asarray(world.evidence_unit_id, dtype=np.int32),
        "selected_root_unit_id": audit.selected_root_unit_id,
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
        "corridor_pair_effect": effect.pair_effect,
        "corridor_necessity": effect.necessity,
        "corridor_conditional_rescue": effect.sufficiency,
        "corridor_blocked_rescue": effect.blocked_sufficiency,
        "corridor_mediated_rescue": effect.mediated_sufficiency,
        "corridor_native_restoration_error": effect.clean_restoration_error,
        "corridor_cut_restoration_error": effect.corrupt_restoration_error,
        "corridor_restoration_error": effect.restoration_error,
        "corridor_restoration_tolerance": effect.restoration_tolerance,
        "corridor_restoration_valid": effect.restoration_valid,
        "corridor_confirmed": audit.corridor_confirmed,
        "carrier_any_confirmed": carrier_any_confirmed,
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
        "edge_source_root_lineage_fraction": source_root_share,
        "edge_root_lineage_action": root_lineage_action,
        "edge_native_message_norm": edges.clean_message_norm,
        "edge_root_cut_message_norm": edges.corrupt_message_norm,
        "edge_delta_message_norm": edges.delta_message_norm,
        "edge_root_throughput": audit.throughput.edge.index_select(0, edge_index),
        "edge_on_backbone": edge_on_backbone,
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
        "route_head_transport": dynamics.head_transport,
        "route_head_action": dynamics.head_gradient_action,
        "route_head_direct_evidence": dynamics.head_direct_evidence,
        "route_head_local_response": dynamics.head_local_response,
        "route_head_integration": dynamics.head_integration,
        "route_layer_integration": dynamics.layer_integration,
        "route_cross_head_vector_coherence": (dynamics.cross_head_vector_coherence),
        "route_cross_head_functional_agreement": (
            dynamics.cross_head_functional_agreement
        ),
        "route_head_backward_distance": dynamics.head_backward_distance,
        "route_head_span": dynamics.head_span,
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
    arrays.update(_carrier_arrays(world, audit))
    arrays.update(_event_arrays(events))
    return arrays


def save_native_audit(
    path: str | Path,
    world: NativeWorld,
    audit: NativeTargetAudit,
    metadata: NativeAuditMetadata,
) -> None:
    """Atomically save one schema-2 native mechanism artifact."""

    save_result(path, native_audit_arrays(world, audit, metadata))


def validate_native_audit(
    path: str | Path,
    world: NativeWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    metadata: NativeAuditMetadata,
) -> None:
    """Check the minimal identity needed to resume a frozen target audit."""

    expected = {
        "subset_audit_schema": AUDIT_SCHEMA,
        "method_version": METHOD_VERSION,
        "dataset_sample_id": metadata.dataset_sample_id,
        "sample_id": world.sample_id,
        "source_id": metadata.source_id,
        "split": metadata.split,
        "task_type": metadata.task_type,
        "query_position": target.query_position,
        "prediction_position": target.query_position + 1,
        "positive_token_id": target.positive_token_id,
        "negative_token_id": target.negative_token_id,
        "contrast_origin": target.origin,
        "target_selection_rank": metadata.target_rank,
        "flow_signal": FlowSignal(signal).value,
        "generator_model": metadata.generator_model,
        "tokenizer_id": world.tokenizer_id,
        "model_id": metadata.model_id,
        "model_dtype": metadata.model_dtype,
        "target_selection_policy": metadata.target_policy,
        "edge_coverage": metadata.coverage,
        "carrier_scope": metadata.carrier_scope,
        "query_chunk": metadata.query_chunk,
        "root_screen_limit": metadata.root_screen_limit,
        "carrier_limit": metadata.carrier_limit,
        "local_window": metadata.local_window,
        "edge_save_limit": metadata.saved_edges,
    }
    with np.load(Path(path), allow_pickle=False) as stored:
        for name, value in expected.items():
            if name not in stored.files or stored[name].shape != ():
                raise ValueError(f"subset artifact lacks scalar identity {name}")
            if stored[name].item() != value:
                raise ValueError(f"subset artifact {name} does not match")

        if "token_ids" not in stored.files:
            raise ValueError("subset artifact lacks token_ids identity")
        token_ids = stored["token_ids"]
        world_token_ids = world.token_ids.detach().cpu().numpy()
        if token_ids.shape != world_token_ids.shape or not np.array_equal(
            token_ids, world_token_ids
        ):
            raise ValueError("subset artifact token_ids do not match")
