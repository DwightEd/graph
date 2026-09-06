"""Validate persisted native mechanism-audit artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact_payload import _target_reanchor_arrays
from .artifact_schema import (
    AUDIT_SCHEMA,
    METHOD_VERSION,
    ROUTE_EVENT_LIMIT,
    NativeAuditMetadata,
)
from .flow import SOURCE_LOCATION_BUCKET_NAMES, FlowSignal
from .native_world import NativeWorld
from .reanchor_timeline import SOURCE_KIND_NAMES
from .route_model import CHANNEL_NAMES
from .worlds import TargetContrast


def _stored_scalar(stored, name: str):
    if name not in stored.files or stored[name].shape != ():
        raise ValueError(f"subset artifact lacks scalar {name}")
    return stored[name].item()


def _stored_vector_group(
    stored,
    names: tuple[str, ...],
    *,
    count: int | None = None,
) -> int:
    for name in names:
        if name not in stored.files or stored[name].ndim != 1:
            raise ValueError(f"subset artifact has invalid {name}")
    lengths = {len(stored[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"subset artifact has inconsistent {names[0]} group")
    actual = lengths.pop()
    if count is not None and actual != count:
        raise ValueError(f"subset artifact has invalid {names[0]} count")
    return actual


def _require_shape(stored, name: str, shape: tuple[int, ...]) -> None:
    if name not in stored.files or stored[name].shape != shape:
        raise ValueError(f"subset artifact has invalid {name} shape")


def _require_finite(values: np.ndarray, expected: np.ndarray, name: str) -> None:
    if values.shape != expected.shape or not np.array_equal(
        np.isfinite(values), expected
    ):
        raise ValueError(f"subset artifact has inconsistent {name} availability")


def _same_number(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-5, atol=1e-7, equal_nan=True))


def validate_native_audit(
    path: str | Path,
    world: NativeWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    metadata: NativeAuditMetadata,
) -> None:
    """Reject incomplete or inconsistent artifacts before treating them as done."""

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
        "response_start": world.response_start,
        "edge_coverage": metadata.coverage,
        "carrier_scope": metadata.carrier_scope,
        "query_chunk": metadata.query_chunk,
        "route_budget_edges_per_head": metadata.route_budget.edges_per_head,
        "route_budget_max_rows": metadata.route_budget.max_rows,
        "route_budget_root_candidates": metadata.route_budget.root_candidates,
        "route_budget_hub_candidates": metadata.route_budget.hub_candidates,
        "route_budget_corridor_edges": metadata.route_budget.corridor_edges,
        "route_budget_confirm": metadata.route_budget.confirm,
        "local_window": metadata.local_window,
        "edge_save_limit": metadata.saved_edges,
    }
    with np.load(Path(path), allow_pickle=False) as stored:
        for name, value in expected.items():
            if _stored_scalar(stored, name) != value:
                raise ValueError(f"subset artifact {name} does not match")

        required_scalars = (
            "method",
            "analysis_stage",
            "layer_count",
            "edge_saved_count",
            "selected_root_unit_id",
            "selected_root_selection_fallback",
            "selected_root_evaluated",
            "selected_root_confirmed",
            "selected_root_route_mass",
            "selected_root_value_necessity",
            "selected_root_conditional_sufficiency",
            "selected_root_causal_score",
            "native_margin",
            "root_cut_margin",
            "root_value_effect",
            "all_evidence_cut_margin",
            "corridor_edge_count",
            "corridor_necessity",
            "corridor_conditional_rescue",
            "corridor_blocked_rescue",
            "corridor_mediated_rescue",
            "corridor_native_restoration_error",
            "corridor_cut_restoration_error",
            "corridor_restoration_error",
            "corridor_restoration_tolerance",
            "corridor_restoration_valid",
            "corridor_evaluated",
            "corridor_confirmed",
            "carrier_evaluated_count",
            "carrier_any_confirmed",
            "full_chain_evaluated",
            "full_chain_confirmed",
            "target_reanchor_selection_recorded",
            "target_reanchor_policy",
            "target_reanchor_has_event",
            "target_reanchor_is_center",
            "target_reanchor_fallback",
            "target_reanchor_center_position",
            "target_reanchor_window_offset",
            "target_reanchor_layer",
            "target_reanchor_head",
            "target_reanchor_source_kind",
            "target_reanchor_source_position",
            "target_reanchor_source_unit_id",
            "target_reanchor_score",
            "target_reanchor_support",
            "target_reanchor_previous_anchor_fraction",
            "target_reanchor_anchor_fraction",
            "target_reanchor_relative_anchor_rise",
            "target_reanchor_relative_local_fall",
            "target_reanchor_scan_signal",
        )
        values = {name: _stored_scalar(stored, name) for name in required_scalars}
        expected_selection = _target_reanchor_arrays(world, target)
        selection_float_fields = {
            "target_reanchor_score",
            "target_reanchor_previous_anchor_fraction",
            "target_reanchor_anchor_fraction",
            "target_reanchor_relative_anchor_rise",
            "target_reanchor_relative_local_fall",
        }
        for name, expected_value in expected_selection.items():
            actual = values[name]
            equal = (
                np.isclose(float(actual), float(expected_value), atol=1e-6)
                if name in selection_float_fields
                else actual == expected_value
            )
            if not equal:
                raise ValueError(
                    f"subset artifact {name} disagrees with target selection"
                )

        world_arrays = {
            "token_ids": world.token_ids.detach().cpu().numpy(),
            "token_unit_id": world.units.token_unit_id.detach().cpu().numpy(),
            "unit_name": np.asarray(world.units.name),
            "unit_kind": np.asarray(world.units.kind),
            "evidence_unit_id": np.asarray(world.evidence_unit_id, dtype=np.int32),
        }
        for name, expected_array in world_arrays.items():
            if name not in stored.files:
                raise ValueError(f"subset artifact lacks required array {name}")
            actual = stored[name]
            if actual.shape != expected_array.shape or not np.array_equal(
                actual.astype(str) if expected_array.dtype.kind in "US" else actual,
                expected_array.astype(str)
                if expected_array.dtype.kind in "US"
                else expected_array,
            ):
                raise ValueError(f"subset artifact {name} does not match")

        token_ids = stored["token_ids"]
        source_count = len(token_ids) - 1
        query = target.query_position
        if (
            len(stored["token_unit_id"]) != source_count
            or not world.response_start - 1 <= query < source_count
            or int(token_ids[query + 1]) != target.positive_token_id
        ):
            raise ValueError("subset artifact has inconsistent world coordinates")

        layer_count = int(values["layer_count"])
        if layer_count < 1:
            raise ValueError("subset artifact has invalid layer_count")
        row_position = stored.get("route_row_position")
        if row_position is None or row_position.ndim != 1 or not len(row_position):
            raise ValueError("subset artifact has invalid route_row_position")
        row_count = len(row_position)
        minimum_row = 0 if metadata.carrier_scope == "all" else world.response_start - 1
        expected_first = world.response_start - 1
        if (
            row_count > metadata.route_budget.max_rows
            or int(row_position[0]) < minimum_row
            or int(row_position[0]) > expected_first
            or int(row_position[-1]) != query
            or not np.array_equal(np.diff(row_position), np.ones(row_count - 1))
        ):
            raise ValueError("subset artifact has invalid route row coordinates")

        head_transport = stored.get("route_head_transport")
        if (
            head_transport is None
            or head_transport.ndim != 4
            or head_transport.shape[0] != layer_count
            or head_transport.shape[2:] != (row_count, len(CHANNEL_NAMES))
        ):
            raise ValueError("subset artifact has invalid route_head_transport shape")
        head_count = head_transport.shape[1]
        if head_count < 1:
            raise ValueError("subset artifact has no attention heads")
        route_tokens = query + 1
        route_shapes = {
            "route_head_action": (layer_count, head_count, row_count, 4),
            "route_head_direct_evidence": (layer_count, head_count, row_count, 2),
            "route_head_integration": (layer_count, head_count, row_count, 4),
            "route_layer_integration": (layer_count, row_count, 4),
            "route_cross_head_vector_coherence": (layer_count, row_count),
            "route_cross_head_functional_agreement": (layer_count, row_count),
            "route_evidence_source_reuse": (layer_count, head_count, route_tokens, 2),
            "route_response_source_reuse": (layer_count, head_count, route_tokens, 2),
            "route_node_origin": (layer_count + 1, route_tokens, 4),
            "route_node_throughput": (layer_count + 1, route_tokens),
            "route_row_total": (layer_count, head_count, row_count),
            "route_row_retained": (layer_count, head_count, row_count),
            "route_stage_displacement": (layer_count, row_count, 3),
            "route_stage_action": (layer_count, row_count, 3),
            "route_module_vector_cosine": (layer_count, row_count),
            "route_module_functional_agreement": (layer_count, row_count),
            "route_state_continuity": (layer_count, row_count),
            "reanchor_anchor_fraction": (layer_count, head_count, row_count),
            "reanchor_switch_delta": (layer_count, head_count, row_count),
            "reanchor_relative_anchor_rise": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_relative_local_fall": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_score": (layer_count, head_count, row_count),
            "reanchor_dominance_flip": (layer_count, head_count, row_count),
            "reanchor_anchor_source_kind": (layer_count, head_count, row_count),
            "reanchor_anchor_source_position": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_anchor_source_unit": (layer_count, head_count, row_count),
            "reanchor_anchor_source_transport": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_anchor_source_evidence_fraction": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_local_source_position": (
                layer_count,
                head_count,
                row_count,
            ),
            "reanchor_local_source_unit": (layer_count, head_count, row_count),
            "reanchor_bucket_attention": (layer_count, head_count, row_count, 4),
            "reanchor_bucket_transport": (layer_count, head_count, row_count, 4),
            "reanchor_bucket_downstream_action": (
                layer_count,
                head_count,
                row_count,
                4,
            ),
            "reanchor_source_position": (layer_count, head_count, row_count, 4),
            "reanchor_source_unit": (layer_count, head_count, row_count, 4),
            "reanchor_source_attention": (layer_count, head_count, row_count, 4),
            "reanchor_source_transport": (layer_count, head_count, row_count, 4),
            "reanchor_source_downstream_action": (
                layer_count,
                head_count,
                row_count,
                4,
            ),
        }
        for name, shape in route_shapes.items():
            _require_shape(stored, name, shape)
        _require_shape(stored, "route_stage_position", (row_count,))
        if not np.array_equal(stored["route_stage_position"], row_position):
            raise ValueError("subset artifact route_stage_position does not match")
        _require_shape(stored, "route_origin_name", (len(CHANNEL_NAMES),))
        if not np.array_equal(
            stored["route_origin_name"].astype(str), np.asarray(CHANNEL_NAMES)
        ):
            raise ValueError("subset artifact route_origin_name does not match")
        _require_shape(stored, "reanchor_source_kind_name", (len(SOURCE_KIND_NAMES),))
        if not np.array_equal(
            stored["reanchor_source_kind_name"].astype(str),
            np.asarray(SOURCE_KIND_NAMES),
        ):
            raise ValueError("subset artifact reanchor source kinds do not match")
        _require_shape(
            stored,
            "reanchor_bucket_name",
            (len(SOURCE_LOCATION_BUCKET_NAMES),),
        )
        if not np.array_equal(
            stored["reanchor_bucket_name"].astype(str),
            np.asarray(SOURCE_LOCATION_BUCKET_NAMES),
        ):
            raise ValueError("subset artifact reanchor bucket names do not match")

        edge_fields = (
            "edge_layer",
            "edge_head",
            "edge_source",
            "edge_target",
            "edge_source_unit",
            "edge_attention_native",
            "edge_attention_root_cut",
            "edge_native_functional_score",
            "edge_root_cut_functional_score",
            "edge_source_evidence_lineage_fraction",
            "edge_evidence_lineage_action",
            "edge_native_message_norm",
            "edge_root_cut_message_norm",
            "edge_delta_message_norm",
            "edge_root_throughput",
            "edge_on_backbone",
            "edge_in_frozen_corridor",
        )
        edge_count = _stored_vector_group(
            stored, edge_fields, count=int(values["edge_saved_count"])
        )
        _require_shape(stored, "route_edge_origin", (edge_count, 4))
        if edge_count:
            valid_edge = (
                (stored["edge_layer"] >= 0)
                & (stored["edge_layer"] < layer_count)
                & (stored["edge_head"] >= 0)
                & (stored["edge_head"] < head_count)
                & (stored["edge_source"] >= 0)
                & (stored["edge_source"] < route_tokens)
                & (stored["edge_target"] >= 0)
                & (stored["edge_target"] < route_tokens)
                & (stored["edge_source"] <= stored["edge_target"])
            )
            if not bool(valid_edge.all()):
                raise ValueError("subset artifact has invalid saved edge coordinates")
            expected_unit = stored["token_unit_id"][stored["edge_source"]]
            if not np.array_equal(stored["edge_source_unit"], expected_unit):
                raise ValueError("subset artifact saved edge source units disagree")
            groups = np.stack(
                (stored["edge_layer"], stored["edge_head"], stored["edge_target"]),
                axis=1,
            )
            _, group_size = np.unique(groups, axis=0, return_counts=True)
            if bool((group_size > 2 * metadata.route_budget.edges_per_head).any()):
                raise ValueError("subset artifact exceeds per-head edge budget")

        backbone_count = _stored_vector_group(
            stored,
            (
                "backbone_node_layer",
                "backbone_node_position",
                "backbone_node_throughput",
            ),
        )
        _require_shape(stored, "backbone_node_origin", (backbone_count, 4))
        _stored_vector_group(
            stored,
            ("backbone_step_throughput", "backbone_step_is_residual"),
            count=max(0, backbone_count - 1),
        )
        if backbone_count not in (1, layer_count + 1):
            raise ValueError("subset artifact has invalid backbone length")
        if (
            int(stored["backbone_node_layer"][-1]) != layer_count
            or int(stored["backbone_node_position"][-1]) != query
        ):
            raise ValueError("subset artifact backbone does not reach the target")

        root_fields = (
            "root_unit_id",
            "root_route_mass",
            "root_signed_action",
            "root_absolute_action",
            "root_functional_agreement",
            "root_selection_score",
            "root_value_necessity",
            "root_conditional_sufficiency",
            "root_causal_score",
            "root_evaluated",
        )
        root_count = _stored_vector_group(stored, root_fields)
        expected_root_count = min(
            len(world.evidence_unit_id), metadata.route_budget.root_candidates
        )
        if root_count != expected_root_count:
            raise ValueError("subset artifact has invalid root candidate count")
        selected_root = int(values["selected_root_unit_id"])
        selected_slot = np.flatnonzero(stored["root_unit_id"] == selected_root)
        if (
            len(selected_slot) != 1
            or selected_root not in world.evidence_unit_id
            or len(np.unique(stored["root_unit_id"])) != root_count
        ):
            raise ValueError("subset artifact has invalid selected root")
        selected_slot = int(selected_slot[0])

        hub_fields = (
            "hub_layer",
            "hub_position",
            "hub_route_mass",
            "hub_signed_action",
            "hub_graph_score",
        )
        hub_count = _stored_vector_group(stored, hub_fields)
        if hub_count > metadata.route_budget.hub_candidates:
            raise ValueError("subset artifact exceeds hub candidate budget")
        if hub_count and not bool(
            (
                (stored["hub_layer"] > 0)
                & (stored["hub_layer"] < layer_count)
                & (stored["hub_position"] >= 0)
                & (stored["hub_position"] < route_tokens)
            ).all()
        ):
            raise ValueError("subset artifact has invalid hub coordinates")
        carrier_fields = (
            "carrier_layer",
            "carrier_position",
            "carrier_source_unit",
            "carrier_route_throughput",
            "carrier_state_delta_norm",
            "carrier_target_score",
            "carrier_necessity",
            "carrier_rescue",
            "carrier_block_effect",
            "carrier_blocked_rescue",
            "carrier_mediated_rescue",
            "carrier_block_tolerance",
            "carrier_confirmed",
        )
        _stored_vector_group(stored, carrier_fields, count=hub_count)
        if not np.array_equal(
            stored["carrier_layer"], stored["hub_layer"]
        ) or not np.array_equal(stored["carrier_position"], stored["hub_position"]):
            raise ValueError("subset artifact carrier plan does not match hubs")

        reanchor_candidate_fields = (
            "reanchor_candidate_layer",
            "reanchor_candidate_head",
            "reanchor_candidate_position",
            "reanchor_candidate_source_kind",
            "reanchor_candidate_source_position",
            "reanchor_candidate_source_unit",
            "reanchor_candidate_source_evidence_fraction",
            "reanchor_candidate_previous_local_source_position",
            "reanchor_candidate_previous_local_source_unit",
            "reanchor_candidate_score",
            "reanchor_candidate_switch_delta",
            "reanchor_candidate_previous_anchor_fraction",
            "reanchor_candidate_anchor_fraction",
            "reanchor_candidate_prompt_transport",
            "reanchor_candidate_other_prompt_transport",
            "reanchor_candidate_remote_response_transport",
            "reanchor_candidate_previous_local_transport",
            "reanchor_candidate_local_transport",
            "reanchor_candidate_long_range_downstream_action",
            "reanchor_candidate_bucket_downstream_action",
            "reanchor_candidate_source_downstream_action",
            "reanchor_candidate_local_downstream_action",
            "reanchor_candidate_current_target_match",
            "reanchor_candidate_long_range_immediate_action",
            "reanchor_candidate_bucket_immediate_action",
            "reanchor_candidate_source_immediate_action",
            "reanchor_candidate_selected_root_integration_budget",
            "reanchor_candidate_selected_root_integration_coherence",
            "reanchor_candidate_selected_root_integration_action",
            "reanchor_candidate_dominance_flip",
            "reanchor_candidate_relative_anchor_rise",
            "reanchor_candidate_relative_local_fall",
            "reanchor_candidate_support",
        )
        reanchor_count = _stored_vector_group(stored, reanchor_candidate_fields)
        if reanchor_count > ROUTE_EVENT_LIMIT:
            raise ValueError("subset artifact exceeds reanchor candidate limit")
        if reanchor_count:
            valid_reanchor = (
                (stored["reanchor_candidate_layer"] >= 0)
                & (stored["reanchor_candidate_layer"] < layer_count)
                & (stored["reanchor_candidate_head"] >= 0)
                & (stored["reanchor_candidate_head"] < head_count)
                & (stored["reanchor_candidate_position"] >= world.response_start)
                & (stored["reanchor_candidate_position"] <= query)
                & (stored["reanchor_candidate_source_position"] >= 0)
                & (stored["reanchor_candidate_source_position"] < route_tokens)
                & (
                    stored["reanchor_candidate_previous_local_source_position"]
                    >= world.response_start
                )
                & (
                    stored["reanchor_candidate_previous_local_source_position"]
                    < route_tokens
                )
            )
            if not bool(valid_reanchor.all()):
                raise ValueError("subset artifact has invalid reanchor coordinates")
            source_kind = stored["reanchor_candidate_source_kind"].astype(str)
            if not bool(np.isin(source_kind, SOURCE_KIND_NAMES[1:]).all()):
                raise ValueError("subset artifact has invalid reanchor source kind")
            current_target = stored["reanchor_candidate_current_target_match"].astype(
                bool
            )
            if not np.array_equal(
                current_target, stored["reanchor_candidate_position"] == query
            ):
                raise ValueError("subset artifact has invalid immediate-action flag")
            for name in (
                "reanchor_candidate_long_range_immediate_action",
                "reanchor_candidate_bucket_immediate_action",
                "reanchor_candidate_source_immediate_action",
            ):
                if bool((np.isfinite(stored[name]) != current_target).any()):
                    raise ValueError(
                        "subset artifact has invalid immediate-action values"
                    )

        selection_recorded = bool(values["target_reanchor_selection_recorded"])
        reanchor_policy = metadata.target_policy in {"reanchor", "reanchor-window"}
        if selection_recorded != reanchor_policy:
            raise ValueError("subset artifact target selection provenance is missing")
        if selection_recorded:
            if (
                str(values["target_reanchor_policy"]) != metadata.target_policy
                or str(values["target_reanchor_scan_signal"])
                != "exact_full_row_message_transport"
            ):
                raise ValueError("subset artifact has invalid reanchor scan identity")
            has_event = bool(values["target_reanchor_has_event"])
            fallback = bool(values["target_reanchor_fallback"])
            is_center = bool(values["target_reanchor_is_center"])
            center = int(values["target_reanchor_center_position"])
            offset = int(values["target_reanchor_window_offset"])
            if has_event:
                layer = int(values["target_reanchor_layer"])
                head = int(values["target_reanchor_head"])
                source_position = int(values["target_reanchor_source_position"])
                source_unit = int(values["target_reanchor_source_unit_id"])
                source_kind = str(values["target_reanchor_source_kind"])
                fractions = np.asarray(
                    [
                        values["target_reanchor_previous_anchor_fraction"],
                        values["target_reanchor_anchor_fraction"],
                        values["target_reanchor_relative_anchor_rise"],
                        values["target_reanchor_relative_local_fall"],
                    ],
                    dtype=float,
                )
                valid_selection = (
                    not fallback
                    and center >= world.response_start
                    and query - center == offset
                    and is_center == (offset == 0)
                    and 0 <= layer < layer_count
                    and 0 <= head < head_count
                    and source_kind in SOURCE_KIND_NAMES[1:]
                    and 0 <= source_position <= center
                    and 0 <= source_unit < len(world.units.name)
                    and int(world.units.token_unit_id[source_position]) == source_unit
                    and float(values["target_reanchor_score"]) > 0
                    and int(values["target_reanchor_support"]) > 0
                    and bool(np.isfinite(fractions).all())
                    and bool(((fractions >= 0) & (fractions <= 1)).all())
                )
                if not valid_selection:
                    raise ValueError("subset artifact has invalid reanchor selection")
                if center <= query:
                    match = (
                        (stored["reanchor_candidate_position"] == center)
                        & (stored["reanchor_candidate_layer"] == layer)
                        & (stored["reanchor_candidate_head"] == head)
                    )
                    selected = np.flatnonzero(match)
                    if len(selected) != 1:
                        raise ValueError(
                            "frozen reanchor center is absent from the timeline"
                        )
                    selected = int(selected[0])
                    candidate_pairs = (
                        ("reanchor_candidate_source_kind", source_kind),
                        ("reanchor_candidate_source_position", source_position),
                        ("reanchor_candidate_source_unit", source_unit),
                        (
                            "reanchor_candidate_support",
                            int(values["target_reanchor_support"]),
                        ),
                    )
                    if any(
                        stored[name][selected] != expected
                        for name, expected in candidate_pairs
                    ):
                        raise ValueError(
                            "frozen reanchor center disagrees with the timeline"
                        )
                    numeric_pairs = (
                        (
                            "reanchor_candidate_score",
                            float(values["target_reanchor_score"]),
                        ),
                        (
                            "reanchor_candidate_previous_anchor_fraction",
                            float(values["target_reanchor_previous_anchor_fraction"]),
                        ),
                        (
                            "reanchor_candidate_anchor_fraction",
                            float(values["target_reanchor_anchor_fraction"]),
                        ),
                        (
                            "reanchor_candidate_relative_anchor_rise",
                            float(values["target_reanchor_relative_anchor_rise"]),
                        ),
                        (
                            "reanchor_candidate_relative_local_fall",
                            float(values["target_reanchor_relative_local_fall"]),
                        ),
                    )
                    if any(
                        not np.isclose(stored[name][selected], expected, atol=1e-6)
                        for name, expected in numeric_pairs
                    ):
                        raise ValueError(
                            "frozen reanchor score disagrees with the timeline"
                        )
            elif (
                is_center
                or center != -1
                or offset != 0
                or int(values["target_reanchor_layer"]) != -1
                or int(values["target_reanchor_head"]) != -1
                or str(values["target_reanchor_source_kind"]) != "none"
                or int(values["target_reanchor_source_position"]) != -1
                or int(values["target_reanchor_source_unit_id"]) != -1
                or float(values["target_reanchor_score"]) != 0
                or int(values["target_reanchor_support"]) != 0
            ):
                raise ValueError("subset artifact has invalid no-event selection")
        elif (
            str(values["target_reanchor_policy"]) != "none"
            or str(values["target_reanchor_scan_signal"]) != "none"
            or bool(values["target_reanchor_has_event"])
        ):
            raise ValueError("subset artifact has stray reanchor target metadata")

        corridor_fields = (
            "frozen_corridor_edge_index",
            "frozen_corridor_layer",
            "frozen_corridor_head",
            "frozen_corridor_source",
            "frozen_corridor_target",
            "frozen_corridor_source_unit",
            "frozen_corridor_attention_native",
            "frozen_corridor_attention_root_cut",
            "frozen_corridor_native_functional_score",
            "frozen_corridor_root_cut_functional_score",
            "frozen_corridor_source_evidence_lineage_fraction",
            "frozen_corridor_evidence_lineage_action",
            "frozen_corridor_native_message_norm",
            "frozen_corridor_root_cut_message_norm",
            "frozen_corridor_delta_message_norm",
            "frozen_corridor_root_throughput",
        )
        corridor_count = int(values["corridor_edge_count"])
        _stored_vector_group(stored, corridor_fields, count=corridor_count)
        if corridor_count > metadata.route_budget.corridor_edges:
            raise ValueError("subset artifact exceeds corridor edge budget")
        if corridor_count:
            corridor_valid = (
                (stored["frozen_corridor_layer"] >= 0)
                & (stored["frozen_corridor_layer"] < layer_count)
                & (stored["frozen_corridor_head"] >= 0)
                & (stored["frozen_corridor_head"] < head_count)
                & (stored["frozen_corridor_source"] >= 0)
                & (stored["frozen_corridor_source"] < route_tokens)
                & (stored["frozen_corridor_target"] >= 0)
                & (stored["frozen_corridor_target"] < route_tokens)
                & (stored["frozen_corridor_source"] <= stored["frozen_corridor_target"])
            )
            if not bool(corridor_valid.all()):
                raise ValueError("subset artifact has invalid corridor coordinates")
            corridor_index = stored["frozen_corridor_edge_index"]
            if bool((corridor_index < 0).any()) or len(
                np.unique(corridor_index)
            ) != len(corridor_index):
                raise ValueError("subset artifact has invalid frozen corridor indices")
            expected_unit = stored["token_unit_id"][stored["frozen_corridor_source"]]
            if not np.array_equal(stored["frozen_corridor_source_unit"], expected_unit):
                raise ValueError("subset artifact corridor source units disagree")
        membership = stored["edge_in_frozen_corridor"].astype(bool)
        if int(membership.sum()) != corridor_count:
            raise ValueError("subset artifact omits a frozen corridor edge")
        saved_corridor = {
            tuple(map(int, row))
            for row in np.stack(
                (
                    stored["edge_layer"][membership],
                    stored["edge_head"][membership],
                    stored["edge_source"][membership],
                    stored["edge_target"][membership],
                ),
                axis=1,
            )
        }
        frozen_corridor = {
            tuple(map(int, row))
            for row in np.stack(
                (
                    stored["frozen_corridor_layer"],
                    stored["frozen_corridor_head"],
                    stored["frozen_corridor_source"],
                    stored["frozen_corridor_target"],
                ),
                axis=1,
            )
        }
        if saved_corridor != frozen_corridor or len(frozen_corridor) != corridor_count:
            raise ValueError("subset artifact frozen corridor tables disagree")

        confirm = metadata.route_budget.confirm
        analysis_stage = str(values["analysis_stage"])
        if analysis_stage != ("confirmed" if confirm else "discovery"):
            raise ValueError("subset artifact analysis_stage does not match budget")
        root_evaluated = stored["root_evaluated"].astype(bool)
        selected_evaluated = bool(values["selected_root_evaluated"])
        if (
            selected_evaluated != bool(root_evaluated[selected_slot])
            or selected_evaluated != confirm
            or int(root_evaluated.sum()) != int(confirm)
        ):
            raise ValueError("subset artifact has inconsistent root evaluation state")
        selected_confirmed = bool(values["selected_root_confirmed"])
        if selected_confirmed and not selected_evaluated:
            raise ValueError("subset artifact confirms an unevaluated root")
        selected_scalar_fields = (
            ("selected_root_route_mass", "root_route_mass"),
            ("selected_root_value_necessity", "root_value_necessity"),
            (
                "selected_root_conditional_sufficiency",
                "root_conditional_sufficiency",
            ),
            ("selected_root_causal_score", "root_causal_score"),
        )
        if any(
            not _same_number(
                float(values[scalar_name]), float(stored[array_name][selected_slot])
            )
            for scalar_name, array_name in selected_scalar_fields
        ):
            raise ValueError("subset artifact selected-root summaries disagree")
        root_exact = (
            stored["root_conditional_sufficiency"],
            stored["root_causal_score"],
        )
        for name, array in zip(
            ("root_conditional_sufficiency", "root_causal_score"),
            root_exact,
            strict=True,
        ):
            _require_finite(array, root_evaluated, name)
        _require_finite(
            stored["root_value_necessity"],
            root_evaluated,
            "root_value_necessity",
        )
        if np.isfinite(float(values["all_evidence_cut_margin"])) != confirm:
            raise ValueError("subset artifact all-evidence cut availability disagrees")
        native_margin = float(values["native_margin"])
        root_cut_margin = float(values["root_cut_margin"])
        root_value_effect = float(values["root_value_effect"])
        if not all(
            np.isfinite((native_margin, root_cut_margin, root_value_effect))
        ) or not _same_number(native_margin - root_cut_margin, root_value_effect):
            raise ValueError("subset artifact root-cut margin values disagree")

        corridor_evaluated = bool(values["corridor_evaluated"])
        if corridor_evaluated != bool(confirm and corridor_count):
            raise ValueError(
                "subset artifact has inconsistent corridor evaluation state"
            )
        corridor_exact_names = (
            "corridor_necessity",
            "corridor_conditional_rescue",
            "corridor_blocked_rescue",
            "corridor_mediated_rescue",
            "corridor_native_restoration_error",
            "corridor_cut_restoration_error",
            "corridor_restoration_error",
        )
        if any(
            np.isfinite(float(values[name])) != corridor_evaluated
            for name in corridor_exact_names
        ):
            raise ValueError("subset artifact has inconsistent corridor exact values")
        restoration_valid = bool(values["corridor_restoration_valid"])
        corridor_confirmed = bool(values["corridor_confirmed"])
        if (restoration_valid and not corridor_evaluated) or (
            corridor_confirmed and not corridor_evaluated
        ):
            raise ValueError("subset artifact confirms an unevaluated corridor")
        if corridor_evaluated:
            if not _same_number(
                float(values["corridor_conditional_rescue"])
                - float(values["corridor_blocked_rescue"]),
                float(values["corridor_mediated_rescue"]),
            ):
                raise ValueError("subset artifact corridor mediation disagrees")
            restoration_error = max(
                float(values["corridor_native_restoration_error"]),
                float(values["corridor_cut_restoration_error"]),
            )
            if not _same_number(
                restoration_error, float(values["corridor_restoration_error"])
            ) or restoration_valid != (
                restoration_error <= float(values["corridor_restoration_tolerance"])
            ):
                raise ValueError("subset artifact corridor restoration disagrees")

        carrier_evaluated = int(values["carrier_evaluated_count"])
        expected_carrier_evaluated = int(bool(confirm and hub_count))
        if carrier_evaluated != expected_carrier_evaluated:
            raise ValueError(
                "subset artifact has inconsistent carrier evaluation count"
            )
        carrier_available = np.arange(hub_count) < carrier_evaluated
        for name in (
            "carrier_necessity",
            "carrier_rescue",
            "carrier_block_effect",
            "carrier_blocked_rescue",
            "carrier_mediated_rescue",
        ):
            _require_finite(stored[name], carrier_available, name)
        carrier_confirmed = stored["carrier_confirmed"].astype(bool)
        if bool(carrier_confirmed[carrier_evaluated:].any()):
            raise ValueError("subset artifact confirms an unevaluated carrier")
        if carrier_evaluated and not np.allclose(
            stored["carrier_rescue"][:carrier_evaluated]
            - stored["carrier_blocked_rescue"][:carrier_evaluated],
            stored["carrier_mediated_rescue"][:carrier_evaluated],
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError("subset artifact carrier mediation disagrees")
        carrier_any_confirmed = bool(values["carrier_any_confirmed"])
        if carrier_any_confirmed != bool(carrier_confirmed.any()):
            raise ValueError("subset artifact carrier confirmation summary disagrees")

        full_chain_evaluated = bool(values["full_chain_evaluated"])
        expected_chain_evaluated = bool(
            selected_evaluated and corridor_evaluated and carrier_evaluated
        )
        if full_chain_evaluated != expected_chain_evaluated:
            raise ValueError("subset artifact full-chain evaluation state disagrees")
        full_chain_confirmed = bool(values["full_chain_confirmed"])
        expected_chain_confirmed = bool(
            full_chain_evaluated
            and selected_confirmed
            and corridor_confirmed
            and carrier_any_confirmed
        )
        if full_chain_confirmed != expected_chain_confirmed:
            raise ValueError("subset artifact full-chain confirmation disagrees")
