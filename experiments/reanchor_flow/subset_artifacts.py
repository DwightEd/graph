"""Compact, label-free persistence for native subset corridor audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .artifacts import save_result
from .flow import FlowSignal
from .native import NativeTargetAudit
from .native_world import NativeWorld
from .worlds import TargetContrast

AUDIT_SCHEMA = 1
MANIFEST_SCHEMA = 1


def canonical_capture_config(config: Mapping[str, object]) -> str:
    """Return the stable representation bound into every resumable artifact."""

    return json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def capture_config_sha256(config: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_capture_config(config).encode("utf-8")).hexdigest()


def _validate_config_context(
    config: Mapping[str, object],
    *,
    model_id: str,
    model_dtype: str,
    tokenizer_id: str,
    split: str,
    signal: FlowSignal | str,
    target_policy: str,
    coverage: float,
    carrier_scope: str,
    query_chunk: int,
    root_screen_limit: int,
    carrier_limit: int,
    saved_edges: int,
) -> None:
    expected = {
        "model": model_id,
        "model_dtype": model_dtype,
        "tokenizer": tokenizer_id,
        "split": split,
        "flow_signal": FlowSignal(signal).value,
        "target_policy": target_policy,
        "edge_coverage": coverage,
        "carrier_scope": carrier_scope,
        "query_chunk": query_chunk,
        "root_screen_limit": root_screen_limit,
        "carrier_limit": carrier_limit,
        "saved_edges": saved_edges,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"capture configuration disagrees on {name}")
    for name in ("dataset_manifest_sha256", "source_info_sha256"):
        value = config.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"capture configuration lacks a valid {name}")


def _top_support_edges(audit: NativeTargetAudit, limit: int) -> torch.Tensor:
    support = audit.throughput.edge > 0
    candidate = torch.nonzero(support, as_tuple=False).flatten()
    if limit == 0 or not len(candidate):
        return candidate[:0]
    if len(candidate) <= limit:
        return candidate
    score = audit.throughput.edge.index_select(0, candidate)
    selected = torch.topk(score, k=limit, sorted=True).indices
    return candidate.index_select(0, selected)


def _root_arrays(audit: NativeTargetAudit) -> dict[str, np.ndarray]:
    roots = audit.roots
    return {
        "root_unit_id": np.asarray([item.unit_id for item in roots], dtype=np.int32),
        "root_route_mass": np.asarray(
            [item.route_mass for item in roots], dtype=np.float32
        ),
        "root_functional_score": np.asarray(
            [item.gradient_score for item in roots], dtype=np.float32
        ),
        "root_value_necessity": np.asarray(
            [item.necessity for item in roots], dtype=np.float32
        ),
        "root_conditional_sufficiency": np.asarray(
            [item.sufficiency for item in roots], dtype=np.float32
        ),
        "root_causal_score": np.asarray(
            [item.causal_score for item in roots], dtype=np.float32
        ),
        "root_evaluated": np.asarray([item.evaluated for item in roots], dtype=bool),
    }


def _carrier_arrays(
    world: NativeWorld,
    audit: NativeTargetAudit,
) -> dict[str, np.ndarray]:
    carriers = audit.carriers
    return {
        "carrier_layer": np.asarray([item.layer for item in carriers], dtype=np.int16),
        "carrier_position": np.asarray(
            [item.position for item in carriers], dtype=np.int32
        ),
        "carrier_source_unit": np.asarray(
            [int(world.units.token_unit_id[item.position]) for item in carriers],
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


def _stage_arrays(audit: NativeTargetAudit) -> dict[str, object]:
    stages = audit.flow.stages
    if stages is None:
        return {
            "stage_position": np.empty(0, dtype=np.int32),
            "state_delta_norm": np.empty((0, 0), dtype=np.float32),
            "state_target_score": np.empty((0, 0), dtype=np.float32),
            "attention_write_delta_norm": np.empty((0, 0), dtype=np.float32),
            "attention_write_target_score": np.empty((0, 0), dtype=np.float32),
            "mlp_write_delta_norm": np.empty((0, 0), dtype=np.float32),
            "mlp_write_target_score": np.empty((0, 0), dtype=np.float32),
        }
    return {
        "stage_position": stages.position,
        "state_delta_norm": stages.state_delta_norm,
        "state_target_score": stages.state_score,
        "attention_write_delta_norm": stages.attention_delta_norm,
        "attention_write_target_score": stages.attention_score,
        "mlp_write_delta_norm": stages.mlp_delta_norm,
        "mlp_write_target_score": stages.mlp_score,
    }


def compact_native_audit_arrays(
    world: NativeWorld,
    audit: NativeTargetAudit,
    *,
    dataset_sample_id: str,
    source_id: str,
    split: str,
    task_type: str,
    generator_model: str,
    model_id: str,
    model_dtype: str,
    target_policy: str,
    target_rank: int,
    coverage: float,
    carrier_scope: str,
    query_chunk: int,
    root_screen_limit: int,
    carrier_limit: int,
    saved_edges: int,
    world_sha256: str,
    capture_config: Mapping[str, object],
) -> dict[str, object]:
    """Keep scalar graph evidence after all exact reruns have completed."""

    flow = audit.flow
    _validate_config_context(
        capture_config,
        model_id=model_id,
        model_dtype=model_dtype,
        tokenizer_id=world.tokenizer_id,
        split=split,
        signal=flow.signal,
        target_policy=target_policy,
        coverage=coverage,
        carrier_scope=carrier_scope,
        query_chunk=query_chunk,
        root_screen_limit=root_screen_limit,
        carrier_limit=carrier_limit,
        saved_edges=saved_edges,
    )
    effect = audit.effect
    edge_index = _top_support_edges(audit, saved_edges)
    edges = flow.edges.select(edge_index)
    edge_throughput = audit.throughput.edge.index_select(0, edge_index)
    support_mask = audit.throughput.edge > 0
    support_mass = float(audit.throughput.edge[support_mask].sum())
    saved_mass = float(edge_throughput.sum())
    root_cut_margin = flow.corrupt_margin
    config_json = canonical_capture_config(capture_config)
    config_sha256 = capture_config_sha256(capture_config)
    arrays: dict[str, object] = {
        "subset_audit_schema": AUDIT_SCHEMA,
        "artifact_complete": 1,
        "capture_config_json": config_json,
        "capture_config_sha256": config_sha256,
        "world_sha256": world_sha256,
        "dataset_manifest_sha256": str(capture_config["dataset_manifest_sha256"]),
        "source_info_sha256": str(capture_config["source_info_sha256"]),
        "method": "native evidence-to-target causal corridor",
        "world_kind": "native_source_value_message_cut",
        "claim_scope": "observed-target dependence under a source Value-message cut",
        "factual_correctness_identified": 0,
        "labels_used_for_capture": 0,
        "dataset_sample_id": dataset_sample_id,
        "sample_id": world.sample_id,
        "source_id": source_id,
        "split": split,
        "task_type": task_type,
        "generator_model": generator_model,
        "tokenizer_id": world.tokenizer_id,
        "model_id": model_id,
        "model_dtype": model_dtype,
        "target_selection_policy": target_policy,
        "target_selection_rank": target_rank,
        "response_start": world.response_start,
        "query_position": flow.target.query_position,
        "prediction_position": flow.target.query_position + 1,
        "positive_token_id": flow.target.positive_token_id,
        "negative_token_id": flow.target.negative_token_id,
        "contrast_origin": flow.target.origin,
        "flow_signal": flow.signal.value,
        "transport_score_semantics": (
            "raw_native_attention"
            if flow.signal is FlowSignal.ATTENTION
            else "native_post_WO_message_l2_norm"
        ),
        "functional_score_semantics": ("native_gradient_dot_native_pre_WO_AV_message"),
        "root_cut_functional_score_semantics": (
            "frozen_native_gradient_dot_root_cut_pre_WO_AV_message"
        ),
        "residual_transition_semantics": (
            "head_count_identity_vs_attention_mass"
            if flow.signal is FlowSignal.ATTENTION
            else "native_residual_l2_norm_vs_message_l2_budget"
        ),
        "source_cut_semantics": (
            "delete source Value messages in every layer and query; "
            "do not directly mask Q/K, but source self-message deletion makes "
            "later state and Q/K evolve in the Value-cut world"
        ),
        "edge_coverage": coverage,
        "carrier_scope": carrier_scope,
        "query_chunk": query_chunk,
        "root_screen_limit": root_screen_limit,
        "carrier_limit": carrier_limit,
        "layer_count": flow.clean_cache.layer_count,
        "head_count": flow.row_total.shape[1],
        "head_dim": flow.edges.clean_code.shape[1],
        "hidden_size": flow.clean_cache.final_hidden.shape[1],
        "token_ids": world.token_ids,
        "token_unit_id": world.units.token_unit_id,
        "unit_name": np.asarray(world.units.name),
        "unit_kind": np.asarray(world.units.kind),
        "evidence_unit_id": np.asarray(world.evidence_unit_id, dtype=np.int32),
        "selected_root_unit_id": audit.selected_root_unit_id,
        "selected_root_confirmed": audit.selected_root_confirmed,
        "root_value_confirmed": audit.selected_root_confirmed,
        "selected_root_route_mass": audit.throughput.root_mass,
        "selected_root_value_necessity": (audit.selected_root_effect.necessity),
        "selected_root_conditional_sufficiency": (
            audit.selected_root_effect.sufficiency
        ),
        "selected_root_causal_score": (audit.selected_root_effect.causal_score),
        "causal_effect_tolerance": effect.restoration_tolerance,
        "source_unit_route_mass": audit.throughput.unit_mass,
        "transport_source_unit_route_mass": (audit.transport_throughput.unit_mass),
        "support_token_root_mass": audit.throughput.reverse_visit[0],
        "transport_token_root_mass": (audit.transport_throughput.reverse_visit[0]),
        "row_position": flow.row_position,
        "row_total_transport": flow.row_total,
        "row_retained_transport": flow.row_retained,
        "row_residual_weight": flow.residual_weight,
        "row_message_budget": flow.aggregation.message_budget,
        "row_net_message_norm": flow.aggregation.net_message_norm,
        "row_message_coherence": flow.aggregation.coherence,
        "row_signed_functional_score": flow.aggregation.signed_score,
        "row_positive_functional_score": flow.aggregation.positive_score,
        "row_negative_functional_score": flow.aggregation.negative_score,
        "edge_candidate_count": flow.edges.count,
        "corridor_edge_count": audit.corridor.count,
        "edge_saved_count": len(edge_index),
        "edge_save_limit": saved_edges,
        "edge_save_rule": "top positive-functional root throughput",
        "edge_saved_corridor_mass": saved_mass,
        "edge_total_corridor_mass": support_mass,
        "edge_saved_corridor_fraction": (
            saved_mass / support_mass if support_mass > 0 else 0.0
        ),
        "edge_payload_saved": 0,
        "edge_layer": edges.layer,
        "edge_head": edges.head,
        "edge_source": edges.source,
        "edge_target": edges.target,
        "edge_source_unit": edges.source_unit,
        "edge_attention_native": edges.attention_clean,
        "edge_attention_root_cut": edges.attention_corrupt,
        "edge_transport_score": edges.score,
        "edge_native_functional_score": edges.clean_target_score,
        "edge_root_cut_native_gradient_projection": edges.corrupt_target_score,
        "edge_native_message_norm": edges.clean_message_norm,
        "edge_root_cut_message_norm": edges.corrupt_message_norm,
        "edge_delta_message_norm": edges.delta_message_norm,
        "edge_transition_probability": (
            audit.throughput.edge_probability.index_select(0, edge_index)
        ),
        "edge_root_throughput": edge_throughput,
        "residual_transition_probability": (audit.throughput.residual_probability),
        "reverse_node_visit": audit.throughput.reverse_visit,
        "root_conditioned_node_throughput": audit.throughput.node,
        "native_margin": flow.clean_margin,
        "root_cut_margin": root_cut_margin,
        "root_value_effect": flow.clean_margin - root_cut_margin,
        "all_evidence_cut_margin": audit.all_evidence_cut_margin,
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
        "corridor_native_rescue_confirmed": audit.corridor_confirmed,
        # A carrier can pass its local patch/block diagnostic even when the
        # complete root-to-target corridor does not.  Keep that diagnostic,
        # but expose a carrier-mediated evaluation outcome only for a fully
        # confirmed terminal corridor.
        "carrier_any_confirmed": any(item.confirmed for item in audit.carriers),
        "carrier_value_mediated": (
            audit.corridor_confirmed and any(item.confirmed for item in audit.carriers)
        ),
        "full_chain_confirmed": (
            audit.corridor_confirmed and any(item.confirmed for item in audit.carriers)
        ),
    }
    arrays.update(_root_arrays(audit))
    arrays.update(_carrier_arrays(world, audit))
    arrays.update(_stage_arrays(audit))
    return arrays


def save_compact_native_audit(
    path: str | Path,
    world: NativeWorld,
    audit: NativeTargetAudit,
    **metadata,
) -> None:
    save_result(path, compact_native_audit_arrays(world, audit, **metadata))


def validate_compact_native_audit(
    path: str | Path,
    *,
    dataset_sample_id: str,
    sample_id: str,
    source_id: str,
    split: str,
    task_type: str,
    generator_model: str,
    tokenizer_id: str,
    world_sha256: str,
    target: TargetContrast,
    target_rank: int,
    signal: FlowSignal | str,
    model_id: str,
    model_dtype: str,
    capture_config: Mapping[str, object],
) -> None:
    """Reject any resumed NPZ not bound to this manifest and frozen target."""

    signal = FlowSignal(signal)
    _validate_config_context(
        capture_config,
        model_id=model_id,
        model_dtype=model_dtype,
        tokenizer_id=tokenizer_id,
        split=split,
        signal=signal,
        target_policy=str(capture_config.get("target_policy")),
        coverage=float(capture_config.get("edge_coverage", float("nan"))),
        carrier_scope=str(capture_config.get("carrier_scope")),
        query_chunk=int(capture_config.get("query_chunk", -1)),
        root_screen_limit=int(capture_config.get("root_screen_limit", -1)),
        carrier_limit=int(capture_config.get("carrier_limit", -1)),
        saved_edges=int(capture_config.get("saved_edges", -1)),
    )
    expected_config_json = canonical_capture_config(capture_config)
    expected_config_sha256 = capture_config_sha256(capture_config)
    with np.load(Path(path), allow_pickle=False) as stored:
        required = {
            "subset_audit_schema",
            "artifact_complete",
            "capture_config_json",
            "capture_config_sha256",
            "world_sha256",
            "dataset_manifest_sha256",
            "source_info_sha256",
            "dataset_sample_id",
            "sample_id",
            "source_id",
            "split",
            "task_type",
            "generator_model",
            "tokenizer_id",
            "query_position",
            "prediction_position",
            "positive_token_id",
            "negative_token_id",
            "contrast_origin",
            "flow_signal",
            "model_id",
            "model_dtype",
            "target_selection_policy",
            "target_selection_rank",
            "edge_coverage",
            "carrier_scope",
            "query_chunk",
            "root_screen_limit",
            "carrier_limit",
            "edge_save_limit",
            "labels_used_for_capture",
            "token_ids",
            "response_start",
            "selected_root_confirmed",
            "corridor_confirmed",
            "corridor_restoration_valid",
            "carrier_any_confirmed",
            "carrier_value_mediated",
            "full_chain_confirmed",
            "root_value_effect",
            "corridor_necessity",
            "corridor_conditional_rescue",
            "corridor_mediated_rescue",
        }
        missing = required - set(stored.files)
        if missing:
            raise ValueError(
                f"subset artifact is incomplete: {', '.join(sorted(missing))}"
            )
        if int(stored["subset_audit_schema"]) != AUDIT_SCHEMA:
            raise ValueError("unsupported subset audit schema")
        if int(stored["artifact_complete"]) != 1:
            raise ValueError("subset artifact is not marked complete")

        expected_scalars = {
            "dataset_sample_id": dataset_sample_id,
            "sample_id": sample_id,
            "source_id": source_id,
            "split": split,
            "task_type": task_type,
            "generator_model": generator_model,
            "tokenizer_id": tokenizer_id,
            "world_sha256": world_sha256,
            "query_position": target.query_position,
            "prediction_position": target.query_position + 1,
            "positive_token_id": target.positive_token_id,
            "negative_token_id": target.negative_token_id,
            "contrast_origin": target.origin,
            "flow_signal": signal.value,
            "model_id": model_id,
            "model_dtype": model_dtype,
            "target_selection_policy": capture_config["target_policy"],
            "target_selection_rank": target_rank,
            "edge_coverage": capture_config["edge_coverage"],
            "carrier_scope": capture_config["carrier_scope"],
            "query_chunk": capture_config["query_chunk"],
            "root_screen_limit": capture_config["root_screen_limit"],
            "carrier_limit": capture_config["carrier_limit"],
            "edge_save_limit": capture_config["saved_edges"],
            "dataset_manifest_sha256": capture_config["dataset_manifest_sha256"],
            "source_info_sha256": capture_config["source_info_sha256"],
            "capture_config_json": expected_config_json,
            "capture_config_sha256": expected_config_sha256,
        }
        for name, expected in expected_scalars.items():
            value = stored[name]
            if value.shape != () or value.item() != expected:
                raise ValueError(
                    f"subset artifact {name} does not match the frozen manifest"
                )
        if bool(stored["labels_used_for_capture"]):
            raise ValueError("subset artifact violates the label firewall")

        carrier_any_confirmed = bool(stored["carrier_any_confirmed"])
        corridor_confirmed = bool(stored["corridor_confirmed"])
        full_chain_confirmed = corridor_confirmed and carrier_any_confirmed
        if bool(stored["carrier_value_mediated"]) != full_chain_confirmed:
            raise ValueError(
                "subset artifact carrier_value_mediated is not a full-chain result"
            )
        if bool(stored["full_chain_confirmed"]) != full_chain_confirmed:
            raise ValueError(
                "subset artifact full_chain_confirmed disagrees with its diagnostics"
            )

        token_ids = stored["token_ids"]
        prediction = target.query_position + 1
        if (
            token_ids.ndim != 1
            or not 0 <= prediction < len(token_ids)
            or int(token_ids[prediction]) != target.positive_token_id
        ):
            raise ValueError(
                "subset artifact does not contain the frozen observed token"
            )
