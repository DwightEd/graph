from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.artifact_payload import (
    _route_backbone,
    native_audit_arrays,
    save_native_audit,
)
from experiments.reanchor_flow.artifact_schema import (
    AUDIT_SCHEMA,
    METHOD_VERSION,
    NativeAuditMetadata,
)
from experiments.reanchor_flow.artifact_validation import validate_native_audit
from experiments.reanchor_flow.corridor import (
    CarrierEffect,
    CorridorEffect,
    RootEffect,
)
from experiments.reanchor_flow.flow import FlowSignal, SourceLocationBuckets
from experiments.reanchor_flow.native_world import TargetReanchorSelection
from experiments.reanchor_flow.route_plan import (
    AuditPlan,
    HubCandidate,
    RootRouteScore,
    RouteBudget,
)
from experiments.reanchor_flow.worlds import TargetContrast


def _fixture() -> tuple[SimpleNamespace, SimpleNamespace, TargetContrast]:
    target = TargetContrast(2, 13, 9, "observed_vs_runner")
    world = SimpleNamespace(
        sample_id="sample-1",
        tokenizer_id="tokenizer-1",
        token_ids=torch.tensor([10, 11, 12, 13]),
        response_start=2,
        units=SimpleNamespace(
            token_unit_id=torch.tensor([0, 1, 2]),
            name=("evidence", "prompt", "response"),
            kind=("evidence", "other_prompt", "response"),
        ),
        evidence_unit_id=(0,),
    )
    edges = SimpleNamespace(
        count=3,
        layer=torch.tensor([0, 0, 1]),
        head=torch.tensor([0, 1, 0]),
        source=torch.tensor([0, 1, 1]),
        target=torch.tensor([1, 2, 2]),
        source_unit=torch.tensor([0, 1, 1]),
        attention_clean=torch.tensor([0.4, 0.2, 0.7]),
        attention_corrupt=torch.tensor([0.1, 0.2, 0.3]),
        clean_target_score=torch.tensor([0.5, -0.1, 0.8]),
        corrupt_target_score=torch.tensor([0.1, -0.1, 0.2]),
        clean_message_norm=torch.tensor([0.6, 0.3, 0.9]),
        corrupt_message_norm=torch.tensor([0.2, 0.3, 0.4]),
        delta_message_norm=torch.tensor([0.4, 0.0, 0.5]),
    )
    edges.select = lambda index: SimpleNamespace(
        **{
            name: value.index_select(0, index)
            for name, value in vars(edges).items()
            if isinstance(value, torch.Tensor)
        }
    )

    action = torch.zeros(2, 2, 2, 4)
    action[0, 0, 0, 0] = 0.5
    action[1, 0, 1, 0] = 0.8
    transport = torch.zeros_like(action)
    transport[..., 0] = action[..., 0].abs()
    direct = torch.zeros(2, 2, 2, 2)
    direct[0, 0, 0] = torch.tensor([0.5, 0.5])
    node_register = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.2, 0.0, 0.0, 0.8], [0.6, 0.2, 0.0, 0.2], [0.0, 0.0, 0.5, 0.5]],
            [[0.1, 0.0, 0.0, 0.9], [0.3, 0.2, 0.0, 0.5], [0.5, 0.0, 0.3, 0.2]],
        ]
    )
    dynamics = SimpleNamespace(
        root_unit_id=0,
        local_window=3,
        row_position=torch.tensor([1, 2]),
        node_register=node_register,
        edge_register=torch.tensor(
            [
                [0.4, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.3, 0.0, 0.4, 0.0],
            ]
        ),
        head_transport=transport,
        head_gradient_action=action,
        head_direct_evidence=direct,
        head_integration=torch.rand(2, 2, 2, 4),
        layer_integration=torch.rand(2, 2, 4),
        cross_head_vector_coherence=torch.rand(2, 2),
        cross_head_functional_agreement=torch.rand(2, 2),
        evidence_source_reuse=torch.rand(2, 2, 3, 2),
        response_source_reuse=torch.rand(2, 2, 3, 2),
        stage_position=torch.tensor([1, 2]),
        stage_displacement=torch.rand(2, 2, 3),
        stage_gradient_action=torch.rand(2, 2, 3),
        attention_mlp_vector_cosine=torch.rand(2, 2),
        attention_mlp_functional_agreement=torch.rand(2, 2),
        state_continuity=torch.rand(2, 2),
    )
    effect = CorridorEffect(
        edge_count=3,
        pair_effect=0.6,
        necessity=0.5,
        sufficiency=0.45,
        blocked_sufficiency=0.02,
        mediated_sufficiency=0.43,
        clean_restoration_error=0.0,
        corrupt_restoration_error=0.0,
        restoration_error=0.0,
        restoration_tolerance=1e-5,
        restoration_valid=True,
    )
    root = RootEffect(0, 0.8, 0.7, 0.5, 0.4, 0.4, True)
    carrier = CarrierEffect(1, 2, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.01, 0.39, 1e-5, True)
    source_shape = (2, 2, 2, 4)
    source_location = SourceLocationBuckets(
        local_window=3,
        attention=torch.zeros(source_shape),
        transport=torch.zeros(source_shape),
        downstream_action=torch.zeros(source_shape),
        source_position=torch.full(source_shape, -1, dtype=torch.int32),
        source_unit_id=torch.full(source_shape, -1, dtype=torch.int32),
        source_attention=torch.zeros(source_shape),
        source_transport=torch.zeros(source_shape),
        source_downstream_action=torch.zeros(source_shape),
    )
    flow = SimpleNamespace(
        target=target,
        signal=FlowSignal.MESSAGE,
        edges=edges,
        clean_margin=0.8,
        corrupt_margin=0.2,
        row_total=torch.ones(2, 2, 2),
        row_retained=torch.full((2, 2, 2), 0.75),
        source_location=source_location,
    )
    throughput = SimpleNamespace(
        edge=torch.tensor([0.4, 0.2, 0.7]),
        root_mass=0.8,
        residual_probability=torch.zeros(2, 3),
        reverse_visit=torch.ones(3, 3),
        node=torch.tensor(
            [
                [0.4, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    budget = RouteBudget(
        edges_per_head=2,
        max_rows=8,
        root_candidates=4,
        hub_candidates=2,
        corridor_edges=8,
        confirm=True,
    )
    plan = AuditPlan(
        budget=budget,
        roots=(RootRouteScore(0, 0.8, 0.7, 0.9, 0.7 / 0.9, 0.72),),
        selected_root_unit_id=0,
        root_selection_fallback=False,
        throughput=throughput,
        backbone_edge_index=torch.tensor([0, 2]),
        backbone_position=torch.tensor([0, 1, 2]),
        corridor_edge_index=torch.tensor([0, 1, 2]),
        hubs=(HubCandidate(1, 2, 0.7, 0.6, 0.42),),
    )
    audit = SimpleNamespace(
        flow=flow,
        plan=plan,
        dynamics=dynamics,
        throughput=throughput,
        corridor=SimpleNamespace(count=3),
        effect=effect,
        corridor_evaluated=True,
        corridor_confirmed=True,
        roots=(root,),
        all_evidence_cut_margin=0.1,
        selected_root_unit_id=0,
        selected_root_effect=root,
        selected_root_confirmed=True,
        carriers=(carrier,),
        carrier_evaluated_count=1,
    )
    return world, audit, target


def _metadata() -> NativeAuditMetadata:
    return NativeAuditMetadata(
        dataset_sample_id="dataset-1",
        source_id="source-1",
        split="test",
        task_type="QA",
        generator_model="generator-1",
        model_id="model-1",
        model_dtype="float32",
        target_policy="all",
        target_rank=0,
        coverage=0.9,
        carrier_scope="all",
        query_chunk=2,
        route_budget=RouteBudget(
            edges_per_head=2,
            max_rows=8,
            root_candidates=4,
            hub_candidates=2,
            corridor_edges=8,
            confirm=True,
        ),
        local_window=3,
        saved_edges=1,
    )


def test_schema_three_keeps_plot_ready_head_resolved_ledgers() -> None:
    world, audit, _ = _fixture()
    arrays = native_audit_arrays(world, audit, _metadata())

    assert arrays["subset_audit_schema"] == AUDIT_SCHEMA
    assert arrays["method_version"] == METHOD_VERSION
    assert arrays["analysis_stage"] == "confirmed"
    assert arrays["route_budget_edges_per_head"] == 2
    assert arrays["route_budget_max_rows"] == 8
    assert arrays["route_budget_root_candidates"] == 4
    assert arrays["route_budget_hub_candidates"] == 2
    assert arrays["route_budget_corridor_edges"] == 8
    assert arrays["route_budget_confirm"] is True
    assert torch.equal(arrays["edge_layer"], torch.tensor([0, 0, 1]))
    assert torch.equal(arrays["route_edge_origin"], audit.dynamics.edge_register)
    assert arrays["edge_on_backbone"].tolist() == [True, False, True]
    assert arrays["edge_in_frozen_corridor"].tolist() == [True, True, True]
    assert arrays["edge_saved_count"] == 3
    assert arrays["edge_save_limit"] == 1
    assert arrays["corridor_edge_count"] == 3
    assert arrays["frozen_corridor_edge_index"].tolist() == [0, 1, 2]
    assert arrays["frozen_corridor_layer"].tolist() == [0, 0, 1]
    assert arrays["frozen_corridor_head"].tolist() == [0, 1, 0]
    assert arrays["frozen_corridor_source"].tolist() == [0, 1, 1]
    assert arrays["frozen_corridor_target"].tolist() == [1, 2, 2]
    assert torch.allclose(
        arrays["frozen_corridor_root_throughput"], torch.tensor([0.4, 0.2, 0.7])
    )
    assert len(arrays["frozen_corridor_layer"]) == len(arrays["edge_layer"])
    assert arrays["edge_saved_count"] > arrays["edge_save_limit"]
    assert arrays["backbone_node_layer"].tolist() == [0, 1, 2]
    assert arrays["backbone_node_position"].tolist() == [0, 1, 2]
    assert arrays["backbone_step_is_residual"].tolist() == [False, False]
    assert torch.allclose(
        arrays["edge_source_evidence_lineage_fraction"],
        torch.tensor([1.0, 0.0, 0.6]),
    )
    assert torch.allclose(
        arrays["edge_evidence_lineage_action"], torch.tensor([0.5, 0.0, 0.48])
    )
    assert arrays["route_head_transport"].shape == (2, 2, 2, 4)
    assert torch.allclose(
        arrays["route_row_total"] - arrays["route_row_retained"],
        torch.full((2, 2, 2), 0.25),
    )
    assert arrays["route_head_action"].shape == (2, 2, 2, 4)
    assert arrays["route_head_integration"].shape == (2, 2, 2, 4)
    assert arrays["route_layer_integration"].shape == (2, 2, 4)
    assert arrays["route_stage_displacement"].shape == (2, 2, 3)
    assert arrays["route_module_vector_cosine"].shape == (2, 2)
    assert arrays["reanchor_bucket_name"].tolist() == [
        "prompt_evidence",
        "other_prompt",
        "remote_response",
        "recent_local",
    ]
    assert arrays["reanchor_bucket_transport"].shape == (2, 2, 2, 4)
    assert arrays["reanchor_source_position"].shape == (2, 2, 2, 4)
    assert arrays["reanchor_score"].shape == (2, 2, 2)
    assert arrays["reanchor_candidate_layer"].shape == (0,)
    assert np.allclose(arrays["root_signed_action"], [0.7])
    assert np.allclose(arrays["root_absolute_action"], [0.9])
    assert np.allclose(arrays["root_functional_agreement"], [0.7 / 0.9])
    assert np.allclose(arrays["root_selection_score"], [0.72])
    assert np.allclose(arrays["hub_graph_score"], [0.42])
    assert arrays["selected_root_evaluated"] is True
    assert arrays["corridor_evaluated"] is True


def test_artifact_persists_structured_reanchor_target_fallback() -> None:
    world, audit, target = _fixture()
    selection = TargetReanchorSelection(
        query_position=target.query_position,
        policy="reanchor",
        has_event=False,
        fallback=True,
        center_position=-1,
        window_offset=0,
        layer=-1,
        head=-1,
        source_kind="none",
        source_position=-1,
        source_unit_id=-1,
        score=0,
        support=0,
        previous_anchor_fraction=0,
        anchor_fraction=0,
        relative_anchor_rise=0,
        relative_local_fall=0,
    )
    world.targets = (target,)
    world.target_selection = (selection,)
    metadata = replace(_metadata(), target_policy="reanchor")

    arrays = native_audit_arrays(world, audit, metadata)

    assert arrays["target_reanchor_selection_recorded"] is True
    assert arrays["target_reanchor_policy"] == "reanchor"
    assert arrays["target_reanchor_has_event"] is False
    assert arrays["target_reanchor_fallback"] is True
    assert arrays["target_reanchor_scan_signal"] == ("exact_full_row_message_transport")
    assert arrays["carrier_evaluated_count"] == 1
    assert arrays["full_chain_evaluated"] is True
    assert arrays["carrier_confirmed"].tolist() == [True]
    assert arrays["full_chain_confirmed"] is True
    assert not any("sha" in name or "capture_config" in name for name in arrays)


def test_schema_three_discovery_keeps_missing_exact_values_as_nan() -> None:
    world, audit, _ = _fixture()
    missing = float("nan")
    root = replace(
        audit.selected_root_effect,
        necessity=missing,
        sufficiency=missing,
        causal_score=missing,
        evaluated=False,
    )
    effect = replace(
        audit.effect,
        necessity=missing,
        sufficiency=missing,
        blocked_sufficiency=missing,
        mediated_sufficiency=missing,
        clean_restoration_error=missing,
        corrupt_restoration_error=missing,
        restoration_error=missing,
        restoration_valid=False,
    )
    carrier = replace(
        audit.carriers[0],
        necessity=missing,
        rescue=missing,
        block_effect=missing,
        blocked_rescue=missing,
        mediated_rescue=missing,
        confirmed=False,
    )
    discovery = SimpleNamespace(
        **{
            **vars(audit),
            "plan": replace(
                audit.plan, budget=replace(audit.plan.budget, confirm=False)
            ),
            "effect": effect,
            "corridor_evaluated": False,
            "corridor_confirmed": False,
            "roots": (root,),
            "all_evidence_cut_margin": missing,
            "selected_root_effect": root,
            "selected_root_confirmed": False,
            "carriers": (carrier,),
            "carrier_evaluated_count": 0,
        }
    )
    metadata = replace(
        _metadata(), route_budget=replace(_metadata().route_budget, confirm=False)
    )

    arrays = native_audit_arrays(world, discovery, metadata)

    assert arrays["analysis_stage"] == "discovery"
    assert arrays["route_budget_confirm"] is False
    assert arrays["selected_root_evaluated"] is False
    assert arrays["corridor_evaluated"] is False
    assert arrays["carrier_evaluated_count"] == 0
    assert arrays["full_chain_evaluated"] is False
    assert arrays["full_chain_confirmed"] is False
    assert np.isnan(arrays["selected_root_value_necessity"])
    assert np.isnan(arrays["selected_root_conditional_sufficiency"])
    assert np.isnan(arrays["selected_root_causal_score"])
    assert np.isnan(arrays["corridor_necessity"])
    assert np.isnan(arrays["carrier_necessity"][0])


def test_backbone_connects_root_to_query_across_residual_step() -> None:
    node = torch.zeros(4, 3)
    node[0, 0] = 0.4
    node[1, 1] = 0.4
    node[2, 1] = 0.3
    node[3, 2] = 1.0
    residual_probability = torch.zeros(3, 3)
    residual_probability[1, 1] = 0.75
    world = SimpleNamespace(
        response_start=2,
        units=SimpleNamespace(token_unit_id=torch.tensor([0, 1, 2])),
    )
    audit = SimpleNamespace(
        selected_root_unit_id=0,
        plan=SimpleNamespace(
            backbone_edge_index=torch.tensor([0, 1]),
            backbone_position=torch.tensor([0, 1, 1, 2]),
        ),
        flow=SimpleNamespace(
            target=SimpleNamespace(query_position=2),
            edges=SimpleNamespace(
                layer=torch.tensor([0, 2]),
                source=torch.tensor([0, 1]),
                target=torch.tensor([1, 2]),
            ),
        ),
        throughput=SimpleNamespace(
            edge=torch.tensor([0.4, 0.3]),
            root_mass=0.3,
            residual_probability=residual_probability,
            reverse_visit=torch.ones(4, 3),
            node=node,
        ),
        dynamics=SimpleNamespace(node_register=torch.ones(4, 3, 4) / 4),
    )

    backbone = _route_backbone(world, audit)

    assert backbone.edge_index.tolist() == [0, 1]
    assert backbone.node_layer.tolist() == [0, 1, 2, 3]
    assert backbone.node_position.tolist() == [0, 1, 1, 2]
    assert backbone.step_is_residual.tolist() == [False, True, False]


def test_schema_three_round_trip_validates_frozen_identity(tmp_path) -> None:
    world, audit, target = _fixture()
    path = tmp_path / "audit.npz"
    metadata = _metadata()
    save_native_audit(path, world, audit, metadata)

    validate_native_audit(
        path,
        world,
        target,
        FlowSignal.MESSAGE,
        metadata,
    )
    with np.load(path, allow_pickle=False) as stored:
        assert stored["route_head_transport"].shape == (2, 2, 2, 4)
        assert int(stored["positive_token_id"]) == 13
        assert int(stored["negative_token_id"]) == 9

    with pytest.raises(ValueError, match="source_id"):
        validate_native_audit(
            path,
            world,
            target,
            FlowSignal.MESSAGE,
            replace(metadata, source_id="source-2"),
        )


@pytest.mark.parametrize(
    ("field", "changed", "artifact_name"),
    (
        ("generator_model", "generator-2", "generator_model"),
        ("model_dtype", "float16", "model_dtype"),
        ("target_policy", "uncertain", "target_selection_policy"),
        ("coverage", 0.8, "edge_coverage"),
        ("carrier_scope", "response", "carrier_scope"),
        ("query_chunk", 3, "query_chunk"),
        (
            "route_budget",
            replace(_metadata().route_budget, edges_per_head=3),
            "route_budget_edges_per_head",
        ),
        (
            "route_budget",
            replace(_metadata().route_budget, max_rows=9),
            "route_budget_max_rows",
        ),
        (
            "route_budget",
            replace(_metadata().route_budget, root_candidates=5),
            "route_budget_root_candidates",
        ),
        (
            "route_budget",
            replace(_metadata().route_budget, hub_candidates=3),
            "route_budget_hub_candidates",
        ),
        (
            "route_budget",
            replace(_metadata().route_budget, corridor_edges=9),
            "route_budget_corridor_edges",
        ),
        (
            "route_budget",
            replace(_metadata().route_budget, confirm=False),
            "route_budget_confirm",
        ),
        ("local_window", 4, "local_window"),
        ("saved_edges", 2, "edge_save_limit"),
    ),
)
def test_schema_three_resume_rejects_changed_run_coordinates(
    tmp_path,
    field: str,
    changed: object,
    artifact_name: str,
) -> None:
    world, audit, target = _fixture()
    path = tmp_path / "audit.npz"
    metadata = _metadata()
    save_native_audit(path, world, audit, metadata)

    with pytest.raises(ValueError, match=artifact_name):
        validate_native_audit(
            path,
            world,
            target,
            FlowSignal.MESSAGE,
            replace(metadata, **{field: changed}),
        )


def test_schema_three_resume_rejects_changed_tokenizer(tmp_path) -> None:
    world, audit, target = _fixture()
    path = tmp_path / "audit.npz"
    metadata = _metadata()
    save_native_audit(path, world, audit, metadata)
    changed_world = SimpleNamespace(**{**vars(world), "tokenizer_id": "tokenizer-2"})

    with pytest.raises(ValueError, match="tokenizer_id"):
        validate_native_audit(
            path,
            changed_world,
            target,
            FlowSignal.MESSAGE,
            metadata,
        )


@pytest.mark.parametrize(
    "changed_token_ids",
    (
        torch.tensor([99, 11, 12, 13]),
        torch.tensor([10, 11, 12, 13, 14]),
    ),
)
def test_schema_three_resume_rejects_changed_token_sequence(
    tmp_path,
    changed_token_ids: torch.Tensor,
) -> None:
    world, audit, target = _fixture()
    path = tmp_path / "audit.npz"
    metadata = _metadata()
    save_native_audit(path, world, audit, metadata)
    changed_world = SimpleNamespace(**{**vars(world), "token_ids": changed_token_ids})

    with pytest.raises(ValueError, match="token_ids"):
        validate_native_audit(
            path,
            changed_world,
            target,
            FlowSignal.MESSAGE,
            metadata,
        )


def test_schema_three_save_rejects_local_window_mismatch() -> None:
    world, audit, _ = _fixture()

    with pytest.raises(ValueError, match="local_window"):
        native_audit_arrays(world, audit, replace(_metadata(), local_window=4))


def test_schema_three_save_rejects_route_budget_mismatch() -> None:
    world, audit, _ = _fixture()
    metadata = replace(
        _metadata(),
        route_budget=replace(_metadata().route_budget, corridor_edges=9),
    )

    with pytest.raises(ValueError, match="route_budget"):
        native_audit_arrays(world, audit, metadata)
