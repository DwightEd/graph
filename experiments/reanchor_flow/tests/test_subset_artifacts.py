from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.corridor import (
    CarrierEffect,
    CorridorEffect,
    RootEffect,
)
from experiments.reanchor_flow.flow import FlowSignal
from experiments.reanchor_flow.subset_artifacts import (
    AUDIT_SCHEMA,
    METHOD_VERSION,
    NativeAuditMetadata,
    _route_backbone,
    native_audit_arrays,
    save_native_audit,
    validate_native_audit,
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
    local = torch.zeros_like(direct)
    local[1, 0, 1] = torch.tensor([0.4, 0.3])
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
        head_local_response=local,
        head_integration=torch.rand(2, 2, 2, 4),
        layer_integration=torch.rand(2, 2, 4),
        cross_head_vector_coherence=torch.rand(2, 2),
        cross_head_functional_agreement=torch.rand(2, 2),
        head_backward_distance=torch.rand(2, 2, 2),
        head_span=torch.rand(2, 2),
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
        edge_count=2,
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
    flow = SimpleNamespace(
        target=target,
        signal=FlowSignal.MESSAGE,
        edges=edges,
        clean_margin=0.8,
        corrupt_margin=0.2,
    )
    audit = SimpleNamespace(
        flow=flow,
        dynamics=dynamics,
        throughput=SimpleNamespace(
            edge=torch.tensor([0.4, 0.0, 0.7]),
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
        ),
        corridor=SimpleNamespace(count=2),
        effect=effect,
        corridor_confirmed=True,
        roots=(root,),
        all_evidence_cut_margin=0.1,
        selected_root_unit_id=0,
        selected_root_effect=root,
        selected_root_confirmed=True,
        carriers=(carrier,),
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
        root_screen_limit=4,
        carrier_limit=2,
        local_window=3,
        saved_edges=1,
    )


def test_schema_two_keeps_plot_ready_head_resolved_ledgers() -> None:
    world, audit, _ = _fixture()
    arrays = native_audit_arrays(world, audit, _metadata())

    assert arrays["subset_audit_schema"] == AUDIT_SCHEMA
    assert arrays["method_version"] == METHOD_VERSION
    assert torch.equal(arrays["edge_layer"], torch.tensor([0, 1]))
    assert torch.equal(
        arrays["route_edge_origin"], audit.dynamics.edge_register[[0, 2]]
    )
    assert arrays["edge_on_backbone"].tolist() == [True, True]
    assert arrays["edge_saved_count"] == 2
    assert arrays["edge_save_limit"] == 1
    assert arrays["backbone_node_layer"].tolist() == [0, 1, 2]
    assert arrays["backbone_node_position"].tolist() == [0, 1, 2]
    assert arrays["backbone_step_is_residual"].tolist() == [False, False]
    assert torch.allclose(
        arrays["edge_source_root_lineage_fraction"], torch.tensor([1.0, 0.6])
    )
    assert torch.allclose(arrays["edge_root_lineage_action"], torch.tensor([0.5, 0.48]))
    assert arrays["route_head_transport"].shape == (2, 2, 2, 4)
    assert arrays["route_head_action"].shape == (2, 2, 2, 4)
    assert arrays["route_head_integration"].shape == (2, 2, 2, 4)
    assert arrays["route_layer_integration"].shape == (2, 2, 4)
    assert arrays["route_stage_displacement"].shape == (2, 2, 3)
    assert arrays["route_module_vector_cosine"].shape == (2, 2)
    assert arrays["route_event_direct_fraction"].ndim == 1
    assert arrays["carrier_confirmed"].tolist() == [True]
    assert arrays["full_chain_confirmed"] is True
    assert not any("sha" in name or "capture_config" in name for name in arrays)
    assert not any("max_" in name for name in arrays)


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


def test_schema_two_round_trip_validates_frozen_identity(tmp_path) -> None:
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
        assert stored["target_token_ids"].tolist() == [13, 9]

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
        ("root_screen_limit", 5, "root_screen_limit"),
        ("carrier_limit", 3, "carrier_limit"),
        ("local_window", 4, "local_window"),
        ("saved_edges", 2, "edge_save_limit"),
    ),
)
def test_schema_two_resume_rejects_changed_run_coordinates(
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


def test_schema_two_resume_rejects_changed_tokenizer(tmp_path) -> None:
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
def test_schema_two_resume_rejects_changed_token_sequence(
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


def test_schema_two_save_rejects_local_window_mismatch() -> None:
    world, audit, _ = _fixture()

    with pytest.raises(ValueError, match="local_window"):
        native_audit_arrays(world, audit, replace(_metadata(), local_window=4))
