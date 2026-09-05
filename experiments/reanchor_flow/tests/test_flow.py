from __future__ import annotations

import numpy as np
import torch

from experiments.reanchor_flow.audit import audit_target, save_audit
from experiments.reanchor_flow.flow import (
    FlowSignal,
    attention_rows,
    capture_paired_flow,
    project_selected_messages,
)
from experiments.reanchor_flow.message_norm import output_gram
from experiments.reanchor_flow.throughput import compute_throughput

from .etcc_helpers import paired_world, tiny_model


def test_recomputed_attention_uses_the_native_probability_dtype() -> None:
    query = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    probability = attention_rows(query, key, torch.tensor([1, 2]), 0.5)
    assert probability.dtype == torch.bfloat16


def test_compact_gram_norms_match_materialized_messages() -> None:
    torch.manual_seed(5)
    output = torch.randn(6, 6)
    head = torch.tensor([0, 1, 0, 1])
    clean = torch.randn(4, 3)
    corrupt = torch.randn(4, 3)
    materialized = project_selected_messages(
        output,
        head,
        clean,
        corrupt,
        2,
        materialize=True,
    )
    compact = project_selected_messages(
        output,
        head,
        clean,
        corrupt,
        2,
        materialize=False,
        gram=output_gram(output, 2, 3),
    )
    for materialized_norm, compact_norm in zip(
        materialized[:3], compact[:3], strict=True
    ):
        torch.testing.assert_close(materialized_norm, compact_norm)
    assert compact[3].shape == (4, 0)


def test_attention_backend_is_raw_routing_data() -> None:
    model = tiny_model()
    world = paired_world()
    target = world.targets[0]
    flow = capture_paired_flow(
        model,
        world,
        target,
        FlowSignal.ATTENTION,
        coverage=1.0,
        query_chunk=2,
    )
    torch.testing.assert_close(flow.edges.score, flow.edges.attention_clean)
    assert bool(torch.isnan(flow.edges.clean_target_score).all())
    assert bool(torch.isnan(flow.edges.corrupt_target_score).all())
    assert bool(torch.isnan(flow.edges.selector_score).all())
    assert bool(torch.isnan(flow.edges.content_score).all())
    assert flow.stages is None
    assert flow.clean_cache.attention_write == {}
    assert flow.clean_cache.mlp_write == {}
    assert bool(torch.isnan(flow.aggregation.signed_score).all())
    torch.testing.assert_close(flow.row_retained, flow.row_total)


def test_message_backend_preserves_signed_decomposition_and_real_messages() -> None:
    model = tiny_model()
    world = paired_world()
    flow = capture_paired_flow(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        coverage=1.0,
        query_chunk=2,
        materialize_messages=True,
    )
    torch.testing.assert_close(
        flow.edges.selector_score + flow.edges.content_score,
        flow.edges.score,
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        flow.edges.clean_target_score - flow.edges.corrupt_target_score,
        flow.edges.score,
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        flow.aggregation.positive_score + flow.aggregation.negative_score,
        flow.aggregation.signed_score,
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        flow.aggregation.selector_score + flow.aggregation.content_score,
        flow.aggregation.signed_score,
        atol=2e-7,
        rtol=2e-6,
    )
    assert bool((flow.aggregation.coherence <= 1.0 + 2e-5).all())
    assert flow.edges.clean_code.dtype == torch.float32
    assert flow.edges.delta_message_vector.shape == (flow.edges.count, 32)
    for index in range(min(flow.edges.count, 32)):
        layer = int(flow.edges.layer[index])
        head = int(flow.edges.head[index])
        output = model.model.layers[layer].self_attn.o_proj.weight.detach().float()
        head_dim = flow.edges.clean_code.shape[1]
        block = output[:, head * head_dim : (head + 1) * head_dim]
        expected_clean = block @ flow.edges.clean_code[index]
        expected_corrupt = block @ flow.edges.corrupt_code[index]
        expected_delta = expected_clean - expected_corrupt
        torch.testing.assert_close(
            flow.edges.clean_message_vector[index], expected_clean
        )
        torch.testing.assert_close(
            flow.edges.corrupt_message_vector[index], expected_corrupt
        )
        torch.testing.assert_close(
            flow.edges.delta_message_vector[index], expected_delta
        )
        torch.testing.assert_close(
            flow.edges.delta_message_norm[index], expected_delta.norm()
        )


def test_root_conditioned_throughput_is_conserved_at_full_coverage() -> None:
    model = tiny_model()
    world = paired_world()
    flow = capture_paired_flow(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        coverage=1.0,
        query_chunk=2,
    )
    throughput = compute_throughput(
        flow,
        world.units.token_unit_id,
        world.units.count,
        (1,),
    )
    torch.testing.assert_close(throughput.unit_mass.sum(), torch.tensor(1.0))
    assert throughput.root_mass > 0
    torch.testing.assert_close(
        throughput.unit_mass[1], torch.tensor(throughput.root_mass)
    )
    torch.testing.assert_close(
        throughput.node.sum(dim=1),
        torch.ones(flow.clean_cache.layer_count + 1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert bool((throughput.edge >= 0).all())


def test_coverage_pruning_goes_to_sink_without_breaking_conditioned_flow() -> None:
    model = tiny_model()
    world = paired_world().isolate(1)
    flow = capture_paired_flow(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        coverage=0.5,
        query_chunk=2,
    )
    throughput = compute_throughput(
        flow,
        world.units.token_unit_id,
        world.units.count,
        (1,),
    )
    assert throughput.unit_mass.sum() < 1
    torch.testing.assert_close(
        throughput.unit_mass[1], torch.tensor(throughput.root_mass)
    )
    torch.testing.assert_close(
        throughput.node.sum(dim=1),
        torch.ones(flow.clean_cache.layer_count + 1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_response_scope_keeps_candidate_root_gradients_but_not_prompt_rows() -> None:
    model = tiny_model()
    world = paired_world()
    flow = capture_paired_flow(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        carrier_scope="response",
        coverage=0.9,
        query_chunk=2,
    )
    assert flow.stages is not None
    assert 1 in flow.stages.position.tolist()
    assert 2 in flow.stages.position.tolist()
    assert 1 not in flow.row_position.tolist()
    assert 2 not in flow.row_position.tolist()


def test_full_audit_selects_and_causally_confirms_a_root(tmp_path) -> None:
    model = tiny_model()
    world = paired_world()
    audit = audit_target(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        carrier_scope="all",
        coverage=1.0,
        gradient_steps=1,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=2,
        materialize_messages=True,
    )
    assert audit.selected_root_unit_id in world.candidate_unit_id
    assert audit.selected_root_confirmed
    assert audit.world.candidate_unit_id == (audit.selected_root_unit_id,)
    nonselected = (
        set(world.candidate_unit_id) - {audit.selected_root_unit_id}
    ).pop()
    nonselected_position = world.units.positions((nonselected,))
    torch.testing.assert_close(
        audit.world.corrupt_token_ids.index_select(0, nonselected_position),
        world.clean_token_ids.index_select(0, nonselected_position),
    )
    assert all(root.evaluated for root in audit.roots)
    assert audit.effect.edge_count == audit.corridor.count
    assert audit.effect.clean_restoration_error <= 1e-6
    assert audit.effect.corrupt_restoration_error <= 1e-6
    assert audit.effect.restoration_error <= 1e-6
    assert audit.effect.restoration_valid
    assert audit.throughput.root_mass > 0
    candidate_position = set(
        world.units.positions((audit.selected_root_unit_id,)).tolist()
    )
    assert all(
        carrier.position not in candidate_position for carrier in audit.carriers
    )
    for carrier in audit.carriers:
        assert carrier.route_throughput > 0
        assert carrier.target_score * audit.effect.pair_effect > 0
        assert abs(
            carrier.mediated_rescue
            - (carrier.rescue - carrier.blocked_rescue)
        ) < 1e-8

    path = tmp_path / "audit.npz"
    save_audit(
        path,
        world,
        audit,
        model_id="tiny",
        model_dtype="float32",
        coverage=1.0,
        gradient_steps=1,
        carrier_scope="all",
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=2,
        materialize_messages=True,
    )
    with np.load(path, allow_pickle=False) as stored:
        assert stored["flow_signal"].item() == "message"
        torch.testing.assert_close(
            torch.from_numpy(stored["screen_corrupt_token_ids"]),
            world.corrupt_token_ids,
        )
        assert stored["edge_layer"].shape == stored["edge_score"].shape
        assert stored["edge_clean_code"].shape[0] == stored["edge_score"].shape[0]
        assert int(stored["selected_root_unit_id"]) in world.candidate_unit_id
        assert bool(stored["selected_root_confirmed"])
        assert float(stored["corridor_clean_restoration_error"]) <= 1e-6
        assert float(stored["corridor_corrupt_restoration_error"]) <= 1e-6
        assert float(stored["corridor_restoration_error"]) <= 1e-6
        assert bool(stored["corridor_restoration_valid"])
        assert bool(stored["corridor_confirmed"]) == audit.corridor_confirmed
        assert stored["carrier_confirmed"].shape == stored["carrier_layer"].shape
        assert stored["carrier_route_throughput"].shape == stored[
            "carrier_layer"
        ].shape


def test_bfloat16_corridor_restoration_uses_dtype_tolerance() -> None:
    model = tiny_model().to(torch.bfloat16)
    world = paired_world()
    audit = audit_target(
        model,
        world,
        world.targets[0],
        FlowSignal.MESSAGE,
        carrier_scope="all",
        coverage=0.9,
        gradient_steps=1,
        query_chunk=2,
        root_screen_limit=1,
        carrier_limit=0,
        materialize_messages=False,
    )
    assert audit.flow.edges.clean_code.dtype == torch.float32
    assert audit.effect.restoration_tolerance == 2e-2
    assert audit.effect.restoration_valid


def test_attention_audit_uses_message_patches_without_gradient_proxies() -> None:
    model = tiny_model()
    world = paired_world().isolate(1)
    audit = audit_target(
        model,
        world,
        world.targets[0],
        "attention",
        carrier_scope="all",
        coverage=0.9,
        gradient_steps=1,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=2,
        materialize_messages=False,
    )
    assert audit.flow.signal is FlowSignal.ATTENTION
    assert audit.flow.stages is None
    assert audit.carriers
    assert all(carrier.route_throughput > 0 for carrier in audit.carriers)
    assert all(np.isnan(carrier.target_score) for carrier in audit.carriers)
    assert audit.effect.restoration_valid
