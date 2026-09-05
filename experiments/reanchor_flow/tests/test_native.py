from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from experiments.common.llama_message_intervention import (
    baseline_forward,
    forward_layers,
)
from experiments.reanchor_flow import attribution as attribution_module
from experiments.reanchor_flow import corridor as corridor_module
from experiments.reanchor_flow import native as native_module
from experiments.reanchor_flow.attribution import (
    GradientObserver,
    contrast_direction,
    native_target_gradients,
)
from experiments.reanchor_flow.corridor import (
    CarrierEffect,
    complete_mediation_confirmed,
    confirm_carriers,
)
from experiments.reanchor_flow.flow import FlowEdges
from experiments.reanchor_flow.native import audit_native_target
from experiments.reanchor_flow.native_flow import native_flow_screen
from experiments.reanchor_flow.native_world import (
    NativeWorld,
    load_native_world,
    save_native_world,
)
from experiments.reanchor_flow.subset_artifacts import (
    save_compact_native_audit,
    validate_compact_native_audit,
)
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model
from experiments.reanchor_flow.throughput import (
    compute_throughput,
    transition_probabilities,
)
from experiments.reanchor_flow.worlds import TargetContrast


def native_world(model=None) -> NativeWorld:
    pair = paired_world()
    targets = pair.targets
    if model is not None:
        cache = baseline_forward(
            model,
            pair.clean_token_ids,
            pair.response_start,
            checkpoint_layers=range(len(model.model.layers)),
            attention_query_chunk=2,
        )
        query = pair.targets[0].query_position
        slot = int(torch.nonzero(cache.query == query, as_tuple=False).flatten()[0])
        targets = (
            TargetContrast(
                query,
                int(cache.target[slot]),
                int(cache.runner[slot]),
                "label_free_test_observed_token_vs_native_runner",
            ),
        )
    return NativeWorld(
        pair.sample_id,
        pair.tokenizer_id,
        pair.clean_token_ids,
        pair.response_start,
        pair.units,
        pair.candidate_unit_id,
        targets,
    ).check()


def compact_capture_config(world: NativeWorld) -> dict:
    return {
        "model": "tiny",
        "model_dtype": "float32",
        "tokenizer": world.tokenizer_id,
        "dataset_manifest_sha256": "d" * 64,
        "source_info_sha256": "s" * 64,
        "split": "test",
        "target_policy": "evenly-spaced",
        "flow_signal": "message",
        "edge_coverage": 1.0,
        "carrier_scope": "all",
        "query_chunk": 2,
        "root_screen_limit": 0,
        "carrier_limit": 1,
        "saved_edges": 4,
    }


def test_native_world_round_trip_and_observed_target_contract(tmp_path) -> None:
    world = native_world()
    path = tmp_path / "world.npz"
    save_native_world(path, world)
    loaded = load_native_world(path)
    assert torch.equal(loaded.token_ids, world.token_ids)
    assert loaded.evidence_unit_id == world.evidence_unit_id
    assert loaded.targets == world.targets

    invalid = NativeWorld(
        world.sample_id,
        world.tokenizer_id,
        world.token_ids,
        world.response_start,
        world.units,
        world.evidence_unit_id,
        (
            type(world.targets[0])(
                world.targets[0].query_position,
                13,
                world.targets[0].negative_token_id,
                world.targets[0].origin,
            ),
        ),
    )
    with pytest.raises(ValueError, match="observed target"):
        invalid.check()

    with pytest.raises(ValueError, match="duplicates"):
        replace(
            world,
            targets=world.targets
            + (replace(world.targets[0], origin="another_selection_origin"),),
        ).check()


def test_native_world_rejects_an_unfrozen_target_position() -> None:
    model = tiny_model()
    world = native_world(model)
    query = world.response_start - 1
    positive = int(world.token_ids[query + 1])
    unfrozen = TargetContrast(
        query,
        positive,
        (positive + 1) % model.config.vocab_size,
        "post_hoc_target",
    )
    assert unfrozen not in world.targets
    with pytest.raises(ValueError, match="not frozen"):
        world.prefix(unfrozen)


def test_complete_mediation_uses_a_two_sided_block_and_effect_floor() -> None:
    tolerance = 0.02
    assert complete_mediation_confirmed(
        0.10,
        0.12,
        0.01,
        direction=1.0,
        tolerance=tolerance,
    )
    assert not complete_mediation_confirmed(
        0.10,
        0.12,
        -0.10,
        direction=1.0,
        tolerance=tolerance,
    )
    assert not complete_mediation_confirmed(
        0.01,
        0.12,
        0.0,
        direction=1.0,
        tolerance=tolerance,
    )


def test_native_support_throughput_connects_edges_via_an_implicit_residual() -> None:
    count = 2
    scalar = torch.ones(count)
    nan = torch.full((count,), float("nan"))
    code = torch.zeros(count, 1)
    vector = torch.empty(count, 0)
    edges = FlowEdges(
        layer=torch.tensor([0, 2], dtype=torch.int16),
        head=torch.zeros(count, dtype=torch.int16),
        source=torch.tensor([0, 1], dtype=torch.int32),
        target=torch.tensor([1, 2], dtype=torch.int32),
        source_unit=torch.tensor([0, 1], dtype=torch.int32),
        attention_clean=scalar,
        attention_corrupt=scalar,
        score=scalar,
        clean_target_score=scalar,
        corrupt_target_score=scalar,
        selector_score=nan,
        content_score=nan,
        clean_message_norm=scalar,
        corrupt_message_norm=scalar,
        delta_message_norm=torch.zeros(count),
        clean_code=code,
        corrupt_code=code,
        clean_message_vector=vector,
        corrupt_message_vector=vector,
        delta_message_vector=vector,
    )
    row_total = torch.zeros(3, 1, 2)
    row_total[0, 0, 0] = 1.0
    row_total[2, 0, 1] = 1.0
    flow = SimpleNamespace(
        edges=edges,
        row_total=row_total,
        row_position=torch.tensor([1, 2]),
        residual_weight=torch.ones(3, 2),
        clean_cache=SimpleNamespace(layer_count=3),
        target=TargetContrast(2, 3, 4, "test"),
    )
    throughput = compute_throughput(
        flow,
        torch.tensor([0, 1, 2]),
        3,
        (0,),
    )

    assert throughput.root_mass > 0
    torch.testing.assert_close(throughput.edge, torch.ones(2))
    torch.testing.assert_close(throughput.node[1:3, 1], torch.ones(2))
    assert int(edges.layer[1]) - int(edges.layer[0]) == 2
    assert float(throughput.residual_probability[1, 1]) == pytest.approx(1.0)


def test_native_carrier_reruns_keep_the_root_source_cut(monkeypatch) -> None:
    root_mask = torch.tensor([True, False, False])
    clean_cache = SimpleNamespace(
        layer_count=1,
        layer_input={0: torch.ones(3, 2)},
    )
    cut_cache = SimpleNamespace(
        layer_count=1,
        layer_input={0: torch.zeros(3, 2)},
    )
    flow = SimpleNamespace(
        pair_effect=1.0,
        clean_margin=1.0,
        corrupt_margin=0.0,
        clean_cache=clean_cache,
        corrupt_cache=cut_cache,
        clean_source_mask=None,
        corrupt_source_mask=root_mask,
        row_position=torch.tensor([1]),
        target=TargetContrast(2, 3, 4, "test"),
        stages=None,
    )
    throughput = SimpleNamespace(node=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=1),
        dtype=torch.float32,
    )
    margins = iter((0.8, 0.2, 0.0, 0.0))
    seen_masks = []

    def fake_rerun(_model, _cache, _gate, _target, *, base_source_mask=None):
        seen_masks.append(
            None
            if base_source_mask is None
            else tuple(bool(value) for value in base_source_mask.tolist())
        )
        return next(margins)

    monkeypatch.setattr(corridor_module, "rerun_margin", fake_rerun)
    carriers = confirm_carriers(
        model,
        flow,
        throughput,
        torch.tensor([0]),
        limit=1,
        effect_direction=1.0,
    )

    assert seen_masks == [
        None,
        (True, False, False),
        (True, False, False),
        (True, False, False),
    ]
    assert len(carriers) == 1
    assert carriers[0].confirmed


def test_layer_local_native_vjp_matches_monolithic_gradient() -> None:
    model = tiny_model()
    world = native_world(model)
    target = world.targets[0]
    layers = len(model.model.layers)
    cache = baseline_forward(
        model,
        world.token_ids,
        world.response_start,
        checkpoint_layers=range(layers),
        attention_query_chunk=2,
    )
    positions = torch.arange(target.query_position + 1)
    actual = native_target_gradients(
        model,
        cache,
        target,
        positions,
        query_chunk=2,
    )

    hidden = cache.layer_input[0].clone()[None].detach().requires_grad_(True)
    observer = GradientObserver()
    final = forward_layers(
        model,
        hidden,
        0,
        observer=observer,
        attention_query_chunk=2,
    )
    direction, bias = contrast_direction(model, target)
    objective = torch.dot(final[0, target.query_position].float(), direction) + bias
    objective.backward()
    expected = observer.gradients(positions, layers)

    assert actual.position.equal(expected.position)
    for name in ("head_output", "layer_input", "attention_write", "mlp_write"):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            atol=2e-6,
            rtol=2e-5,
        )


def test_native_vjp_rejects_a_non_native_runner() -> None:
    model = tiny_model()
    world = native_world(model)
    target = world.targets[0]
    cache = baseline_forward(
        model,
        world.token_ids,
        world.response_start,
        checkpoint_layers=range(len(model.model.layers)),
        attention_query_chunk=2,
    )
    invalid_runner = next(
        token
        for token in range(model.config.vocab_size)
        if token not in {target.positive_token_id, target.negative_token_id}
    )
    invalid = TargetContrast(
        target.query_position,
        target.positive_token_id,
        invalid_runner,
        target.origin,
    )
    with pytest.raises(ValueError, match="frozen native runner"):
        native_target_gradients(
            model,
            cache,
            invalid,
            torch.tensor([target.query_position]),
            query_chunk=2,
        )


def test_native_vjp_recomputes_exactly_one_layer_per_reverse_step(monkeypatch) -> None:
    model = tiny_model()
    world = native_world(model)
    target = world.targets[0]
    layers = len(model.model.layers)
    cache = baseline_forward(
        model,
        world.token_ids,
        world.response_start,
        checkpoint_layers=range(layers),
        attention_query_chunk=2,
    )
    actual_forward = attribution_module.forward_layers
    calls = []

    def traced_forward(model, hidden, start_layer, **kwargs):
        calls.append(
            (
                start_layer,
                kwargs.get("end_layer"),
                kwargs.get("apply_final_norm"),
            )
        )
        return actual_forward(model, hidden, start_layer, **kwargs)

    monkeypatch.setattr(attribution_module, "forward_layers", traced_forward)
    native_target_gradients(
        model,
        cache,
        target,
        torch.tensor([target.query_position]),
        query_chunk=2,
    )
    assert calls == [(layer, layer + 1, False) for layer in range(layers - 1, -1, -1)]


def test_native_attention_and_message_are_distinct_transport_backends() -> None:
    model = tiny_model()
    world = native_world(model)
    target = world.targets[0]
    attention, _ = native_flow_screen(
        model,
        world,
        target,
        "attention",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
    )
    message, _ = native_flow_screen(
        model,
        world,
        target,
        "message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
    )
    assert torch.equal(attention.edges.layer, message.edges.layer)
    assert torch.allclose(attention.edges.score, attention.edges.attention_clean)
    assert torch.allclose(message.edges.score, message.edges.clean_message_norm)
    assert not torch.allclose(attention.edges.score, message.edges.score)
    assert torch.allclose(
        attention.edges.clean_target_score,
        message.edges.clean_target_score,
        atol=1e-6,
    )
    assert torch.all(attention.residual_weight == model.config.num_attention_heads)
    assert torch.all(message.residual_weight > 0)

    for flow in (attention, message):
        probability, residual = transition_probabilities(flow, len(world.token_ids) - 1)
        for layer in range(flow.clean_cache.layer_count):
            for position in flow.row_position.tolist():
                selected = (flow.edges.layer == layer) & (flow.edges.target == position)
                total = probability[selected].sum() + residual[layer, position]
                assert float(total) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("signal", ["attention", "message"])
def test_native_corridor_restores_both_base_worlds(signal: str) -> None:
    model = tiny_model()
    world = native_world(model)
    result = audit_native_target(
        model,
        world,
        world.targets[0],
        signal,
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
    )
    assert result.flow.corrupt_source_mask is not None
    assert result.effect.restoration_error <= 1e-6
    assert result.effect.restoration_valid
    assert result.effect.edge_count == result.corridor.count
    assert all(float(score) > 0 for score in result.corridor.clean_target_score)


def test_native_roots_are_screened_on_positive_functional_support(monkeypatch) -> None:
    model = tiny_model()
    world = native_world(model)
    target = world.targets[0]
    actual_compute = native_module.compute_throughput
    calls = []

    def traced_compute(flow, token_unit_id, unit_count, root_unit_id):
        result = actual_compute(flow, token_unit_id, unit_count, root_unit_id)
        calls.append((flow.edges.score.clone(), tuple(root_unit_id), result))
        return result

    monkeypatch.setattr(native_module, "compute_throughput", traced_compute)
    result = audit_native_target(
        model,
        world,
        target,
        "message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        root_screen_limit=1,
        carrier_limit=0,
    )

    assert len(calls) == 3
    raw_score, raw_roots, _ = calls[0]
    support_score, support_roots, support_candidates = calls[1]
    assert raw_roots == world.evidence_unit_id
    assert support_roots == world.evidence_unit_id
    assert torch.equal(
        support_score,
        torch.where(
            result.flow.edges.clean_target_score > 0,
            raw_score,
            torch.zeros_like(raw_score),
        ),
    )
    for root in result.roots:
        assert root.route_mass == pytest.approx(
            float(support_candidates.unit_mass[root.unit_id])
        )


def test_compact_native_artifact_omits_message_codes(tmp_path) -> None:
    model = tiny_model()
    world = native_world(model)
    result = audit_native_target(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
    )
    path = tmp_path / "audit.npz"
    capture_config = compact_capture_config(world)
    save_compact_native_audit(
        path,
        world,
        result,
        dataset_sample_id="dataset-tiny",
        source_id="source-tiny",
        split="test",
        task_type="QA",
        generator_model="tiny",
        model_id="tiny",
        model_dtype="float32",
        target_policy="evenly-spaced",
        target_rank=0,
        coverage=1.0,
        carrier_scope="all",
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
        saved_edges=4,
        world_sha256="w" * 64,
        capture_config=capture_config,
    )
    validate_compact_native_audit(
        path,
        dataset_sample_id="dataset-tiny",
        sample_id=world.sample_id,
        source_id="source-tiny",
        split="test",
        task_type="QA",
        generator_model="tiny",
        tokenizer_id=world.tokenizer_id,
        world_sha256="w" * 64,
        target=world.targets[0],
        target_rank=0,
        signal=result.flow.signal,
        model_id="tiny",
        model_dtype="float32",
        capture_config=capture_config,
    )
    with np.load(path, allow_pickle=False) as stored:
        assert int(stored["edge_saved_count"]) <= 4
        assert not any("code" in name for name in stored.files)
        assert "edge_root_cut_native_gradient_projection" in stored.files
        assert "edge_root_cut_functional_score" not in stored.files
        assert str(stored["world_kind"].item()) == ("native_source_value_message_cut")
        assert str(stored["root_cut_functional_score_semantics"].item()) == (
            "frozen_native_gradient_dot_root_cut_pre_WO_AV_message"
        )
        assert "do not directly mask Q/K" in str(stored["source_cut_semantics"].item())
        assert float(stored["causal_effect_tolerance"]) == pytest.approx(
            result.effect.restoration_tolerance
        )
        full_chain = bool(stored["corridor_confirmed"]) and bool(
            stored["carrier_any_confirmed"]
        )
        assert bool(stored["carrier_value_mediated"]) == full_chain
        assert bool(stored["full_chain_confirmed"]) == full_chain
        assert not bool(stored["labels_used_for_capture"])
        assert bool(stored["corridor_restoration_valid"])

    local_carrier = CarrierEffect(
        layer=0,
        position=world.response_start,
        route_throughput=1.0,
        state_delta_norm=1.0,
        target_score=1.0,
        necessity=1.0,
        rescue=1.0,
        block_effect=1.0,
        blocked_rescue=0.0,
        mediated_rescue=1.0,
        block_tolerance=1e-5,
        confirmed=True,
    )
    local_only = replace(
        result,
        corridor_confirmed=False,
        carriers=(local_carrier,),
    )
    local_only_path = tmp_path / "local-carrier-only.npz"
    save_compact_native_audit(
        local_only_path,
        world,
        local_only,
        dataset_sample_id="dataset-tiny",
        source_id="source-tiny",
        split="test",
        task_type="QA",
        generator_model="tiny",
        model_id="tiny",
        model_dtype="float32",
        target_policy="evenly-spaced",
        target_rank=0,
        coverage=1.0,
        carrier_scope="all",
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
        saved_edges=4,
        world_sha256="w" * 64,
        capture_config=capture_config,
    )
    with np.load(local_only_path, allow_pickle=False) as stored:
        assert bool(stored["carrier_any_confirmed"])
        assert not bool(stored["carrier_value_mediated"])
        assert not bool(stored["full_chain_confirmed"])


def test_compact_native_artifact_rejects_swapped_resume_identity(tmp_path) -> None:
    model = tiny_model()
    world = native_world(model)
    result = audit_native_target(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
    )
    path = tmp_path / "audit.npz"
    capture_config = compact_capture_config(world)
    save_compact_native_audit(
        path,
        world,
        result,
        dataset_sample_id="dataset-tiny",
        source_id="source-tiny",
        split="test",
        task_type="QA",
        generator_model="tiny",
        model_id="tiny",
        model_dtype="float32",
        target_policy="evenly-spaced",
        target_rank=0,
        coverage=1.0,
        carrier_scope="all",
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
        saved_edges=4,
        world_sha256="w" * 64,
        capture_config=capture_config,
    )
    validation = {
        "dataset_sample_id": "dataset-tiny",
        "sample_id": world.sample_id,
        "source_id": "source-tiny",
        "split": "test",
        "task_type": "QA",
        "generator_model": "tiny",
        "tokenizer_id": world.tokenizer_id,
        "world_sha256": "w" * 64,
        "target": world.targets[0],
        "target_rank": 0,
        "signal": result.flow.signal,
        "model_id": "tiny",
        "model_dtype": "float32",
        "capture_config": capture_config,
    }
    with pytest.raises(ValueError, match="source_id"):
        validate_compact_native_audit(
            path,
            **{**validation, "source_id": "another-source"},
        )

    target = world.targets[0]
    another_negative = target.negative_token_id + 1
    if another_negative == target.positive_token_id:
        another_negative += 1
    swapped_target = TargetContrast(
        target.query_position,
        target.positive_token_id,
        another_negative,
        target.origin,
    )
    with pytest.raises(ValueError, match="negative_token_id"):
        validate_compact_native_audit(
            path,
            **{**validation, "target": swapped_target},
        )

    stale_config = {**capture_config, "saved_edges": 8}
    with pytest.raises(ValueError, match="edge_save_limit|capture_config"):
        validate_compact_native_audit(
            path,
            **{**validation, "capture_config": stale_config},
        )


def test_target_policy_uses_only_clean_margin() -> None:
    from experiments.reanchor_flow.subset_data import target_slots

    cache = SimpleNamespace(
        query=torch.arange(5),
        full_margin=torch.tensor([3.0, -0.2, 0.1, -4.0, 1.0]),
    )
    assert target_slots(cache, count=2, policy="uncertain") == (2, 1)
    assert target_slots(cache, count=2, policy="low-margin") == (3, 1)
    assert target_slots(cache, count=2, policy="evenly-spaced") == (1, 3)
    assert target_slots(cache, count=2, policy="all") == (0, 1, 2, 3, 4)
