from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from experiments.common.llama_message_intervention import baseline_forward
from experiments.reanchor_flow import native_flow as native_flow_module
from experiments.reanchor_flow.flow import SOURCE_LOCATION_BUCKET_NAMES
from experiments.reanchor_flow.native_flow import (
    SourceLocationAccumulator,
    _budgeted_edge_mask,
    attach_cut_edge_codes,
    capture_source_location_buckets,
    native_flow_screen,
    represented_positions,
)
from experiments.reanchor_flow.native_world import (
    NativeWorld,
    gated_forward_cache,
    source_gate,
)
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model
from experiments.reanchor_flow.worlds import TargetContrast


def _valid_native_world(model) -> NativeWorld:
    pair = paired_world()
    cache = baseline_forward(
        model,
        pair.clean_token_ids,
        pair.response_start,
        checkpoint_layers=range(len(model.model.layers)),
        attention_query_chunk=2,
    )
    query = pair.targets[0].query_position
    slot = int(torch.nonzero(cache.query == query, as_tuple=False).flatten()[0])
    target = TargetContrast(
        query,
        int(cache.target[slot]),
        int(cache.runner[slot]),
        "label_free_budget_test",
    )
    return NativeWorld(
        pair.sample_id,
        pair.tokenizer_id,
        pair.clean_token_ids,
        pair.response_start,
        pair.units,
        pair.candidate_unit_id,
        (target,),
    ).check()


def test_represented_row_budget_fails_before_arange(monkeypatch) -> None:
    world = paired_world()
    target = world.targets[0]

    def unexpected_arange(*_args, **_kwargs):
        raise AssertionError("row tensor was allocated before checking its budget")

    monkeypatch.setattr(native_flow_module.torch, "arange", unexpected_arange)
    with pytest.raises(
        ValueError,
        match=r"carrier_scope='all' requires 6 represented rows.*max_rows=5",
    ):
        represented_positions(world, target, "all", max_rows=5)


def test_native_screen_checks_row_budget_before_baseline(monkeypatch) -> None:
    model = tiny_model()
    pair = paired_world()
    world = NativeWorld(
        pair.sample_id,
        pair.tokenizer_id,
        pair.clean_token_ids,
        pair.response_start,
        pair.units,
        pair.candidate_unit_id,
        pair.targets,
    ).check()

    def unexpected_baseline(*_args, **_kwargs):
        raise AssertionError("baseline allocated before checking the row budget")

    monkeypatch.setattr(native_flow_module, "baseline_forward", unexpected_baseline)
    with pytest.raises(ValueError, match="exceeding max_rows=5"):
        native_flow_screen(
            model,
            world,
            world.targets[0],
            "message",
            carrier_scope="all",
            coverage=0.9,
            query_chunk=2,
            max_rows=5,
            max_edges_per_head_row=1,
        )


def test_edge_budget_unions_transport_and_functional_routes() -> None:
    transport = torch.tensor([[[0.60, 0.25, 0.10, 0.05]]])
    functional = torch.tensor([[[0.0, 0.0, -3.0, 2.0]]])
    keep = _budgeted_edge_mask(
        transport,
        functional,
        coverage=0.95,
        max_edges_per_head_row=1,
    )

    assert torch.nonzero(keep[0, 0], as_tuple=False).flatten().tolist() == [0, 2]
    assert float(transport.masked_fill(~keep, 0).sum()) == pytest.approx(0.70)
    assert int(keep.sum(dim=-1).max()) <= 2


def test_full_rows_are_bucketed_before_pruning_without_head_averaging() -> None:
    token_unit = torch.tensor([0, 1, 1, 2, 3, 4, 5])
    accumulator = SourceLocationAccumulator(
        layers=1,
        heads=1,
        rows=3,
        token_unit_id=token_unit,
        response_start=4,
        evidence_unit_id=(1,),
        local_window=1,
        with_action=True,
        device=torch.device("cpu"),
    )
    attention = torch.tensor(
        [
            [
                [0.10, 0.20, 0.10, 0.10, 0.50, 0.00, 0.00],
                [0.05, 0.10, 0.05, 0.10, 0.20, 0.50, 0.00],
                [0.05, 0.05, 0.10, 0.10, 0.30, 0.15, 0.25],
            ]
        ]
    )
    source_scale = torch.arange(1, 8, dtype=torch.float32)
    message = attention * source_scale
    action = attention * (source_scale - 3)
    accumulator.observe(
        0,
        0,
        torch.tensor([4, 5, 6]),
        attention,
        message,
        action,
    )
    summary = accumulator.summary()

    assert SOURCE_LOCATION_BUCKET_NAMES == (
        "prompt_evidence",
        "other_prompt",
        "remote_response",
        "recent_local",
    )
    assert summary.attention.shape == (1, 1, 3, 4)
    torch.testing.assert_close(summary.attention.sum(-1), torch.ones(1, 1, 3))
    torch.testing.assert_close(summary.transport.sum(-1), message.sum(-1).unsqueeze(0))
    torch.testing.assert_close(
        summary.downstream_action.sum(-1), action.sum(-1).unsqueeze(0)
    )
    assert summary.source_position[0, 0].tolist() == [
        [1, 3, -1, 4],
        [1, 3, -1, 5],
        [2, 3, 4, 6],
    ]
    assert summary.source_unit_id[0, 0].tolist() == [
        [1, 2, -1, 3],
        [1, 2, -1, 4],
        [1, 2, 3, 5],
    ]
    assert float(summary.attention[0, 0, 0, 3]) == pytest.approx(0.5)
    assert float(summary.source_transport[0, 0, 0, 2]) == 0.0
    assert int(summary.source_position[0, 0, 0, 3]) == 4


def test_full_row_transport_is_independent_of_sparse_edge_signal() -> None:
    model = tiny_model()
    world = _valid_native_world(model)
    arguments = {
        "carrier_scope": "response",
        "coverage": 0.5,
        "query_chunk": 2,
        "max_rows": 16,
        "max_edges_per_head_row": 1,
        "local_window": 1,
    }
    attention_flow, _ = native_flow_screen(
        model, world, world.targets[0], "attention", **arguments
    )
    message_flow, _ = native_flow_screen(
        model, world, world.targets[0], "message", **arguments
    )

    attention_location = attention_flow.source_location
    message_location = message_flow.source_location
    assert attention_location is not None
    assert message_location is not None
    torch.testing.assert_close(
        attention_location.transport,
        message_location.transport,
    )
    torch.testing.assert_close(
        attention_location.attention.sum(-1),
        torch.ones_like(attention_location.attention[..., 0]),
        atol=1e-6,
        rtol=0,
    )
    prefix = world.prefix(world.targets[0])
    transport_only = capture_source_location_buckets(
        model,
        attention_flow.clean_cache,
        prefix.units,
        attention_flow.row_position,
        response_start=prefix.response_start,
        evidence_unit_id=prefix.evidence_unit_id,
        local_window=1,
        query_chunk=2,
    )
    assert transport_only.downstream_action is None
    assert transport_only.source_downstream_action is None
    torch.testing.assert_close(transport_only.attention, attention_location.attention)
    torch.testing.assert_close(transport_only.transport, attention_location.transport)


def test_discovery_codes_are_empty_and_cut_materializes_retained_edges() -> None:
    model = tiny_model()
    world = _valid_native_world(model)
    target = world.targets[0]
    flow, gradients = native_flow_screen(
        model,
        world,
        target,
        "message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        max_rows=16,
        max_edges_per_head_row=1,
    )

    assert flow.edges.clean_code.shape == (flow.edges.count, 0)
    assert flow.edges.corrupt_code.shape == (flow.edges.count, 0)
    coordinates = torch.stack(
        (
            flow.edges.layer.long(),
            flow.edges.head.long(),
            flow.edges.target.long(),
        ),
        dim=1,
    )
    _, counts = torch.unique(coordinates, dim=0, return_counts=True)
    assert int(counts.max()) <= 2
    assert bool((flow.row_retained <= flow.row_total + 1e-6).all())

    prefix = world.prefix(target)
    gate = source_gate(prefix, (prefix.evidence_unit_id[0],))
    cut = gated_forward_cache(model, flow.clean_cache, gate)
    materialized = attach_cut_edge_codes(
        model,
        flow.edges,
        cut,
        gradients,
        gate.source_mask,
        clean_cache=flow.clean_cache,
        query_chunk=1,
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert materialized.clean_code.shape == (flow.edges.count, head_dim)
    assert materialized.corrupt_code.shape == (flow.edges.count, head_dim)
    torch.testing.assert_close(
        materialized.attention_clean,
        flow.edges.attention_clean,
    )
    torch.testing.assert_close(
        materialized.clean_target_score,
        flow.edges.clean_target_score,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        materialized.clean_message_norm,
        flow.edges.clean_message_norm,
        atol=1e-6,
        rtol=1e-5,
    )
    deleted = gate.source_mask.index_select(0, materialized.source.long())
    assert bool((materialized.corrupt_code[deleted] == 0).all())


def test_response_scope_gradients_cover_only_represented_rows() -> None:
    model = tiny_model()
    world = _valid_native_world(model)
    flow, gradients = native_flow_screen(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="response",
        coverage=0.9,
        query_chunk=2,
        max_rows=16,
        max_edges_per_head_row=1,
    )

    torch.testing.assert_close(gradients.position, flow.row_position)
    assert int(gradients.position.min()) == world.response_start - 1
