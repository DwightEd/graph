from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.head_resolved_shortcut_route.route_artifact import (
    RouteArtifact,
    RouteReadout,
    load_route_artifact,
    save_route_artifact,
    validate_artifact,
)
from experiments.head_resolved_shortcut_route.route_capture import (
    NativeRouteObserver,
    _logits_keep_arguments,
    _select_event_logits,
    endpoint_boundary_error,
)
from experiments.head_resolved_shortcut_route.route_pipeline import (
    CapturedRouteOperators,
    build_route_artifact,
)
from experiments.head_resolved_shortcut_route.route_shortcut import (
    EVIDENCE,
    EVIDENCE_PROMPT,
    NUMERIC,
    OTHER_PROMPT,
    QUESTION,
    RESPONSE,
    RESPONSE_HISTORY,
    SUPPORT,
    VETO,
    LayerRoutes,
    concatenate_sparse_routes,
    measure_layer_routes,
    moments_from_layers,
    moments_from_sparse,
    prediction_events,
    route_axes,
    route_axes_from_sparse,
    sparsify_routes,
    token_carriers,
)
from experiments.head_resolved_shortcut_route.route_suffix import (
    ObservedSuffix,
    injection_contribution,
    reverse_observed_suffix,
    symmetric_swiglu_adjoint,
    symmetric_swiglu_root_write,
)


def layer_routes(
    root_phi: torch.Tensor,
    *,
    layer: int = 0,
    query_position: torch.Tensor | None = None,
    source_position: torch.Tensor | None = None,
    carrier: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    value_energy: torch.Tensor | None = None,
    message_norm: torch.Tensor | None = None,
) -> LayerRoutes:
    """Build a transparent exact route table for score-level tests."""

    root_phi = root_phi.float()
    events, heads, sources, roots = root_phi.shape
    assert roots == 4
    if source_position is None:
        source_position = torch.arange(sources)
    if query_position is None:
        query_position = torch.full((events,), int(source_position.max()) + 1)
    if carrier is None:
        carrier = torch.full((sources,), OTHER_PROMPT, dtype=torch.long)
    causal = source_position[None] < query_position[:, None]
    shape = (events, heads, sources)
    if attention is None:
        attention = causal[:, None].expand(shape).float()
    if value_energy is None:
        value_energy = causal[:, None].expand(shape).float()
    if message_norm is None:
        message_norm = root_phi[..., :NUMERIC].sum(-1).abs()
    return LayerRoutes(
        layer=layer,
        query_position=query_position,
        source_position=source_position,
        carrier=carrier,
        causal=causal,
        attention=attention.float(),
        value_energy=value_energy.float(),
        physical_message_norm=message_norm.float(),
        root_phi=root_phi,
    )


def test_prediction_events_are_q_to_q_plus_one():
    events = prediction_events(torch.tensor([10, 11, 12, 13, 14]), response_start=2)

    assert events.query_position.tolist() == [1, 2, 3]
    assert events.prediction_position.tolist() == [2, 3, 4]
    assert events.target_token_id.tolist() == [12, 13, 14]
    assert torch.equal(events.prediction_position, events.query_position + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_prediction_events_stay_on_the_input_device():
    token_ids = torch.tensor([10, 11, 12], device="cuda")

    events = prediction_events(token_ids, response_start=1)

    assert events.query_position.device == token_ids.device
    assert events.prediction_position.device == token_ids.device
    assert events.target_token_id.device == token_ids.device


def test_operator_boundary_validity_is_endpoint_local_not_prefix_global():
    boundary_error = [
        torch.tensor([0.9, 0.02, 0.03]),
        torch.tensor([0.8, 0.04, 0.01]),
    ]

    error = endpoint_boundary_error(boundary_error, torch.tensor([1, 2]))

    torch.testing.assert_close(error, torch.tensor([0.04, 0.03]))


def test_logits_keep_arguments_support_transformers_api_generations():
    class CurrentModel:
        def forward(self, *, logits_to_keep=0, **_kwargs):
            return logits_to_keep

    class EarlierModel:
        def forward(self, *, num_logits_to_keep=0, **_kwargs):
            return num_logits_to_keep

    class LegacyModel:
        def forward(self, **_kwargs):
            return None

    assert _logits_keep_arguments(CurrentModel(), 296) == {"logits_to_keep": 296}
    assert _logits_keep_arguments(EarlierModel(), 296) == {"num_logits_to_keep": 296}
    assert _logits_keep_arguments(LegacyModel(), 296) == {}


def test_event_logits_accept_compact_rows_or_select_full_query_rows():
    full = torch.arange(5 * 7, dtype=torch.float32).reshape(5, 7)
    query = torch.tensor([2, 3, 4])
    compact = full.index_select(0, query)

    assert _select_event_logits(compact, query, source_count=5) is compact
    torch.testing.assert_close(
        _select_event_logits(full, query, source_count=5), compact
    )
    with pytest.raises(RuntimeError, match="actual=4, events=3, sources=5"):
        _select_event_logits(full[:4], query, source_count=5)


def test_true_avwo_edges_use_native_gqa_and_strict_first_arrival():
    generator = torch.Generator().manual_seed(17)
    events, heads, sources, roots = 2, 4, 4, 4
    kv_heads, head_dim, hidden = 2, 2, 6
    attention = torch.rand(events, heads, sources, generator=generator)
    root_values = torch.randn(sources, roots, kv_heads, head_dim, generator=generator)
    output_weight = torch.randn(hidden, heads * head_dim, generator=generator)
    suffix_adjoint = torch.randn(events, hidden, generator=generator)
    query = torch.tensor([2, 3])
    source = torch.tensor([0, 1, 2, 4])
    carrier = token_carriers(
        source,
        response_start=2,
        evidence_mask=torch.tensor([True, False, False, False]),
    )

    actual = measure_layer_routes(
        layer=3,
        attention=attention,
        root_values=root_values,
        output_weight=output_weight,
        suffix_adjoint=suffix_adjoint,
        query_position=query,
        source_position=source,
        carrier=carrier,
    )

    expected_phi = torch.zeros_like(actual.root_phi)
    expected_norm = torch.zeros_like(actual.physical_message_norm)
    expected_energy = torch.zeros_like(actual.value_energy)
    expected_attention = torch.zeros_like(actual.attention)
    for event in range(events):
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            block = output_weight[:, head * head_dim : (head + 1) * head_dim]
            for source_index in range(sources):
                if source[source_index] >= query[event]:
                    continue
                weight = attention[event, head, source_index]
                rooted_message = weight * F.linear(
                    root_values[source_index, :, kv_head], block
                )
                expected_phi[event, head, source_index] = (
                    rooted_message * suffix_adjoint[event]
                ).sum(-1)
                expected_norm[event, head, source_index] = rooted_message.sum(0).norm()
                expected_energy[event, head, source_index] = (
                    root_values[source_index, :, kv_head].sum(0).norm()
                )
                expected_attention[event, head, source_index] = weight

    assert actual.layer == 3
    assert torch.equal(
        actual.causal,
        torch.tensor([[True, True, False, False], [True, True, True, False]]),
    )
    torch.testing.assert_close(actual.root_phi, expected_phi)
    torch.testing.assert_close(actual.physical_message_norm, expected_norm)
    torch.testing.assert_close(actual.value_energy, expected_energy)
    torch.testing.assert_close(actual.attention, expected_attention)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_layer_route_stream_restores_cpu_capture_to_weight_device():
    generator = torch.Generator().manual_seed(18)
    attention = torch.rand(2, 4, 4, generator=generator)
    root_values = torch.randn(4, 4, 2, 2, generator=generator)
    output_weight = torch.randn(6, 8, generator=generator)
    suffix_adjoint = torch.randn(2, 6, generator=generator)
    query = torch.tensor([2, 3])
    source = torch.arange(4)
    carrier = token_carriers(
        source,
        response_start=2,
        evidence_mask=torch.tensor([True, False, False, False]),
    )
    arguments = {
        "layer": 0,
        "attention": attention,
        "root_values": root_values,
        "suffix_adjoint": suffix_adjoint,
        "query_position": query,
        "source_position": source,
        "carrier": carrier,
    }

    expected = measure_layer_routes(output_weight=output_weight, **arguments)
    actual = measure_layer_routes(
        output_weight=output_weight.to("cuda"),
        **arguments,
    )

    for field in fields(expected):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        if isinstance(expected_value, torch.Tensor):
            assert actual_value.device.type == "cuda"
            torch.testing.assert_close(actual_value.cpu(), expected_value)
        else:
            assert actual_value == expected_value


def test_suffix_adjoint_keeps_predictor_self_mlp_and_residual_paths():
    generator = torch.Generator().manual_seed(23)
    events, layers, hidden, heads, kv_heads = 2, 2, 4, 4, 2
    head_dim, intermediate, sources = 1, 5, 3
    query = torch.tensor([1, 2])
    attention, attention_scale, mlp_scale = [], [], []
    native_gate, native_up = [], []
    value_weight, output_weight = [], []
    gate_weight, up_weight, down_weight = [], [], []
    for _ in range(layers):
        row = torch.rand(events, heads, sources, generator=generator)
        source = torch.arange(sources)
        row *= source[None, None] <= query[:, None, None]
        row /= row.sum(dim=-1, keepdim=True)
        attention.append(row)
        attention_scale.append(torch.rand(events, hidden, generator=generator) + 0.5)
        mlp_scale.append(torch.rand(events, hidden, generator=generator) + 0.5)
        native_gate.append(torch.randn(events, intermediate, generator=generator))
        native_up.append(torch.randn(events, intermediate, generator=generator))
        value_weight.append(
            torch.randn(kv_heads * head_dim, hidden, generator=generator)
        )
        output_weight.append(torch.randn(hidden, heads * head_dim, generator=generator))
        gate_weight.append(torch.randn(intermediate, hidden, generator=generator))
        up_weight.append(torch.randn(intermediate, hidden, generator=generator))
        down_weight.append(torch.randn(hidden, intermediate, generator=generator))
    final_scale = torch.rand(events, hidden, generator=generator) + 0.5
    direction = torch.randn(events, hidden, generator=generator)
    observed = ObservedSuffix(
        query,
        attention,
        attention_scale,
        mlp_scale,
        native_gate,
        native_up,
        value_weight,
        output_weight,
        gate_weight,
        up_weight,
        down_weight,
        final_scale,
        direction,
    )
    actual = reverse_observed_suffix(observed)

    state = torch.randn(events, hidden, generator=generator, requires_grad=True)
    arrivals = [
        torch.randn(events, hidden, generator=generator, requires_grad=True)
        for _ in range(layers)
    ]
    layer_injections = [
        torch.randn(events, hidden, generator=generator, requires_grad=True)
        for _ in range(layers)
    ]
    current = state
    head_to_kv = torch.arange(heads) // (heads // kv_heads)
    for layer in range(layers):
        normalized = current * attention_scale[layer]
        value = F.linear(normalized, value_weight[layer]).reshape(
            events, kv_heads, head_dim
        )
        self_weight = attention[layer][
            torch.arange(events)[:, None], torch.arange(heads)[None], query[:, None]
        ]
        context = self_weight[..., None] * value[:, head_to_kv]
        current = current + F.linear(context.flatten(1), output_weight[layer])
        current = current + arrivals[layer]
        normalized = current * mlp_scale[layer]
        gate_delta = F.linear(normalized, gate_weight[layer])
        up_delta = F.linear(normalized, up_weight[layer])
        interaction = (
            0.5
            * torch.sigmoid(native_gate[layer])
            * (gate_delta * native_up[layer] + native_gate[layer] * up_delta)
        )
        current = current + F.linear(interaction, down_weight[layer])
        current = current + layer_injections[layer]
    score = (current * final_scale * direction).sum()
    gradients = torch.autograd.grad(score, [state, *arrivals, *layer_injections])
    expected_input = gradients[0]
    expected_arrival = gradients[1 : 1 + layers]
    expected_layer_output = gradients[1 + layers :]

    torch.testing.assert_close(actual.input, expected_input)
    for observed_adjoint, expected in zip(
        actual.attention_write, expected_arrival, strict=True
    ):
        torch.testing.assert_close(observed_adjoint, expected)
    for observed_adjoint, expected in zip(
        actual.layer_output, expected_layer_output, strict=True
    ):
        torch.testing.assert_close(observed_adjoint, expected)

    wrong_layer_order = reverse_observed_suffix(
        replace(observed, output_weight=tuple(reversed(observed.output_weight)))
    )
    assert not torch.allclose(wrong_layer_order.input, expected_input)

    roots = torch.randn(events, 4, hidden, generator=generator)
    torch.testing.assert_close(
        injection_contribution(roots, actual.input).sum(dim=1),
        (roots.sum(dim=1) * expected_input).sum(dim=1),
    )


def test_blocked_swiglu_root_write_matches_dense_formula_without_mutation():
    generator = torch.Generator().manual_seed(2301)
    tokens, roots, hidden, intermediate = 7, 4, 5, 11
    normalized = torch.randn(tokens, roots, hidden, generator=generator)
    gate_weight = torch.randn(intermediate, hidden, generator=generator)
    up_weight = torch.randn(intermediate, hidden, generator=generator)
    down_weight = torch.randn(hidden, intermediate, generator=generator)
    native_gate = torch.randn(tokens, intermediate, generator=generator)
    native_up = torch.randn(tokens, intermediate, generator=generator)
    inputs = (
        normalized,
        native_gate,
        native_up,
        gate_weight,
        up_weight,
        down_weight,
    )
    saved = tuple(value.clone() for value in inputs)
    gate_roots = normalized @ gate_weight.T
    up_roots = normalized @ up_weight.T
    dense = (
        0.5
        * torch.sigmoid(native_gate).unsqueeze(1)
        * (gate_roots * native_up.unsqueeze(1) + native_gate.unsqueeze(1) * up_roots)
    ) @ down_weight.T

    blocked = symmetric_swiglu_root_write(
        *inputs,
        token_block_size=3,
        intermediate_block_size=4,
    )

    assert blocked.shape == (tokens, roots, hidden)
    assert blocked.dtype == torch.float32
    torch.testing.assert_close(blocked, dense, rtol=1e-5, atol=1e-6)
    for current, original in zip(inputs, saved, strict=True):
        assert torch.equal(current, original)


def test_blocked_swiglu_forward_is_transpose_of_observed_adjoint():
    generator = torch.Generator().manual_seed(2302)
    tokens, roots, hidden, intermediate = 5, 4, 6, 13
    input_roots = torch.randn(
        tokens, roots, hidden, generator=generator, requires_grad=True
    )
    multiplier = torch.rand(tokens, hidden, generator=generator) + 0.5
    gate_weight = torch.randn(intermediate, hidden, generator=generator)
    up_weight = torch.randn(intermediate, hidden, generator=generator)
    down_weight = torch.randn(hidden, intermediate, generator=generator)
    complete = input_roots.detach().sum(dim=1) * multiplier
    native_gate = F.linear(complete, gate_weight).detach()
    native_up = F.linear(complete, up_weight).detach()
    downstream = torch.randn(tokens, hidden, generator=generator)
    normalized_roots = input_roots * multiplier[:, None]
    root_write = symmetric_swiglu_root_write(
        normalized_roots,
        native_gate,
        native_up,
        gate_weight,
        up_weight,
        down_weight,
        token_block_size=2,
        intermediate_block_size=5,
    )
    score = ((input_roots + root_write) * downstream[:, None]).sum()
    (actual_gradient,) = torch.autograd.grad(score, input_roots)
    expected = symmetric_swiglu_adjoint(
        downstream,
        multiplier,
        native_gate,
        native_up,
        gate_weight,
        up_weight,
        down_weight,
        token_block_size=2,
        intermediate_block_size=5,
    )

    torch.testing.assert_close(
        actual_gradient,
        expected[:, None].expand_as(actual_gradient),
        rtol=1e-5,
        atol=1e-6,
    )
    native_write = F.linear(F.silu(native_gate) * native_up, down_weight)
    torch.testing.assert_close(
        root_write.sum(dim=1), native_write, rtol=3e-5, atol=5e-6
    )


def test_route_pipeline_closes_injection_and_strict_arrival_to_terminal_roots():
    query = torch.tensor([1])
    events = prediction_events(torch.tensor([7, 8, 9]), response_start=2)
    attention = [torch.tensor([[[0.4, 0.6]]])]
    attention_scale = [torch.tensor([[2.0]])]
    mlp_scale = [torch.tensor([[7.0]])]
    native_gate = [torch.tensor([[0.2]])]
    native_up = [torch.tensor([[0.4]])]
    value_weight = [torch.tensor([[3.0]])]
    output_weight = [torch.tensor([[5.0]])]
    gate_weight = [torch.tensor([[11.0]])]
    up_weight = [torch.tensor([[13.0]])]
    down_weight = [torch.tensor([[17.0]])]
    final_scale = torch.tensor([[19.0]])
    direction = torch.tensor([[23.0]])
    suffix = ObservedSuffix(
        query,
        attention,
        attention_scale,
        mlp_scale,
        native_gate,
        native_up,
        value_weight,
        output_weight,
        gate_weight,
        up_weight,
        down_weight,
        final_scale,
        direction,
    )
    root_values = [torch.zeros(2, 4, 1, 1)]
    root_values[0][0, EVIDENCE, 0, 0] = 2
    input_roots = torch.zeros(1, 4, 1)
    input_roots[0, QUESTION, 0] = 1.5

    lam = final_scale[0, 0] * direction[0, 0]
    sigmoid = torch.sigmoid(native_gate[0][0, 0])
    eta = lam + 0.5 * mlp_scale[0][0, 0] * (
        sigmoid * native_up[0][0, 0] * lam * down_weight[0][0, 0] * gate_weight[0][0, 0]
        + sigmoid
        * native_gate[0][0, 0]
        * lam
        * down_weight[0][0, 0]
        * up_weight[0][0, 0]
    )
    input_adjoint = eta + (
        attention_scale[0][0, 0]
        * attention[0][0, 0, 1]
        * eta
        * output_weight[0][0, 0]
        * value_weight[0][0, 0]
    )
    terminal = torch.zeros(1, 4)
    terminal[0, EVIDENCE] = (
        attention[0][0, 0, 0]
        * output_weight[0][0, 0]
        * root_values[0][0, EVIDENCE, 0, 0]
        * eta
    )
    terminal[0, QUESTION] = input_roots[0, QUESTION, 0] * input_adjoint
    captured = CapturedRouteOperators(
        response_start=2,
        events=events,
        source_token_id=torch.tensor([7, 8]),
        competitor_token_id=torch.tensor([0]),
        target_logprob=torch.tensor([-1.0]),
        source_position=torch.tensor([0, 1]),
        evidence_mask=torch.tensor([True, False]),
        root_values=root_values,
        input_roots=input_roots,
        suffix=suffix,
        self_value_numeric_input=torch.zeros(1, 1, 1, 1),
        post_attention_numeric_write=torch.zeros(1, 1, 1),
        layer_numeric_write=torch.zeros(1, 1, 1),
        final_rms_numeric_write=torch.zeros(1, 1),
        terminal_root_margin=terminal,
        native_margin=terminal.sum(dim=1),
        operator_error=torch.zeros(1),
        operator_valid=torch.ones(1, dtype=torch.bool),
    )

    artifact = build_route_artifact(captured, top_k=2, cover_mass=1.0)

    torch.testing.assert_close(artifact.readout.root_closure_error, torch.zeros(1, 4))
    torch.testing.assert_close(artifact.readout.terminal_root_margin, terminal)
    assert artifact.readout.operator_valid.tolist() == [True]
    assert artifact.routes.source_position.tolist() == [0]

    near_zero_terminal = terminal.clone()
    near_zero_terminal[:, NUMERIC] = 5e-8
    near_zero = build_route_artifact(
        replace(
            captured,
            terminal_root_margin=near_zero_terminal,
            native_margin=near_zero_terminal.sum(dim=1),
        ),
        top_k=2,
        cover_mass=1.0,
    )
    assert near_zero.readout.operator_valid.tolist() == [True]
    torch.testing.assert_close(
        near_zero.readout.root_closure_error[:, NUMERIC], torch.tensor([5e-8])
    )
    missing_route = build_route_artifact(
        replace(captured, root_values=[torch.zeros_like(root_values[0])]),
        top_k=2,
        cover_mass=1.0,
    )
    assert missing_route.readout.operator_valid.tolist() == [False]
    local_terminal = terminal.clone()
    local_terminal[:, NUMERIC] = 0.25 * eta
    with_local_write = replace(
        captured,
        post_attention_numeric_write=torch.full((1, 1, 1), 0.25),
        terminal_root_margin=local_terminal,
        native_margin=local_terminal.sum(dim=1),
    )
    closed_local = build_route_artifact(
        with_local_write,
        top_k=2,
        cover_mass=1.0,
    )
    assert closed_local.readout.operator_valid.tolist() == [True]
    torch.testing.assert_close(
        closed_local.readout.root_closure_error, torch.zeros(1, 4)
    )
    dropped_local = build_route_artifact(
        replace(
            with_local_write,
            post_attention_numeric_write=torch.zeros(1, 1, 1),
        ),
        top_k=2,
        cover_mass=1.0,
    )
    torch.testing.assert_close(
        dropped_local.readout.root_closure_error[:, NUMERIC],
        (0.25 * eta).reshape(1),
    )
    assert dropped_local.readout.operator_valid.tolist() == [False]
    for field, invalid in (
        ("self_value_numeric_input", torch.zeros(1, 2, 1, 1)),
        ("post_attention_numeric_write", torch.zeros(1, 1, 2)),
        ("layer_numeric_write", torch.zeros(1, 1, 2)),
        ("final_rms_numeric_write", torch.zeros(2, 1)),
    ):
        with pytest.raises(ValueError):
            build_route_artifact(replace(captured, **{field: invalid}))


def test_self_value_numeric_variation_does_not_cancel_across_heads():
    events = prediction_events(torch.tensor([7, 8, 9]), response_start=2)
    zeros = [torch.zeros(1, 1)]
    suffix = ObservedSuffix(
        query_position=events.query_position,
        attention=[torch.zeros(1, 2, 2)],
        attention_rms_multiplier=[torch.ones(1, 1)],
        mlp_rms_multiplier=[torch.ones(1, 1)],
        native_gate=zeros,
        native_up=zeros,
        value_weight=[torch.zeros(1, 1)],
        output_weight=[torch.ones(1, 2)],
        gate_weight=zeros,
        up_weight=zeros,
        down_weight=zeros,
        final_rms_multiplier=torch.ones(1, 1),
        readout_direction=torch.ones(1, 1),
    )
    self_value_write = torch.tensor([[[[1.0], [-1.0]]]])
    captured = CapturedRouteOperators(
        response_start=2,
        events=events,
        source_token_id=torch.tensor([7, 8]),
        competitor_token_id=torch.tensor([0]),
        target_logprob=torch.tensor([-1.0]),
        source_position=torch.tensor([0, 1]),
        evidence_mask=torch.tensor([True, False]),
        root_values=[torch.zeros(2, 4, 1, 1)],
        input_roots=torch.zeros(1, 4, 1),
        suffix=suffix,
        self_value_numeric_input=self_value_write,
        post_attention_numeric_write=torch.zeros(1, 1, 1),
        layer_numeric_write=torch.zeros(1, 1, 1),
        final_rms_numeric_write=torch.zeros(1, 1),
        terminal_root_margin=torch.zeros(1, 4),
        native_margin=torch.zeros(1),
        operator_error=torch.zeros(1),
        operator_valid=torch.ones(1, dtype=torch.bool),
    )

    artifact = build_route_artifact(captured, top_k=0)

    torch.testing.assert_close(
        artifact.readout.numeric_self_v_phi,
        torch.tensor([[[1.0, -1.0]]]),
    )
    torch.testing.assert_close(
        artifact.readout.injection_phi[:, NUMERIC], torch.zeros(1)
    )
    torch.testing.assert_close(
        artifact.readout.numeric_total_variation, torch.tensor([2.0])
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_native_observer_closes_local_numeric_writes_in_one_forward(
    dtype: torch.dtype,
    tmp_path,
):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    config = transformers.LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.LlamaForCausalLM(config).to(dtype).eval()
    observer = NativeRouteObserver(
        model,
        root_token_block_size=2,
        root_intermediate_block_size=3,
    )
    forward_calls: list[int] = []
    native_values: list[torch.Tensor] = []
    handles = [
        model.register_forward_hook(lambda *_: forward_calls.append(1)),
        *[
            layer.self_attn.v_proj.register_forward_hook(
                lambda _module, _args, output: native_values.append(
                    output[0].detach().float()
                )
            )
            for layer in model.model.layers
        ],
    ]
    hooks_before = sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )
    try:
        captured = observer.capture(
            [1, 2, 3, 4, 5, 6, 7],
            response_start=3,
            evidence_mask=[True, False, False],
        )
        hooks_after = sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks)
            for module in model.modules()
        )
    finally:
        for handle in handles:
            handle.remove()

    assert len(forward_calls) == 1
    assert hooks_after == hooks_before
    assert captured.events.query_position.tolist() == [2, 3, 4, 5]
    assert captured.events.prediction_position.tolist() == [3, 4, 5, 6]
    assert captured.self_value_numeric_input.shape == (4, 2, 4, 4)
    assert captured.post_attention_numeric_write.shape == (4, 2, 16)
    assert captured.layer_numeric_write.shape == (4, 2, 16)
    assert captured.final_rms_numeric_write.shape == (4, 16)
    for rooted, native in zip(captured.root_values, native_values, strict=True):
        torch.testing.assert_close(
            rooted.sum(dim=1),
            native.reshape(len(captured.source_token_id), 2, 4),
            rtol=0,
            atol=1e-6,
        )

    artifact = build_route_artifact(captured, top_k=8, cover_mass=1.0)
    assert captured.operator_valid.all()
    assert artifact.readout.operator_valid.all()
    torch.testing.assert_close(
        artifact.readout.root_closure_error,
        torch.zeros_like(artifact.readout.root_closure_error),
        rtol=0,
        atol=1e-6,
    )
    native_error = (
        artifact.readout.terminal_root_margin.sum(dim=1)
        - artifact.readout.native_margin
    ).abs()
    torch.testing.assert_close(native_error, captured.operator_error)
    closure_limit = max(1e-3, float(torch.finfo(dtype).eps))
    assert (
        native_error
        <= closure_limit * artifact.readout.native_margin.abs().clamp_min(1.0)
    ).all()
    assert (
        artifact.readout.numeric_total_variation
        >= artifact.readout.injection_phi[:, NUMERIC].abs() - 1e-7
    ).all()
    torch.testing.assert_close(
        artifact.axes.resolution,
        artifact.readout.numeric_total_variation
        + captured.operator_error
        + artifact.readout.root_closure_error.abs().sum(dim=1),
    )

    path = tmp_path / f"native_{dtype}.npz"
    save_route_artifact(path, artifact)
    loaded = load_route_artifact(path)
    torch.testing.assert_close(
        loaded.readout.numeric_total_variation,
        artifact.readout.numeric_total_variation,
    )
    recomputed = route_axes_from_sparse(
        loaded.routes,
        loaded.readout.injection_phi,
        event_valid=loaded.readout.operator_valid,
        resolution=(loaded.axes.resolution - loaded.readout.numeric_total_variation),
        numeric_total_variation=loaded.readout.numeric_total_variation,
    )
    for field in fields(loaded.axes):
        torch.testing.assert_close(
            getattr(recomputed, field.name),
            getattr(loaded.axes, field.name),
            equal_nan=True,
        )

    wrong_suffix = replace(
        captured.suffix,
        output_weight=tuple(reversed(captured.suffix.output_weight)),
    )
    wrong_projection = build_route_artifact(
        replace(captured, suffix=wrong_suffix),
        top_k=8,
        cover_mass=1.0,
    )
    wrong_arrival = (
        wrong_projection.readout.terminal_root_margin
        - wrong_projection.readout.root_closure_error
    )
    wrong_tolerance = 1e-6 + 1e-3 * (
        wrong_projection.readout.terminal_root_margin.abs() + wrong_arrival.abs()
    )
    violated = (
        wrong_projection.readout.root_closure_error.abs() > wrong_tolerance
    ).any(dim=1)
    assert violated.any()
    assert (~wrong_projection.readout.operator_valid[violated]).all()


def test_native_observer_selects_queries_when_model_returns_full_logits(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(11)
    config = transformers.LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    arguments = {
        "token_ids": [1, 2, 3, 4, 5, 6, 7],
        "response_start": 4,
        "evidence_mask": [True, False, False, False],
    }
    expected = NativeRouteObserver(
        model,
        root_token_block_size=2,
        root_intermediate_block_size=3,
    ).capture(**arguments)
    native_forward = model.forward
    returned_rows: list[int] = []

    def full_logits_forward(*args, **kwargs):
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("num_logits_to_keep", None)
        result = native_forward(*args, **kwargs)
        returned_rows.append(result.logits.shape[1])
        return result

    monkeypatch.setattr(model, "forward", full_logits_forward)
    actual = NativeRouteObserver(
        model,
        root_token_block_size=2,
        root_intermediate_block_size=3,
    ).capture(**arguments)

    assert returned_rows == [len(arguments["token_ids"]) - 1]
    assert actual.events.query_position.tolist() == [3, 4, 5]
    assert actual.events.prediction_position.tolist() == [4, 5, 6]
    assert actual.events.target_token_id.tolist() == [5, 6, 7]
    assert torch.equal(actual.competitor_token_id, expected.competitor_token_id)
    assert torch.equal(actual.operator_valid, expected.operator_valid)
    torch.testing.assert_close(actual.target_logprob, expected.target_logprob)
    torch.testing.assert_close(actual.native_margin, expected.native_margin)
    torch.testing.assert_close(
        actual.terminal_root_margin, expected.terminal_root_margin
    )


def test_current_target_embedding_cannot_change_its_predictor_trace():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    config = transformers.LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    observer = NativeRouteObserver(transformers.LlamaForCausalLM(config).eval())
    original = observer.capture(
        [1, 2, 3, 4, 5, 6, 7],
        response_start=3,
        evidence_mask=[True, False, False],
    )
    embedding = observer.model.get_input_embeddings().weight
    saved_embedding = embedding[5].clone()
    try:
        with torch.no_grad():
            embedding[5].add_(
                torch.linspace(0.1, 0.2, embedding.shape[1], dtype=embedding.dtype)
            )
        changed = observer.capture(
            [1, 2, 3, 4, 5, 6, 7],
            response_start=3,
            evidence_mask=[True, False, False],
        )
    finally:
        with torch.no_grad():
            embedding[5].copy_(saved_embedding)
    event = 1
    prediction = int(original.events.prediction_position[event])

    assert prediction == 4
    assert original.events.query_position[event] == prediction - 1
    assert torch.equal(original.events.target_token_id, changed.events.target_token_id)
    torch.testing.assert_close(
        original.input_roots[event], changed.input_roots[event], rtol=0, atol=0
    )
    for left, right in zip(original.root_values, changed.root_values, strict=True):
        torch.testing.assert_close(
            left[:prediction], right[:prediction], rtol=0, atol=0
        )
    assert any(
        not torch.equal(left[prediction], right[prediction])
        for left, right in zip(original.root_values, changed.root_values, strict=True)
    )
    for field in (
        "attention",
        "attention_rms_multiplier",
        "mlp_rms_multiplier",
        "native_gate",
        "native_up",
    ):
        for left, right in zip(
            getattr(original.suffix, field),
            getattr(changed.suffix, field),
            strict=True,
        ):
            torch.testing.assert_close(left[event], right[event], rtol=0, atol=0)
    for field in (
        "self_value_numeric_input",
        "post_attention_numeric_write",
        "layer_numeric_write",
        "final_rms_numeric_write",
    ):
        torch.testing.assert_close(
            getattr(original, field)[event],
            getattr(changed, field)[event],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        original.suffix.final_rms_multiplier[event],
        changed.suffix.final_rms_multiplier[event],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.suffix.readout_direction[event],
        changed.suffix.readout_direction[event],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.terminal_root_margin[event],
        changed.terminal_root_margin[event],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.native_margin[event], changed.native_margin[event], rtol=0, atol=0
    )


def test_support_and_veto_do_not_cancel_across_heads():
    phi = torch.zeros(1, 2, 1, 4)
    phi[0, 0, 0, EVIDENCE] = 3
    phi[0, 1, 0, EVIDENCE] = -2
    routes = layer_routes(
        phi,
        source_position=torch.tensor([0]),
        query_position=torch.tensor([1]),
        carrier=torch.tensor([EVIDENCE_PROMPT]),
    )

    torch.testing.assert_close(routes.physical_phi.sum(), torch.tensor(1.0))
    torch.testing.assert_close(routes.support[:, :, 0], torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(routes.veto[:, :, 0], torch.tensor([[0.0, 2.0]]))

    axes = route_axes([routes], torch.zeros(1, 4))
    torch.testing.assert_close(
        axes.root_carrier_mass[0, EVIDENCE_PROMPT, EVIDENCE],
        torch.tensor([3.0, 2.0]),
    )


def test_root_and_carrier_coordinates_separate_D_G_B_and_mixed_prompt():
    phi = torch.zeros(1, 1, 3, 4)
    phi[0, 0, 0, EVIDENCE] = 1  # D: evidence root, evidence carrier
    phi[0, 0, 1, EVIDENCE] = 2  # mixed: evidence root, other-prompt carrier
    phi[0, 0, 1, QUESTION] = 5
    phi[0, 0, 2, EVIDENCE] = 3  # G: evidence root, response carrier
    phi[0, 0, 2, QUESTION] = 7
    phi[0, 0, 2, RESPONSE] = 4  # B: response root, response carrier
    routes = layer_routes(
        phi,
        query_position=torch.tensor([3]),
        source_position=torch.tensor([0, 1, 2]),
        carrier=torch.tensor([EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY]),
    )

    axes = route_axes([routes], torch.zeros(1, 4))

    torch.testing.assert_close(axes.direct_evidence[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        axes.grounded_response_relay[0], torch.tensor([3.0, 0.0])
    )
    torch.testing.assert_close(axes.response_born_history[0], torch.tensor([4.0, 0.0]))
    assert axes.root_carrier_mass[0, OTHER_PROMPT, EVIDENCE, SUPPORT] == 2
    assert axes.root_carrier_mass[0, OTHER_PROMPT, NUMERIC].sum() == 0
    assert axes.response_born_takeover_defined[0, SUPPORT]
    torch.testing.assert_close(
        axes.response_born_takeover[0, SUPPORT], torch.tensor(4 / 14)
    )


def test_undefined_axes_are_nan_and_have_false_masks():
    phi = torch.zeros(1, 1, 3, 4)
    common = {
        "query_position": torch.tensor([3]),
        "source_position": torch.tensor([0, 1, 2]),
        "carrier": torch.tensor([EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY]),
    }
    layers = [layer_routes(phi, layer=index, **common) for index in range(2)]

    axes = route_axes(layers, torch.zeros(1, 4))

    for value, defined in (
        (axes.carrier_drift, axes.carrier_drift_defined),
        (axes.carrier_drift_by_head, axes.carrier_drift_head_defined),
        (axes.prompt_source_dispersion, axes.prompt_source_dispersion_defined),
        (
            axes.prompt_source_dispersion_by_layer_head,
            axes.prompt_source_dispersion_row_defined,
        ),
        (axes.response_born_takeover, axes.response_born_takeover_defined),
    ):
        assert not defined.any()
        assert torch.isnan(value).all()

    assert not axes.carrier_drift_map_defined.any()
    assert torch.isnan(axes.carrier_drift_map).all()
    assert not axes.response_born_takeover_row_defined.any()
    assert torch.isnan(axes.response_born_takeover_by_layer_head).all()


def test_prompt_dispersion_is_zero_for_one_source_and_one_for_uniform_sources():
    phi = torch.zeros(2, 1, 3, 4)
    phi[0, 0, :, EVIDENCE] = torch.tensor([3.0, 0.0, 0.0])
    phi[1, 0, :, EVIDENCE] = 1
    routes = layer_routes(
        phi,
        query_position=torch.tensor([3, 3]),
        source_position=torch.tensor([0, 1, 2]),
        carrier=torch.tensor([EVIDENCE_PROMPT, OTHER_PROMPT, OTHER_PROMPT]),
    )

    axes = route_axes([routes], torch.zeros(2, 4))

    assert axes.prompt_source_dispersion_defined[:, SUPPORT].all()
    torch.testing.assert_close(
        axes.prompt_source_dispersion[:, SUPPORT], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        axes.prompt_source_dispersion_by_layer_head[:, 0, 0, SUPPORT],
        torch.tensor([0.0, 1.0]),
    )
    assert not axes.prompt_source_dispersion_defined[:, VETO].any()
    assert torch.isnan(axes.prompt_source_dispersion[:, VETO]).all()


def test_prompt_dispersion_never_pools_sources_across_prediction_events():
    phi = torch.zeros(2, 1, 2, 4)
    phi[0, 0, 0, EVIDENCE] = 1
    phi[1, 0, 1, EVIDENCE] = 1
    routes = layer_routes(
        phi,
        query_position=torch.tensor([2, 3]),
        source_position=torch.tensor([0, 1]),
        carrier=torch.tensor([EVIDENCE_PROMPT, OTHER_PROMPT]),
    )

    axes = route_axes([routes], torch.zeros(2, 4))

    # Each prediction uses exactly one prompt endpoint. Pooling the two
    # events first would instead create a false uniform distribution H=1.
    assert axes.prompt_source_dispersion_defined[:, SUPPORT].all()
    torch.testing.assert_close(
        axes.prompt_source_dispersion[:, SUPPORT], torch.zeros(2)
    )
    torch.testing.assert_close(
        axes.prompt_source_dispersion_by_layer_head[:, 0, 0, SUPPORT],
        torch.zeros(2),
    )


@pytest.mark.parametrize(
    ("root", "expected"),
    ((EVIDENCE, 0.0), (QUESTION, 0.0), (RESPONSE, 1.0)),
)
@pytest.mark.parametrize(("sign", "side"), ((1.0, SUPPORT), (-1.0, VETO)))
def test_response_takeover_requires_response_ancestry_not_response_carrier(
    root: int,
    expected: float,
    sign: float,
    side: int,
):
    phi = torch.zeros(1, 1, 1, 4)
    phi[0, 0, 0, root] = 2 * sign
    routes = layer_routes(
        phi,
        query_position=torch.tensor([3]),
        source_position=torch.tensor([2]),
        carrier=torch.tensor([RESPONSE_HISTORY]),
    )

    axes = route_axes([routes], torch.zeros(1, 4))

    assert axes.response_born_takeover_defined[0, side]
    torch.testing.assert_close(
        axes.response_born_takeover[0, side], torch.tensor(expected)
    )
    other_side = VETO if side == SUPPORT else SUPPORT
    assert not axes.response_born_takeover_defined[0, other_side]
    assert torch.isnan(axes.response_born_takeover[0, other_side])


def test_response_takeover_is_computed_per_prediction_event():
    phi = torch.zeros(2, 1, 1, 4)
    phi[0, 0, 0, EVIDENCE] = 4
    phi[1, 0, 0, RESPONSE] = 4
    routes = layer_routes(
        phi,
        query_position=torch.tensor([3, 4]),
        source_position=torch.tensor([2]),
        carrier=torch.tensor([RESPONSE_HISTORY]),
    )

    axes = route_axes([routes], torch.zeros(2, 4))

    # A pooled response carrier would report 1/2 for both endpoints. The first
    # endpoint is a grounded E relay; only the second is response-born.
    assert axes.response_born_takeover_defined[:, SUPPORT].all()
    torch.testing.assert_close(
        axes.response_born_takeover[:, SUPPORT], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        axes.response_born_takeover_by_layer_head[:, 0, 0, SUPPORT],
        torch.tensor([0.0, 1.0]),
    )


def test_response_takeover_does_not_rescue_individually_unresolved_rows():
    phi = torch.zeros(1, 1, 1, 4)
    phi[0, 0, 0, RESPONSE] = 0.75
    common = {
        "query_position": torch.tensor([3]),
        "source_position": torch.tensor([2]),
        "carrier": torch.tensor([RESPONSE_HISTORY]),
    }
    layers = [
        layer_routes(phi, layer=0, **common),
        layer_routes(phi, layer=1, **common),
    ]

    axes = route_axes(layers, torch.zeros(1, 4), resolution=1.0)

    # Each row is below resolution even though their pooled mass is 1.5.
    assert not axes.response_born_takeover_row_defined.any()
    assert not axes.response_born_takeover_defined.any()
    assert torch.isnan(axes.response_born_takeover).all()


@pytest.mark.parametrize("layer_ids", ((1, 0), (0, 2)))
def test_route_axes_reject_reordered_or_incomplete_layers(layer_ids):
    phi = torch.zeros(1, 1, 1, 4)
    common = {
        "query_position": torch.tensor([1]),
        "source_position": torch.tensor([0]),
        "carrier": torch.tensor([EVIDENCE_PROMPT]),
    }
    layers = [layer_routes(phi, layer=layer, **common) for layer in layer_ids]

    with pytest.raises(ValueError, match="complete and ordered"):
        route_axes(layers, torch.zeros(1, 4))


def test_carrier_drift_keeps_support_and_veto_as_separate_depth_flows():
    early = torch.zeros(1, 1, 2, 4)
    late = torch.zeros_like(early)
    early[0, 0, 0, EVIDENCE] = 2
    early[0, 0, 1, RESPONSE] = -3
    late[0, 0, 0, EVIDENCE] = -5
    late[0, 0, 1, RESPONSE] = 4
    common = {
        "query_position": torch.tensor([3]),
        "source_position": torch.tensor([0, 2]),
        "carrier": torch.tensor([EVIDENCE_PROMPT, RESPONSE_HISTORY]),
    }

    axes = route_axes(
        [layer_routes(early, layer=0, **common), layer_routes(late, layer=1, **common)],
        torch.zeros(1, 4),
    )

    assert axes.carrier_drift_defined.all()
    torch.testing.assert_close(axes.carrier_drift, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(
        axes.carrier_drift_by_head, torch.tensor([[[1.0, -1.0]]])
    )


def test_carrier_drift_centroids_are_prediction_event_local():
    early = torch.zeros(2, 1, 2, 4)
    late = torch.zeros_like(early)
    early[0, 0, 0, EVIDENCE] = 100
    early[0, 0, 1, RESPONSE] = 1
    late[1, 0, 0, EVIDENCE] = 1
    late[1, 0, 1, RESPONSE] = 100
    common = {
        "query_position": torch.tensor([3, 4]),
        "source_position": torch.tensor([0, 2]),
        "carrier": torch.tensor([EVIDENCE_PROMPT, RESPONSE_HISTORY]),
    }

    axes = route_axes(
        [layer_routes(early, layer=0, **common), layer_routes(late, layer=1, **common)],
        torch.zeros(2, 4),
    )

    # Within each endpoint the prompt and response mass has the same depth,
    # hence both drifts are zero. Pooling endpoints first yields the spurious
    # Simpson-style drift 99/101.
    assert axes.carrier_drift_defined[:, SUPPORT].all()
    torch.testing.assert_close(axes.carrier_drift[:, SUPPORT], torch.zeros(2))
    torch.testing.assert_close(
        axes.carrier_drift_by_head[:, 0, SUPPORT], torch.zeros(2)
    )
    pooled_false_drift = torch.tensor(99 / 101)
    assert not torch.isclose(axes.carrier_drift[:, SUPPORT], pooled_false_drift).any()


def test_first_response_event_has_no_history_route_or_tail():
    response_start = 3
    events = prediction_events(
        torch.tensor([10, 11, 12, 13, 14, 15]),
        response_start=response_start,
    )
    source = torch.arange(6)
    carrier = token_carriers(
        source,
        response_start,
        torch.tensor([True, False, False, False, False, False]),
    )
    phi = torch.zeros(3, 1, 6, 4)
    phi[:, 0, 0, EVIDENCE] = 1
    phi[:, 0, 3, RESPONSE] = 50
    routes = layer_routes(
        phi,
        query_position=events.query_position,
        source_position=source,
        carrier=carrier,
    )

    axes = route_axes([routes], torch.zeros(3, 4))
    sparse = sparsify_routes([routes], top_k=2, cover_mass=1.0)

    assert events.query_position[0] == response_start - 1
    assert not (routes.causal[0] & (carrier == RESPONSE_HISTORY)).any()
    assert not (routes.causal[1] & (carrier == RESPONSE_HISTORY)).any()
    assert axes.root_carrier_mass[0, RESPONSE_HISTORY].count_nonzero() == 0
    assert axes.root_carrier_mass[1, RESPONSE_HISTORY].count_nonzero() == 0
    assert not axes.response_born_takeover_defined[0].any()
    assert torch.isnan(axes.response_born_takeover[0]).all()
    first_row = int(torch.nonzero(sparse.row_event == 0, as_tuple=False)[0])
    first_edges = slice(
        int(sparse.row_ptr[first_row]), int(sparse.row_ptr[first_row + 1])
    )
    assert not (sparse.carrier[first_edges] == RESPONSE_HISTORY).any()
    assert sparse.tail_count[first_row, RESPONSE_HISTORY] == 0

    # The last event can read response position 3, proving that the fixture
    # exercises history rather than globally removing response carriers.
    assert axes.response_born_takeover_defined[-1, SUPPORT]
    assert axes.response_born_takeover[-1, SUPPORT] == 1


def test_sparse_selection_uses_message_norm_and_tail_closes_by_carrier_and_root():
    phi = torch.tensor(
        [
            [
                [
                    [100.0, -1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [3.0, 0.0, -5.0, 0.0],
                    [-7.0, 0.0, 11.0, -13.0],
                ],
                [
                    [-6.0, 1.0, 0.0, 0.0],
                    [0.0, -4.0, 0.0, 0.0],
                    [2.0, 0.0, 3.0, 0.0],
                    [1.0, -2.0, -4.0, 5.0],
                ],
            ]
        ]
    )
    attention = torch.tensor([[[0.9, 0.1, 0.4, 0.3], [0.2, 0.8, 0.1, 0.4]]])
    energy = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    norm = torch.tensor([[[0.009, 10.0, 2.0, 3.0], [1.0, 2.0, 9.0, 8.0]]])
    carrier = torch.tensor(
        [EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY, RESPONSE_HISTORY]
    )
    routes = layer_routes(
        phi,
        layer=0,
        query_position=torch.tensor([4]),
        source_position=torch.arange(4),
        carrier=carrier,
        attention=attention,
        value_energy=energy,
        message_norm=norm,
    )

    sparse = sparsify_routes([routes], top_k=1)

    assert sparse.row_event.tolist() == [0, 0]
    assert sparse.row_layer.tolist() == [0, 0]
    assert sparse.row_head.tolist() == [0, 1]
    assert sparse.row_ptr.tolist() == [0, 1, 2]
    # Head 0 rejects the largest attention and largest |phi| in favour of the
    # true-message norm. Head 1 independently selects its own strongest edge.
    assert sparse.source_position.tolist() == [1, 2]

    for head, kept_source in enumerate((1, 2)):
        selected = slice(int(sparse.row_ptr[head]), int(sparse.row_ptr[head + 1]))
        omitted = torch.ones(4, dtype=torch.bool)
        omitted[kept_source] = False
        for role in (EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY):
            mask = omitted & (carrier == role)
            root_tail = phi[0, head, mask]
            physical_tail = root_tail[:, :NUMERIC].sum(-1)
            positive = physical_tail.clamp_min(0)
            negative = (-physical_tail).clamp_min(0)
            assert sparse.tail_count[head, role] == mask.sum()
            torch.testing.assert_close(
                sparse.tail_attention_sum[head, role], attention[0, head, mask].sum()
            )
            torch.testing.assert_close(
                sparse.tail_value_energy_sum[head, role], energy[0, head, mask].sum()
            )
            torch.testing.assert_close(
                sparse.tail_message_norm_sum[head, role], norm[0, head, mask].sum()
            )
            expected_max = (
                norm[0, head, mask].max() if mask.any() else norm.new_zeros(())
            )
            torch.testing.assert_close(
                sparse.tail_message_norm_max[head, role], expected_max
            )
            torch.testing.assert_close(
                sparse.tail_root_positive[head, role], root_tail.clamp_min(0).sum(0)
            )
            torch.testing.assert_close(
                sparse.tail_root_negative[head, role],
                (-root_tail).clamp_min(0).sum(0),
            )
            torch.testing.assert_close(
                sparse.tail_physical_positive[head, role], positive.sum()
            )
            torch.testing.assert_close(
                sparse.tail_physical_negative[head, role], negative.sum()
            )
            positive_xlogx = sum(
                value * math.log(value) for value in positive.tolist() if value > 0
            )
            negative_xlogx = sum(
                value * math.log(value) for value in negative.tolist() if value > 0
            )
            torch.testing.assert_close(
                sparse.tail_physical_pos_xlogx[head, role],
                sparse.tail_physical_pos_xlogx.new_tensor(positive_xlogx),
            )
            torch.testing.assert_close(
                sparse.tail_physical_neg_xlogx[head, role],
                sparse.tail_physical_neg_xlogx.new_tensor(negative_xlogx),
            )

        selected_phi = sparse.root_phi[selected]
        torch.testing.assert_close(
            selected_phi.clamp_min(0).sum(0) + sparse.tail_root_positive[head].sum(0),
            phi[0, head].clamp_min(0).sum(0),
        )
        torch.testing.assert_close(
            (-selected_phi).clamp_min(0).sum(0)
            + sparse.tail_root_negative[head].sum(0),
            (-phi[0, head]).clamp_min(0).sum(0),
        )
        torch.testing.assert_close(
            sparse.physical_message_norm[selected].sum()
            + sparse.tail_message_norm_sum[head].sum(),
            norm[0, head].sum(),
        )


def test_layerwise_sparse_stream_has_the_same_canonical_table_as_dense_call():
    generator = torch.Generator().manual_seed(2303)
    source = torch.arange(5)
    query = torch.tensor([3, 5])
    carrier = torch.tensor(
        [
            EVIDENCE_PROMPT,
            OTHER_PROMPT,
            OTHER_PROMPT,
            RESPONSE_HISTORY,
            RESPONSE_HISTORY,
        ]
    )
    layers = []
    for layer in range(3):
        phi = torch.randn(2, 2, 5, 4, generator=generator)
        message_norm = torch.rand(2, 2, 5, generator=generator)
        layers.append(
            layer_routes(
                phi,
                layer=layer,
                query_position=query,
                source_position=source,
                carrier=carrier,
                message_norm=message_norm,
            )
        )

    direct = sparsify_routes(layers, top_k=2, cover_mass=0.8)
    streamed = concatenate_sparse_routes(
        [sparsify_routes([route], top_k=2, cover_mass=0.8) for route in layers]
    )

    for field in fields(direct):
        torch.testing.assert_close(
            getattr(streamed, field.name), getattr(direct, field.name)
        )


def test_layerwise_moment_stream_matches_dense_signed_formula():
    generator = torch.Generator().manual_seed(2304)
    events, layer_count, heads, sources, roots = 19, 3, 2, 7, 4
    source = torch.arange(sources)
    query = torch.arange(events).remainder(sources - 1) + 1
    carrier = torch.tensor(
        [
            EVIDENCE_PROMPT,
            OTHER_PROMPT,
            OTHER_PROMPT,
            RESPONSE_HISTORY,
            RESPONSE_HISTORY,
            RESPONSE_HISTORY,
            RESPONSE_HISTORY,
        ]
    )
    layers = [
        layer_routes(
            torch.randn(events, heads, sources, roots, generator=generator),
            layer=layer,
            query_position=query,
            source_position=source,
            carrier=carrier,
        )
        for layer in range(layer_count)
    ]

    actual = moments_from_layers(layers)
    phi = torch.stack([route.root_phi for route in layers], dim=1)
    causal = source[None] < query[:, None]
    physical = phi[..., :NUMERIC].sum(dim=-1)
    expected_physical = torch.zeros_like(actual.physical_mass)
    expected_root = torch.zeros_like(actual.root_mass)
    expected_xlogx = torch.zeros_like(actual.physical_xlogx)
    expected_count = torch.zeros_like(actual.eligible_source_count)
    for role in range(3):
        role_mask = causal & (carrier == role)[None]
        physical_mask = role_mask[:, None, None, :, None]
        root_mask = role_mask[:, None, None, :, None, None]
        physical_signed = torch.stack(
            (physical.clamp_min(0), (-physical).clamp_min(0)), dim=-1
        )
        root_signed = torch.stack((phi.clamp_min(0), (-phi).clamp_min(0)), dim=-1)
        role_physical = physical_signed * physical_mask
        expected_physical[..., role, :] = role_physical.sum(dim=3)
        expected_root[..., role, :, :] = (root_signed * root_mask).sum(dim=3)
        expected_xlogx[..., role, :] = torch.where(
            role_physical > 0,
            role_physical * role_physical.clamp_min(torch.finfo(phi.dtype).tiny).log(),
            0,
        ).sum(dim=3)
        expected_count[..., role] = role_mask.sum(dim=1)[:, None, None]

    torch.testing.assert_close(actual.physical_mass, expected_physical)
    torch.testing.assert_close(actual.root_mass, expected_root)
    torch.testing.assert_close(actual.physical_xlogx, expected_xlogx)
    torch.testing.assert_close(actual.eligible_source_count, expected_count)


def test_sparse_prefix_reaches_capacity_cover_without_saving_zero_edges():
    phi = torch.zeros(1, 1, 4, 4)
    phi[0, 0, :3, EVIDENCE] = 1
    routes = layer_routes(
        phi,
        query_position=torch.tensor([4]),
        source_position=torch.arange(4),
        carrier=torch.tensor(
            [EVIDENCE_PROMPT, OTHER_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY]
        ),
        message_norm=torch.tensor([[[6.0, 3.0, 1.0, 0.0]]]),
    )

    sparse = sparsify_routes([routes], top_k=4, cover_mass=0.8)

    assert sparse.source_position.tolist() == [0, 1]
    torch.testing.assert_close(sparse.physical_message_norm.sum(), torch.tensor(9.0))
    torch.testing.assert_close(sparse.tail_message_norm_sum.sum(), torch.tensor(1.0))


@pytest.mark.parametrize(
    ("top_k", "cover_mass", "source_count"),
    ((0, 0.95, 9), (1, 0.95, 9), (64, 1.0, 70), (64, 1.0, 9)),
)
def test_sparse_tail_recomputes_the_exact_dense_axes(
    top_k: int,
    cover_mass: float,
    source_count: int,
):
    generator = torch.Generator().manual_seed(1000 + source_count + top_k)
    events, heads = 2, 3
    response_start = source_count // 2
    source = torch.arange(source_count)
    carrier = torch.full((source_count,), RESPONSE_HISTORY, dtype=torch.long)
    carrier[: response_start // 2] = EVIDENCE_PROMPT
    carrier[response_start // 2 : response_start] = OTHER_PROMPT
    query = torch.full((events,), source_count)
    layers = []
    for layer in range(2):
        phi = torch.randn(events, heads, source_count, 4, generator=generator)
        phi[..., NUMERIC] *= 1e-4
        layers.append(
            layer_routes(
                phi,
                layer=layer,
                query_position=query,
                source_position=source,
                carrier=carrier,
                message_norm=torch.rand(
                    events, heads, source_count, generator=generator
                )
                + 0.01,
            )
        )
    injection = torch.randn(events, 4, generator=generator)
    injection[:, NUMERIC] *= 1e-4
    event_valid = torch.tensor([True, True])
    resolution = torch.tensor([0.01, 0.02])

    expected = route_axes(
        layers,
        injection,
        event_valid=event_valid,
        resolution=resolution,
    )
    sparse = sparsify_routes(layers, top_k=top_k, cover_mass=cover_mass)
    actual = route_axes_from_sparse(
        sparse,
        injection,
        event_valid=event_valid,
        resolution=resolution,
    )

    for field in fields(expected):
        torch.testing.assert_close(
            getattr(actual, field.name),
            getattr(expected, field.name),
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
        )


def test_explicit_numeric_variation_cannot_understate_saved_n_atoms():
    phi = torch.zeros(1, 1, 2, 4)
    phi[0, 0, 0, NUMERIC] = 0.2
    phi[0, 0, 1, NUMERIC] = -0.1
    routes = layer_routes(
        phi,
        query_position=torch.tensor([2]),
        source_position=torch.arange(2),
    )
    injection = torch.zeros(1, 4)
    injection[0, NUMERIC] = 0.4
    with pytest.raises(ValueError):
        route_axes(
            layers=[routes],
            injection_phi=injection,
            numeric_total_variation=torch.tensor([0.6]),
        )

    explicit = torch.tensor([0.9])
    dense = route_axes(
        [routes],
        injection,
        resolution=torch.tensor([0.1]),
        numeric_total_variation=explicit,
    )
    sparse = route_axes_from_sparse(
        sparsify_routes([routes], top_k=0),
        injection,
        resolution=torch.tensor([0.1]),
        numeric_total_variation=explicit,
    )
    torch.testing.assert_close(dense.resolution, torch.tensor([1.0]))
    torch.testing.assert_close(sparse.resolution, dense.resolution)


def test_npz_roundtrip_preserves_ragged_routes_axes_and_label_boundary(tmp_path):
    events = prediction_events(torch.tensor([5, 6, 7, 8, 9]), response_start=2)
    phi = torch.zeros(3, 2, 4, 4)
    phi[:, :, 0, EVIDENCE] = 2
    phi[1:, :, 2, EVIDENCE] = 3
    phi[1:, :, 2, RESPONSE] = -1
    phi[2:, :, 3, RESPONSE] = 4
    common = {
        "query_position": events.query_position,
        "source_position": torch.arange(4),
        "carrier": torch.tensor(
            [EVIDENCE_PROMPT, OTHER_PROMPT, RESPONSE_HISTORY, RESPONSE_HISTORY]
        ),
    }
    layers = [
        layer_routes(phi, layer=0, **common),
        layer_routes(phi * 0.5, layer=1, **common),
    ]
    axes = route_axes(layers, torch.zeros(3, 4))
    routes = sparsify_routes(layers, top_k=2, cover_mass=0.8)
    terminal = (
        axes.root_carrier_mass[..., SUPPORT] - axes.root_carrier_mass[..., VETO]
    ).sum(dim=1)
    readout = RouteReadout(
        competitor_token_id=torch.tensor([0, 1, 2]),
        target_logprob=torch.tensor([-1.0, -2.0, -3.0]),
        injection_phi=torch.zeros(3, 4),
        terminal_root_margin=terminal,
        native_margin=terminal.sum(dim=1),
        root_closure_error=torch.zeros(3, 4),
        numeric_self_v_phi=torch.zeros(3, 2, 2),
        numeric_post_attention_phi=torch.zeros(3, 2),
        numeric_layer_phi=torch.zeros(3, 2),
        numeric_final_phi=torch.zeros(3),
        numeric_total_variation=torch.zeros(3),
        operator_error=torch.zeros(3),
        operator_valid=torch.ones(3, dtype=torch.bool),
    )
    artifact = RouteArtifact(
        response_start=2,
        source_token_id=torch.tensor([5, 6, 7, 8]),
        evidence_mask=torch.tensor([True, False, False, False]),
        top_k=2,
        cover_mass=0.8,
        events=events,
        routes=routes,
        axes=axes,
        readout=readout,
    )
    path = tmp_path / "route_sample.npz"

    save_route_artifact(path, artifact)
    loaded = load_route_artifact(path)

    assert loaded.response_start == artifact.response_start
    assert torch.equal(loaded.source_token_id, artifact.source_token_id)
    assert torch.equal(loaded.evidence_mask, artifact.evidence_mask)
    assert loaded.top_k == artifact.top_k
    assert loaded.cover_mass == pytest.approx(artifact.cover_mass)
    for name in ("events", "routes", "axes", "readout"):
        expected_object = getattr(artifact, name)
        actual_object = getattr(loaded, name)
        for field in fields(expected_object):
            torch.testing.assert_close(
                getattr(actual_object, field.name),
                getattr(expected_object, field.name),
                equal_nan=True,
            )
    with np.load(path, allow_pickle=False) as stored:
        assert not any(
            "label" in name.casefold() or "halluc" in name.casefold()
            for name in stored.files
        )

    wrong_source = loaded.source_token_id.clone()
    wrong_source[loaded.response_start] += 1
    with pytest.raises(ValueError, match="canonical source tokens"):
        validate_artifact(replace(loaded, source_token_id=wrong_source))

    skipped_events = replace(
        loaded.events,
        query_position=torch.tensor([1, 3, 3]),
        prediction_position=torch.tensor([2, 4, 4]),
    )
    with pytest.raises(ValueError, match="complete response"):
        validate_artifact(replace(loaded, events=skipped_events))

    duplicate_rows = loaded.routes.row_event.clone()
    duplicate_rows[0] = 1
    with pytest.raises(ValueError, match="cover each event, layer, and head once"):
        validate_artifact(
            replace(loaded, routes=replace(loaded.routes, row_event=duplicate_rows))
        )

    wrong_closure = replace(
        loaded.readout,
        root_closure_error=loaded.readout.root_closure_error + 0.25,
    )
    with pytest.raises(ValueError, match="root_closure_error"):
        validate_artifact(replace(loaded, readout=wrong_closure))

    wrong_axes = replace(
        loaded.axes,
        root_carrier_mass=loaded.axes.root_carrier_mass + 1,
    )
    with pytest.raises(ValueError, match="axis root_carrier_mass"):
        validate_artifact(replace(loaded, axes=wrong_axes))

    empty_tail = loaded.routes.tail_count == 0
    empty_row, empty_carrier = torch.nonzero(empty_tail, as_tuple=True)
    wrong_tail_attention = loaded.routes.tail_attention_sum.clone()
    wrong_tail_attention[empty_row[0], empty_carrier[0]] = 1
    with pytest.raises(ValueError, match="empty carrier tail"):
        validate_artifact(
            replace(
                loaded,
                routes=replace(
                    loaded.routes,
                    tail_attention_sum=wrong_tail_attention,
                ),
            )
        )

    with pytest.raises(ValueError, match="not the minimal coverage prefix"):
        validate_artifact(replace(loaded, cover_mass=0.1))

    nonfinite_readout = replace(
        loaded.readout,
        target_logprob=torch.full_like(loaded.readout.target_logprob, torch.nan),
    )
    with pytest.raises(ValueError, match="must be finite"):
        validate_artifact(replace(loaded, readout=nonfinite_readout))


def test_artifact_accepts_bounded_dense_to_sparse_accumulation_roundoff():
    torch.manual_seed(0)
    source_count = 2048
    response_start = source_count - 2
    events = prediction_events(torch.arange(source_count + 1), response_start)
    evidence_mask = torch.zeros(source_count, dtype=torch.bool)
    evidence_mask[: source_count // 2] = True
    carrier = token_carriers(torch.arange(source_count), response_start, evidence_mask)
    layers = []
    for layer in range(4):
        phi = torch.randn(3, 8, source_count, 4) * 0.01
        phi[..., NUMERIC] = 0
        layers.append(
            layer_routes(
                phi,
                layer=layer,
                query_position=events.query_position,
                source_position=torch.arange(source_count),
                carrier=carrier,
            )
        )
    injection = torch.zeros(3, 4)
    dense_axes = route_axes(layers, injection)
    routes = sparsify_routes(layers, top_k=2, cover_mass=0.95)
    axes = route_axes_from_sparse(routes, injection)
    terminal = (
        dense_axes.root_carrier_mass[..., SUPPORT]
        - dense_axes.root_carrier_mass[..., VETO]
    ).sum(dim=1)
    sparse_moments = moments_from_sparse(routes)
    sparse_terminal = (
        sparse_moments.root_mass[..., SUPPORT] - sparse_moments.root_mass[..., VETO]
    ).sum(dim=(1, 2, 3))
    assert (terminal - sparse_terminal).abs().max() > 1e-5
    artifact = RouteArtifact(
        response_start=response_start,
        source_token_id=torch.arange(source_count),
        evidence_mask=evidence_mask,
        top_k=2,
        cover_mass=0.95,
        events=events,
        routes=routes,
        axes=axes,
        readout=RouteReadout(
            competitor_token_id=torch.zeros(3, dtype=torch.long),
            target_logprob=torch.zeros(3),
            injection_phi=injection,
            terminal_root_margin=terminal,
            native_margin=terminal.sum(dim=1),
            root_closure_error=torch.zeros(3, 4),
            numeric_self_v_phi=torch.zeros(3, 4, 8),
            numeric_post_attention_phi=torch.zeros(3, 4),
            numeric_layer_phi=torch.zeros(3, 4),
            numeric_final_phi=torch.zeros(3),
            numeric_total_variation=torch.zeros(3),
            operator_error=torch.zeros(3),
            operator_valid=torch.ones(3, dtype=torch.bool),
        ),
    )

    validate_artifact(artifact)
