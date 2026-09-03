import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch.nn import functional as F

from experiments.attention_mechanism_audit.capture import (
    BRANCH_NAMES,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    FunctionalTraceReplay,
)


def _tiny_replay(*, layers=2, dtype=torch.float32):
    torch.manual_seed(41)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).to(dtype=dtype)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    return model, FunctionalTraceReplay(model)


def _inputs():
    return (
        torch.tensor([1, 5, 7, 9, 11, 13, 15]),
        3,
        torch.tensor([True, False, False]),
    )


def test_replay_rejects_pretraining_tp_that_bypasses_value_hook():
    model, _replay = _tiny_replay()
    model.config.pretraining_tp = 2
    with pytest.raises(ValueError, match="pretraining_tp"):
        FunctionalTraceReplay(model)


def _one_shot_scores(model, token_ids, response_start, evidence_mask, removal):
    """Independent full-sequence oracle for one causal branch."""

    layers = tuple(model.model.layers)
    length = len(token_ids) - 1
    kv_heads = model.config.num_key_value_heads
    repeats = model.config.num_attention_heads // kv_heads
    source = torch.arange(length)
    query = torch.arange(length)
    causal = source[None] <= query[:, None]
    response_query = query >= response_start - 1
    mask = torch.zeros(length, length, dtype=torch.bool)
    if removal in {"evidence", "both"}:
        evidence = torch.zeros(length, dtype=torch.bool)
        evidence[:response_start] = evidence_mask
        mask |= (
            causal
            & response_query[:, None]
            & evidence[None]
            & (source[None] != query[:, None])
        )
    if removal in {"history", "both"}:
        mask |= (
            causal
            & response_query[:, None]
            & (source[None] >= response_start)
            & (source[None] < query[:, None])
        )
    values, handles = [None] * len(layers), []

    def save_value(index):
        def hook(_module, _args, output):
            values[index] = output.reshape(1, length, kv_heads, -1)

        return hook

    def delete_messages(index, layer):
        def hook(_module, _args, output):
            parts = list(output)
            if removal is not None:
                value = values[index].repeat_interleave(repeats, dim=2)
                context = torch.einsum(
                    "bhqs,bshd->bqhd", parts[1] * mask[None, None], value
                )
                parts[0] = parts[0] - F.linear(
                    context.flatten(-2), layer.self_attn.o_proj.weight, None
                )
            parts[1] = None
            return tuple(parts)

        return hook

    for index, layer in enumerate(layers):
        handles.extend(
            (
                layer.self_attn.v_proj.register_forward_hook(save_value(index)),
                layer.self_attn.register_forward_hook(delete_messages(index, layer)),
            )
        )
    try:
        with torch.inference_mode():
            logits = (
                model(
                    input_ids=token_ids[:-1][None],
                    attention_mask=torch.ones(1, length, dtype=torch.long),
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
                .logits[0, response_start - 1 :]
                .float()
            )
    finally:
        for handle in handles:
            handle.remove()
    targets = token_ids[response_start:]
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    competitor = logits.scatter(1, targets[:, None], -torch.inf).max(1).values
    return selected - logits.logsumexp(1), selected - competitor


def _assert_zero_register(trace, register, rows):
    assert torch.all(trace["register_route_source_index"][:, rows, register] == -1)
    assert torch.all(trace["register_route_head_index"][:, rows, register] == -1)
    fields = [
        trace[name][:, rows, register]
        for name in (
            "register_route_magnitude",
            "register_route_contribution",
            "register_route_root_contribution",
            "register_route_carrier_contribution",
            "register_route_gate_contribution",
            "register_route_remainder_magnitude",
            "register_route_remainder_contribution",
            "register_route_remainder_root_contribution",
            "register_route_remainder_carrier_contribution",
            "register_route_remainder_gate_contribution",
            "register_route_cover_size",
            "register_norm",
            "register_mlp_alignment",
            "register_conservation_error",
            "register_attention_edge_error",
            "register_step_gram",
        )
    ]
    fields.extend(
        trace[name][:, rows, :, register]
        for name in (
            "register_role_mass",
            "register_role_contribution",
            "register_role_root_contribution",
            "register_role_carrier_contribution",
            "register_role_gate_contribution",
        )
    )
    fields.append(trace["register_role_effective_routes"][:, rows, register])
    fields.append(trace["final_register_norm"][:, rows, register])
    for value in fields:
        assert torch.all(torch.isfinite(value))
        torch.testing.assert_close(
            value.float(),
            torch.zeros_like(value, dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )


def test_capture_uses_one_four_branch_pass_and_saves_aligned_state():
    model, replay = _tiny_replay()
    token_ids, response_start, evidence_mask = _inputs()
    calls = 0

    def count_call(_module, _args):
        nonlocal calls
        calls += 1

    handle = model.model.register_forward_pre_hook(count_call)
    try:
        artifact = replay.capture(
            token_ids,
            response_start,
            evidence_mask,
            predictor_chunk=len(token_ids),
            top_k=10,
        )
    finally:
        handle.remove()

    assert calls == 1
    assert BRANCH_NAMES == (
        "full",
        "no_evidence",
        "no_history",
        "no_evidence_history",
    )
    assert torch.equal(artifact["token_ids"], token_ids)
    assert torch.equal(artifact["evidence_mask"], evidence_mask)
    assert artifact["evidence_mask"].dtype == torch.bool
    assert artifact["response_start"] == response_start
    assert set(artifact) == {
        "token_ids",
        "response_start",
        "evidence_mask",
        "trace",
        "score_inputs",
        "peak_cuda_reserved_bytes",
    }
    assert set(artifact["score_inputs"]) == {
        f"{branch}_{statistic}"
        for branch in BRANCH_NAMES
        for statistic in ("logprob", "margin")
    }
    trace = artifact["trace"]
    for family in ("attention", "edge"):
        assert trace[f"prompt_{family}_effective_sources"].shape == (2, 4)
        assert trace[f"prompt_{family}_effective_rank"].shape == (2, 4)
        assert trace[f"prompt_{family}_anchor_index"].shape == (2, 4, 4)
    # The global route universe is head x source (4 x 6), not just 6 sources.
    assert trace["register_route_source_index"].shape == (2, 4, 2, 10)
    assert trace["register_route_head_index"].shape == (2, 4, 2, 10)
    assert trace["register_route_magnitude"].shape == (2, 4, 2, 10)
    assert trace["register_route_contribution"].shape == (2, 4, 2, 10)
    for component in ("root", "carrier", "gate"):
        assert trace[f"register_route_{component}_contribution"].shape == (
            2,
            4,
            2,
            10,
        )
        assert trace[f"register_route_remainder_{component}_contribution"].shape == (
            2,
            4,
            2,
        )
        assert trace[f"register_role_{component}_contribution"].shape == (
            2,
            4,
            4,
            2,
            4,
        )
    assert trace["register_route_remainder_magnitude"].shape == (2, 4, 2)
    assert trace["register_route_remainder_contribution"].shape == (2, 4, 2)
    assert trace["register_route_cover_size"].shape == (2, 4, 2)
    assert trace["register_role_mass"].shape == (2, 4, 4, 2, 4)
    assert trace["register_role_contribution"].shape == (2, 4, 4, 2, 4)
    assert trace["register_role_effective_routes"].shape == (2, 4, 2, 4)
    assert trace["register_norm"].shape == (2, 4, 2, 4)
    assert trace["register_mlp_alignment"].shape == (2, 4, 2)
    assert trace["register_conservation_error"].shape == (2, 4, 2)
    assert trace["register_attention_edge_error"].shape == (2, 4, 2)
    assert trace["register_step_gram"].shape == (2, 4, 2, 2)
    assert trace["interaction_norm"].shape == (2, 4, 4)
    assert trace["final_register_norm"].shape == (1, 4, 2)
    assert trace["shortcut_route_gram"].shape == (2, 4, 7, 7)
    assert trace["shortcut_head_gram"].shape == (2, 4, 4, 7, 7)
    assert trace["shortcut_rewire_valid"].shape == (2, 4)
    assert torch.allclose(
        trace["shortcut_route_gram"],
        trace["shortcut_route_gram"].transpose(-1, -2),
    )
    assert torch.allclose(
        trace["shortcut_head_gram"],
        trace["shortcut_head_gram"].transpose(-1, -2),
    )
    assert ROLE_NAMES == (
        "evidence",
        "other_prompt",
        "response_history",
        "predictor_self",
    )
    assert REGISTER_NAMES == ("evidence_adoption", "autonomous_history")
    assert REGISTER_STAGE_NAMES == (
        "input_state",
        "attention_write",
        "mlp_write",
        "output_state",
    )
    assert torch.all(trace["register_conservation_error"] < 1e-5)
    assert torch.all(trace["register_attention_edge_error"] < 1e-5)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bfloat16_capture_keeps_route_diagnostics_dtype_aligned():
    model, replay = _tiny_replay(layers=1, dtype=torch.bfloat16)
    artifact = replay.capture(*_inputs(), predictor_chunk=2, top_k=3)
    trace = artifact["trace"]

    assert model.model.layers[0].self_attn.o_proj.weight.dtype == torch.bfloat16
    assert torch.any(trace["register_route_cover_size"] > 0)
    assert torch.all(torch.isfinite(trace["register_role_mass"]))
    assert torch.all(trace["register_attention_edge_error"] < 2e-3)


def test_tiny_full_sequence_oracle_matches_all_four_branch_scores():
    model, replay = _tiny_replay(layers=2)
    token_ids, response_start, evidence_mask = _inputs()
    actual = replay.capture(
        token_ids,
        response_start,
        evidence_mask,
        predictor_chunk=2,
    )["score_inputs"]
    for branch, removal in zip(
        BRANCH_NAMES, (None, "evidence", "history", "both"), strict=True
    ):
        logprob, margin = _one_shot_scores(
            model, token_ids, response_start, evidence_mask, removal
        )
        torch.testing.assert_close(
            actual[f"{branch}_logprob"], logprob, atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            actual[f"{branch}_margin"], margin, atol=3e-5, rtol=3e-5
        )
    assert not torch.allclose(
        actual["full_logprob"], actual["no_evidence_logprob"], atol=1e-5
    )


def test_final_register_norm_matches_the_actual_normalized_branch_states():
    model, replay = _tiny_replay(layers=2)
    token_ids, response_start, evidence_mask = _inputs()
    observed = []

    def save_final_state(_module, _args, output):
        observed.append(output.detach().cpu())

    handle = model.model.norm.register_forward_hook(save_final_state)
    try:
        trace = replay.capture(
            token_ids,
            response_start,
            evidence_mask,
            predictor_chunk=len(token_ids),
        )["trace"]
    finally:
        handle.remove()

    assert len(observed) == 1
    response_state = observed[0][:, response_start - 1 :]
    expected = (
        torch.stack(
            (
                response_state[0] - response_state[1],
                response_state[1] - response_state[3],
            ),
            dim=1,
        )
        .float()
        .norm(dim=-1)
    )
    torch.testing.assert_close(trace["final_register_norm"][0], expected)


def test_structurally_zero_registers_have_no_routes_and_strict_history_onset():
    _model, replay = _tiny_replay(layers=2)
    token_ids, response_start, evidence_mask = _inputs()
    trace = replay.capture(
        token_ids, response_start, evidence_mask, predictor_chunk=2, top_k=3
    )["trace"]
    autonomous = REGISTER_NAMES.index("autonomous_history")
    _assert_zero_register(trace, autonomous, slice(0, 2))
    assert torch.any(trace["register_norm"][:, 2:, autonomous] > 1e-6)

    no_evidence = replay.capture(
        token_ids,
        response_start,
        torch.zeros_like(evidence_mask),
        predictor_chunk=2,
        top_k=3,
    )["trace"]
    evidence = REGISTER_NAMES.index("evidence_adoption")
    _assert_zero_register(no_evidence, evidence, slice(None))


def test_register_routes_use_global_head_source_cover_and_signed_tail():
    model, _replay = _tiny_replay(layers=1)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    with torch.no_grad():
        weight = model.model.layers[0].self_attn.o_proj.weight
        weight.zero_()
        weight[0, 0] = 1
        weight[0, head_dim] = 1
    replay = FunctionalTraceReplay(model)
    sources = 4
    attention = torch.zeros(4, 1, replay.heads, sources)
    value = torch.zeros(4, sources, replay.kv_heads, replay.head_dim)

    # Opposing edges from different heads write to the same output direction:
    # +3 and -2. The complete write is +1, so contributions are signed +3/-2.
    for branch in (0, 1, 3):
        attention[branch, 0, 1, 0] = 1
        attention[branch, 0, 0, 1] = 1
    value[0, 0, 0, 0] = 5
    value[0, 1, 0, 0] = -4
    value[1, 0, 0, 0] = value[3, 0, 0, 0] = 2
    value[1, 1, 0, 0] = value[3, 1, 0, 0] = -2
    value_by_head = value[:, :, replay.q_to_kv]
    context = torch.einsum("brhs,bshd->brhd", attention, value_by_head)
    attention_write = replay._removed_write(0, context)
    attention_register = replay._registers(attention_write)
    roles = (
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[False, False, True, False]]),
        torch.tensor([[False, False, False, True]]),
    )

    actual = replay._register_routes(
        0,
        attention,
        value,
        attention_register,
        roles,
        top_k=1,
        cover_mass=0.5,
    )

    assert actual["source_index"].shape == (1, 2, 1)
    assert actual["head_index"].shape == (1, 2, 1)
    assert actual["source_index"][0, 0, 0] == 0
    assert actual["head_index"][0, 0, 0] == 1
    assert actual["cover_size"][0, 0] == 1
    torch.testing.assert_close(actual["magnitude"][0, 0, 0], torch.tensor(3.0))
    torch.testing.assert_close(actual["remainder_magnitude"][0, 0], torch.tensor(2.0))
    torch.testing.assert_close(actual["contribution"][0, 0, 0], torch.tensor(3.0))
    torch.testing.assert_close(
        actual["remainder_contribution"][0, 0], torch.tensor(-2.0)
    )
    torch.testing.assert_close(actual["root_contribution"][0, 0, 0], torch.tensor(3.0))
    assert actual["carrier_contribution"][0, 0, 0] == 0
    assert actual["gate_contribution"][0, 0, 0] == 0
    assert actual["remainder_root_contribution"][0, 0] == 0
    torch.testing.assert_close(
        actual["remainder_carrier_contribution"][0, 0], torch.tensor(-2.0)
    )
    assert actual["remainder_gate_contribution"][0, 0] == 0
    torch.testing.assert_close(
        actual["contribution"].sum(-1) + actual["remainder_contribution"],
        torch.tensor([[1.0, 0.0]]),
    )
    explicit_components = sum(
        actual[f"{name}_contribution"] for name in ("root", "carrier", "gate")
    )
    remainder_components = sum(
        actual[f"remainder_{name}_contribution"] for name in ("root", "carrier", "gate")
    )
    torch.testing.assert_close(explicit_components, actual["contribution"])
    torch.testing.assert_close(remainder_components, actual["remainder_contribution"])
    expected_mass = torch.zeros(replay.heads, 4)
    expected_mass[0, 1] = 2
    expected_mass[1, 0] = 3
    torch.testing.assert_close(actual["role_mass"][0, :, 0], expected_mass)
    expected_contribution = torch.zeros(replay.heads, 4)
    expected_contribution[0, 1] = -2
    expected_contribution[1, 0] = 3
    torch.testing.assert_close(
        actual["role_contribution"][0, :, 0], expected_contribution
    )
    expected_root = torch.zeros(replay.heads, 4)
    expected_root[1, 0] = 3
    expected_carrier = torch.zeros(replay.heads, 4)
    expected_carrier[0, 1] = -2
    torch.testing.assert_close(actual["role_root_contribution"][0, :, 0], expected_root)
    torch.testing.assert_close(
        actual["role_carrier_contribution"][0, :, 0], expected_carrier
    )
    assert torch.count_nonzero(actual["role_gate_contribution"][0, :, 0]) == 0
    torch.testing.assert_close(
        actual["role_effective_routes"][0, 0],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    assert torch.count_nonzero(actual["edge_error"]) == 0
    assert torch.equal(actual["source_index"][0, 1], torch.tensor([-1]))
    assert torch.equal(actual["head_index"][0, 1], torch.tensor([-1]))
    for name in (
        "magnitude",
        "contribution",
        "root_contribution",
        "carrier_contribution",
        "gate_contribution",
        "remainder_magnitude",
        "remainder_contribution",
        "remainder_root_contribution",
        "remainder_carrier_contribution",
        "remainder_gate_contribution",
        "cover_size",
    ):
        assert torch.count_nonzero(actual[name][0, 1]) == 0
    for name in (
        "role_mass",
        "role_contribution",
        "role_root_contribution",
        "role_carrier_contribution",
        "role_gate_contribution",
    ):
        assert torch.count_nonzero(actual[name][0, :, 1]) == 0
    assert torch.count_nonzero(actual["role_effective_routes"][0, 1]) == 0


def test_effective_routes_mass_weight_heads_instead_of_averaging_them():
    mass = torch.tensor([[[9.0, 0.0], [0.0, 1.0]]])
    role = torch.ones(1, 2, dtype=torch.bool)
    result = FunctionalTraceReplay._effective_routes(mass, (role,))

    expected_joint_routes = math.exp(-0.9 * math.log(0.9) - 0.1 * math.log(0.1))
    torch.testing.assert_close(result[0, 0], torch.tensor(expected_joint_routes))
    assert result[0, 0] < 2


def test_adaptive_cover_retains_every_dense_source_in_saved_plus_remainder():
    mass = torch.tensor(
        [
            [
                [
                    [0.19, 0.18, 0.17, 0.16, 0.15, 0.15],
                    [0.75, 0.10, 0.06, 0.04, 0.03, 0.02],
                ]
            ]
        ]
    )
    index, magnitude, remainder, count = FunctionalTraceReplay._mass_cover(
        mass, top_k=3, cover_mass=0.8
    )

    assert torch.equal(count, torch.tensor([[[5, 2]]]))
    assert torch.equal(index[0, 0, 0], torch.tensor([0, 1, 2]))
    assert torch.equal(index[0, 0, 1], torch.tensor([0, 1, -1]))
    torch.testing.assert_close(magnitude.sum(-1) + remainder, mass.sum(-1))
    torch.testing.assert_close(remainder, torch.tensor([[[0.46, 0.15]]]))


def test_register_statistics_obey_branch_differences_and_residual_equation():
    torch.manual_seed(7)
    layer_input = torch.randn(4, 2, 5)
    attention = torch.randn(4, 2, 5)
    mlp = torch.randn(4, 2, 5)
    output = layer_input + attention + mlp
    actual = FunctionalTraceReplay._register_statistics(
        layer_input, attention, mlp, output
    )

    states = (layer_input, attention, mlp, output)
    registers = tuple(
        torch.stack((state[0] - state[1], state[1] - state[3]), dim=1)
        for state in states
    )
    expected_norm = torch.stack(registers, dim=2).norm(dim=-1)
    expected_step = registers[1] + registers[2]
    torch.testing.assert_close(actual["norm"], expected_norm)
    torch.testing.assert_close(actual["step"], expected_step)
    torch.testing.assert_close(registers[3], registers[0] + expected_step)
    torch.testing.assert_close(
        actual["conservation_error"],
        torch.zeros_like(actual["conservation_error"]),
        atol=1e-6,
        rtol=0,
    )

    pre_mlp = registers[0] + registers[1]
    denominator = pre_mlp.norm(dim=-1) * registers[2].norm(dim=-1)
    expected_alignment = (pre_mlp * registers[2]).sum(-1) / denominator
    torch.testing.assert_close(actual["mlp_alignment"], expected_alignment)
    expected_interaction = torch.stack(
        tuple(state[0] - state[1] - state[2] + state[3] for state in states),
        dim=1,
    )
    torch.testing.assert_close(
        actual["interaction_norm"], expected_interaction.norm(dim=-1)
    )

    perturbed_mlp = mlp.clone()
    perturbed_mlp[0] += 0.1
    perturbed = FunctionalTraceReplay._register_statistics(
        layer_input, attention, perturbed_mlp, output
    )
    assert torch.all(perturbed["conservation_error"][:, 0] > 0)
    torch.testing.assert_close(
        perturbed["conservation_error"][:, 1],
        torch.zeros_like(perturbed["conservation_error"][:, 1]),
        atol=1e-6,
        rtol=0,
    )


def test_register_alignment_is_zero_and_finite_when_either_vector_is_zero():
    zeros = torch.zeros(4, 2, 5)
    coefficient = torch.tensor([1.0, 0.0, 1.0, 0.0])[:, None, None]
    pre_only = coefficient.expand_as(zeros)
    no_effect = FunctionalTraceReplay._register_statistics(zeros, zeros, zeros, zeros)
    complete_cancellation = FunctionalTraceReplay._register_statistics(
        pre_only, zeros, -pre_only, zeros
    )
    no_pre_effect = FunctionalTraceReplay._register_statistics(
        zeros, zeros, pre_only, pre_only
    )

    for actual in (no_effect, complete_cancellation, no_pre_effect):
        torch.testing.assert_close(
            actual["conservation_error"],
            torch.zeros_like(actual["conservation_error"]),
            atol=1e-6,
            rtol=0,
        )
        assert torch.all(torch.isfinite(actual["mlp_alignment"]))

    assert torch.count_nonzero(no_effect["mlp_alignment"]) == 0
    assert torch.count_nonzero(no_pre_effect["mlp_alignment"]) == 0
    torch.testing.assert_close(
        complete_cancellation["mlp_alignment"][:, 0],
        -torch.ones(2),
    )
    assert torch.count_nonzero(complete_cancellation["mlp_alignment"][:, 1]) == 0


def test_deletions_are_symmetric_and_predictor_self_is_never_removed():
    _model, replay = _tiny_replay(layers=1)
    masks = replay._removal_mask(
        (None, "evidence", "history", "both"),
        torch.tensor([True, True, True]),
        response_start=3,
        query_start=0,
        query_stop=6,
    )
    full, evidence, history, both = masks
    assert torch.count_nonzero(full) == 0
    assert torch.count_nonzero(masks[:, :2]) == 0
    assert torch.equal(
        evidence[2], torch.tensor([True, True, False, False, False, False])
    )
    assert torch.count_nonzero(history[2]) == 0
    assert torch.equal(
        history[4], torch.tensor([False, False, False, True, False, False])
    )
    assert torch.equal(both, evidence | history)
    assert torch.count_nonzero(torch.diagonal(masks, dim1=1, dim2=2)) == 0


def test_each_layer_uses_its_own_output_projection():
    model, _ = _tiny_replay(layers=2)
    identity = torch.eye(model.config.hidden_size)
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight.copy_(identity)
        model.model.layers[1].self_attn.o_proj.weight.copy_(2 * identity)
    replay = FunctionalTraceReplay(model)
    context = torch.randn(2, 3, replay.heads, replay.head_dim)

    for index, layer in enumerate(replay.layers):
        expected = F.linear(
            context.flatten(-2), layer.self_attn.o_proj.weight, bias=None
        )
        torch.testing.assert_close(replay._removed_write(index, context), expected)
    torch.testing.assert_close(
        replay._removed_write(1, context), 2 * replay._removed_write(0, context)
    )


def test_gqa_source_norm_maps_each_query_head_to_its_kv_head():
    _model, replay = _tiny_replay(layers=2)
    value = torch.randn(5, replay.kv_heads, replay.head_dim)

    assert torch.equal(replay.q_to_kv.cpu(), torch.tensor([0, 0, 1, 1]))
    for index, layer in enumerate(replay.layers):
        weight = layer.self_attn.o_proj.weight.detach().float()
        expected = torch.stack(
            [
                F.linear(
                    value[:, replay.q_to_kv[head]].float(),
                    weight[:, head * replay.head_dim : (head + 1) * replay.head_dim],
                ).norm(dim=-1)
                for head in range(replay.heads)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            replay._source_norm(index, value), expected, atol=1e-5, rtol=1e-5
        )


def test_register_step_gram_is_symmetric_psd_and_predictor_chunk_invariant():
    _model, replay = _tiny_replay(layers=2)
    serial = replay.capture(*_inputs(), predictor_chunk=1, top_k=3)
    batched = replay.capture(*_inputs(), predictor_chunk=4, top_k=3)

    for artifact in (serial, batched):
        gram = artifact["trace"]["register_step_gram"].permute(1, 2, 0, 3)
        torch.testing.assert_close(gram, gram.transpose(-1, -2))
        assert torch.all(torch.linalg.eigvalsh(gram) >= -1e-6)

    for name in serial["score_inputs"]:
        torch.testing.assert_close(
            serial["score_inputs"][name],
            batched["score_inputs"][name],
            atol=3e-5,
            rtol=3e-5,
        )
    for name in serial["trace"]:
        left, right = serial["trace"][name], batched["trace"][name]
        if left.dtype.is_floating_point:
            torch.testing.assert_close(left, right, atol=3e-3, rtol=3e-3)
        else:
            assert torch.equal(left, right)
