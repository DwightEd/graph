import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch.nn import functional as F

from experiments.attention_mechanism_audit.capture import (
    BRANCH_NAMES,
    PATHWAY_CONTRAST_NAMES,
    PATHWAY_STAGE_NAMES,
    ROLE_NAMES,
    ROUTE_ROLE_NAMES,
    FunctionalTraceReplay,
)


def _tiny_replay(*, layers=2):
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
    model = transformers.LlamaForCausalLM(config)
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


def _dense_route_oracle(model, replay, token_ids, response_start, evidence_mask):
    """Return dense full-branch A and A||W_OV|| without capture helpers."""

    layers = tuple(model.model.layers)
    length = len(token_ids) - 1
    kv_heads = model.config.num_key_value_heads
    heads = model.config.num_attention_heads
    repeats = heads // kv_heads
    head_dim = model.config.hidden_size // heads
    values, handles = [None] * len(layers), []

    def save_value(index):
        def hook(_module, _args, output):
            values[index] = output.reshape(1, length, kv_heads, head_dim)[0].float()

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(save_value(index)))
    try:
        with torch.inference_mode():
            output = model.model(
                input_ids=token_ids[:-1][None],
                attention_mask=torch.ones(1, length, dtype=torch.long),
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    attention_rows, edge_rows = [], []
    for index, layer in enumerate(layers):
        attention = output.attentions[index][0, :, response_start - 1 :].permute(
            1, 0, 2
        )
        weight = layer.self_attn.o_proj.weight.detach().float()
        source_norm = torch.empty(length, heads)
        for head in range(heads):
            block = weight[:, head * head_dim : (head + 1) * head_dim]
            source_norm[:, head] = F.linear(
                values[index][:, head // repeats], block
            ).norm(dim=-1)
        attention_rows.append(attention.float())
        edge_rows.append(attention.float() * source_norm.T[None])
    return attention_rows, edge_rows


def _roles(query, response_start, evidence_mask):
    source = torch.arange(query + 1)
    self_source = source == query
    evidence = torch.zeros(query + 1, dtype=torch.bool)
    stop = min(response_start, query + 1)
    evidence[:stop] = evidence_mask[:stop]
    return (
        evidence & ~self_source,
        (source < response_start) & ~evidence & ~self_source,
        (source >= response_start) & (source < query),
        self_source,
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
            top_k=3,
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
    assert trace["attention_role_mass"].shape == (2, 4, 4, 4)
    assert trace["edge_role_mass"].shape == (2, 4, 4, 4)
    assert trace["route_source_index"].shape == (2, 4, 4, 2, 3)
    assert trace["route_source_magnitude"].shape == (2, 4, 4, 2, 3)
    assert trace["route_source_remainder"].shape == (2, 4, 4, 2)
    assert trace["route_source_cover_size"].shape == (2, 4, 4, 2)
    assert trace["pathway_effect_norm"].shape == (2, 4, 3, 5)
    assert trace["pathway_residual_error"].shape == (2, 4, 4)
    assert trace["pathway_valid"].shape == (2, 4, 3)
    assert trace["pathway_cosine_valid"].shape == (2, 4, 3)
    assert ROLE_NAMES == (
        "evidence",
        "other_prompt",
        "response_history",
        "predictor_self",
    )
    assert PATHWAY_CONTRAST_NAMES == ("evidence", "history", "interaction")
    assert ROUTE_ROLE_NAMES == ("evidence", "response_history")
    assert PATHWAY_STAGE_NAMES == (
        "input",
        "attention",
        "pre_mlp",
        "mlp",
        "output",
    )
    assert torch.all(trace["pathway_residual_error"] < 1e-3)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


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


def test_dense_route_oracle_preserves_head_source_identity_and_mass():
    model, replay = _tiny_replay(layers=1)
    token_ids, response_start, evidence_mask = _inputs()
    artifact = replay.capture(
        token_ids,
        response_start,
        evidence_mask,
        predictor_chunk=2,
        top_k=3,
        route_cover_mass=0.8,
    )
    attention, edges = _dense_route_oracle(
        model, replay, token_ids, response_start, evidence_mask
    )
    trace = artifact["trace"]

    for row, query in enumerate(range(response_start - 1, len(token_ids) - 1)):
        role_masks = _roles(query, response_start, evidence_mask)
        current_attention = attention[0][row, :, : query + 1]
        current_edge = edges[0][row, :, : query + 1]
        for role, role_mask in enumerate(role_masks):
            expected_attention = (current_attention * role_mask).sum(-1)
            expected_edge = (current_edge * role_mask).sum(-1)
            torch.testing.assert_close(
                trace["attention_role_mass"][0, row, :, role].float(),
                expected_attention,
                atol=5e-4,
                rtol=5e-4,
            )
            torch.testing.assert_close(
                trace["edge_role_mass"][0, row, :, role].float(),
                expected_edge,
                atol=5e-3,
                rtol=5e-3,
            )
            for head in range(replay.heads):
                selected = current_edge[head] * role_mask
                total = selected.sum()
                if total == 0:
                    assert trace["edge_role_anchor_index"][0, row, head, role] == -1
                    continue
                probability = selected / total
                expected_entropy = -(
                    probability * probability.clamp_min(1e-12).log()
                ).sum()
                torch.testing.assert_close(
                    trace["edge_role_source_entropy"][0, row, head, role].float(),
                    expected_entropy,
                    atol=2e-3,
                    rtol=2e-3,
                )
                assert (
                    trace["edge_role_anchor_index"][0, row, head, role].item()
                    == selected.argmax().item()
                )

        for head in range(replay.heads):
            for route_role, role in enumerate((0, 2)):
                dense = current_edge[head] * role_masks[role]
                values, indices = dense.sort(descending=True)
                if values.sum() == 0:
                    required = 0
                else:
                    required = int((values.cumsum(0) < 0.8 * values.sum()).sum()) + 1
                retained = min(required, 3)
                assert (
                    trace["route_source_cover_size"][0, row, head, route_role]
                    == required
                )
                assert torch.equal(
                    trace["route_source_index"][0, row, head, route_role, :retained],
                    indices[:retained].int(),
                )
                saved = (
                    trace["route_source_magnitude"][0, row, head, route_role]
                    .float()
                    .sum()
                )
                remainder = trace["route_source_remainder"][
                    0, row, head, route_role
                ].float()
                torch.testing.assert_close(
                    saved + remainder, dense.sum(), atol=5e-3, rtol=5e-3
                )
                torch.testing.assert_close(
                    saved + remainder,
                    trace["edge_role_mass"][0, row, head, role].float(),
                    atol=5e-3,
                    rtol=5e-3,
                )


def test_routing_statistics_mass_weight_heads_instead_of_averaging_them():
    mass = torch.tensor([[[9.0, 0.0], [0.0, 1.0]]])
    role = torch.ones(1, 2, dtype=torch.bool)
    result = FunctionalTraceReplay._routing_statistics(mass, (role,))

    assert torch.equal(result["anchor_index"][0, :, 0], torch.tensor([0, 1]))
    torch.testing.assert_close(result["top1"][0, :, 0], torch.ones(2))
    expected_joint_routes = math.exp(-0.9 * math.log(0.9) - 0.1 * math.log(0.1))
    torch.testing.assert_close(
        result["effective_routes"][0, 0], torch.tensor(expected_joint_routes)
    )
    expected_rank = (0.9**2 + 0.1**2) ** 2 / (0.9**4 + 0.1**4)
    torch.testing.assert_close(
        result["effective_rank"][0, 0], torch.tensor(expected_rank)
    )
    assert result["effective_routes"][0, 0] < 2


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


def test_pathway_decomposition_obeys_residual_and_factorial_equations():
    torch.manual_seed(7)
    layer_input = torch.randn(4, 2, 5)
    attention = torch.randn(4, 2, 5)
    mlp = torch.randn(4, 2, 5)
    output = layer_input + attention + mlp
    actual = FunctionalTraceReplay._pathway_statistics(
        layer_input, attention, mlp, output
    )

    pre_mlp = layer_input + attention
    stages = torch.stack((layer_input, attention, pre_mlp, mlp, output), dim=2)
    full, no_evidence, no_history, no_both = stages
    expected_effects = torch.stack(
        (
            0.5 * ((full - no_evidence) + (no_history - no_both)),
            0.5 * ((full - no_history) + (no_evidence - no_both)),
            full - no_evidence - no_history + no_both,
        ),
        dim=1,
    )
    torch.testing.assert_close(actual["effect_norm"], expected_effects.norm(dim=-1))
    assert torch.count_nonzero(actual["residual_error"]) == 0

    pre = expected_effects[:, :, 2]
    mlp_effect = expected_effects[:, :, 3]
    out = expected_effects[:, :, 4]
    pre_norm = pre.norm(dim=-1)
    expected_projection = (mlp_effect * pre).sum(-1) / pre_norm.square()
    expected_cosine = (pre * out).sum(-1) / (pre_norm * out.norm(dim=-1))
    expected_gain = out.norm(dim=-1) / pre_norm
    torch.testing.assert_close(actual["mlp_projection"], expected_projection)
    torch.testing.assert_close(actual["pre_output_cosine"], expected_cosine)
    torch.testing.assert_close(actual["pre_output_gain"], expected_gain)
    assert torch.all(actual["valid"])
    assert torch.all(actual["cosine_valid"])

    perturbed = FunctionalTraceReplay._pathway_statistics(
        layer_input, attention, mlp + 0.1, output
    )
    assert torch.all(perturbed["residual_error"] > 0)


def test_pathway_ratios_are_zero_and_finite_when_contrast_norm_is_tiny():
    zeros = torch.zeros(4, 2, 5)
    coefficient = torch.tensor([1.0, 0.0, 1.0, 0.0])[:, None, None]
    pre_only = coefficient.expand_as(zeros)
    output_only = coefficient.expand_as(zeros)
    no_effect = FunctionalTraceReplay._pathway_statistics(zeros, zeros, zeros, zeros)
    complete_cancellation = FunctionalTraceReplay._pathway_statistics(
        pre_only, zeros, -pre_only, zeros
    )
    no_pre_effect = FunctionalTraceReplay._pathway_statistics(
        zeros, zeros, output_only, output_only
    )

    for actual in (no_effect, complete_cancellation, no_pre_effect):
        assert torch.count_nonzero(actual["residual_error"]) == 0
        for name in ("mlp_projection", "pre_output_cosine", "pre_output_gain"):
            assert torch.all(torch.isfinite(actual[name]))

    assert not no_effect["valid"][:, 0].any()
    assert not no_pre_effect["valid"][:, 0].any()
    for actual in (no_effect, no_pre_effect):
        assert torch.count_nonzero(actual["mlp_projection"][:, 0]) == 0
        assert torch.count_nonzero(actual["pre_output_gain"][:, 0]) == 0

    assert complete_cancellation["valid"][:, 0].all()
    assert not complete_cancellation["cosine_valid"][:, 0].any()
    torch.testing.assert_close(
        complete_cancellation["mlp_projection"][:, 0],
        -torch.ones(2),
    )
    assert torch.count_nonzero(complete_cancellation["pre_output_gain"][:, 0]) == 0
    assert torch.count_nonzero(complete_cancellation["pre_output_cosine"][:, 0]) == 0


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


def test_predictor_chunk_size_does_not_change_scores_or_trace():
    _model, replay = _tiny_replay(layers=2)
    serial = replay.capture(*_inputs(), predictor_chunk=1, top_k=3)
    batched = replay.capture(*_inputs(), predictor_chunk=4, top_k=3)

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
