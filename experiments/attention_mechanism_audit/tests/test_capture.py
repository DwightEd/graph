import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch.nn import functional as F

from experiments.attention_mechanism_audit.capture import (
    ROLE_NAMES,
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


def _one_shot_scores(model, token_ids, response_start, evidence_mask, removal):
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
                    "bhqs,bshd->bqhd",
                    parts[1] * mask[None, None],
                    value,
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


def _explicit_full_sequence_trace(
    model, token_ids, response_start, evidence_mask, *, top_k
):
    """Recompute every saved trace field from raw HF attention and V tensors."""

    layers = tuple(model.model.layers)
    source_tokens = len(token_ids) - 1
    response_tokens = len(token_ids) - response_start
    heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // heads
    repeats = heads // kv_heads
    roles = len(ROLE_NAMES)
    k = min(top_k, source_tokens)
    values, handles = [None] * len(layers), []

    def save_value(index):
        def hook(_module, _args, output):
            values[index] = (
                output.detach()
                .reshape(1, source_tokens, kv_heads, head_dim)[0]
                .float()
                .cpu()
            )

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(save_value(index)))
    try:
        with torch.inference_mode():
            output = model.model(
                input_ids=token_ids[:-1][None],
                attention_mask=torch.ones(1, source_tokens, dtype=torch.long),
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    expected = {
        "role_attention_mass": torch.zeros(
            len(layers), response_tokens, heads, roles
        ),
        "edge_role_energy": torch.zeros(
            len(layers), response_tokens, heads, roles
        ),
        "head_role_write_norm": torch.zeros(
            len(layers), response_tokens, heads, roles
        ),
        "head_source_entropy": torch.zeros(
            len(layers), response_tokens, heads
        ),
        "role_head_coherence": torch.zeros(
            len(layers), response_tokens, roles
        ),
        "top_source_index": torch.full(
            (len(layers), response_tokens, k), -1, dtype=torch.int32
        ),
        "top_source_magnitude": torch.zeros(
            len(layers), response_tokens, k
        ),
    }
    evidence_mask = evidence_mask.bool()
    for layer_index, layer in enumerate(layers):
        attention = output.attentions[layer_index][0].detach().float().cpu()
        value = values[layer_index]
        weight = layer.self_attn.o_proj.weight.detach().float().cpu()
        blocks = [
            weight[:, head * head_dim : (head + 1) * head_dim]
            for head in range(heads)
        ]
        for row in range(response_tokens):
            query = response_start - 1 + row
            source_energy = torch.zeros(query + 1)
            net_role_writes = [torch.zeros(weight.shape[0]) for _ in range(roles)]
            for head in range(heads):
                kv_head = head // repeats
                role_contexts = [torch.zeros(head_dim) for _ in range(roles)]
                head_source_energy = torch.zeros(query + 1)
                for source in range(query + 1):
                    if source == query:
                        role = 3
                    elif source < response_start:
                        role = 0 if evidence_mask[source] else 1
                    else:
                        role = 2
                    mass = attention[head, query, source]
                    source_write = F.linear(
                        value[source, kv_head], blocks[head], bias=None
                    )
                    energy = mass * source_write.norm()
                    expected["role_attention_mass"][
                        layer_index, row, head, role
                    ] += mass
                    expected["edge_role_energy"][
                        layer_index, row, head, role
                    ] += energy
                    role_contexts[role] += mass * value[source, kv_head]
                    head_source_energy[source] = energy
                    source_energy[source] += energy
                probability = head_source_energy / head_source_energy.sum().clamp_min(
                    1e-12
                )
                expected["head_source_entropy"][layer_index, row, head] = -(
                    probability * probability.clamp_min(1e-12).log()
                ).sum() / math.log(max(query + 1, 2))
                for role, context in enumerate(role_contexts):
                    role_write = F.linear(context, blocks[head], bias=None)
                    expected["head_role_write_norm"][
                        layer_index, row, head, role
                    ] = role_write.norm()
                    net_role_writes[role] += role_write
            for role, net_write in enumerate(net_role_writes):
                denominator = expected["head_role_write_norm"][
                    layer_index, row, :, role
                ].sum()
                expected["role_head_coherence"][layer_index, row, role] = (
                    net_write.norm() / denominator.clamp_min(1e-12)
                )
            current_k = min(k, query + 1)
            magnitude, index = source_energy.topk(current_k)
            expected["top_source_index"][
                layer_index, row, :current_k
            ] = index.int()
            expected["top_source_magnitude"][
                layer_index, row, :current_k
            ] = magnitude
    return expected


def test_capture_saves_the_rich_mechanism_state_and_all_branch_scores():
    model, replay = _tiny_replay()
    token_ids, response_start, evidence_mask = _inputs()
    predictors = torch.arange(response_start - 1, len(token_ids) - 1)

    with torch.inference_mode():
        logits = (
            model(
                input_ids=token_ids[None],
                attention_mask=torch.ones_like(token_ids)[None],
                use_cache=False,
                return_dict=True,
            )
            .logits[0]
            .index_select(0, predictors)
            .float()
        )
    targets = token_ids[response_start:]
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    expected_logprob = selected - logits.logsumexp(1)

    artifact = replay.capture(
        token_ids, response_start, evidence_mask, predictor_chunk=2, top_k=8
    )

    assert artifact["response_start"] == response_start
    assert set(artifact) == {
        "token_ids",
        "response_start",
        "trace",
        "score_inputs",
        "peak_cuda_reserved_bytes",
    }
    torch.testing.assert_close(
        artifact["score_inputs"]["full_logprob"],
        expected_logprob,
        atol=2e-5,
        rtol=2e-5,
    )
    assert set(artifact["score_inputs"]) == {
        "full_logprob",
        "full_margin",
        "no_evidence_logprob",
        "no_evidence_margin",
        "no_history_logprob",
        "no_history_margin",
        "no_evidence_history_logprob",
        "no_evidence_history_margin",
    }
    assert {
        "role_attention_mass",
        "edge_role_energy",
        "head_role_write_norm",
        "head_source_entropy",
        "role_head_coherence",
        "top_source_index",
        "top_source_magnitude",
    }.issubset(artifact["trace"])
    for family in ("attention", "edge"):
        for statistic in (
            "effective_sources",
            "mean_head_entropy",
            "head_jsd",
            "effective_rank",
            "mean_top1",
        ):
            assert artifact["trace"][f"prompt_{family}_{statistic}"].shape == (
                2,
                4,
            )
        assert artifact["trace"][f"prompt_{family}_anchor_index"].shape == (
            2,
            4,
            4,
        )
    assert ROLE_NAMES == (
        "evidence",
        "other_prompt",
        "response_history",
        "predictor_self",
    )
    assert artifact["trace"]["role_attention_mass"].shape == (2, 4, 4, 4)
    assert artifact["trace"]["edge_role_energy"].shape == (2, 4, 4, 4)
    assert artifact["trace"]["head_role_write_norm"].shape == (2, 4, 4, 4)
    assert artifact["trace"]["head_source_entropy"].shape == (2, 4, 4)
    assert artifact["trace"]["role_head_coherence"].shape == (2, 4, 4)
    assert artifact["trace"]["top_source_index"].shape == (2, 4, 6)
    assert all(value.shape == (4,) for value in artifact["score_inputs"].values())
    trace = artifact["trace"]
    evidence, other_prompt, response_history, predictor_self = range(4)
    assert torch.count_nonzero(
        trace["edge_role_energy"][:, :2, :, response_history]
    ) == 0
    assert torch.count_nonzero(
        trace["role_attention_mass"][:, :2, :, response_history]
    ) == 0
    assert torch.all(trace["edge_role_energy"][:, 2:, :, response_history] > 0)
    assert torch.all(trace["edge_role_energy"][:, :, :, predictor_self] > 0)
    assert torch.all(trace["role_attention_mass"][:, 0, :, predictor_self] > 0)
    assert torch.all(trace["edge_role_energy"][:, 0, :, evidence] > 0)
    assert torch.all(trace["edge_role_energy"][:, 0, :, other_prompt] > 0)
    assert torch.all(trace["head_role_write_norm"] >= 0)
    assert torch.all((trace["head_source_entropy"] >= 0))
    assert torch.all((trace["head_source_entropy"] <= 1 + 2e-3))
    assert torch.all((trace["role_head_coherence"] >= 0))
    assert torch.all((trace["role_head_coherence"] <= 1 + 2e-3))
    for family in ("attention", "edge"):
        assert torch.all(trace[f"prompt_{family}_effective_sources"] >= 1)
        assert torch.all(trace[f"prompt_{family}_effective_rank"] >= 1)
        assert torch.all((trace[f"prompt_{family}_mean_top1"] >= 0))
        assert torch.all((trace[f"prompt_{family}_mean_top1"] <= 1 + 2e-3))
    torch.testing.assert_close(
        trace["role_attention_mass"].float().sum(-1),
        torch.ones(2, 4, 4),
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        trace["edge_role_energy"].float().sum((2, 3)),
        trace["top_source_magnitude"].float().sum(-1),
        atol=4e-3,
        rtol=4e-3,
    )

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_sequence_oracle_reconstructs_every_saved_trace_equation():
    model, replay = _tiny_replay(layers=2)
    token_ids, response_start, evidence_mask = _inputs()
    artifact = replay.capture(
        token_ids,
        response_start,
        evidence_mask,
        predictor_chunk=1,
        top_k=3,
    )
    expected = _explicit_full_sequence_trace(
        model,
        token_ids,
        response_start,
        evidence_mask,
        top_k=3,
    )

    for name in (
        "role_attention_mass",
        "edge_role_energy",
        "head_role_write_norm",
        "head_source_entropy",
        "role_head_coherence",
        "top_source_magnitude",
    ):
        torch.testing.assert_close(
            artifact["trace"][name].float(),
            expected[name],
            atol=3e-3,
            rtol=3e-3,
        )
    assert torch.equal(
        artifact["trace"]["top_source_index"], expected["top_source_index"]
    )


def test_deletions_are_symmetric_response_query_interventions():
    _model, replay = _tiny_replay(layers=1)
    masks = replay._removal_mask(
        ("evidence", "history", "both"),
        torch.tensor([True, True, True]),
        response_start=3,
        query_start=0,
        query_stop=6,
    )

    evidence, history, both = masks
    assert torch.count_nonzero(masks[:, :2]) == 0
    assert torch.equal(
        evidence[2], torch.tensor([True, True, False, False, False, False])
    )
    assert torch.count_nonzero(history[2]) == 0
    assert torch.equal(
        evidence[3], torch.tensor([True, True, True, False, False, False])
    )
    assert torch.equal(
        history[3], torch.tensor([False, False, False, False, False, False])
    )
    assert torch.equal(
        history[4], torch.tensor([False, False, False, True, False, False])
    )
    assert torch.equal(both, evidence | history)


def test_interventions_project_each_layer_with_that_layers_own_o_proj():
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

    called_layers = set()
    project = replay._removed_write

    def record_layer(index, current):
        called_layers.add(index)
        return project(index, current)

    replay._removed_write = record_layer
    replay.capture(*_inputs(), predictor_chunk=2, intervention_batch=3)
    assert called_layers == {0, 1}


def test_chunked_interventions_match_full_sequence_hook_replay():
    model, _ = _tiny_replay(layers=2)
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        model.model.layers[1].self_attn.o_proj.weight.zero_()
    replay = FunctionalTraceReplay(model)
    token_ids, response_start, evidence_mask = _inputs()
    actual = replay.capture(
        token_ids,
        response_start,
        evidence_mask,
        predictor_chunk=len(token_ids),
        intervention_batch=1,
    )["score_inputs"]
    full = _one_shot_scores(model, token_ids, response_start, evidence_mask, None)
    no_evidence = _one_shot_scores(
        model, token_ids, response_start, evidence_mask, "evidence"
    )
    no_history = _one_shot_scores(
        model, token_ids, response_start, evidence_mask, "history"
    )
    no_evidence_history = _one_shot_scores(
        model, token_ids, response_start, evidence_mask, "both"
    )
    expected = {
        "full_logprob": full[0],
        "full_margin": full[1],
        "no_evidence_logprob": no_evidence[0],
        "no_evidence_margin": no_evidence[1],
        "no_history_logprob": no_history[0],
        "no_history_margin": no_history[1],
        "no_evidence_history_logprob": no_evidence_history[0],
        "no_evidence_history_margin": no_evidence_history[1],
    }
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, atol=3e-5, rtol=3e-5)
    assert not torch.allclose(no_evidence[0], full[0], atol=1e-5, rtol=1e-5)


def test_source_norm_is_dynamic_value_through_each_head_output_block():
    _model, replay = _tiny_replay(layers=2)
    value = torch.randn(5, replay.kv_heads, replay.head_dim)

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


def test_prompt_carrier_statistics_preserve_concentration_and_head_rank():
    mass = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    prompt = torch.ones(2, 3, dtype=torch.bool)
    result = FunctionalTraceReplay._prompt_carriers(mass, prompt)
    torch.testing.assert_close(result["effective_sources"], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(result["effective_rank"], torch.ones(2))
    torch.testing.assert_close(result["mean_top1"], torch.tensor([0.5, 1.0]))
    assert torch.equal(result["anchor_index"], torch.zeros(2, 2, dtype=torch.long))


def test_chunk_and_intervention_batch_sizes_do_not_change_results():
    _model, replay = _tiny_replay(layers=2)
    serial = replay.capture(
        *_inputs(), predictor_chunk=1, top_k=3, intervention_batch=1
    )
    batched = replay.capture(
        *_inputs(), predictor_chunk=4, top_k=3, intervention_batch=3
    )

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
            torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
        else:
            assert torch.equal(left, right)
