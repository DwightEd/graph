import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch.nn import functional as F

from experiments.attention_mechanism_audit.capture import FunctionalTraceReplay


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
    removed = torch.zeros(length, dtype=torch.bool)
    if removal in {"evidence", "both"}:
        removed[:response_start] |= evidence_mask
    if removal in {"response", "both"}:
        removed[response_start:] = True
    mask = torch.ones(length, length, dtype=torch.bool).tril() & removed[None]
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


def test_capture_keeps_only_the_exact_inputs_needed_by_the_four_scores():
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
        token_ids, response_start, evidence_mask, predictor_chunk=2, top_k=3
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
        "no_evidence_logprob",
        "no_response_logprob",
        "no_evidence_response_margin",
    }
    assert set(artifact["trace"]) == {
        "total_message_magnitude",
        "evidence_message_magnitude",
        "response_message_magnitude",
        "source_message_entropy",
        "top_source_index",
        "top_source_magnitude",
    }
    assert artifact["trace"]["total_message_magnitude"].shape == (2, 4)
    assert artifact["trace"]["source_message_entropy"].shape == (2, 4)
    assert artifact["trace"]["top_source_index"].shape == (2, 4, 3)
    assert all(value.shape == (4,) for value in artifact["score_inputs"].values())
    trace = artifact["trace"]
    assert torch.count_nonzero(trace["response_message_magnitude"][:, 0]) == 0
    assert torch.all(trace["response_message_magnitude"][:, 1:] > 0)
    assert torch.all(
        trace["total_message_magnitude"] + 1e-6
        >= trace["evidence_message_magnitude"] + trace["response_message_magnitude"]
    )

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


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


def test_multilayer_interventions_match_an_independent_full_sequence_oracle():
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
    no_response = _one_shot_scores(
        model, token_ids, response_start, evidence_mask, "response"
    )
    no_evidence_response = _one_shot_scores(
        model, token_ids, response_start, evidence_mask, "both"
    )
    expected = {
        "full_logprob": full[0],
        "no_evidence_logprob": no_evidence[0],
        "no_response_logprob": no_response[0],
        "no_evidence_response_margin": no_evidence_response[1],
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
