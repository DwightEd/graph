import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch.nn import functional as F

from experiments.attention_mechanism_audit.capture import (
    FunctionalTraceReplay,
    HISTORY,
    SELF,
    predictor_positions,
)
from experiments.attention_mechanism_audit.data import (
    CONSTRAINT,
    EVIDENCE,
    QUESTION,
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
    return model, FunctionalTraceReplay(model, checkpoint="tiny-llama")


def _inputs():
    token_ids = torch.tensor([1, 5, 7, 9, 11, 13, 15])
    response_start = 3
    prompt_roles = torch.tensor([EVIDENCE, QUESTION, CONSTRAINT])
    return token_ids, response_start, prompt_roles


def _scores(model, hidden, targets):
    logits = model.lm_head(hidden).float()
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    competitor = logits.scatter(1, targets[:, None], -torch.inf).max(1).values
    return {
        "target_logit": selected,
        "target_logprob": selected - logits.logsumexp(1),
        "target_margin": selected - competitor,
        "top1_token_id": logits.argmax(1),
    }


def test_tiny_llama_uses_previous_positions_and_preserves_full_logits():
    model, replay = _tiny_replay()
    token_ids, response_start, prompt_roles = _inputs()
    predictors = predictor_positions(response_start, len(token_ids))

    with torch.inference_mode():
        direct = model(
            input_ids=token_ids[None],
            attention_mask=torch.ones_like(token_ids)[None],
            use_cache=False,
            return_dict=True,
        ).logits[0].index_select(0, predictors).float()
    targets = token_ids[response_start:]
    selected = direct.gather(1, targets[:, None]).squeeze(1)
    competitor = direct.scatter(1, targets[:, None], -torch.inf).max(1).values
    expected = {
        "target_logit": selected,
        "target_logprob": selected - direct.logsumexp(1),
        "target_margin": selected - competitor,
        "top1_token_id": direct.argmax(1),
    }

    artifact = replay.capture(
        token_ids,
        response_start,
        prompt_roles,
        predictor_chunk=2,
        top_k=3,
    )

    assert torch.equal(artifact["predictor_positions"], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(artifact["target_ids"], torch.tensor([9, 11, 13, 15]))
    source_role = artifact["trace"]["source_role"]
    assert source_role[0, 2] == SELF
    assert not torch.any(source_role[0] == HISTORY)
    assert source_role[2, 3] == HISTORY
    assert source_role[2, 4] == SELF
    for name, value in expected.items():
        actual = artifact["scores"]["full"][name]
        if value.dtype.is_floating_point:
            torch.testing.assert_close(actual, value.cpu(), atol=2e-5, rtol=2e-5)
        else:
            assert torch.equal(actual, value.cpu())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_saved_attention_and_values_reconstruct_the_actual_oproj_write():
    model, replay = _tiny_replay()
    token_ids, response_start, prompt_roles = _inputs()
    artifact = replay.capture(
        token_ids, response_start, prompt_roles, predictor_chunk=2, top_k=3
    )
    trace = artifact["trace"]

    for layer_index, layer in enumerate(replay.layers):
        attention = trace["attention"][layer_index].permute(1, 0, 2)
        value = trace["value_states"][layer_index]
        value_by_head = value[:, replay.q_to_kv]
        context = torch.einsum("rhn,nhd->rhd", attention, value_by_head)
        torch.testing.assert_close(
            trace["o_proj_input"][layer_index],
            context.reshape(len(context), replay.hidden),
            atol=2e-5,
            rtol=2e-5,
        )
        expected = F.linear(
            context.reshape(len(context), replay.hidden),
            layer.self_attn.o_proj.weight.detach().cpu(),
            (
                layer.self_attn.o_proj.bias.detach().cpu()
                if layer.self_attn.o_proj.bias is not None
                else None
            ),
        )
        torch.testing.assert_close(
            trace["attention_update"][layer_index],
            expected,
            atol=2e-5,
            rtol=2e-5,
        )
        output_weight = layer.self_attn.o_proj.weight.detach().cpu().float()
        source_norm = torch.stack(
            [
                F.linear(
                    value_by_head[:, head].float(),
                    output_weight[
                        :, head * replay.head_dim : (head + 1) * replay.head_dim
                    ],
                ).norm(dim=-1)
                for head in range(replay.heads)
            ],
            dim=1,
        )
        edge_magnitude = attention.float() * source_norm.T[None]
        expected_roles = torch.stack(
            [
                (edge_magnitude * (trace["source_role"] == role)[:, None]).sum(-1)
                for role in range(len(artifact["role_names"]))
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            trace["role_edge_magnitude"][layer_index],
            expected_roles,
            atol=2e-5,
            rtol=2e-5,
        )


def test_one_layer_message_ablations_equal_manual_residual_updates():
    model, replay = _tiny_replay(layers=1)
    token_ids, response_start, prompt_roles = _inputs()
    artifact = replay.capture(
        token_ids, response_start, prompt_roles, predictor_chunk=2, top_k=3
    )
    trace = artifact["trace"]
    layer = replay.layers[0]
    attention = trace["attention"][0].permute(1, 0, 2)
    value_by_head = trace["value_states"][0][:, replay.q_to_kv]
    source_role = trace["source_role"]

    branch_roles = {
        "evidence_removed": (EVIDENCE,),
        "response_removed": (HISTORY, SELF),
        "evidence_response_removed": (EVIDENCE, HISTORY, SELF),
    }
    for branch, removed_roles in branch_roles.items():
        removed_sources = torch.zeros_like(source_role, dtype=torch.bool)
        for role in removed_roles:
            removed_sources |= source_role == role
        if SELF in removed_roles:
            removed_sources[0] &= source_role[0] != SELF
        removed_context = torch.einsum(
            "rhn,nhd->rhd",
            attention * removed_sources[:, None],
            value_by_head,
        )
        removed_write = F.linear(
            removed_context.reshape(len(removed_context), replay.hidden),
            layer.self_attn.o_proj.weight.detach().cpu(),
            None,
        )
        post_attention = (
            trace["residual_input"][0]
            + trace["attention_update"][0]
            - removed_write
        )
        with torch.inference_mode():
            layer_output = post_attention + layer.mlp(
                layer.post_attention_layernorm(post_attention)
            )
            hidden = model.model.norm(layer_output)
            expected = _scores(model, hidden, artifact["target_ids"])
        for name, value in expected.items():
            actual = artifact["scores"][branch][name]
            if value.dtype.is_floating_point:
                torch.testing.assert_close(actual, value, atol=2e-5, rtol=2e-5)
            else:
                assert torch.equal(actual, value)


def test_multilayer_scores_are_chunk_and_intervention_batch_invariant():
    _model, replay = _tiny_replay(layers=2)
    token_ids, response_start, prompt_roles = _inputs()
    serial = replay.capture(
        token_ids,
        response_start,
        prompt_roles,
        predictor_chunk=1,
        top_k=3,
        intervention_batch=1,
    )
    batched = replay.capture(
        token_ids,
        response_start,
        prompt_roles,
        predictor_chunk=4,
        top_k=3,
        intervention_batch=3,
    )

    assert serial["scores"].keys() == batched["scores"].keys()
    for branch in serial["scores"]:
        for name in serial["scores"][branch]:
            left = serial["scores"][branch][name]
            right = batched["scores"][branch][name]
            if left.dtype.is_floating_point:
                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
            else:
                assert torch.equal(left, right)


def test_mechanism_trace_keeps_registered_routes_without_dense_raw_states():
    _model, replay = _tiny_replay(layers=2)
    token_ids, response_start, prompt_roles = _inputs()

    artifact = replay.capture(
        token_ids,
        response_start,
        prompt_roles,
        predictor_chunk=2,
        top_k=3,
        retain_raw=False,
    )

    trace = artifact["trace"]
    assert {
        "role_attention",
        "role_edge_magnitude",
        "source_message_entropy",
        "message_coherence",
        "top_source_index",
        "top_source_magnitude",
        "source_role",
    } <= set(trace)
    assert {
        "attention",
        "value_states",
        "residual_input",
        "attention_update",
        "mlp_update",
        "final_hidden",
    }.isdisjoint(trace)
