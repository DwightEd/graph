import numpy as np
import pytest

from experiments.attention_mechanism_audit.replay import (
    FrozenMarginReplay,
    history_source_positions,
    predictor_allowed_attention,
    prompt_source_allowed_attention,
)


def test_prompt_source_ablation_removes_direct_and_relay_routes():
    allowed = prompt_source_allowed_attention(7, 6, [1, 3])
    causal = np.tri(7, 7, dtype=np.bool_)

    assert allowed[1, 1]
    assert allowed[3, 3]
    assert not allowed[2:, 1].any()
    assert not allowed[4:, 3].any()
    untouched = causal.copy()
    untouched[2:, 1] = False
    untouched[4:, 3] = False
    np.testing.assert_array_equal(allowed, untouched)


def test_history_ablation_changes_only_predictor_row_and_keeps_diagonal():
    history = history_source_positions(3, 7, predictor_index=6)
    np.testing.assert_array_equal(history, [3, 4, 5])

    allowed = predictor_allowed_attention(7, 6, history)
    causal = np.tri(7, 7, dtype=np.bool_)
    causal[6, 3:6] = False
    np.testing.assert_array_equal(allowed, causal)
    assert allowed[6, 6]


def _tiny_replay():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(41)
    config = transformers.LlamaConfig(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
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
    return torch, model, FrozenMarginReplay(model, checkpoint="tiny-llama")


def _direct_margin(torch, replay, tokens, allowed, candidate_b, candidate_a):
    ids = torch.as_tensor(tokens, dtype=torch.long)[None]
    if allowed is None:
        mask = torch.ones_like(ids)
    else:
        mask = torch.zeros(allowed.shape, dtype=replay.model.dtype)
        mask.masked_fill_(
            ~torch.as_tensor(allowed), torch.finfo(replay.model.dtype).min
        )
        mask = mask[None, None]
    with torch.inference_mode():
        hidden = replay.backbone(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0, -1]
        logits = replay.model.lm_head(hidden[None, None])[0, 0]
        return float((logits[candidate_b].float() - logits[candidate_a].float()).item())


def test_tiny_llama_margin_and_two_mask_semantics_are_exact():
    torch, model, replay = _tiny_replay()
    tokens = [1, 5, 7, 9, 11, 13, 15]
    predictor = len(tokens) - 1
    candidate_b, candidate_a = 17, 19

    expected_full = _direct_margin(
        torch, replay, tokens, None, candidate_b, candidate_a
    )
    actual_full = replay.score_margin(
        tokens, predictor, candidate_b, candidate_a
    )
    assert actual_full == pytest.approx(expected_full, abs=1e-7)

    source_allowed = prompt_source_allowed_attention(len(tokens), predictor, [1, 2])
    expected_without_source = _direct_margin(
        torch, replay, tokens, source_allowed, candidate_b, candidate_a
    )
    actual_without_source = replay.score_without_prompt_sources_margin(
        tokens, predictor, candidate_b, candidate_a, [1, 2]
    )
    assert actual_without_source == pytest.approx(expected_without_source, abs=1e-7)

    history_allowed = predictor_allowed_attention(
        len(tokens), predictor, [3, 4, 5]
    )
    expected_without_history = _direct_margin(
        torch, replay, tokens, history_allowed, candidate_b, candidate_a
    )
    actual_without_history = replay.score_without_history_margin(
        tokens, predictor, candidate_b, candidate_a, 3, 7
    )
    assert actual_without_history == pytest.approx(expected_without_history, abs=1e-7)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_tiny_llama_full_path_mask_zeroes_the_declared_attention_edges():
    torch, _model, replay = _tiny_replay()
    tokens = [1, 5, 7, 9, 11, 13, 15]
    predictor = len(tokens) - 1
    sources = [1, 2]
    allowed = prompt_source_allowed_attention(len(tokens), predictor, sources)
    ids = torch.as_tensor(tokens, dtype=torch.long)[None]

    with torch.inference_mode():
        output = replay.backbone(
            input_ids=ids,
            attention_mask=replay._attention_mask(ids[0], allowed),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    for attention in output.attentions:
        for source in sources:
            assert torch.count_nonzero(attention[0, :, source + 1 :, source]) == 0
            assert torch.count_nonzero(attention[0, :, source, source]) > 0


def test_tiny_llama_prior_history_kv_patch_is_exact_and_excludes_predictor():
    torch, model, replay = _tiny_replay()
    prior_tokens = [1, 4, 6, 8, 10, 12, 14]
    counter_tokens = [1, 21, 23, 8, 10, 12, 14]
    predictor = len(prior_tokens) - 1
    candidate_b, candidate_a = 16, 18

    history_kv, captured_margin = replay.capture_history_kv(
        prior_tokens,
        predictor,
        candidate_b,
        candidate_a,
        history_start=3,
        history_stop=7,
    )
    assert history_kv.history_start == 3
    assert history_kv.history_stop == predictor
    assert len(history_kv.keys) == len(replay.layers) == 2
    assert len(history_kv.values) == 2
    assert history_kv.keys[0].shape == (3, 12)
    assert history_kv.values[0].shape == (3, 12)

    # Patching an identical branch must be numerically identical to the
    # ordinary forward; this binds capture and replacement at every layer.
    prior_full = replay.score_margin(
        prior_tokens, predictor, candidate_b, candidate_a
    )
    assert captured_margin == pytest.approx(prior_full, abs=1e-7)
    prior_hybrid = replay.score_hybrid_history_margin(
        prior_tokens, predictor, candidate_b, candidate_a, history_kv
    )
    assert prior_hybrid == pytest.approx(prior_full, abs=1e-7)

    counter_hybrid = replay.score_hybrid_history_margin(
        counter_tokens, predictor, candidate_b, candidate_a, history_kv
    )
    assert np.isfinite(counter_hybrid)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
