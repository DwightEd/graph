import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.adoption import (
    TerminalSuffix,
    distribution_effect,
    grouped_head_messages,
    opposition,
    outgoing_key_mask,
    probe_messages,
)


def test_opposition_retains_sign_and_zero_mass():
    assert opposition([2, 3])["competition"] == 0
    assert opposition([2, -2])["competition"] == 1
    assert opposition([0, 0])["competition"] is None
    assert opposition([2, -1])["competition"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        opposition([float("nan")])


def test_gqa_messages_close_and_preserve_opposing_heads():
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    values = torch.tensor([[[2.0]], [[4.0]]])
    projection = torch.tensor([[1.0, -1.0]])
    messages = grouped_head_messages(weights, values, projection, [0, 1])
    assert messages.shape == (2, 2, 1)
    assert messages.sum() == pytest.approx(0.5)
    assert messages[0, 0] > 0 and messages[0, 1] < 0
    with pytest.raises(ValueError):
        grouped_head_messages(weights, values, projection, [0, 2])


def test_access_cut_changes_only_declared_causal_rows():
    sham = outgoing_key_mask(9, [], 5)[0, 0]
    cut = outgoing_key_mask(9, [2, 3], 5)[0, 0]
    assert torch.equal(sham[:5], cut[:5])
    assert (cut[5:, 2:4] < -1e20).all()
    assert torch.equal(cut.diag(), torch.zeros(9))
    assert (cut[torch.ones(9, 9, dtype=torch.bool).triu(1)] < -1e20).all()
    with pytest.raises(ValueError):
        outgoing_key_mask(9, [5], 5)


def test_full_distribution_effect_is_symmetric_and_zero_for_sham():
    first, second = torch.tensor([[2.0, -2.0]]), torch.tensor([[-2.0, 2.0]])
    zero = distribution_effect(first, first, [0])
    assert zero["js_nats"] == pytest.approx([0], abs=1e-7)
    assert zero["saved_token_logp_change"] == [0]
    effect = distribution_effect(first, second, [0])
    assert effect["argmax_changed"] == [True]
    assert 0 < effect["js_nats"][0] < np.log(2)
    assert effect["js_nats"] == distribution_effect(second, first, [0])["js_nats"]


def test_terminal_suffix_matches_actual_fp32_model_and_derivative():
    torch.manual_seed(3)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    captured = {}

    def hook(module, args):
        captured["state"] = args[0][0, -1].detach().clone()

    handle = model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(
        hook
    )
    with torch.no_grad():
        native = model(torch.tensor([[1, 3, 4, 8]]), use_cache=False).logits[0, -1]
    handle.remove()
    candidates = native.topk(4).indices
    suffix = TerminalSuffix(model, candidates)
    assert torch.allclose(suffix(captured["state"]), native[candidates], atol=1e-7)
    messages = torch.randn(3, 4, 16) * 1e-3
    summary, arrays = probe_messages(
        suffix, captured["state"], messages, native[candidates]
    )
    step = 1e-3
    message = messages[0].sum(0)
    plus = suffix(captured["state"] + step * message)
    minus = suffix(captured["state"] - step * message)
    finite_difference = ((plus[0] - plus[1]) - (minus[0] - minus[1])) / (2 * step)
    estimate = arrays["candidate_group_head_contributions"][0, 0].sum()
    assert estimate == pytest.approx(float(finite_difference), abs=2e-5, rel=2e-3)
    assert summary["surrogate_pool_top1_matches_native"]
    assert arrays["finite_margin_changes_0.1"].shape == (3, 3)
    assert arrays["finite_margin_changes_0.01"].shape == (3, 3)
