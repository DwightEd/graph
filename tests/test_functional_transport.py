"""Native identities, complete original-token coverage, and one-command execution."""


import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

from state_audit.functional_capture import capture_observed, log_probability, rms_readout
from state_audit.functional_hooks import functional_hooks
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.storage import write_json


@pytest.fixture
def model():
    torch.manual_seed(37)
    torch.set_num_threads(1)
    return ModelAdapter(LlamaForCausalLM(LlamaConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)))


def test_rms_scaling_can_suppress_margin_without_direct_write(model):
    with torch.no_grad():
        model.native.lm_head.weight[:, -1] = 0
        before = torch.randn(2, 24)
        before[:, -1] = 0
        message = torch.zeros_like(before)
        message[:, -1] = 5
        logits = model.native.lm_head(model.native.model.norm(before + message))
        target = logits.argmax(-1)
        rows = rms_readout(model, before, message, target, logits)
    np.testing.assert_allclose(rows["rms_direct_margin"], 0, atol=1e-7)
    assert np.all(rows["rms_rescale_margin"] < 0)
    np.testing.assert_allclose(rows["rms_identity_error"], 0, atol=1e-7)


def test_capture_matches_native_sequence_and_root_directional_derivative(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13, 15]
    captured = capture_observed(model, prefix, phrase, [0, 0, 1, 1], 2)
    ids = prefix + phrase[:-1]
    with attention_backend(model, "sdpa"):
        embeddings = model.native.model.embed_tokens(model.input_ids(ids)).detach()
        def objective(values):
            hidden = model.native.model(inputs_embeds=values, use_cache=False).last_hidden_state
            logits = model.native.lm_head(hidden[0, len(prefix) - 1:])
            return log_probability(logits, torch.tensor(phrase)).sum()
        with torch.no_grad():
            expected = float(objective(embeddings))
            direction = torch.zeros_like(embeddings)
            direction[0, 1] = embeddings[0, 1]
            finite = float((objective(embeddings + .01 * direction) - objective(embeddings - .01 * direction)) / .02)
    assert captured["log_probability"].sum() == pytest.approx(expected, abs=2e-6)
    assert captured["prefix_root_sensitivity"][1] == pytest.approx(finite, abs=2e-4)
    np.testing.assert_allclose(captured["head_total"], captured["head_residual"] + captured["head_ffn_mediated"], atol=1e-7)
    assert captured["head_total"].shape == (3, 2, 4, 3)
    assert captured["prefix_root_sensitivity"].shape == (4,)
    assert captured["within_unit_root_sensitivity"].shape == (2,)
    np.testing.assert_allclose(captured["head_attention"].sum(-1), 1, atol=2e-7)


def test_ffn_channel_matches_native_local_jacobian(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13]
    row = capture_observed(model, prefix, phrase, [0, 0, 1, 1], 2)
    with attention_backend(model, "sdpa"), functional_hooks(model) as records:
        output = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]), use_cache=False)
        logits = model.native.lm_head(output.last_hidden_state[0, 3:])
        objective = log_probability(logits, torch.tensor(phrase)).sum()
        mid, write = records[0]["mid"], records[0]["ffn"]
        upstream, downstream = torch.autograd.grad(objective, (mid, write), retain_graph=True)
        via_ffn, = torch.autograd.grad(write, mid, grad_outputs=downstream, retain_graph=True)
        torch.testing.assert_close(upstream, downstream + via_ffn)
        head = records[0]["head"][0, 3].view(4, 6)
        direction = (via_ffn[0, 3] @ model.layers[0].self_attn.o_proj.weight).view(4, 6)
        expected = (head * direction).sum(-1).detach().numpy()
    np.testing.assert_allclose(row["head_ffn_mediated"][0, 0].sum(-1), expected, atol=1e-6)


def test_native_hook_resources_restore_on_failure(model):
    original = model.layers[0].mlp.forward
    with pytest.raises(RuntimeError), functional_hooks(model):
        raise RuntimeError("stop")
    assert model.layers[0].mlp.forward == original
    assert not model.layers[0].mlp._forward_hooks
    capture_observed(model, [1, 5, 7], [9], [0, 0, 1], 2)
    assert all(parameter.requires_grad for parameter in model.native.parameters())
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_sliding_window_rows_match_native_attention():
    model = ModelAdapter(MistralForCausalLM(MistralConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, sliding_window=3)))
    prefix, phrase = [1, 5, 7, 9, 11], [13, 15]
    groups = list(range(len(prefix)))
    captured = capture_observed(model, prefix, phrase, groups, len(prefix))
    with torch.no_grad(), attention_backend(model, "eager"):
        native = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]),
                                    use_cache=False, output_attentions=True)
    for layer, attention in enumerate(native.attentions):
        np.testing.assert_allclose(captured["head_attention"][:, layer],
            attention[0, :, 4:6].transpose(0, 1).numpy(), atol=2e-7)


def test_native_entropy_is_full_vocabulary_and_causal(model):
    prefix, targets = [1, 5, 7], [9, 11, 13]
    captured = capture_observed(model, prefix, targets, [0, 0, 1], 2)
    with torch.no_grad(), attention_backend(model, "sdpa"):
        hidden = model.forward(prefix + targets[:-1])[2:]
        logp, entropy = model.score(hidden, torch.tensor(targets))
    np.testing.assert_allclose(captured["entropy"], entropy.numpy(), atol=1e-7)
    np.testing.assert_allclose(captured["log_probability"], logp.numpy(), atol=1e-7)
    # Later target changes may affect the unit gradient, never earlier native probabilities.
    changed = capture_observed(model, prefix, [9, 15, 17], [0, 0, 1], 2)
    np.testing.assert_allclose(captured["entropy"][:2], changed["entropy"][:2], atol=1e-7)
    np.testing.assert_allclose(captured["log_probability"][:1], changed["log_probability"][:1], atol=1e-7)
