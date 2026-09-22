"""Native GQA, one-target derivatives, causal alignment, and resource restoration."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from state_audit.attribution import (
    aggregate_sources,
    candidate_margin,
    capture_target_attribution,
)
from state_audit.capture import capture_targets
from state_audit.model.adapter import ModelAdapter
from state_audit.operations import Target


@pytest.fixture
def attribution_model():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    config = LlamaConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0,
    )
    model = ModelAdapter(LlamaForCausalLM(config))
    return model, [1, 3, 5, 7, 9, 11, 13, 15], 4, 2


def test_native_logits_gqa_and_additive_readout(attribution_model):
    model, token_ids, prompt, target = attribution_model
    prefix = token_ids[: prompt + target]
    with torch.no_grad(), capture_targets(model, {"value": Target("value", (0,))}) as captured:
        expected = model.native(model.input_ids(prefix), use_cache=False).logits[0, -1]
    result = capture_target_attribution(model, token_ids, prompt, target)
    # lm_head on one position versus a full sequence can round differently.
    np.testing.assert_allclose(result["logits"], expected.numpy(), atol=1e-7)
    assert result["attention"].shape == (2, 4, 6)
    np.testing.assert_allclose(result["attention"].sum(-1), 1, atol=1e-7)
    np.testing.assert_allclose(
        result["contribution"].sum(-1),
        (result["head_gradient"] * result["head_readout"]).sum(-1),
        rtol=1e-5,
        atol=1e-7,
    )
    values = np.repeat(captured["value"][0], 2, axis=0).transpose(1, 0, 2)
    with torch.no_grad():
        writes = model.project_heads(0, values, tuple(range(4))).numpy()
    messages = writes * result["attention"][0].T[..., None]
    np.testing.assert_allclose(
        result["value_energy"][0], np.square(messages).sum(-1).T, rtol=1e-5, atol=1e-9
    )


@pytest.mark.parametrize("layers", [(0, 1), (1,)])
def test_single_edge_gate_finite_difference(attribution_model, layers):
    model, token_ids, prompt, target = attribution_model
    result = capture_target_attribution(model, token_ids, prompt, target, layers=layers)
    layer_index, head, key = np.unravel_index(
        np.abs(result["contribution"]).argmax(), result["contribution"].shape
    )
    layer = int(result["layers"][layer_index])
    query = int(result["query"])
    epsilon = 0.01
    margins = []
    for perturbation in (-epsilon, epsilon):
        def gate(weights):
            changed = weights.clone()
            changed[head, query, key] *= 1 + perturbation
            return changed

        with torch.no_grad(), model.bind("attention", layer, gate, edit=True):
            hidden = model.forward(token_ids[: prompt + target])
            logits = model.native.lm_head(hidden[-1]).float()
            margins.append(float(candidate_margin(logits, token_ids[prompt + target])))
    derivative = (margins[1] - margins[0]) / (2 * epsilon)
    expected = result["contribution"][layer_index, head, key]
    assert derivative == pytest.approx(expected, rel=0.02, abs=2e-5)


def test_subset_preserves_derivatives_without_upstream_graph(attribution_model):
    model, token_ids, prompt, target = attribution_model
    full = capture_target_attribution(model, token_ids, prompt, target)
    early_requires_grad = []

    def observe_early_layer(module, inputs, output):
        early_requires_grad.append(output.requires_grad)

    hook = model.layers[0].register_forward_hook(observe_early_layer)
    try:
        subset = capture_target_attribution(model, token_ids, prompt, target, layers=(1,))
    finally:
        hook.remove()
    # History prefill and the final query both stay outside the upstream graph.
    assert early_requires_grad and not any(early_requires_grad)
    for name in ("head_gradient", "contribution", "head_readout", "attention", "value_energy"):
        np.testing.assert_array_equal(subset[name], full[name][1:], err_msg=name)
    np.testing.assert_array_equal(subset["logits"], full["logits"])


def test_target_alignment_and_future_exclusion(attribution_model):
    model, token_ids, prompt, target = attribution_model
    result = capture_target_attribution(model, token_ids, prompt, target, layers=(1,))
    changed = token_ids[: prompt + target + 1] + [2, 4, 6, 8, 10]
    other = capture_target_attribution(model, changed, prompt, target, layers=(1,))
    for name in result:
        np.testing.assert_array_equal(result[name], other[name], err_msg=name)
    assert result["query"] == prompt + target - 1
    assert result["target_id"] == token_ids[prompt + target]
    assert result["key_positions"][-1] == result["query"]
    np.testing.assert_array_equal(result["layers"], [1])


def test_special_tokens_excluded_only_from_group_statistics(attribution_model):
    model, token_ids, prompt, target = attribution_model
    baseline = capture_target_attribution(model, token_ids, prompt, target)
    result = capture_target_attribution(
        model, token_ids, prompt, target, special_token_ids=(1, 7)
    )
    np.testing.assert_array_equal(result["logits"], baseline["logits"])
    np.testing.assert_array_equal(result["attention"], baseline["attention"])
    groups = np.array([0, 0, 0, 0, 1, 1])
    aggregated = aggregate_sources(result, groups, 2)
    np.testing.assert_allclose(
        aggregated["route_mass"][..., 0], result["attention"][..., [1, 2]].sum(-1)
    )
    np.testing.assert_allclose(
        aggregated["contribution_positive"] - aggregated["contribution_negative"],
        result["contribution"] @ (np.eye(2)[groups] * result["ordinary_keys"][:, None]),
        atol=1e-7,
    )
    assert np.all(aggregated["value_energy"] >= 0)


def test_parameter_flags_grads_and_hooks_restored_even_on_failure(attribution_model):
    model, token_ids, prompt, target = attribution_model
    parameters = tuple(model.native.parameters())
    parameters[0].requires_grad_(False)
    parameters[1].grad = torch.ones_like(parameters[1])
    saved_grad = parameters[1].grad.clone()
    flags = tuple(parameter.requires_grad for parameter in parameters)
    capture_target_attribution(model, token_ids, prompt, target)
    with patch.object(model.native.lm_head, "forward", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            capture_target_attribution(model, token_ids, prompt, target)
    assert tuple(parameter.requires_grad for parameter in parameters) == flags
    assert torch.equal(parameters[1].grad, saved_grad)
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
