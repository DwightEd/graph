"""Cached final-query derivatives agree with full causal replay, without a T² graph."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from state_audit.attribution import (
    capture_target_attribution,
    capture_target_attribution_full,
    iter_target_attributions,
)
from state_audit.model.adapter import ModelAdapter


@pytest.fixture(params=["llama", "mistral", "qwen2"])
def causal_model(request):
    torch.set_num_threads(1)
    torch.manual_seed(83)
    config_class, model_class = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    }[request.param]
    options = dict(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0,
    )
    if request.param == "mistral":
        options["sliding_window"] = 4
    return ModelAdapter(model_class(config_class(**options)))


def assert_attribution_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if actual[name].dtype.kind in "biu":
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        else:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=3e-5, atol=3e-7, err_msg=name
            )


@pytest.mark.parametrize("prompt,target", [(1, 0), (4, 0), (9, 3)])
@pytest.mark.parametrize("layers", [None, (1,), (0, 2)])
def test_cached_attribution_matches_full_replay(causal_model, prompt, target, layers):
    token_ids = [(2 * position + 1) % 31 for position in range(17)]
    options = dict(layers=layers, special_token_ids=(1, 7))
    expected = capture_target_attribution_full(causal_model, token_ids, prompt, target, **options)
    actual = capture_target_attribution(
        causal_model, token_ids, prompt, target, prefill_chunk_size=3, **options
    )
    assert_attribution_equal(actual, expected)
    assert actual["attention"].shape[-1] == prompt + target
    if causal_model.native.config.model_type == "mistral" and prompt + target > 4:
        # Retain full key indexing; masked older sources have exactly zero mass.
        assert not actual["attention"][..., :-4].any()


def test_bounded_no_grad_prefill_and_single_query_backward(causal_model):
    token_ids = [(2 * position + 1) % 31 for position in range(53)]
    calls, saved_shapes, saved_bytes = [], [], []
    original = torch.nn.functional.scaled_dot_product_attention

    def observe_sdpa(query, key, value, *args, **kwargs):
        calls.append((query.shape[-2], key.shape[-2], torch.is_grad_enabled()))
        return original(query, key, value, *args, **kwargs)

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        saved_bytes.append(tensor.numel() * tensor.element_size())
        return tensor

    with patch("torch.nn.functional.scaled_dot_product_attention", observe_sdpa):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = capture_target_attribution(
                causal_model, token_ids, 40, 7, prefill_chunk_size=5
            )
    assert calls and max(query for query, _, _ in calls) <= 5
    assert max(key for _, key, _ in calls) == 46
    assert all(not requires_grad for _, _, requires_grad in calls)
    assert saved_shapes and all(shape[-2:] != (47, 47) for shape in saved_shapes)
    assert result["attention"].shape == (3, 4, 47)

    totals = []
    for length in (23, 35, 47):
        saved_bytes.clear()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            capture_target_attribution(causal_model, token_ids, length, 0, prefill_chunk_size=5)
        totals.append(sum(saved_bytes))
    # This measures saved backward tensors, not total resident GPU memory.
    assert totals[1] - totals[0] == totals[2] - totals[1] > 0

    saved_shapes.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        capture_target_attribution_full(causal_model, token_ids, 40, 7)
    assert any(shape[-2:] == (47, 47) for shape in saved_shapes)


def test_bfloat16_kernel_rounding_is_small_on_tiny_models(causal_model):
    causal_model.native.to(dtype=torch.bfloat16)
    token_ids = [(2 * position + 1) % 31 for position in range(53)]
    expected = capture_target_attribution_full(causal_model, token_ids, 40, 7)
    actual = capture_target_attribution(causal_model, token_ids, 40, 7, prefill_chunk_size=5)
    # SDPA and eager need not be bitwise equal in BF16. Check aggregate deviation;
    # near-zero individual edges cannot support a meaningful relative tolerance.
    for name in ("logits", "attention", "contribution", "value_energy", "head_gradient"):
        deviation = np.linalg.norm(actual[name] - expected[name])
        scale = np.linalg.norm(expected[name])
        assert deviation / scale < 0.02, name


def test_repeated_calls_restore_backend_and_do_not_reuse_future(causal_model):
    token_ids = [(2 * position + 1) % 31 for position in range(17)]
    expected = capture_target_attribution_full(causal_model, token_ids, 9, 3)
    for chunk_size in (1, 4, 256):
        actual = capture_target_attribution(
            causal_model,
            token_ids[:13] + [21, 22, 23],
            9,
            3,
            prefill_chunk_size=chunk_size,
        )
        assert_attribution_equal(actual, expected)
        assert causal_model.native.config._attn_implementation == "eager"
        for module in causal_model.native.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks


def test_prefill_failure_restores_backend_and_parameter_flags(causal_model):
    flags = [parameter.requires_grad for parameter in causal_model.native.parameters()]
    with patch.object(
        causal_model.native.model, "forward", side_effect=RuntimeError("prefill stopped")
    ):
        with pytest.raises(RuntimeError, match="prefill stopped"):
            capture_target_attribution(causal_model, [1, 2, 3, 4, 5], 3, 1)
    assert causal_model.native.config._attn_implementation == "eager"
    assert [parameter.requires_grad for parameter in causal_model.native.parameters()] == flags
    for module in causal_model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize("targets", [(0, 1, 2, 3), (0, 3, 7), (2, 5)])
def test_stream_matches_independent_targets_and_computes_each_prefix_token_once(
    causal_model, targets
):
    token_ids = [(2 * position + 1) % 31 for position in range(25)]
    calls = []
    original = causal_model.native.model.forward

    def observe(*args, **kwargs):
        cache = kwargs["past_key_values"]
        # At every new forward, earlier target graphs must already be released.
        assert all(
            layer.keys.grad_fn is None and layer.values.grad_fn is None for layer in cache.layers
        )
        calls.append(kwargs["input_ids"].shape[-1])
        return original(*args, **kwargs)

    flags = [parameter.requires_grad for parameter in causal_model.native.parameters()]
    iterator = iter_target_attributions(
        causal_model, token_ids, 9, targets, layers=(1, 2), prefill_chunk_size=3
    )
    with patch.object(causal_model.native.model, "forward", side_effect=observe):
        actual = []
        for result in iterator:
            actual.append(result)
            assert [
                parameter.requires_grad for parameter in causal_model.native.parameters()
            ] == flags
            assert causal_model.native.config._attn_implementation == "eager"
    assert sum(calls) == 9 + targets[-1]
    for target, result in zip(targets, actual):
        expected = capture_target_attribution_full(
            causal_model, token_ids, 9, target, layers=(1, 2)
        )
        assert_attribution_equal(result, expected)


def test_stream_does_not_read_future_tokens_or_keep_hooks_when_closed(causal_model):
    token_ids = list(range(1, 17))
    first = iter_target_attributions(causal_model, token_ids, 5, (0, 1, 4))
    changed = iter_target_attributions(causal_model, token_ids[:6] + [23] * 10, 5, (0, 1, 4))
    assert_attribution_equal(next(first), next(changed))
    first.close()
    changed.close()
    for module in causal_model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.requires_grad for parameter in causal_model.native.parameters())


@pytest.mark.parametrize("targets", [(2, 1), (1, 1), (-1,), (20,)])
def test_stream_rejects_noncausal_target_plans(causal_model, targets):
    with pytest.raises(ValueError):
        list(iter_target_attributions(causal_model, list(range(12)), 5, targets))
