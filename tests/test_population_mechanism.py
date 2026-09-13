import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.population_mechanism import (
    CONDITIONS,
    compare_states,
    endpoint_permutation,
    response_forward,
)


def test_population_model_sham_causality_and_actual_messages():
    torch.manual_seed(11)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    ids, prompt, mask, permutation = (
        [1, 2, 3, 4, 5, 6, 7],
        4,
        [False, True, True, False],
        [1, 0],
    )
    base, profiles, _ = response_forward(
        model, ids, prompt, mask, "base", permutation, history_window=1, chunk=2
    )
    assert profiles["source_attention"].shape == (2, 4, 4)
    with torch.inference_mode():
        original = model.model(torch.tensor([ids]), use_cache=False).last_hidden_state[
            0
        ]
    assert torch.equal(base, original)
    for condition in CONDITIONS[1:]:
        altered, _, _ = response_forward(
            model, ids, prompt, mask, condition, permutation, history_window=1, chunk=2
        )
        assert torch.equal(altered[: prompt - 1], base[: prompt - 1])
        if condition == "sham":
            assert torch.equal(altered, base)
        else:
            assert not torch.equal(altered[-1], base[-1])
        effects = compare_states(
            model, base[prompt - 1 :], altered[prompt - 1 :], [5, 6, 7, 8], chunk=2
        )
        assert np.isfinite(effects["js"]).all()
        if condition == "sham":
            assert not effects["argmax_changed"].any()
            assert not effects["js"].any()
        if condition.startswith("history_"):
            assert torch.equal(
                altered[prompt - 1 : prompt + 1], base[prompt - 1 : prompt + 1]
            )
    # Future observed tokens must not enter a preceding computation or its permutation.
    changed, _, _ = response_forward(
        model, ids[:-1] + [9], prompt, mask, "source_permute", permutation, chunk=2
    )
    original, _, _ = response_forward(
        model, ids, prompt, mask, "source_permute", permutation, chunk=2
    )
    assert torch.equal(changed[:-1], original[:-1])
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )


def test_population_identity_and_invalid_inputs():
    perm = endpoint_permutation(20, "1234")
    assert np.array_equal(perm, endpoint_permutation(20, "1234"))
    assert sorted(perm) == list(range(20))
    assert not np.array_equal(perm, endpoint_permutation(20, "1235"))
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    with pytest.raises(ValueError, match="permutation"):
        response_forward(model, [1, 2, 3], 2, [True, True], "base", [0, 0])
