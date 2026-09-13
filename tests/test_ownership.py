import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.ownership import (
    capture_source_factors,
    factorial_effects,
    intervene_source_factors,
    source_context_delta,
)


def test_node_edge_factorization_preserves_mass_and_keeps_interaction():
    a0 = torch.tensor([[[0.3, 0.1]], [[0.2, 0.2]]])
    a1 = torch.tensor([[[0.1, 0.3]], [[0.3, 0.1]]]) * 2
    v0 = torch.tensor([[[1.0]], [[3.0]]])
    v1 = torch.tensor([[[4.0]], [[1.0]]])
    x, _ = source_context_delta(a0, v0, a1, v1, "x")
    e, diag = source_context_delta(a0, v0, a1, v1, "e")
    xe, _ = source_context_delta(a0, v0, a1, v1, "xe")
    assert torch.allclose(x[:, :, 0], torch.tensor([[0.7, 0.2]]))
    assert torch.allclose(e[:, :, 0], torch.tensor([[0.4, -0.2]]))
    assert not torch.allclose(xe, x + e)
    assert diag["source_mass_max_error"] < 1e-7
    sham, _ = source_context_delta(a0, v0, a0, v0, "xe")
    assert torch.equal(sham, torch.zeros_like(sham))


def test_factorial_effects_match_explicit_contrasts():
    effects = factorial_effects([[0, 2], [3, 9]])
    assert effects == {
        "source_main_effect": 5,
        "history_main_effect": 4,
        "interaction": 4,
    }
    with pytest.raises(ValueError):
        factorial_effects([1, 2])


def test_real_model_sham_is_exact_and_patches_preserve_earlier_queries():
    torch.manual_seed(8)
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
    ids, donor_ids = [1, 2, 3, 4, 5], [1, 3, 2, 4, 5]
    queries, sources = [2, 3, 4], [0, 1, 2]
    baseline, base = capture_source_factors(model, ids, queries, sources, [0, 1])
    _, donor = capture_source_factors(model, donor_ids, queries, sources, [0, 1])
    for factor in ["xe", "mlp"]:
        same, _ = intervene_source_factors(
            model,
            ids,
            queries,
            sources,
            [{"layer": 0, "queries": queries, "factor": factor, **base[0]}],
        )
        assert torch.equal(same, baseline)
    for factor in ["x", "e", "xe", "mlp", "endpoint_swap"]:
        patch = {
            "layer": 0,
            "queries": [4],
            "factor": factor,
            "attention": donor[0]["attention"][:, [2]],
            "values": donor[0]["values"],
            "mlp_update": donor[0]["mlp_update"][[2]],
            "source_indices": [0, 1],
        }
        changed, _ = intervene_source_factors(model, ids, queries, sources, [patch])
        assert torch.equal(changed[:2], baseline[:2])
        assert not np.array_equal(changed[2].numpy(), baseline[2].numpy())


def test_failed_capture_or_patch_removes_all_registered_hooks():
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    ids, queries, sources = [1, 2, 3, 4], [2, 3], [0, 1]
    baseline, factors = capture_source_factors(model, ids, queries, sources, [0])
    valid = {"layer": 0, "queries": queries, "factor": "xe", **factors[0]}
    # The invalid second specification fails after the first registered hooks.
    with pytest.raises(ValueError, match="already be visible"):
        intervene_source_factors(
            model, ids, queries, sources, [valid, {**valid, "layer": 1, "queries": [0]}]
        )
    with pytest.raises(IndexError):
        capture_source_factors(model, ids, queries, sources, [0, 99])
    # A hook-time failure must also release every handle.
    with pytest.raises(ValueError, match="incompatible"):
        intervene_source_factors(
            model, ids, queries, sources, [{**valid, "values": torch.zeros(1)}]
        )
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in model.modules()
    )
    restored, _ = capture_source_factors(model, ids, queries, sources, [])
    assert torch.equal(baseline, restored)
