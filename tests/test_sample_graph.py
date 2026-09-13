import copy

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.sample_graph import capture_sample_graph, validate_graph


@pytest.fixture
def model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval().requires_grad_(False)


def test_full_graph_preserves_source_edges_raw_features_and_causal_prefix(model):
    first = capture_sample_graph(model, [1, 2, 3, 4, 5], 3, [False, True, True])
    longer = capture_sample_graph(model, [1, 2, 3, 4, 5, 6], 3, [False, True, True])
    assert first["residual"].shape == (3, 4, 16)
    assert first["attention"].shape == (2, 4, 4, 4)
    assert first["attention"][0, :, 2, :2].sum() > 0
    assert np.allclose(first["residual"], longer["residual"][:, :4], atol=1e-7)
    assert np.allclose(first["attention"], longer["attention"][:, :, :4, :4], atol=1e-7)
    checks = validate_graph(first)
    assert checks["maximum_raw_mlp_residual_addition_error"] < 1e-7
    with torch.no_grad():
        actual = model(
            torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True, use_cache=False
        )
        raw = torch.from_numpy(first["residual"][-1])
        normalized = model.model.norm(raw)
        assert torch.allclose(normalized, actual.hidden_states[-1][0], atol=1e-7)
        assert not torch.allclose(raw, actual.hidden_states[-1][0])
    assert list(first["target_ids"]) == [4, 5]


def test_graph_rejects_future_edges_and_target_misalignment(model):
    graph = capture_sample_graph(model, [1, 2, 3, 4, 5], 3, [False, True, True])
    bad = copy.deepcopy(graph)
    bad["attention"][0, 0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="future"):
        validate_graph(bad)
    bad = copy.deepcopy(graph)
    bad["target_ids"][0] = 8
    with pytest.raises(ValueError, match="identities"):
        validate_graph(bad)
