import pytest
import torch

from route_graph.relation_readout import compatibility


def test_compatibility_node_unchanged_when_edges_swap():
    query = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    attention = torch.tensor([[[0.2, 0.8]]])
    before = compatibility(query, values, attention, torch.eye(2))
    after = compatibility(query, values, attention.flip(-1), torch.eye(2))
    assert torch.equal(before["node"], after["node"])
    assert torch.allclose(before["node"], torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(before["graph"], torch.tensor([[0.2, 0.0]]))
    assert torch.allclose(after["graph"], torch.tensor([[0.8, 0.0]]))
    assert torch.equal(before["attention"].sum(-1), after["attention"].sum(-1))
    with pytest.raises(ValueError, match="unaligned"):
        compatibility(query, values, attention, torch.eye(3))


def test_capture_preserves_native_forward_and_cleans_hooks():
    from transformers import LlamaConfig, LlamaForCausalLM

    from route_graph.relation_readout import capture_relation

    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    ids, queries = [1, 2, 3, 4, 5, 6], [3, 4, 5]
    with torch.no_grad():
        expected = model(torch.tensor([ids])).logits[0, queries]
    logits, factors, scores = capture_relation(model, ids, queries, [1, 2], [1, 2])
    assert torch.equal(logits, expected)
    assert factors[0]["query_residual"].shape == (3, 16)
    assert factors[0]["q_before_rope"].shape == (3, 4, 4)
    assert factors[0]["k_before_rope"].shape == (2, 2, 4)
    assert all(value.shape == (2, 3, 2) for value in scores.values())
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )
