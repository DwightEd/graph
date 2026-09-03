import pytest
import torch

from experiments.attention_mechanism_audit.capture import FunctionalMessageReplay

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM


def test_tiny_llama_builds_functional_message_graph():
    torch.manual_seed(9)
    config = LlamaConfig(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_bias=False,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    replay = FunctionalMessageReplay(model)
    token_ids = torch.tensor([1, 5, 7, 3, 9, 11, 2])
    evidence = torch.tensor([False, True, True, False])
    graph = replay.capture(
        token_ids,
        response_start=4,
        evidence_mask=evidence,
        predictor_batch=1,
        edge_cover=0.9,
        edge_budget=0,
    )
    batched = replay.capture(
        token_ids,
        response_start=4,
        evidence_mask=evidence,
        predictor_batch=2,
        edge_cover=0.9,
        edge_budget=0,
    )

    assert graph["schema"] == "functional-message-graph-v2"
    assert graph["evidence_mask"].tolist() == [False, True, True, False]
    assert graph["node_profile"].shape[:3] == (3, 2, 4)
    assert graph["token_flow"].shape == (3, 7, 4)
    assert graph["node_embedding"].shape[0] == 3
    assert graph["edge_index"].shape[0] == 2
    assert graph["edge_head_message"].shape[1] == 6
    assert graph["edge_index"].shape[1] == 2 * 4 * (4 + 5 + 6)
    assert torch.isfinite(graph["target_logprob"]).all()
    assert torch.isfinite(graph["edge_function"]).all()

    torch.testing.assert_close(graph["target_logprob"], batched["target_logprob"])
    torch.testing.assert_close(graph["token_flow"], batched["token_flow"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(graph["edge_function"], batched["edge_function"], atol=2e-6, rtol=2e-6)

    with torch.no_grad():
        reference = model(token_ids[None]).logits[0]
        targets = torch.tensor([9, 11, 2])
        expected = reference[3:6].log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(graph["target_logprob"], expected, atol=2e-5, rtol=2e-5)