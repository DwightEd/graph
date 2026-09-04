import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common.llama_message_intervention import baseline_forward
from experiments.reanchor_flow.routes import RouteAccumulator


def test_tiny_llama_emits_complete_rhythm_trace():
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    observer = RouteAccumulator(
        model,
        response_start=4,
        prompt_evidence_mask=[True, True, False, False],
        route_window=2,
        future_horizon=3,
        far_lag=2,
        detail=True,
    )
    baseline_forward(
        model,
        ids,
        response_start=4,
        observer=observer,
        attention_query_chunk=2,
    )
    trace = observer.finish()
    assert trace.prompt_share.shape == (2, 4)
    assert trace.future_influence.shape == (2, 4)
    assert trace.detail["edge_map"].shape == (4, 7)
