import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common.llama_message_intervention import baseline_forward
from experiments.reanchor_flow.mechanism import capture_mechanism


def test_grouped_cuts_return_layerwise_presence_and_exact_final_control():
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
    cache = baseline_forward(
        model,
        ids,
        response_start=4,
        checkpoint_layers=range(2),
        attention_query_chunk=2,
    )
    result = capture_mechanism(
        model,
        cache,
        response_start=4,
        evidence_mask=[True, True, False, False],
    )
    assert result["evidence_state_presence"].shape == (3, 4)
    assert result["evidence_state_control"].shape == (3, 4)
    assert result["evidence_effect"].shape == (4,)
    np.testing.assert_allclose(
        result["evidence_state_control"][-1],
        result["evidence_effect"],
        rtol=2e-4,
        atol=2e-4,
    )
    assert np.isfinite(result["evidence_readout_gain"]).all()
    assert np.isfinite(result["evidence_prompt_interaction"]).all()
