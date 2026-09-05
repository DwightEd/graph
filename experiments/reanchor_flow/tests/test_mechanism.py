import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common.llama_message_intervention import baseline_forward
from experiments.reanchor_flow.mechanism import capture_mechanism, vocabulary_effect


def tiny_case():
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
    return model, cache


def test_grouped_cuts_return_layerwise_presence_and_exact_final_control():
    model, cache = tiny_case()
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
    assert result["context_candidate_id"].shape == (4, 5)
    assert result["context_candidate_logprob_gain"].shape == (4, 5)
    assert not np.any(
        result["context_candidate_id"] == cache.target.numpy()[:, None]
    )
    assert (result["context_target_rank"] >= 1).all()
    assert (result["context_distribution_js"] >= -1e-7).all()
    assert np.isfinite(result["context_target_logprob_gain"]).all()
    assert np.isfinite(result["context_adoption_margin"]).all()


def test_functional_cut_skips_extra_grouped_reruns():
    model, cache = tiny_case()
    result = capture_mechanism(
        model,
        cache,
        response_start=4,
        evidence_mask=[True, True, False, False],
        grouped=False,
    )
    assert result["functional"] == 1
    assert result["mechanism"] == 0
    assert result["context_candidate_id"].shape == (4, 5)
    assert "history_effect" not in result
    assert "evidence_state_presence" not in result


def test_vocabulary_event_batching_is_exact():
    model, cache = tiny_case()
    baseline = cache.final_hidden
    cut = baseline.clone()
    cut[cache.query] += 0.01
    one = vocabulary_effect(model, cache, baseline, cut, chunk=1)
    three = vocabulary_effect(model, cache, baseline, cut, chunk=3)
    assert one.keys() == three.keys()
    for name in one:
        np.testing.assert_allclose(one[name], three[name], rtol=1e-6, atol=1e-6)
