import numpy as np
import pytest
import torch

from experiments.reanchor_flow.message_dag.counterfactual_pairs import (
    CounterfactualPair,
)
from experiments.reanchor_flow.message_dag.message_exchange import (
    Carrier,
    MessageExchangeWitness,
    WitnessConfig,
)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(61)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval()


def pair():
    return CounterfactualPair(
        pair_id="pair-1",
        constraint_kind="entity",
        token_ids=np.array([[1, 2, 3, 4, 5, 6], [1, 7, 3, 4, 5, 6]]),
        response_start=3,
        changed_positions=(1,),
        candidate_plus_ids=(8, 9),
        candidate_minus_ids=(10, 11),
        constraint_plus="A",
        constraint_minus="B",
        response_prefix="shared prefix",
        source_id="source",
        dataset="tiny",
    ).check()


def hf_log_probability(model, prefix, candidate):
    ids = torch.tensor([*prefix, *candidate])[None]
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[0]
    positions = torch.arange(len(prefix) - 1, len(prefix) + len(candidate) - 1)
    targets = torch.tensor(candidate)
    return float(logits[positions].log_softmax(-1)[torch.arange(len(targets)), targets].sum())


def test_bidirectional_post_wo_exchange_preserves_baseline_and_scores_sequences():
    model, spec = tiny_model(), pair()
    result = MessageExchangeWitness(
        model, WitnessConfig(doses=(0.0, 0.5, 1.0), query_chunk=2)
    ).run(spec, Carrier(layer=0, head=1, position=4))

    assert result["sequence_log_odds"].shape == (2, 3)
    assert result["evidence_log_odds"].shape == (2, 3)
    assert result["candidate_log_probability"].shape == (2, 3, 2)
    assert result["forward_calls"] == 16
    assert result["baseline_reproduction_error"] < 1e-6
    assert result["labels_used"] is False
    assert result["intervention_kind"] == "post_wo_head_message_exchange"

    expected = [
        hf_log_probability(model, spec.token_ids[world], candidate)
        for world in range(2)
        for candidate in (spec.candidate_plus_ids, spec.candidate_minus_ids)
    ]
    np.testing.assert_allclose(result["baseline_candidate_log_probability"].reshape(-1), expected, atol=2e-5)
    np.testing.assert_allclose(
        result["evidence_log_odds"][:, 0],
        [
            expected[0] - expected[1],
            expected[3] - expected[2],
        ],
        atol=2e-5,
    )
    assert np.linalg.norm(result["native_head_message_delta"]) > 0
    assert np.max(np.abs(result["sequence_log_odds"][:, -1] - result["sequence_log_odds"][:, 0])) > 1e-8


def test_witness_rejects_invalid_doses_and_carrier_coordinates():
    model, spec = tiny_model(), pair()
    with pytest.raises(ValueError, match="include 0 and 1"):
        WitnessConfig(doses=(0.0, 0.5)).check()
    with pytest.raises(ValueError, match="strictly increasing"):
        WitnessConfig(doses=(0.0, 0.5, 0.5, 1.0)).check()
    witness = MessageExchangeWitness(model, WitnessConfig(doses=(0.0, 1.0)))
    with pytest.raises(ValueError, match="response-prefix"):
        witness.run(spec, Carrier(layer=0, head=0, position=1))
