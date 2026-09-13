import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.causal_contrast import ContinuationContrast
from route_graph.causal_groups import CausalOracle, propose_native_groups


def oracle(budget=40):
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    contrast = ContinuationContrast.from_sequences(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 8, 9, 10]], 4
    )
    return CausalOracle(model, contrast, budget)


def group():
    return {
        "kind": "content",
        "role": "source",
        "keys": [1],
        "domain": [1, 2],
        "queries": [3, 4],
        "layers": [0, 1],
    }


def test_oracle_complete_event_sham_donor_and_actual_forward_accounting():
    engine = oracle()
    assert engine.calls == 2
    assert engine.base_f == pytest.approx(
        sum(engine.base["token_logps"][0]) - sum(engine.base["token_logps"][1])
    )
    sham = engine.group(group(), 1)
    assert sham["sham_exact"] is True and sham["delta"] == 0
    assert engine.calls == 4
    assert engine.group(group(), 1) == sham and engine.calls == 4
    changed = engine.group(group(), 0)
    assert changed["sham_exact"] is None and changed["prefix_exact"]
    mediated = engine.mediated(group(), [1], 1)
    assert mediated["delta"] == 0 and mediated["sham_exact"] is True
    assert engine.calls == 10
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks
        for m in engine.model.modules()
    )


def test_budget_rejects_donor_before_any_partial_execution():
    engine = oracle(budget=5)
    with pytest.raises(RuntimeError, match="budget"):
        engine.mediated(group(), [1], 0)
    assert engine.calls == 2


def test_native_search_is_bounded_and_preserves_real_signed_measurements():
    engine = oracle(budget=20)
    result = propose_native_groups(engine, [1, 2], screen_calls=12)
    assert result["forward_calls"] <= 12
    assert any(
        m["group"].get("scope") == "broad_role_route" for m in result["measured"]
    )
    assert all(
        m["measurement"] in {r["key"] for r in engine.records}
        for m in result["measured"]
        if "measurement" in m
    )
    assert engine.calls == 2 + result["forward_calls"]


def test_public_gate_interfaces_reject_postdivergence_queries_without_forward():
    engine = oracle()
    bad = {**group(), "queries": [len(engine.contrast.prefix)]}
    for operation in (
        lambda: engine.gates(bad),
        lambda: engine.group(bad),
        lambda: engine.mass(bad),
        lambda: engine.mediated(bad, [1], 0),
    ):
        with pytest.raises(ValueError, match="shared prefix"):
            operation()
    assert engine.calls == 2
