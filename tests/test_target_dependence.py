import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.target_dependence import (
    TargetOracle,
    localize_queries,
    observed_target,
)


def model():
    torch.manual_seed(73)
    config = LlamaConfig(vocab_size=30, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def target():
    row = {"prompt_length": 3, "token_ids": [1, 2, 3, 4, 5, 6, 7], "response": "a b c d"}
    return observed_target(row, {"response_offsets": [(0, 1), (1, 3), (3, 5), (5, 7)]}, [2, 5])


def test_full_original_target_sum_not_first_token_or_length_average():
    llm, t = model(), target()
    assert t["input_ids"] == [1, 2, 3, 4, 5]
    assert t["targets"] == [5, 6]
    oracle = TargetOracle(llm, t)
    with torch.no_grad():
        logps = llm(torch.tensor([t["input_ids"]])).logits[0].log_softmax(-1)
    expected = float(logps[3, 5] + logps[4, 6])
    assert math.isclose(oracle.base["logp"], expected, abs_tol=1e-6)
    assert oracle.calls == 1


def test_origin_donor_and_recipient_shams_and_real_replay_budgets():
    oracle = TargetOracle(model(), target(), budget=12)
    g = {"keys": [1], "domain": [1, 2], "queries": [2, 3, 4], "layers": [0, 1]}
    assert oracle.group(g, 1)["sham_exact"] is True
    sham = oracle.origin(g, [1], 1)
    assert sham["sham_exact"] and sham["donor_sham_exact"]
    actual = oracle.origin(g, [1], 0)
    assert actual["prefix_exact"] and actual["delta"] != 0
    assert oracle.calls == 6
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in oracle.model.modules())


def test_bounded_localization_keeps_unsearched_groups_and_exact_query_domain():
    oracle = TargetOracle(model(), target(), budget=12)
    g = {"keys": [1], "domain": [1, 2], "queries": [2, 3, 4], "layers": [0, 1]}
    found = localize_queries(oracle, g, budget=2)
    assert len(found["measured"]) == 2 and found["unsearched"] and not found["complete"]
    with pytest.raises(ValueError, match="escape"):
        oracle.group({**g, "queries": [0]})


def test_actual_forward_budget_is_enforced_and_hooks_are_clean():
    oracle = TargetOracle(model(), target(), budget=1)
    with pytest.raises(RuntimeError, match="budget"):
        oracle.group({"keys": [1], "domain": [1, 2], "queries": [2, 3, 4], "layers": [0, 1]})
    assert oracle.calls == 1
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in oracle.model.modules())


def test_origin_pair_reserves_two_calls_before_donor_and_localization_records_invisibility():
    oracle = TargetOracle(model(), target(), budget=2)
    g = {"keys": [3], "domain": [1, 2, 3], "queries": [2, 3, 4], "layers": [0, 1]}
    with pytest.raises(RuntimeError, match="before_origin_pair"):
        oracle.origin(g, [3], 0)
    assert oracle.calls == 1
    result = localize_queries(oracle, g, budget=1)
    assert result["excluded"][0]["group"]["queries"] == [2]
    with pytest.raises(ValueError, match="contiguous"):
        localize_queries(oracle, {**g, "queries": [2, 4]})
