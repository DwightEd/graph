import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph import soft_graph_phases as phases
from route_graph.soft_graph_energy import positive_dependence
from route_graph.soft_graph_structure import (
    assignment_marginals,
    matched_controls,
    pointer_graph,
    response_claims,
    words_with_scores,
)
from route_graph.source_event_graph import raw_inventory, validate_pointer_graph


class BadReader:
    def ask(self, instruction, payload, **kwargs):
        return {"reader_error": "invalid_json"}


def test_invalid_structural_reader_still_preserves_full_response_words():
    inv = raw_inventory("Alice paid 7.\nBob paid 9.", side="response", sample_id="r", unit_tokens=4)
    graph, _ = pointer_graph(BadReader(), inv)
    graph, claims, diagnostics = response_claims(inv, graph)
    validate_pointer_graph(inv, graph)
    assert claims and diagnostics["fallback_anchors"]
    assert all(c["event_ids"] for c in claims)
    assert all(e["status"] == "partial_event" for e in graph["events"])
    assert len(words_with_scores(inv["text"], [])) == 6


def fixture(monkeypatch):
    torch.manual_seed(89)
    cfg = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    monkeypatch.setitem(phases.PROTOCOL, "native_layers", [0, 1])
    response = "a b c d e f g h"
    row = {"id": "1", "source_id": "s", "response": response, "prompt_length": 15,
           "token_ids": list(range(1, 24))}
    alignment = {"response_offsets": [(i, i + 1) for i in range(0, 16, 2)], "source_keys": list(range(1, 13))}
    def edge(sid, keys, controls):
        return {"keys": keys, "selected_control_ids": [c["id"] for c in controls],
                "controls": {"candidates": controls}, "candidate": {"source_node_id": sid}, "key": [sid, sid, "event"]}
    controls = [{"id": "cs1", "keys": [7, 8]}, {"id": "cs2", "keys": [9, 10]}]
    source = [edge("s1", [1, 2], controls), edge("s2", [3, 4], controls)]
    history = edge("h3", [19, 20], [{"id": "h1", "keys": [15, 16]}, {"id": "h2", "keys": [17, 18]}])
    history["previous_claim_id"] = "h3"
    claim = {"claim_id": "current", "span": [12, 15], "source_terms": source, "history_terms": [history]}
    return model, row, {"alignment": alignment}, {"claims": [claim]}


def test_native_all_three_edges_are_real_controlled_origin_measurements_within_budget(tmp_path, monkeypatch):
    model, row, features, semantics = fixture(monkeypatch)
    before = copy.deepcopy(semantics)
    result = phases.native_phase(model, row, features, semantics, tmp_path)
    assert semantics == before
    claim = result["claims"][0]
    assert not claim["failures"]
    assert 40 < claim["forward_calls"] <= 80
    assert claim["donor_artifacts"]
    for item in list(claim["source"].values()) + list(claim["history"].values()):
        assert item["status"] == "measured_with_controls"
        assert item["origin_artifacts_verified"] and item["prefix_exact"]
        assert item["sham_exact"] and item["origin_donor_sham_exact"] and item["origin_recipient_sham_exact"]
        assert item["repeat_valid"] and len(item["control_deltas"]) == 2
        assert 0 <= positive_dependence(item) <= 1
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_unavailable_controls_remain_position_only_without_backfill(tmp_path, monkeypatch):
    model, row, features, semantics = fixture(monkeypatch)
    for item in semantics["claims"][0]["source_terms"] + semantics["claims"][0]["history_terms"]:
        item["selected_control_ids"] = []
    result = phases.native_phase(model, row, features, semantics, tmp_path)
    claim = result["claims"][0]
    assert not claim["donor_artifacts"]
    for item in list(claim["source"].values()) + list(claim["history"].values()):
        assert item["status"] == "measured_position_no_two_fixed_controls"
        assert positive_dependence(item) == 0


def test_native_failure_budget_is_retained_in_full_claim_denominator(tmp_path, monkeypatch):
    model, row, features, semantics = fixture(monkeypatch)
    monkeypatch.setitem(phases.PROTOCOL, "native_forwards_per_claim", 4)
    result = phases.native_phase(model, row, features, semantics, tmp_path)
    assert len(result["claims"]) == 1
    assert result["native_forward_calls"] == 4
    assert len(result["claims"][0]["failures"]) == 3


def test_finite_reader_failure_is_unknown_in_global_local_and_history_interfaces():
    for mapping in [{k: k for k in "SCNU"}, {k: k for k in "SCIU"},
                    {"R": "reuse", "C": "correction", "Q": "quote", "T": "new_topic", "U": "unknown"}]:
        scores, raw = phases.finite(BadReader(), "instruction", {"text": "data"}, mapping)
        assert sum(scores.values()) == 1
        assert scores.get("U", scores.get("unknown")) == 1
        assert "reader_error" in raw


def test_control_pool_cannot_replace_literal_schema_with_easy_raw_tokens():
    import numpy as np
    nodes = [
        {"id": "s", "kind": "literal_field", "span": [0, 1], "path": ["items", 0, "rating"], "value_type": "float", "memberships": ["r1"], "literal_unknown": False},
        {"id": "good", "kind": "literal_field", "span": [2, 3], "path": ["items", 1, "rating"], "value_type": "float", "memberships": ["r2"], "literal_unknown": False},
        {"id": "raw", "kind": "raw_token", "span": [4, 5]},
        {"id": "wrong_field", "kind": "literal_field", "span": [6, 7], "path": ["items", 2, "price"], "value_type": "float", "memberships": ["r3"], "literal_unknown": False},
        {"id": "wrong_type", "kind": "literal_field", "span": [8, 9], "path": ["items", 3, "rating"], "value_type": "str", "memberships": ["r4"], "literal_unknown": False},
    ]
    keys = {n["id"]: [i] for i, n in enumerate(nodes)}
    vectors = {n["id"]: np.ones((2, 4)) for n in nodes}
    controls = matched_controls("s", [0], nodes, keys, vectors)
    assert [c["id"] for c in controls["candidates"]] == ["good"]
    assert controls["structurally_rejected"] == 3


def test_none_field_is_retained_as_unknown_provenance_in_marginals():
    candidate = {"status": "null_literal", "source_node_id": "s", "membership": {"id": "e"}, "unary_cost": .2}
    match = {"event_assignments": [{"total_cost": .2, "role_links": [{"response_role_id": "r", "candidate": candidate}]}],
             "role_candidate_pools": [{"response_role_id": "r", "unsearched_raw_unit_ids": [], "same_value_outside_pool": 0,
                                        "near_tie_pruned_candidates": 0, "scored_occurrences": 1, "pruned_occurrences": 0,
                                        "same_value_competition": False}]}
    terms = assignment_marginals(match)
    assert len(terms) == 1 and terms[0]["candidate"]["status"] == "null_literal"
    assert terms[0]["pi"] == 1
