"""CPU checks for bounded typed native measurements, not semantic claims."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from next_iteration import typed_native_measure as typed_measure
from next_iteration.typed_native_measure import (
    match_controls,
    measure_path,
    root_group,
    search_queries_layers,
)
from route_graph.causal_contrast import ContinuationContrast
from route_graph.causal_groups import CausalOracle


def _oracle():
    torch.manual_seed(907)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    contrast = ContinuationContrast.from_sequences(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 8, 9]], 4
    )
    return CausalOracle(model, contrast, budget=128)


def test_typed_search_keeps_joint_parent_and_reserves_bounded_query_layer_work():
    oracle = _oracle()
    found = search_queries_layers(oracle, [1])
    assert found["actual_forward_calls"] <= 64
    root = next(item for item in found["measured"] if item["id"] == found["root"])
    assert root["group"]["queries"] == list(oracle.contrast.shared_queries)
    assert root["group"]["layers"] == [0, 1]
    assert found["searched_joint_groups"] >= 1
    assert found["searched_single_layer_groups"] >= 1
    assert found["localization_scope"].endswith("no unique lookback claim")
    assert all(item["group"]["keys"] == [1] for item in found["measured"] + found["pending"])
    assert oracle.calls == 2 + found["actual_forward_calls"]


def test_typed_root_rejects_nonprompt_A_key_before_search():
    oracle = _oracle()
    with pytest.raises(ValueError, match="source-prompt"):
        root_group(oracle, [oracle.contrast.prompt_length])
    assert oracle.calls == 2


def test_fixed_control_ids_are_filtered_without_postsearch_backfill(monkeypatch):
    oracle = _oracle()
    group = root_group(oracle, [1])
    census = [
        {"id": "c1", "keys": [0], "grade": "schema_unrelated_to_hours_and_wifi"},
        {"id": "c2", "keys": [2], "grade": "schema_unrelated_to_hours_and_wifi"},
        {"id": "c3", "keys": [3], "grade": "schema_unrelated_to_hours_and_wifi"},
    ]
    monkeypatch.setattr(typed_measure, "ratio_matches", lambda *_: True)
    monkeypatch.setattr(typed_measure, "visibility", lambda *_: 1.0)
    initial = match_controls(oracle, group, census)
    assert [item["id"] for item in initial["position"]] == ["c1", "c2"]
    restricted = match_controls(
        oracle,
        group,
        census,
        fixed_ids={"position": ["c1"], "origin": []},
    )
    assert [item["id"] for item in restricted["position"]] == ["c1"]
    assert restricted["origin"] == []
    assert not restricted["backfill_permitted"]


def test_uncontrolled_typed_A_measurement_stays_raw_and_cleans_tiny_llama_hooks():
    oracle = _oracle()
    found = search_queries_layers(oracle, [1])
    masks = [
        [False] * len(oracle.contrast.continuations[0]),
        [False] * len(oracle.contrast.continuations[1]),
    ]
    measured = measure_path(
        oracle,
        found,
        {"position": [], "origin": [], "census": [], "target": {}},
        masks,
    )
    assert not measured["passed"]
    assert measured["status"] == "raw_native_measurements_uncontrolled_or_not_selective"
    assert measured["routing"] == "unresolved_no_independent_B_occurrence"
    assert measured["aggregation"] == "unresolved"
    assert measured["full_continuation_preference"] in {"A_preferred", "B_preferred", "near_tied"}
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in oracle.model.modules()
    )


def test_pair_census_retains_only_disjoint_unrelated_members_with_stable_member_metadata():
    prepared = {
        "schedule": {
            "source_graph": {
                "occurrences": [
                    {"occurrence_id": "z", "field_path": ["notes", "z"], "span": [0, 1], "epistemic_status": "observed_literal"},
                    {"occurrence_id": "a", "field_path": ["notes", "a"], "span": [3, 4], "epistemic_status": "observed_literal"},
                    {"occurrence_id": "hours", "field_path": ["hours", "Mon", "close"], "span": [5, 6], "epistemic_status": "observed_literal"},
                    {"occurrence_id": "unknown", "field_path": ["notes", "unknown"], "span": [7, 8], "epistemic_status": "unknown"},
                    {"occurrence_id": "overlap_left", "field_path": ["notes", "left"], "span": [9, 11], "epistemic_status": "observed_literal"},
                    {"occurrence_id": "overlap_right", "field_path": ["notes", "right"], "span": [10, 12], "epistemic_status": "observed_literal"},
                ]
            }
        }
    }
    # The shared character is intentionally an un-tokenized separator.  Token-key
    # disjointness alone therefore cannot prove that two literal leaves are disjoint.
    alignment = {"source_offsets": [(0, 1), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]}
    census = typed_measure.source_control_census(prepared, alignment, prefix_length=20)
    pairs = [entry for entry in census if entry.get("kind") == "noncontiguous_control_only_union"]

    assert all("hours" not in entry["member_ids"] and "unknown" not in entry["member_ids"] for entry in pairs)
    assert all(
        right[1] <= left[0] or left[1] <= right[0]
        for entry in pairs
        for left, right in [entry["spans"]]
    )
    pair = next(entry for entry in pairs if entry["member_ids"] == ["a", "z"])
    assert pair["spans"] == [[3, 4], [0, 1]]
    assert pair["field_paths"] == [["notes", "a"], ["notes", "z"]]
    assert pair["keys"] == [0, 2]
