"""Engineering invariants only; handcrafted vectors are not model evidence."""

import copy
import itertools

import numpy as np
import pytest

from route_graph.event_matcher import (
    MatchConfig,
    _relation,
    feature_ids,
    feature_manifest,
    match_catalog,
    propose_event,
)
from route_graph.source_event_graph import (
    compile_literal_fields,
    compile_pointer_events,
    raw_inventory,
)
from route_graph.span_feature_capture import capture_record, span_mapping


def make_graph(text, side, sample_id, specifications):
    inventory = raw_inventory(text, side=side, sample_id=sample_id)
    def pointer(a, b):
        return {"unit_id": "u0", "start_token": a, "end_token": b}
    events = [{"anchor": pointer(a, b), "roles": [
        {"role": role, "pointer": pointer(x, y)} for role, x, y in roles
    ]} for a, b, roles in specifications]
    envelope = {"inventory_sha256": inventory["sha256"], "side": side,
                "sample_id": sample_id, "prediction": {"events": events}}
    graph = compile_pointer_events(inventory, envelope)
    assert not graph["failures"]
    return inventory, graph


def fixture():
    s, sg = make_graph("Crane lifted three. Pilot lifted three.", "source", "s", [
        (0, 4, [("subject", 0, 1), ("predicate", 1, 2), ("quantity", 2, 3)]),
        (4, 8, [("subject", 4, 5), ("predicate", 5, 6), ("quantity", 6, 7)]),
    ])
    r, rg = make_graph("Crane lifted three.", "response", "r", [
        (0, 4, [("subject", 0, 1), ("predicate", 1, 2), ("quantity", 2, 3)]),
    ])
    sc, rc = match_catalog(s, sg), match_catalog(r, rg)
    # Artificial equal states isolate bookkeeping and structural-search tests.
    sv = np.ones((len(feature_ids(sc)), 2, 3), dtype=np.float32)
    rv = np.ones((len(feature_ids(rc)), 2, 3), dtype=np.float32)
    encoder = {k: "a" * 64 for k in ("model_sha256", "tokenizer_sha256", "source_input_sha256",
                                    "response_input_sha256", "capture_code_sha256")}
    encoder.update(layers=[0, 1], representation="pre_block_residual_mean")
    sm = span_mapping(s, sc, list(range(len(s["text"]))), [[i, i + 1] for i in range(len(s["text"]))],
                      {"prompt": s["text"], "response": "", "source_span": [0, len(s["text"])]})
    rm = span_mapping(r, rc, list(range(len(s["text"]) + len(r["text"]))),
                      [None] * len(s["text"]) + [[i, i + 1] for i in range(len(r["text"]))],
                      {"prompt": s["text"], "response": r["text"], "source_span": [0, len(s["text"])]})
    encoder.update(source_input_sha256=sm["input_sha256"], response_input_sha256=rm["input_sha256"])
    captures = {"source": capture_record(sc, sm, sv, encoder), "response": capture_record(rc, rm, rv, encoder)}
    manifest = feature_manifest(sc, rc, sv, rv, encoder, captures)
    return [s, sg, r, rg, rg["events"][0]["id"], sv, rv, manifest]


def test_same_value_different_events_are_distinct_and_not_classified():
    args = fixture()
    result = propose_event(*args, MatchConfig(per_channel=10))
    quantity = next(p for p in result["role_candidate_pools"] if p["role"] == "quantity")
    same = quantity["same_value_candidates"]
    assert len(same) == 2 and len({n["source_node_id"] for n in same}) == 2
    assert len({n["membership"]["id"] for n in same}) == 2
    assert quantity["same_value_competition"]
    assert not result["absence_inference_allowed"] and not result["native_executed"]
    assert all(a["identity_status"] == "not_verified" for a in result["event_assignments"])
    assert all(a["downstream_allowed"] == "relation_verification_only" for a in result["event_assignments"])
    assert propose_event(*args, MatchConfig(per_channel=10)) == result
    assert not result["source_capacity_used"]


def test_beam_with_sufficient_budget_matches_enumerated_objective():
    result = propose_event(*fixture(), MatchConfig(per_channel=10, beam_width=2000))
    pools = result["role_candidate_pools"]
    costs = []
    for candidates in itertools.product(*(p["candidates"] for p in pools)):
        unary = sum(c["unary_cost"] for c in candidates) / len(candidates)
        pairs = [_relation(a, b)[0] for a, b in itertools.combinations(candidates, 2)]
        costs.append(unary + .25 * sum(pairs) / len(pairs))
    assert result["event_assignments"][0]["total_cost"] == pytest.approx(min(costs))
    assert result["denominators"]["beam_states_pruned"] == 0


def test_pruned_pool_and_beam_remain_in_denominators():
    result = propose_event(*fixture(), MatchConfig(per_channel=1, beam_width=1))
    assert result["denominators"]["beam_states_pruned"] > 0
    assert any(p["pruned_occurrences"] > 0 for p in result["role_candidate_pools"])
    assert all(any(c["status"] == "null" for c in p["candidates"]) for p in result["role_candidate_pools"])
    assert all(any(c["status"] == "unknown" for c in p["candidates"]) for p in result["role_candidate_pools"])


def test_foreign_features_and_modified_raw_coordinates_are_rejected():
    args = fixture()
    args[5][0, 0, 0] = 2
    with pytest.raises(ValueError, match="integrity|captured arrays"):
        propose_event(*args)
    args = fixture()
    args[0]["units"][0]["tokens"][0]["span"] = [0, 4]
    with pytest.raises(ValueError, match="integrity"):
        propose_event(*args)


@pytest.mark.parametrize("field,value", [("beam_width", 0), ("per_channel", True),
                                        ("null_cost", float("nan")), ("relation_weight", -1)])
def test_invalid_search_config_is_not_silently_repaired(field, value):
    with pytest.raises(ValueError):
        propose_event(*fixture(), MatchConfig(**{field: value}))


def test_feature_identity_and_space_fail_closed():
    args = fixture()
    sc, rc = match_catalog(args[0], args[1]), match_catalog(args[2], args[3])
    encoder = copy.deepcopy(args[-1]["encoder"])
    encoder["model_sha256"] = ""
    with pytest.raises(ValueError, match="SHA256"):
        feature_manifest(sc, rc, args[5], args[6], encoder, args[-1]["captures"])
    with pytest.raises(ValueError, match="float32"):
        feature_manifest(sc, rc, args[5].astype(np.float16), args[6], args[-1]["encoder"], args[-1]["captures"])


def test_raw_unit_membership_is_not_invented_as_event_membership():
    inventory = raw_inventory("Crane lifted three. Pilot lifted three.", side="source", sample_id="s")
    catalog = match_catalog(inventory)
    assert len(catalog["nodes"]) == 6
    assert all(n["kind"] == "raw_token" and n["memberships"] == [] for n in catalog["nodes"])
    assert not catalog["semantic_extraction_complete"]


def test_literal_unknown_paths_controls_and_raw_search_contract():
    args = fixture()
    s = raw_inventory("{'records': [{'quantity': None}, {'quantity': 3}, {'quantity': 3}]}", side="source", sample_id="s2")
    sg = compile_literal_fields(s)
    sc, rc = match_catalog(s, sg), match_catalog(args[2], args[3])
    sv = np.ones((len(feature_ids(sc)), 2, 3), dtype=np.float32)
    encoder = copy.deepcopy(args[-1]["encoder"])
    sm = span_mapping(s, sc, list(range(len(s["text"]))), [[i, i + 1] for i in range(len(s["text"]))],
                      {"prompt": s["text"], "response": "", "source_span": [0, len(s["text"])]})
    encoder["source_input_sha256"] = sm["input_sha256"]
    rm = span_mapping(args[2], rc, list(range(len(s["text"]) + len(args[2]["text"]))),
                      [None] * len(s["text"]) + [[i, i + 1] for i in range(len(args[2]["text"]))],
                      {"prompt": s["text"], "response": args[2]["text"], "source_span": [0, len(s["text"])]})
    encoder["response_input_sha256"] = rm["input_sha256"]
    captures = {"response": capture_record(rc, rm, args[6], encoder), "source": capture_record(sc, sm, sv, encoder)}
    manifest = feature_manifest(sc, rc, sv, args[6], encoder, captures)
    result = propose_event(s, sg, args[2], args[3], args[4], sv, args[6], manifest,
                           MatchConfig(per_channel=100, output_assignments=32))
    candidates = [n for p in result["role_candidate_pools"] for n in p["candidates"]]
    unknown = [n for n in candidates if n.get("literal_unknown")]
    assert unknown and all(n["status"] == "null_literal" for n in unknown)
    assert all("field_path" in n["cost_terms"] for n in candidates if n.get("source_kind") == "literal_field")
    pools = result["role_candidate_pools"]
    assert all(p["searched_raw_unit_ids"] == ["u0"] and not p["unsearched_raw_unit_ids"] for p in pools)
    assert all(p["raw_token_candidates_scored"] > 0 and p["graph_node_candidates_scored"] > 0 for p in pools)
    controls = [c for a in result["event_assignments"] for c in a["candidate_controls"]]
    assert any(c["type"] == "same_field_other_record" and c["eligible_count"] > 0 for c in controls)


def test_foreign_capture_input_hash_cannot_be_relabelled_in_encoder():
    args = fixture()
    args[-1]["encoder"]["source_input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="encoder input differs"):
        propose_event(*args)
