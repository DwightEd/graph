import copy

import pytest

from route_graph.soft_graph_energy import infer_claim, positive_dependence


def native(delta=1):
    return {"delta": delta, "control_deltas": [.1, .2], "controls_valid": True,
                "sham_exact": True, "prefix_exact": True, "repeat_valid": True,
                "selected_id": "selected", "control_ids": ["c1", "c2"], "origin_donor_sham_exact": True,
                "origin_recipient_sham_exact": True, "origin_artifacts_verified": True,
                "origin_measurement_ids": ["m", "m1", "m2"],
                "measurement_scope": "raw_origin_mediated_target_dependence"}


def claim(q=None):
    return {"claim_id": "a", "span": [0, 5], "q_global": q or {"S": .6, "C": .2, "N": .1, "U": .1},
                "source_terms": [{"key": ["r", "s", "e"], "pi": .7, "pool_quality": 1,
                                   "q_local": {"S": .05, "C": .9, "U": .05}, "native": native()}]}


def test_unknown_is_exactly_neutral_and_local_unknown_cannot_invent_absence():
    c = claim({"S": 0., "C": 0., "N": 0., "U": 1.})
    c["source_terms"][0]["q_local"] = {"S": 0., "C": 0., "U": 1.}
    out = infer_claim(c, {})
    assert all(v["risk_score"] == .5 for v in out["variants"].values())


def test_positive_controlled_conflict_changes_risk_and_negative_does_not():
    c = claim()
    out = infer_claim(c, {})["variants"]
    assert out["full_graph"]["risk_score"] > out["qwen_fullsource_no_graph"]["risk_score"]
    c["source_terms"][0]["native"] = native(-4)
    out = infer_claim(c, {})["variants"]
    assert out["full_graph"] == out["qwen_fullsource_no_graph"]
    assert out["qwen_matcher_no_native"]["risk_score"] > out["full_graph"]["risk_score"]


@pytest.mark.parametrize("field,value", [("controls_valid", False), ("repeat_valid", False),
                                          ("sham_exact", False), ("measurement_scope", "position_only")])
def test_uncontrolled_edge_never_contributes(field, value):
    n = native(); n[field] = value
    assert positive_dependence(n) == 0


def test_duplicate_source_edges_rejected_instead_of_amplifying_beam_paths():
    c = claim(); c["source_terms"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        infer_claim(c, {})


def test_same_role_cannot_manufacture_marginal_mass_or_duplicate_controls():
    c = claim(); other = copy.deepcopy(c["source_terms"][0]); other["key"][1] = "s2"
    c["source_terms"].append(other)
    with pytest.raises(ValueError, match="mass"):
        infer_claim(c, {})
    n = native(); n["control_ids"] = ["c1", "c1"]
    assert positive_dependence(n) == 0


@pytest.mark.parametrize("relation", ["correction", "quote", "new_topic", "unknown"])
def test_history_nonreuse_blocks_without_flipping_risk(relation):
    prior = infer_claim(claim(), {})
    c = claim(); c.update(claim_id="b", span=[7, 11], source_terms=[])
    c["history_terms"] = [{"previous_claim_id": "a", "native": native(), "origin_scope": "mapped_previous_event",
                               "q_relation": {k: float(k == relation) for k in ["reuse", "correction", "quote", "new_topic", "unknown"]}}]
    out = infer_claim(c, {"a": prior})
    assert out["variants"]["full_graph"] == out["variants"]["graph_no_history"]
    c["history_terms"][0]["previous_claim_id"] = "future"
    with pytest.raises(ValueError, match="earlier"):
        infer_claim(c, {"a": prior})


def test_confidence_zero_previous_claim_cannot_transmit_large_raw_energy():
    prior = infer_claim(claim({"S": 0., "C": 0., "N": 0., "U": 1.}), {})
    prior = copy.deepcopy(prior)
    prior["variants"]["full_graph"].update(Lraw=10., confidence=0.)
    c = claim(); c.update(claim_id="b", span=[7, 11], source_terms=[])
    c["history_terms"] = [{"previous_claim_id": "a", "native": native(), "origin_scope": "mapped_previous_slot",
                               "q_relation": {"reuse": 1., "correction": 0., "quote": 0., "new_topic": 0., "unknown": 0.}}]
    assert infer_claim(c, {"a": prior})["variants"]["full_graph"]["Lhistory"] == 0


def test_uniform_history_relation_is_unknown_even_if_reuse_is_first_label():
    previous = infer_claim(claim(), {})
    c = claim(); c.update(claim_id="b", span=[7, 11], source_terms=[])
    c["history_terms"] = [{"previous_claim_id": "a", "native": native(), "origin_scope": "mapped_previous_slot",
                               "q_relation": {"reuse": .2, "correction": .2, "quote": .2, "new_topic": .2, "unknown": .2}}]
    result = infer_claim(c, {"a": previous})
    assert result["history_terms"][0]["edge_label"] == "unknown"
    assert result["variants"]["full_graph"]["Lhistory"] == 0
