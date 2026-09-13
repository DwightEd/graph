"""Representation/inference checks, not research datasets or truth evaluation."""

import copy

import pytest

from next_iteration.fact_scope import compile_graph, incidence_windows, solve_owners


def graph():
    text = "Alice Beka Cora at 8. Later 9."
    return compile_graph(text, "fixture", [
        {"id": "x", "mentions": [[0, 5]]}, {"id": "y", "mentions": [[6, 10]]},
        {"id": "z", "mentions": [[11, 15]]}], [
        {"id": "f", "members": [[0, 21]], "target": [[19, 20]], "variables": ["x", "y", "z"]},
        {"id": "other", "members": [[0, 5], [22, 30]], "target": [[28, 29]], "variables": ["x"]}])


def table(tid, variables, allowed, forbidden=(), **kwargs):
    return {"id": tid, "variables": variables, "allowed": [list(x) for x in allowed],
        "forbidden": [list(x) for x in forbidden], "status": "tested", "scope_fact_ids": ["f"],
        "read_spans": [[0, 15]], "value_dependencies": [], **kwargs}


def solve(tables=(), **kwargs):
    return solve_owners(graph(), {"x": ["A", "B"], "y": ["C", "D"], "z": ["E"]}, list(tables),
        target_fact="f", target_variable="x", **kwargs)


def test_overlapping_noncontiguous_members_do_not_merge_truth():
    result = incidence_windows(graph())
    assert result["windows"][1]["members"] == [[0, 5], [22, 30]]
    assert result["windows"][1]["envelope"] == [0, 30]
    assert result["relations"][0]["shared_variables"] == ["x"]
    assert not result["relations"][0]["licenses_error_propagation"]
    assert not result["risk_scores_emitted"]


def test_joint_relations_resolve_binding_not_independent_top_one():
    result = solve([
        table("xy", ["x", "y"], [["A", "C"], ["B", "D"]], [["A", "D"], ["B", "C"]]),
        table("y", ["y"], [["D"]], [["C"]])])
    assert result["status"] == "unique_under_supplied_constraints"
    assert result["possible_owners_found"] == ["B"]
    assert result["possible_values_found"]["y"] == ["D"]
    assert not result["owner_semantically_certified"]


def test_unlisted_tuple_remains_unknown_not_forbidden():
    result = solve([table("x", ["x"], [["A"]])])
    assert result["status"] == "unverified_constraints_remain"
    assert result["possible_owners_found"] == ["A", "B"]
    assert result["unknown_relation_solution_count"] == 1


def test_unverified_table_does_not_rule_out_candidate():
    result = solve([table("x", ["x"], [["A"]], [["B"]], status="unverified")])
    assert result["possible_owners_found"] == ["A", "B"]
    assert result["fully_allowed_solution_count"] == 0


@pytest.mark.parametrize("changes", [{"read_spans": [[19, 20]]}, {"value_dependencies": ["f"]}])
def test_direct_or_indirect_target_value_cannot_choose_owner(changes):
    result = solve([table("leaking", ["x"], [["A"]], [["B"]], **changes)])
    assert result["possible_owners_found"] == ["A", "B"]
    assert result["excluded_tables"] == [{"id": "leaking", "reason": "observed_current_target_value"}]


def test_separate_cooccurring_fact_does_not_constrain_current_binding():
    result = solve([table("other", ["x"], [["A"]], [["B"]], scope_fact_ids=["other"])])
    assert result["possible_owners_found"] == ["A", "B"]
    assert result["excluded_tables"][0]["reason"] == "outside_focal_conditioning_scope"


def test_unsealed_composite_scope_cannot_constrain_current_factor():
    result = solve([table("wide", ["x"], [["A"]], [["B"]], scope_fact_ids=["f", "other"])])
    assert result["possible_owners_found"] == ["A", "B"]
    assert result["provenance_status"] == "supplied_assumptions_not_reader_or_symbolic_certificates"


def test_explicit_contradictory_tables_produce_conflict():
    result = solve([table("one", ["x"], [["A"]], [["B"]]), table("two", ["x"], [["B"]], [["A"]])])
    assert result["status"] == "constraint_conflict"
    assert result["search_exhausted"]
    assert result["solution_count"] == 0


def test_partial_search_cannot_promote_first_solution_to_unique_owner():
    result = solve(max_visits=1)
    assert result["status"] == "search_incomplete"
    assert result["possible_owners_found"] == ["A"]
    assert result["candidate_domains"]["x"] == ["A", "B"]
    assert not result["search_exhausted"]


def test_exact_budget_at_end_is_exhaustive():
    result = solve(max_visits=2)
    assert result["status"] == "ambiguous_under_supplied_constraints"
    assert result["search_exhausted"]


def test_different_variables_may_reference_same_source_entity():
    result = solve_owners(graph(), {"x": ["A"], "y": ["A"], "z": ["E"]},
        [table("alias", ["x", "y"], [["A", "A"]])], target_fact="f", target_variable="x")
    assert result["solution_count"] == 1
    assert result["possible_values_found"]["x"] == result["possible_values_found"]["y"] == ["A"]


def test_empty_candidate_domain_is_not_evidence_of_factual_contradiction():
    result = solve_owners(graph(), {"x": [], "y": ["A"], "z": ["E"]}, [],
        target_fact="f", target_variable="x")
    assert result["status"] == "candidate_domain_empty"


def test_same_tuple_cannot_be_both_allowed_and_forbidden():
    with pytest.raises(ValueError, match="both allowed and forbidden"):
        solve([table("bad", ["x"], [["A"]], [["A"]])])


def test_mutated_original_text_cannot_keep_old_graph_seal():
    original = copy.deepcopy(graph())
    original["text"] = original["text"].replace("8", "7")
    with pytest.raises(ValueError):
        incidence_windows(original)


def test_target_outside_fact_member_mask_is_rejected():
    with pytest.raises(ValueError, match="target incidence"):
        compile_graph("Alice 8", "fixture", [{"id": "x", "mentions": [[0, 5]]}],
            [{"id": "f", "members": [[0, 5]], "target": [[6, 7]], "variables": ["x"]}])
