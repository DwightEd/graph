"""Exact, overlapping fact incidence and conditional owner-set inference.

This module does not extract facts, judge entailment or issue native certificates.
It joins explicitly supplied relation tables, without pretending their provenance
or semantic correctness follows from a valid graph. Tables are assumptions.
"""

import itertools

from next_iteration.surface_graph import _check, _sealed


def _spans(text, spans):
    if not isinstance(spans, list) or not spans:
        raise ValueError("require nonempty original character spans")
    result = []
    for pair in spans:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair)):
            raise ValueError("invalid span pair")
        a, b = pair
        if not 0 <= a < b <= len(text) or not text[a:b].strip():
            raise ValueError("span outside original text or blank")
        result.append([a, b])
    if result != sorted(result) or any(b > c for (_, b), (c, _) in itertools.pairwise(result)):
        raise ValueError("spans must be sorted and nonoverlapping within one member set")
    return result


def _overlap(left, right):
    return any(a < d and c < b for a, b in left for c, d in right)


def _covered(spans, members):
    return all(any(c <= a < b <= d for c, d in members) for a, b in spans)


def compile_graph(text, sample_id, variables, facts):
    """Compile proposed fact masks; exact spans are integrity, not semantics.

    Input variable: {id, mentions:[[a,b],...]}.
    Input fact: {id, members, target, variables:[id,...]}.
    Shared members and variables never imply error propagation or equal truth.
    """
    if not isinstance(text, str) or not isinstance(variables, list) or not isinstance(facts, list):
        raise TypeError("invalid original text or graph lists")
    vs, fs = {}, {}
    for variable in variables:
        vid = variable["id"]
        if not isinstance(vid, str) or not vid or vid in vs:
            raise ValueError("duplicate or invalid variable identity")
        vs[vid] = {"id": vid, "mentions": _spans(text, variable["mentions"])}
    for fact in facts:
        fid = fact["id"]
        if not isinstance(fid, str) or not fid or fid in fs:
            raise ValueError("duplicate or invalid fact identity")
        members, target = _spans(text, fact["members"]), _spans(text, fact["target"])
        vids = fact["variables"]
        if (not isinstance(vids, list) or len(vids) != len(set(vids))
                or not set(vids) <= set(vs) or not _covered(target, members)):
            raise ValueError("fact variables or target incidence invalid")
        if any(not _overlap(vs[v]["mentions"], members) for v in vids):
            raise ValueError("fact variable lacks an original member mention")
        fs[fid] = {"id": fid, "members": members, "target": target, "variables": vids,
            "envelope": [members[0][0], members[-1][1]],
            "member_texts": [text[a:b] for a, b in members], "semantic_status": "proposed"}
    edges = [{"kind": "variable_in_fact", "variable": v, "fact": f["id"]}
             for f in fs.values() for v in f["variables"]]
    return _sealed({"schema": "fact-incidence-graph@1", "sample_id": str(sample_id), "text": text,
        "variables": list(vs.values()), "facts": list(fs.values()), "edges": edges,
        "labels_used": False, "semantic_validation": "not_performed",
        "native_certificate_count": 0})


def solve_owners(graph, domains, tables, *, target_fact, target_variable, max_visits=100000):
    """Enumerate owner assignments conditional on supplied candidate/table sets.

    Table: {id, variables, allowed:[tuple,...], forbidden:[tuple,...],
            status:tested|unverified, scope_fact_ids:[fact_id,...],
            read_spans:[[a,b],...], value_dependencies:[fact_id,...]}.
    Every unlisted tuple is unknown; an allowed-list is NOT a closed world.
    Exclude tables that observed the target's value, including indirect copies.
    Only the target variable's connected constraint component is searched.
    Table provenance/semantics and completeness of domains are caller obligations;
    this output never licenses a factual or causal certificate by itself.
    """
    _check(graph)
    if type(max_visits) is not int or max_visits < 1:
        raise ValueError("positive visit budget required")
    facts = {f["id"]: f for f in graph["facts"]}
    variables = {v["id"] for v in graph["variables"]}
    if (target_fact not in facts or target_variable not in facts[target_fact]["variables"]
            or set(domains) != variables):
        raise ValueError("target or candidate variable identity differs")
    candidates = {}
    for v, values in domains.items():
        if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
                or len(values) != len(set(values))):
            raise ValueError("candidate domains require distinct nonempty occurrence IDs")
        candidates[v] = sorted(values)
    ids, usable, unresolved, excluded = set(), [], [], []
    for table in tables:
        tid, tv = table["id"], table["variables"]
        if (not isinstance(tid, str) or not tid or tid in ids or not isinstance(tv, list)
                or not tv or len(tv) != len(set(tv)) or not set(tv) <= variables):
            raise ValueError("invalid relation table identity or variable scope")
        ids.add(tid)
        dependencies = table["value_dependencies"]
        if not isinstance(dependencies, list) or not set(dependencies) <= set(facts):
            raise ValueError("unknown value dependency")
        read_spans = _spans(graph["text"], table["read_spans"])
        if table["status"] not in ("tested", "unverified"):
            raise ValueError("table status must distinguish unverified constraints")
        scope = table["scope_fact_ids"]
        if not isinstance(scope, list) or not scope or not set(scope) <= set(facts):
            raise ValueError("invalid conditioning factor scope")
        for label in ("allowed", "forbidden"):
            rows = table[label]
            if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != len(tv)
                    or any(val not in candidates[v] for v, val in zip(tv, row, strict=True)) for row in rows):
                raise ValueError("relation tuple differs from candidate domains")
        if {tuple(x) for x in table["allowed"]} & {tuple(x) for x in table["forbidden"]}:
            raise ValueError("same relation tuple both allowed and forbidden")
        if scope != [target_fact]:
            excluded.append({"id": tid, "reason": "outside_focal_conditioning_scope"})
        elif target_fact in dependencies or _overlap(read_spans, facts[target_fact]["target"]):
            excluded.append({"id": tid, "reason": "observed_current_target_value"})
        elif table["status"] == "unverified":
            unresolved.append(table)
        else:
            usable.append(table)
    component = {target_variable}
    while True:
        extended = component | {v for t in usable + unresolved
                                 if component.intersection(t["variables"]) for v in t["variables"]}
        if extended == component:
            break
        component = extended
    usable = [t for t in usable if component.intersection(t["variables"])]
    unresolved = [t for t in unresolved if component.intersection(t["variables"])]
    missing = sorted(v for v in component if not candidates[v])
    ordered = sorted(component, key=lambda v: (len(candidates[v]), v))
    possible = {v: set() for v in component}
    visits, solutions, supported_solutions, exhausted = 0, 0, 0, True

    def consistent(assignment):
        # Unknown tuples remain possible. Only an explicit forbidden tuple prunes.
        return all(not all(v in assignment for v in t["variables"])
                   or [assignment[v] for v in t["variables"]] not in t["forbidden"] for t in usable)

    def supported(assignment):
        return not unresolved and all([assignment[v] for v in t["variables"]] in t["allowed"] for t in usable)

    # The cap counts every attempted variable assignment, including pruned nodes.
    # Thus a huge infeasible Cartesian product cannot hide behind solution count.
    def search(index, assignment):
        nonlocal visits, solutions, supported_solutions, exhausted
        if index == len(ordered):
            solutions += 1
            supported_solutions += int(supported(assignment))
            for v, value in assignment.items():
                possible[v].add(value)
            return
        v = ordered[index]
        for value in candidates[v]:
            if visits == max_visits:
                exhausted = False
                return
            visits += 1
            assignment[v] = value
            if consistent(assignment):
                search(index + 1, assignment)
            del assignment[v]
            if not exhausted:
                return

    if not missing:
        search(0, {})
    owners = sorted(possible[target_variable])
    if missing:
        status = "candidate_domain_empty"
    elif not exhausted:
        status = "search_incomplete"
    elif not solutions:
        status = "constraint_conflict"
    elif unresolved or supported_solutions != solutions:
        status = "unverified_constraints_remain"
    elif len(owners) == 1:
        status = "unique_under_supplied_constraints"
    else:
        status = "ambiguous_under_supplied_constraints"
    return _sealed({"schema": "conditional-owner-sets@1", "graph_sha256": graph["sha256"],
        "target_fact": target_fact, "target_variable": target_variable, "status": status,
        "component_variables": sorted(component), "possible_owners_found": owners,
        "possible_values_found": {v: sorted(values) for v, values in sorted(possible.items())},
        "candidate_domains": {v: candidates[v] for v in sorted(component)},
        "solution_count": solutions, "fully_allowed_solution_count": supported_solutions,
        "unknown_relation_solution_count": solutions - supported_solutions,
        "visited_assignments": visits, "search_exhausted": exhausted,
        "empty_domains": missing, "used_tables": sorted(t["id"] for t in usable),
        "unverified_tables": sorted(t["id"] for t in unresolved), "excluded_tables": excluded,
        "assumptions_sha256": _sealed({"domains": domains, "tables": tables})["sha256"],
        "semantic_validity": "not_established_by_constraint_solver", "source_search_complete": "not_established",
        "provenance_status": "supplied_assumptions_not_reader_or_symbolic_certificates",
        "labels_used": False, "owner_semantically_certified": False, "native_certificate_count": 0})


def incidence_windows(graph):
    """Return exact overlapping masks and shared variables, never risk spread."""
    _check(graph)
    windows = [{k: f[k] for k in ("id", "members", "target", "envelope", "variables")}
               for f in graph["facts"]]
    relations = []
    for a, b in itertools.combinations(windows, 2):
        shared = sorted(set(a["variables"]).intersection(b["variables"]))
        if shared:
            relations.append({"left": a["id"], "right": b["id"], "shared_variables": shared,
                "kind": "shared_reference_only", "licenses_error_propagation": False})
    return _sealed({"graph_sha256": graph["sha256"], "windows": windows, "relations": relations,
        "native_context_truncated": False, "risk_scores_emitted": False})
