"""Same-event cross-role transport and genuine evidence x MLP interactions."""

from route_graph.audit_certificates import certificate
from route_graph.audit_pools import ratio_matches


def paired_role(oracle, source, history, source_origin, history_origin):
    if not (
        source.get("passed")
        and history.get("passed")
        and source["delta"] <= -0.5
        and history["delta"] >= 0.5
    ):
        return {"status": "required_opposing_positions_not_certified", "passed": False}
    sg, hg = source["group"], history["group"]
    queries = sorted(set(sg["queries"]) & set(hg["queries"]))
    if not queries or sg["layers"] != hg["layers"]:
        return {"status": "no_common_query_layer_scope", "passed": False}
    group = {
        "kind": "route",
        "role": "history",
        "keys": hg["keys"],
        "destination": sg["keys"],
        "domain": sorted(set(hg["keys"]) | set(sg["domain"])),
        "queries": queries,
        "layers": sg["layers"],
    }
    control_groups = [{**group, "destination": c["keys"]} for c in source["controls"]]
    target_mass = oracle.mass({**sg, "queries": queries})
    if not all(
        ratio_matches(
            oracle.mass({**sg, "queries": queries}, c["destination"]), target_mass
        )
        for c in control_groups
    ):
        return {"status": "controls_not_matched_in_common_scope", "passed": False}
    sham = oracle.group(group, 1)
    full, half = oracle.group(group), oracle.group(group, 0.5)
    controls = [oracle.group(c) for c in control_groups]
    repeat = oracle.group(group, force=True)
    result = certificate(full, half, controls, repeat, sham["sham_exact"] is True)
    # This hypothesis has a specified positive direction; preserve negative data.
    result["passed"] &= full["delta"] >= 0.5
    exact_origin_scope = (
        source_origin.get("passed")
        and history_origin.get("passed")
        and sg["queries"] == hg["queries"] == queries
    )
    result.update(
        status=(
            "selective_cross_role_effect"
            if exact_origin_scope
            else "position_selective_cross_role_effect"
        )
        if result["passed"]
        else "cross_role_effect_not_certified",
        origin_scope_matched=bool(exact_origin_scope),
        group=group,
        controls=control_groups,
        measurements=[r["key"] for r in [sham, full, half, *controls, repeat]],
        estimand="same_H_same_outgoing_mass_only_destination_changes",
    )
    return result


def partition(measurement, masks):
    slot = sum(
        p
        for p, selected in zip(measurement["token_logps"][0], masks[0], strict=True)
        if selected
    )
    slot -= sum(
        p
        for p, selected in zip(measurement["token_logps"][1], masks[1], strict=True)
        if selected
    )
    return {"slot": slot, "context": measurement["F"] - slot}


def mlp_interaction(oracle, evidence, mlp, origin, masks):
    if not (evidence.get("passed") and evidence["delta"] <= -0.5 and mlp is not None):
        return {"status": "opposing_evidence_or_native_MLP_missing", "passed": False}
    eg, mg = evidence["group"], mlp["group"]
    sham = oracle.group(eg, 1, extra=oracle.gates(mg, 1))
    scope_contained = set(mg["queries"]).issubset(eg["queries"]) and set(
        mg["layers"]
    ).issubset(eg["layers"])
    base, no_e = oracle.base, oracle.group(eg)
    no_m = oracle.group(mg)
    both = oracle.group(eg, extra=oracle.gates(mg, 0))
    omega = (base["F"] - no_m["F"]) - (no_e["F"] - both["F"])
    control_omegas, control_records = [], []
    for cg in evidence["controls"]:
        no_c = oracle.group(cg)
        joint = oracle.group(cg, extra=oracle.gates(mg, 0))
        control_omegas.append((base["F"] - no_m["F"]) - (no_c["F"] - joint["F"]))
        control_records.extend([no_c["key"], joint["key"]])
    half_m = oracle.group(mg, 0.5)
    half_both = oracle.group(eg, extra=oracle.gates(mg, 0.5))
    half_omega = (base["F"] - half_m["F"]) - (no_e["F"] - half_both["F"])
    repeated = oracle.group(eg, force=True, extra=oracle.gates(mg, 0))
    parts = [partition(r, masks) for r in [base, no_m, no_e, both]]
    decomposition = {
        key: {
            "delta_M": parts[0][key] - parts[1][key],
            "omega": parts[0][key] - parts[1][key] - parts[2][key] + parts[3][key],
        }
        for key in ("slot", "context")
    }
    broad = any(
        abs(decomposition["context"][key]) > abs(decomposition["slot"][key])
        for key in ("delta_M", "omega")
    )
    checks = {
        "evidence_opposes": evidence["delta"] <= -0.5,
        "MLP_supports": no_m["delta"] >= 0.5,
        "interaction": omega >= 0.5,
        "two_controls": len(control_omegas) == 2,
        "selectivity": omega - max(map(abs, control_omegas), default=float("inf"))
        >= 0.25,
        "half": omega * half_omega > 0 and abs(half_omega) <= 2 * abs(omega),
        "repeat": abs(both["F"] - repeated["F"]) <= 1e-5,
        "sham": sham["sham_exact"] is True,
    }
    passed = all(checks.values())
    status = "interaction_not_certified"
    if checks["evidence_opposes"] and checks["MLP_supports"]:
        status = "opposing_paths"
    if passed:
        status = (
            "evidence_MLP_antagonism"
            if origin.get("passed")
            else "position_message_MLP_interaction"
        )
        if broad or not scope_contained:
            status = "broad_MLP_event_interaction"
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "delta_E": evidence["delta"],
        "delta_M": no_m["delta"],
        "omega": omega,
        "half_omega": half_omega,
        "control_omegas": control_omegas,
        "decomposition": decomposition,
        "joint_sham_measurement": sham["key"],
        "evidence_group": eg,
        "MLP_group": mg,
        "mlp_scope_relative_to_evidence": "contained"
        if scope_contained
        else "broader_or_different",
        "four_states": {
            "F11": base["F"],
            "F01": no_e["F"],
            "F10": no_m["F"],
            "F00": both["F"],
        },
        "measurements": [
            r["key"] for r in [base, no_e, no_m, both, half_m, half_both, repeated]
        ]
        + control_records,
        "all_routing_correct_claim": False,
    }
