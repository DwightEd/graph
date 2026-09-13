"""Operational certificates over a frozen event, with signed effects and failures."""

from route_graph.audit_pools import select_controls


def certificate(full, half, controls, repeat, sham):
    delta = full["delta"]
    specificity = abs(delta) - max(
        (abs(c["delta"]) for c in controls), default=float("inf")
    )
    checks = {
        "effect": abs(delta) >= 0.5,
        "two_controls": len(controls) == 2,
        "selectivity": specificity >= 0.25,
        "half_direction": delta * half["delta"] > 0
        and abs(half["delta"]) <= 2 * abs(delta),
        "repeat": abs(delta - repeat["delta"]) <= 1e-5,
        "sham": sham,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "delta": delta,
        "specificity": specificity if controls else None,
        "half_delta": half["delta"],
        "control_deltas": [c["delta"] for c in controls],
        "repeat_delta": repeat["delta"],
        "direction": "supports_original"
        if delta > 0
        else ("opposes_original" if delta < 0 else "zero"),
    }


def validate_group(oracle, item, labels):
    controls = select_controls(item, labels)
    if len(controls) != 2:
        return {
            "status": "missing_matched_unrelated_controls",
            "passed": False,
            "id": item["id"],
            "screen_delta": item["delta"],
            "selected_control_ids": [c["id"] for c in controls],
        }
    return validate_fixed_group(
        oracle, item["group"], [c["group"] for c in controls], item["id"]
    )


def validate_fixed_group(oracle, group, controls, group_id):
    sham = oracle.group(group, 1)
    full = oracle.group(group)
    half = oracle.group(group, 0.5)
    control_results = [oracle.group(c) for c in controls]
    repeat = oracle.group(group, force=True)
    # Global full-shape sham is independently checked by the phase runner.
    result = certificate(
        full, half, control_results, repeat, sham["sham_exact"] is True
    )
    result.update(
        id=group_id,
        status="position_effect" if result["passed"] else "effect_not_certified",
        measurements=[r["key"] for r in [sham, full, half, *control_results, repeat]],
        group=group,
        controls=controls,
    )
    complement = sorted(set(oracle.contrast.shared_queries) - set(group["queries"]))
    if complement:
        try:
            kept = oracle.group({**group, "queries": complement})
            result["keep"] = {
                "delta": kept["delta"],
                "measurement": kept["key"],
                "preserved_within_U": abs(kept["delta"]) <= 0.25,
                "scope": "same_keys_all_shared_queries_same_layers",
            }
        except ValueError as error:
            result["keep"] = {"invalid": str(error)}
    else:
        result["keep"] = {
            "delta": 0,
            "preserved_within_U": True,
            "scope": "U_equals_G_no_additional_sufficiency_evidence",
        }
    return result


def validate_origin(oracle, group, origin, controls, position):
    if not position.get("passed"):
        return {"status": "position_effect_not_certified", "passed": False}
    if len(controls) != 2:
        return {
            "status": "missing_matched_origin_controls",
            "passed": False,
            "origin": origin,
            "controls": controls,
        }
    full = oracle.mediated(group, origin, 0)
    half = oracle.mediated(group, origin, 0.5)
    sham = oracle.mediated(group, origin, 1)
    control_results = [oracle.mediated(group, c["keys"], 0) for c in controls]
    repeat = oracle.mediated(group, origin, 0)
    result = certificate(
        full,
        half,
        control_results,
        repeat,
        sham["sham_exact"] is True and sham["delta"] == 0,
    )
    result["position_direction_compatible"] = full["delta"] * position["delta"] > 0
    result["passed"] &= result["position_direction_compatible"]
    result.update(
        status="origin_mediated_effect" if result["passed"] else "origin_mismatch",
        origin=origin,
        controls=controls,
        origin_nonexclusive=True,
        measurements=[r["key"] for r in [full, half, sham, *control_results, repeat]],
        scope="input_embedding_scaling_through_selected_V_messages_only",
    )
    return result


def validate_layer_bands(oracle, position):
    group, controls = position["group"], position["controls"]
    count = len(oracle.model.model.layers)
    results = []
    for name, left, right in (
        ("early", 0, count * 10 // 32),
        ("middle", count * 10 // 32, count * 22 // 32),
        ("late", count * 22 // 32, count),
    ):
        layers = list(range(left, right))
        if not layers:
            continue
        if oracle.calls + 14 > oracle.budget:
            results.append(
                {
                    "id": position["id"] + ":" + name,
                    "passed": False,
                    "status": "budget_unresolved",
                }
            )
            continue
        result = validate_fixed_group(
            oracle,
            {**group, "layers": layers},
            [{**c, "layers": layers} for c in controls],
            position["id"] + ":" + name,
        )
        results.append(result)
    return {
        "bands": results,
        "status": "has_certified_layer_band"
        if any(r["passed"] for r in results)
        else (
            "layer_scope_unresolved"
            if any(r["status"] == "budget_unresolved" for r in results)
            else "distributed_all_layer_effect_in_tested_family"
        ),
    }
