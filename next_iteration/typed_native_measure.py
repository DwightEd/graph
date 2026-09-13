"""Bounded query/layer localization of an exactly supplied typed source path.

This is evidence-conditioned localization on a frozen observer, not discovery
of an unknown owner or a unique original-generator lookback node.
"""

import heapq

import numpy as np
import torch

from route_graph.audit_alignment import span_keys
from route_graph.audit_certificates import certificate
from route_graph.audit_interactions import partition
from route_graph.audit_pools import ratio_matches, visibility
from route_graph.frozen_reader import digest

PROTOCOL = {"schema": "typed-native-query-layer@1", "budget_actual_forwards": 128,
    "search_caps": {"layer": 16, "query": 40, "union": 8}, "effect_threshold": 0.5,
    "selectivity": 0.25, "query_layer_seeds": 2, "position_controls": 2,
    "origin_controls": 2, "selection": "smallest tested negative-effect group, else strongest absolute raw group",
    "control_census": "all source leaves and all disjoint pairs of complete schema-unrelated leaves, pre-effect; pair is control-only",
    "search_scope": "all shared prefix queries and all layers; typed source A fixed before effects",
    "claim_scope": "conditional typed source path in observer replay; no unique lookback, routing or aggregation claim"}


def _id(group):
    return digest(group)[:24]


def _size(group):
    return len(group["queries"]) * len(group["layers"])


def root_group(oracle, keys):
    if (not keys or len(set(keys)) != len(keys)
            or any(type(k) is not int or not 0 <= k < oracle.contrast.prompt_length for k in keys)):
        raise ValueError("typed A keys must be unique original source-prompt positions")
    return {"kind": "content", "role": "source", "keys": list(keys), "domain": list(keys),
        "queries": list(oracle.contrast.shared_queries), "layers": list(range(len(oracle.model.model.layers)))}


def search_queries_layers(oracle, keys):
    """Reserve query work before layer effects, retaining every measured parent."""
    root = root_group(oracle, keys)
    measured, pending_records, known = [], [], {}
    start = oracle.calls

    def measure(group, parent, axis):
        gid = _id(group)
        if gid in known:
            return known[gid]
        before = oracle.calls
        result = oracle.group(group)
        item = {"id": gid, "group": group, "parent_id": parent, "axis": axis,
            "delta": result["delta"], "measurement": result["key"], "actual_forward_calls": oracle.calls - before}
        measured.append(item)
        known[gid] = item
        return item

    def tree(seed, dimension, cap):
        phase_start, queue, enqueued = oracle.calls, [], set()

        def enqueue(item):
            values = item["group"][dimension]
            if len(values) > 1 and item["id"] not in enqueued:
                enqueued.add(item["id"])
                heapq.heappush(queue, (-abs(item["delta"]), item["id"], item))

        enqueue(seed)
        while queue and oracle.calls - phase_start + 4 <= cap:
            _, _, item = heapq.heappop(queue)
            positions = item["group"][dimension]
            mid = len(positions) // 2
            for half in (positions[:mid], positions[mid:]):
                child = measure({**item["group"], dimension: half}, item["id"], dimension)
                enqueue(child)
        for _, _, item in queue:
            positions = item["group"][dimension]
            mid = len(positions) // 2
            for half in (positions[:mid], positions[mid:]):
                group = {**item["group"], dimension: half}
                if _id(group) not in known:
                    pending_records.append({"id": _id(group), "group": group, "parent_id": item["id"], "axis": dimension})

    root_item = measure(root, None, "root")
    tree(root_item, "layers", PROTOCOL["search_caps"]["layer"] - (oracle.calls - start))
    layers = sorted((m for m in measured if m["id"] != root_item["id"]),
        key=lambda m: (-abs(m["delta"]), len(m["group"]["layers"]), m["id"]))
    seeds = [root_item] + layers[:1]
    query_cap = PROTOCOL["search_caps"]["query"] // PROTOCOL["query_layer_seeds"]
    for seed in seeds:
        tree(seed, "queries", query_cap)
    # Preserve cross-span unions when separately searched children are disjoint
    # and leave a real gap. The actual union is measured, never score-added.
    unions = {}
    for i, left in enumerate(measured):
        for right in measured[i + 1:]:
            a, b = left["group"], right["group"]
            if a["layers"] != b["layers"] or set(a["queries"]) & set(b["queries"]):
                continue
            q = sorted(set(a["queries"]) | set(b["queries"]))
            if len(q) == q[-1] - q[0] + 1:
                continue
            group = {**a, "queries": q}
            gid = _id(group)
            if gid not in known:
                unions[gid] = (-(abs(left["delta"]) + abs(right["delta"])), group, [left["id"], right["id"]])
    union_start = oracle.calls
    for gid, (_, group, parents) in sorted(unions.items(), key=lambda kv: (kv[1][0], kv[0])):
        if oracle.calls - union_start + 2 <= PROTOCOL["search_caps"]["union"]:
            measure(group, parents, "nonadjacent_query_union")
        else:
            pending_records.append({"id": gid, "group": group, "parent_id": parents, "axis": "nonadjacent_query_union"})
    candidates = [m for m in measured if m["delta"] <= -PROTOCOL["effect_threshold"]]
    selected = min(candidates, key=lambda m: (_size(m["group"]), len(m["group"]["queries"]), m["id"])) if candidates else min(
        measured, key=lambda m: (-abs(m["delta"]), _size(m["group"]), m["id"]))
    pending = {p["id"]: p for p in pending_records if p["id"] not in known}
    return {"root": root_item["id"], "measured": measured, "selected": selected,
        "selection_has_A_directed_effect": bool(candidates), "pending": list(pending.values()),
        "actual_forward_calls": oracle.calls - start, "searched_joint_groups": sum(len(m["group"]["layers"]) > 1 for m in measured),
        "searched_single_layer_groups": sum(len(m["group"]["layers"]) == 1 for m in measured),
        "nonadjacent_query_unions": sum(m["axis"] == "nonadjacent_query_union" for m in measured),
        "unsearched_layer_children": sum(p["axis"] == "layers" for p in pending.values()),
        "unsearched_query_children": sum(p["axis"] == "queries" for p in pending.values()),
        "unsearched_domain": "pending frontier plus all unenumerated subsets; not an exhaustive domain count",
        "localization_scope": "bounded evidence-conditioned effect search; no unique lookback claim"}


def source_control_census(prepared, alignment, prefix_length):
    """All non-target source leaves retained, graded by the hours algebra.

    Counts/masses/norms are used later for structural matching. This census
    contains no intervention effect and cannot substitute a semantic B owner.
    """
    records = []
    for occurrence in prepared["schedule"]["source_graph"]["occurrences"]:
        path = occurrence["field_path"]
        keys = span_keys(alignment, "source", occurrence["span"])
        keys = [k for k in keys if k < prefix_length]
        if not keys:
            continue
        # Full hours fields are related to both endpoints. Business identity
        # determines the record, and WiFi can be a preserved event conjunct.
        excluded = path[0] in {"hours", "name"} or path[:2] == ["attributes", "WiFi"]
        grade = "schema_related" if excluded else "schema_unrelated_to_hours_and_wifi"
        if occurrence["epistemic_status"] != "observed_literal":
            grade = "unknown_source_literal"
        records.append({"id": occurrence["occurrence_id"], "field_path": path, "keys": keys,
            "span": occurrence["span"], "grade": grade,
            "rationale": "typed algebra reads root identity, all hours and optional WiFi only; no general semantic unrelatedness claim"})
    leaves = [r for r in records if r["grade"] == "schema_unrelated_to_hours_and_wifi"]
    for index, left in enumerate(leaves):
        for right in leaves[index + 1:]:
            if (set(left["keys"]) & set(right["keys"])
                    or max(left["span"][0], right["span"][0]) < min(left["span"][1], right["span"][1])):
                continue
            ordered = sorted((left, right), key=lambda r: r["id"])
            members = [r["id"] for r in ordered]
            records.append({"id": "control_pair:" + digest(members)[:24],
                "field_paths": [r["field_path"] for r in ordered],
                "keys": sorted(set(left["keys"]) | set(right["keys"])),
                "spans": [r["span"] for r in ordered], "member_ids": members,
                "grade": "schema_unrelated_to_hours_and_wifi", "kind": "noncontiguous_control_only_union",
                "rationale": "two complete disjoint unrelated leaves in this source; one joint intervention, not an owner/fact or sum of member effects"})
    return records


def match_controls(oracle, group, census, fixed_ids=None):
    """Freeze both raw-input and position controls before measuring controls."""
    with torch.no_grad():
        ids = torch.tensor(oracle.contrast.prefix, device=oracle.model.device)
        norms = oracle.model.get_input_embeddings()(ids).float().norm(dim=-1).cpu().numpy()
    target = group["keys"]
    target_mass, target_norm = oracle.mass(group), float(norms[target].mean())
    target_visibility = visibility(target, group["queries"])
    # Raw-input reach is into the fixed receiving V positions, not queries.
    target_reach = float((np.asarray(target)[:, None] <= np.asarray(target)[None, :]).mean())
    audited = []
    for entry in census:
        keys = entry["keys"]
        mass, norm = oracle.mass(group, keys), float(norms[keys].mean())
        reach = float((np.asarray(keys)[:, None] <= np.asarray(target)[None, :]).mean())
        eligible = entry["grade"] == "schema_unrelated_to_hours_and_wifi" and not set(keys) & set(target)
        common = eligible and ratio_matches(len(keys), len(target))
        position = common and ratio_matches(mass, target_mass) and abs(visibility(keys, group["queries"]) - target_visibility) <= .1
        origin = common and ratio_matches(norm, target_norm) and abs(reach - target_reach) <= .1
        audited.append({**entry, "position_matched": bool(position), "origin_matched": bool(origin),
            "attention_mass": mass, "embedding_norm": norm, "causal_reach": reach,
            "key_count": len(keys), "visibility": visibility(keys, group["queries"]),
            "matching_checks": {"grade_and_target_disjoint": bool(eligible),
                "key_count": bool(ratio_matches(len(keys), len(target))),
                "position_mass": bool(ratio_matches(mass, target_mass)),
                "position_visibility": abs(visibility(keys, group["queries"]) - target_visibility) <= .1,
                "origin_norm": bool(ratio_matches(norm, target_norm)),
                "origin_reach": abs(reach - target_reach) <= .1}})
    def pick(field, kind):
        chosen, used = [], set()
        for entry in sorted(audited, key=lambda c: c["id"]):
            allowed = fixed_ids is None or entry["id"] in fixed_ids[kind]
            if allowed and entry[field] and not used & set(entry["keys"]):
                chosen.append(entry)
                used.update(entry["keys"])
                if len(chosen) == 2:
                    break
        return chosen
    return {"census": audited, "position": pick("position_matched", "position"), "origin": pick("origin_matched", "origin"),
        "fixed_ids": fixed_ids, "backfill_permitted": False,
        "target": {"mass": target_mass, "norm": target_norm, "causal_reach": target_reach,
            "visibility": target_visibility, "keys": target},
        "matching_scope": "three-band mean mass and input norms, not per-head route-destination matching"}


def measure_path(oracle, search, controls, masks):
    """Run raw position and input-to-V trials even without selective controls."""
    group = search["selected"]["group"]
    position = [oracle.group(group, 1), oracle.group(group), oracle.group(group, .5)]
    position_controls = [oracle.group({**group, "keys": c["keys"], "domain": c["keys"]}) for c in controls["position"]]
    position.append(oracle.group(group, force=True))
    position_cert = certificate(position[1], position[2], position_controls, position[3], position[0]["sham_exact"] is True)
    position_cert["passed"] &= position[1]["delta"] <= -.5
    origin = [oracle.mediated(group, group["keys"], strength) for strength in (1, 0, .5)]
    origin_controls = [oracle.mediated(group, c["keys"], 0) for c in controls["origin"]]
    origin.append(oracle.mediated(group, group["keys"], 0))
    origin_cert = certificate(origin[1], origin[2], origin_controls, origin[3], origin[0]["sham_exact"] is True and origin[0]["delta"] == 0)
    origin_cert["passed"] &= origin[1]["delta"] <= -.5
    complements = {}
    for axis, universe in (("queries", list(oracle.contrast.shared_queries)), ("layers", list(range(len(oracle.model.model.layers))))):
        rest = sorted(set(universe) - set(group[axis]))
        if rest:
            result = oracle.group({**group, axis: rest})
            complements[axis] = {"delta": result["delta"], "measurement": result["key"], "positions": rest,
                "scope": "complement on one axis holding the other axis and source A fixed"}
    certified = bool(position_cert["passed"] and origin_cert["passed"])
    return {"group": group, "position": position_cert, "origin": origin_cert,
        "position_measurements": [r["key"] for r in position + position_controls],
        "origin_measurements": [r["key"] for r in origin + origin_controls],
        "raw_position_delta": position[1]["delta"], "raw_origin_delta": origin[1]["delta"],
        "partition": {"baseline": partition(oracle.base, masks), "position_removed": partition(position[1], masks),
            "origin_removed": partition(origin[1], masks)}, "complements": complements,
        "passed": certified, "status": "conditional_typed_correct_source_V_contribution" if certified else "raw_native_measurements_uncontrolled_or_not_selective",
        "routing": "unresolved_no_independent_B_occurrence", "aggregation": "unresolved",
        "origin_scope": "branch-matched input scaling through same A V receiver keys; other recipient messages untouched",
        "position_scope": "joint_multilayer" if len(group["layers"]) > 1 else "single_layer",
        "full_continuation_preference": "B_preferred" if oracle.base_f >= .5 else "A_preferred" if oracle.base_f <= -.5 else "near_tied",
        "original_generator_causal_claim": False, "unique_lookback_claim": False}
