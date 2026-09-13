"""Frozen-feature candidate matching with local provenance constraints.

This is a proposer, not a fact classifier. It has no source capacity constraint,
learned weights, label input, source answer generator, or native-effect input.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from route_graph.source_event_graph import (
    _digest,
    validate_inventory,
    validate_pointer_graph,
)
from route_graph.span_feature_capture import capture_record


@dataclass(frozen=True)
class MatchConfig:
    per_channel: int = 4
    beam_width: int = 32
    output_assignments: int = 8
    relation_weight: float = 0.25
    null_cost: float = 0.8
    unknown_cost: float = 0.8


def _words(text):
    return set(re.findall(r"\w+", text.casefold()))


def match_catalog(inventory, graph=None):
    """Add raw-token fallbacks; addressing units never become semantic events."""
    validate_inventory(inventory)
    if graph is not None:
        validate_pointer_graph(inventory, graph)
    nodes, events = {}, {}
    if graph and graph["schema"] == "pointer-event-graph@1":
        for event in graph["events"]:
            events[event["id"]] = {"id": event["id"], "span": event["anchor_span"]}
            for role, nid in event["roles"].items():
                original = next(n for n in graph["nodes"] if n["id"] == nid)
                item = nodes.setdefault(nid, {
                    "id": nid, "span": original["span"], "role": role,
                    "kind": "role", "memberships": [], "path": [],
                    "literal_unknown": False,
                })
                item["memberships"].append({"id": event["id"], "kind": "parsed_event"})
    elif graph:
        for node in graph["nodes"]:
            if node["kind"] != "scalar":
                continue
            # List elements remain different records even when values match.
            index_positions = [i for i, value in enumerate(node["path"]) if type(value) is int]
            record_path = node["path"][:index_positions[-1] + 1] if index_positions else []
            record = _digest([inventory["sha256"], "literal_record", record_path])
            nodes[node["id"]] = {
                "id": node["id"], "span": node["span"], "role": None,
                "kind": "literal_field", "path": node["path"],
                "value_type": node["value_type"],
                "memberships": [{"id": record, "kind": "literal_record"}],
                "literal_unknown": node["epistemic_status"] == "unknown",
            }
    if inventory["side"] == "source":
        existing = {tuple(n["span"]) for n in nodes.values()}
        for unit in inventory["units"]:
            for token in unit["tokens"]:
                if tuple(token["span"]) in existing or not _words(token["text"]):
                    continue
                nid = _digest([inventory["sha256"], "raw_token", token["span"]])
                nodes[nid] = {
                    "id": nid, "span": token["span"], "role": None,
                    "kind": "raw_token", "memberships": [], "path": [],
                    "literal_unknown": False,
                }
    for node in nodes.values():
        node["quote"] = inventory["text"][slice(*node["span"])]
    result = {
        "schema": "event-match-catalog@1", "inventory_sha256": inventory["sha256"],
        "graph_sha256": graph["sha256"] if graph else None, "side": inventory["side"],
        "text_sha256": inventory["text_sha256"], "text_length": len(inventory["text"]),
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "events": sorted(events.values(), key=lambda n: n["id"]),
        "raw_tokens": inventory["token_count"], "raw_units": len(inventory["units"]),
        "semantic_extraction_complete": False,
    }
    return {**result, "sha256": _digest(result)}


def feature_ids(catalog):
    return [n["id"] for n in catalog["nodes"]] + [e["id"] for e in catalog["events"]]


def feature_manifest(source, response, source_vectors, response_vectors, encoder, captures):
    """Bind real feature arrays to their ordered catalogs and common encoder.

    Arrays are [catalog node/event, layer, hidden]. This constructor does not
    extract features; callers must retain the actual frozen-model provenance.
    """
    required = {"model_sha256", "tokenizer_sha256", "layers", "representation",
                "source_input_sha256", "response_input_sha256", "capture_code_sha256"}
    if set(encoder) != required or not encoder["layers"]:
        raise ValueError("feature encoder requires exact execution provenance")
    if any(not isinstance(encoder[k], str) or not re.fullmatch(r"[0-9a-f]{64}", encoder[k])
           for k in required - {"layers", "representation"}):
        raise ValueError("feature execution identities must be SHA256 digests")
    if (any(type(i) is not int or i < 0 for i in encoder["layers"])
            or sorted(set(encoder["layers"])) != encoder["layers"]
            or encoder["representation"] not in {"pre_block_residual_mean", "post_block_residual_mean"}):
        raise ValueError("invalid frozen layers or residual representation")
    if set(captures) != {"source", "response"}:
        raise ValueError("feature manifest requires both immutable capture records")
    contexts = [captures[side]["mapping"]["input_context"] for side in ("source", "response")]
    if any(contexts[0][key] != contexts[1][key] for key in ("prompt", "source_span")):
        raise ValueError("source and response captures have different prompt/source contexts")
    result = {"schema": "event-match-features@1", "encoder": encoder, "captures": captures}
    shape = None
    for side, catalog, vectors in (
        ("source", source, source_vectors), ("response", response, response_vectors),
    ):
        array = np.asarray(vectors)
        if (array.ndim != 3 or array.shape[0] != len(feature_ids(catalog))
                or array.shape[1] != len(encoder["layers"]) or array.shape[2] < 1
                or array.dtype != np.float32 or not np.isfinite(array).all()):
            raise ValueError("feature arrays require finite float32 node/layer/hidden values")
        if shape is not None and shape != array.shape[1:]:
            raise ValueError("source/response feature spaces differ")
        shape = array.shape[1:]
        expected_record = capture_record(catalog, captures[side]["mapping"], array, encoder)
        if expected_record != captures[side]:
            raise ValueError("captured arrays or provenance record differ")
        result[side] = {"catalog_sha256": catalog["sha256"], "ids": feature_ids(catalog),
                        "shape": list(array.shape),
                        "array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    return {**result, "sha256": _digest(result)}


def _cosine(a, b):
    # Average distances across the frozen layer set; zero states remain unknown.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (an > 0) & (bn > 0)
    if not valid.any():
        return 1.0
    similarity = np.sum(a[valid] * b[valid], axis=-1) / (an[valid] * bn[valid])
    return float(np.mean((1 - np.clip(similarity, -1, 1)) / 2))


def _relation(left, right):
    if left["status"] != "matched" or right["status"] != "matched":
        return 0.0, "unmatched_pair_not_semantically_checked"
    if left["source_node_id"] == right["source_node_id"]:
        return 1.0, "shared_occurrence_between_distinct_roles_unverified"
    a, b = left["membership"], right["membership"]
    if a is None or b is None or a["kind"] != b["kind"]:
        return 0.5, "event_membership_unknown_or_schema_differs"
    if a["id"] == b["id"]:
        return 0.0, "shared_" + a["kind"] + "_provenance_only"
    return 1.0, "different_" + a["kind"] + "_provenance_only"


def propose_event(source_inventory, source_graph, response_inventory, response_graph,
                  event_id, source_vectors, response_vectors, manifest, config=None):
    """Return bounded role pools and joint candidate assignments, never E/C/N."""
    config = MatchConfig() if config is None else config
    if source_inventory["side"] != "source" or response_inventory["side"] != "response":
        raise ValueError("source and response sides must stay independent")
    for field in ("per_channel", "beam_width", "output_assignments"):
        if type(getattr(config, field)) is not int or getattr(config, field) < 1:
            raise ValueError("positive integer search budgets required")
    if any(not np.isfinite(v) or v < 0 for v in (
        config.relation_weight, config.null_cost, config.unknown_cost,
    )):
        raise ValueError("matching costs must be finite and nonnegative")
    source, response = match_catalog(source_inventory, source_graph), match_catalog(response_inventory, response_graph)
    actual = feature_manifest(source, response, source_vectors, response_vectors, manifest["encoder"], manifest["captures"])
    if actual != manifest:
        raise ValueError("feature manifest/catalog/array integrity mismatch")
    if response_graph["schema"] != "pointer-event-graph@1":
        raise ValueError("response requires independent pointer events")
    event = next((e for e in response_graph["events"] if e["id"] == event_id), None)
    if event is None:
        raise ValueError("unknown response event")
    sv = dict(zip(feature_ids(source), source_vectors, strict=True))
    rv = dict(zip(feature_ids(response), response_vectors, strict=True))
    response_nodes = {n["id"]: n for n in response["nodes"]}
    pools = []
    for role, rid in sorted(event["roles"].items()):
        rnode = response_nodes[rid]
        event_words = _words(event["anchor_quote"])
        scored = []
        for node in source["nodes"]:
            surface = _words(rnode["quote"])
            target = _words(node["quote"])
            terms = {
                "hidden": _cosine(rv[rid], sv[node["id"]]),
                "surface": 1 - len(surface & target) / max(1, len(surface | target)),
                "type": 0.5 if node["role"] is None else float(node["role"] != role),
                "field_path": 0.5,
            }
            if node["path"]:
                path_words = _words(" ".join(str(p).replace("_", " ") for p in node["path"] if isinstance(p, str)))
                terms["field_path"] = 1 - len(event_words & path_words) / max(1, len(path_words))
            for member in node["memberships"] or [None]:
                context = sv.get(member["id"]) if member else None
                terms_here = {**terms, "event_context": _cosine(rv[event_id], context)
                              if context is not None else 0.5}
                scored.append({
                    "candidate_id": _digest([rid, node["id"], member]),
                    "source_node_id": node["id"], "membership": member,
                    "span": node["span"], "quote": node["quote"], "field_path": node["path"],
                    "status": "null_literal" if node["literal_unknown"] else "matched",
                    "literal_unknown": node["literal_unknown"], "source_kind": node["kind"],
                    "unary_cost": sum(terms_here[k] * w for k, w in (
                        ("hidden", .55), ("event_context", .25), ("surface", .1), ("type", .05), ("field_path", .05),
                    )), "cost_terms": terms_here, "retrieval_reasons": [],
                })
        retained = {}
        for channel in ("hidden", "surface", "event_context", "type", "field_path"):
            ordered = sorted(scored, key=lambda n: (n["cost_terms"][channel], n["unary_cost"], n["candidate_id"]))
            for candidate in ordered[:config.per_channel]:
                entry = retained.setdefault(candidate["candidate_id"], candidate)
                entry["retrieval_reasons"].append(channel + "_topk")
        candidates = sorted(retained.values(), key=lambda n: (n["unary_cost"], n["candidate_id"]))
        for status, cost in (("null", config.null_cost), ("unknown", config.unknown_cost)):
            candidates.append({"candidate_id": _digest([rid, status]), "source_node_id": None,
                               "membership": None, "status": status, "unary_cost": cost,
                               "cost_terms": {status: cost}, "retrieval_reasons": ["explicit_abstention"]})
        # All equal-value occurrences are counted even if the bounded pool omits some.
        same_values = [n for n in scored if n["cost_terms"]["surface"] == 0]
        best = min((n["unary_cost"] for n in candidates if n["status"] == "matched"), default=1.)
        near_tie_pruned = sum(n["candidate_id"] not in retained and n["status"] == "matched"
                              and n["unary_cost"] <= best + .05 for n in scored)
        unit_ids = [u["id"] for u in source_inventory["units"] if any(
            n["span"][0] < u["span"][1] and n["span"][1] > u["span"][0] for n in source["nodes"]
        )]
        pools.append({"response_role_id": rid, "role": role, "candidates": candidates,
                      "scored_occurrences": len(scored), "retained_occurrences": len(retained),
                      "pruned_occurrences": len(scored) - len(retained),
                      "same_value_candidates": [{k: n[k] for k in (
                          "candidate_id", "source_node_id", "membership",
                      )} for n in same_values],
                      "same_value_competition": len(same_values) > 1,
                      "same_value_outside_pool": sum(n["candidate_id"] not in retained for n in same_values),
                      "near_tie_pruned_candidates": near_tie_pruned,
                      "searched_raw_unit_ids": unit_ids,
                      "unsearched_raw_unit_ids": [u["id"] for u in source_inventory["units"] if u["id"] not in unit_ids],
                      "retrieval_scope": "all_catalog_nodes_with_raw_lexical_fallback; topk_assignments_pruned",
                      "graph_node_candidates_scored": sum(n["source_kind"] != "raw_token" for n in scored),
                      "raw_token_candidates_scored": sum(n["source_kind"] == "raw_token" for n in scored),
                      "control_catalog": scored})
    beams, pruned = [(0.0, [], [])], 0
    role_count = max(1, len(pools))
    pair_count = max(1, len(pools) * (len(pools) - 1) // 2)
    for pool in pools:
        expanded = []
        for cost, links, pairs in beams:
            for candidate in pool["candidates"]:
                new_pairs = []
                for previous in links:
                    penalty, reason = _relation(previous["candidate"], candidate)
                    new_pairs.append({"roles": [previous["response_role_id"], pool["response_role_id"]],
                                      "penalty": penalty, "reason": reason})
                new_cost = cost + candidate["unary_cost"] / role_count + config.relation_weight * sum(p["penalty"] for p in new_pairs) / pair_count
                expanded.append((new_cost, [*links, {"response_role_id": pool["response_role_id"], "candidate": candidate}], [*pairs, *new_pairs]))
        expanded.sort(key=lambda x: (x[0], tuple(n["candidate"]["candidate_id"] for n in x[1])))
        pruned += max(0, len(expanded) - config.beam_width)
        beams = expanded[:config.beam_width]
    assignments = []
    for cost, links, pairs in beams[:config.output_assignments]:
        controls = []
        for pool, link in zip(pools, links, strict=True):
            selected = link["candidate"]
            if selected["status"] != "matched":
                continue
            kinds = {"same_value_other_membership": [], "same_field_other_record": []}
            for other in pool["control_catalog"]:
                if other["status"] != "matched" or not selected["membership"] or not other["membership"]:
                    continue
                if selected["membership"] == other["membership"]:
                    continue
                if _words(other["quote"]) == _words(selected["quote"]):
                    kinds["same_value_other_membership"].append(other)
                if (selected["field_path"] and other["field_path"]
                        and selected["field_path"][-1] == other["field_path"][-1]
                        and selected["membership"]["kind"] == other["membership"]["kind"] == "literal_record"):
                    kinds["same_field_other_record"].append(other)
            retained_ids = {c["candidate_id"] for c in pool["candidates"]}
            for kind, candidates in kinds.items():
                controls.append({
                    "response_role_id": pool["response_role_id"], "type": kind,
                    "eligible_count": len(candidates), "selection": "candidate_id_hash_first_two_before_verification",
                    "candidates": [{**n, "in_assignment_pool": n["candidate_id"] in retained_ids}
                                   for n in sorted(candidates, key=lambda n: n["candidate_id"])[:2]],
                    "applicability": "unverified; different_record_does_not_mean_unrelated",
                })
        assignments.append({
            "total_cost": cost, "role_links": links, "pairwise_penalties": pairs,
            "candidate_controls": controls,
            "identity_status": "not_verified", "downstream_allowed": "relation_verification_only",
            "ambiguity_status": "candidate_competition_not_fact_uncertainty",
        })
    for pool in pools:
        del pool["control_catalog"]
    result = {
        "schema": "event-local-matcher@1", "response_event_id": event_id,
        "source_catalog_sha256": source["sha256"], "response_catalog_sha256": response["sha256"],
        "feature_manifest_sha256": manifest["sha256"], "config": asdict(config),
        "role_candidate_pools": pools, "event_assignments": assignments,
        "source_capacity_used": False, "absence_inference_allowed": False,
        "relation_verification": "not_performed", "native_executed": False,
        "denominators": {"source_units_total": source["raw_units"],
                         "source_catalog_nodes_scored": len(source["nodes"]),
                         "source_raw_tokens": source["raw_tokens"],
                         "response_roles": len(pools), "beam_states_pruned": pruned,
                         "semantic_extraction_complete": False},
        "scope": "candidate_ranking_only; beam_search_is_not_global_optimum; costs_are_not_probabilities",
        "objective": "mean_role_unary + relation_weight * mean_all_role_pair_penalty",
        "unary_weights": {"hidden": .55, "event_context": .25, "surface": .1, "type": .05, "field_path": .05},
    }
    result = json.loads(json.dumps(result, allow_nan=False))
    return {**result, "sha256": _digest(result)}
