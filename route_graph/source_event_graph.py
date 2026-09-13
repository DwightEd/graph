"""Text-grounded pointer graphs; reference integrity is not factual support.

This module performs no model calls, classification, candidate selection or
native interventions. Source and response inventories are constructed separately.
"""

import ast
import hashlib
import json
import math
import re

ROLES = frozenset({
    "subject", "predicate", "object", "origin", "destination", "time",
    "location", "duration", "quantity", "condition", "negation", "attribute",
})


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def raw_inventory(text, *, side, sample_id, unit_tokens=128):
    """Partition raw coordinates for pointer addressing, not semantic segmentation."""
    if side not in {"source", "response"} or not isinstance(text, str):
        raise ValueError("inventory requires text and source/response side")
    if type(unit_tokens) is not int or unit_tokens < 1:
        raise ValueError("unit_tokens must be a positive integer")
    tokens = [list(m.span()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    units, cursor = [], 0
    for start in range(0, len(tokens), unit_tokens):
        spans = tokens[start : start + unit_tokens]
        right = len(text) if start + unit_tokens >= len(tokens) else spans[-1][1]
        units.append({
            "id": f"u{len(units)}", "span": [cursor, right],
            "tokens": [{"span": span, "text": text[slice(*span)]} for span in spans],
        })
        cursor = right
    if not units:
        units = [{"id": "u0", "span": [0, len(text)], "tokens": []}]
    inventory = {
        "schema": "raw-pointer-inventory@1", "side": side,
        "sample_id": str(sample_id), "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "unit_tokens": unit_tokens, "units": units, "token_count": len(tokens),
        "coverage_kind": "raw_characters_not_facts",
    }
    return {**inventory, "sha256": _digest(inventory)}


def validate_inventory(inventory):
    expected = raw_inventory(
        inventory["text"], side=inventory["side"],
        sample_id=inventory["sample_id"], unit_tokens=inventory["unit_tokens"],
    )
    if inventory != expected:
        raise ValueError("raw inventory integrity mismatch")


def _span(inventory, pointer):
    if not isinstance(pointer, dict) or set(pointer) != {
        "unit_id", "start_token", "end_token",
    }:
        raise ValueError("pointer must contain only unit_id/start_token/end_token")
    unit = next((u for u in inventory["units"] if u["id"] == pointer["unit_id"]), None)
    a, b = pointer["start_token"], pointer["end_token"]
    if unit is None or type(a) is not int or type(b) is not int:
        raise ValueError("unknown unit or non-integer token pointer")
    if not 0 <= a < b <= len(unit["tokens"]):
        raise ValueError("pointer escapes raw unit")
    return [unit["tokens"][a]["span"][0], unit["tokens"][b - 1]["span"][1]]


def compile_pointer_events(inventory, envelope):
    """Compile only finite-role pointers; all quote text and IDs come from raw text."""
    validate_inventory(inventory)
    if not isinstance(envelope, dict) or set(envelope) != {
        "inventory_sha256", "side", "sample_id", "prediction",
    }:
        raise ValueError("compiler requires the inventory-bound request envelope")
    if any(envelope[key] != inventory[field] for key, field in (
        ("inventory_sha256", "sha256"), ("side", "side"), ("sample_id", "sample_id"),
    )):
        raise ValueError("prediction request belongs to another inventory")
    # Identity is copied from the frozen request by its caller, not generated
    # by the model or rebound after reading a cached response.
    envelope = json.loads(json.dumps(envelope, ensure_ascii=False, allow_nan=False))
    prediction = envelope["prediction"]
    text = inventory["text"]
    events, nodes, failures = {}, {}, []
    proposals = []
    if (
        not isinstance(prediction, dict) or set(prediction) != {"events"}
        or not isinstance(prediction["events"], list)
    ):
        failures.append({"scope": "root", "proposal": prediction,
                         "reason": "model root must contain only an events list"})
    else:
        proposals = prediction["events"]
    for index, proposal in enumerate(proposals):
        try:
            if not isinstance(proposal, dict) or set(proposal) != {"anchor", "roles"}:
                raise ValueError("event contains free text or unknown fields")
            anchor = _span(inventory, proposal["anchor"])
            if not isinstance(proposal["roles"], list) or not proposal["roles"]:
                raise ValueError("event has no role pointers")
            roles, spans, pending = {}, [], {}
            for item in proposal["roles"]:
                if not isinstance(item, dict) or set(item) != {"role", "pointer"}:
                    raise ValueError("role must contain only role and pointer")
                role = item["role"]
                if not isinstance(role, str) or role not in ROLES or role in roles:
                    raise ValueError("unknown or repeated role")
                span = _span(inventory, item["pointer"])
                if not anchor[0] <= span[0] < span[1] <= anchor[1]:
                    raise ValueError("role escapes event anchor")
                if any(a < span[1] and b > span[0] for a, b in spans):
                    raise ValueError("role pointers overlap")
                if spans and span[0] < spans[-1][0]:
                    raise ValueError("role pointers are not in raw text order")
                spans.append(span)
                nid = _digest([inventory["sha256"], role, span])
                roles[role] = nid
                pending[nid] = {
                    "id": nid, "role": role, "span": span,
                    "quote": text[slice(*span)], "pointer": item["pointer"],
                }
            eid = _digest([inventory["sha256"], anchor, roles])
            if eid in events:
                raise ValueError("duplicate event proposal")
            gaps, cursor = [], anchor[0]
            for a, b in sorted(spans):
                if a > cursor:
                    gaps.append([cursor, a])
                cursor = b
            if cursor < anchor[1]:
                gaps.append([cursor, anchor[1]])
            events[eid] = {
                "id": eid, "anchor_span": anchor, "anchor_quote": text[slice(*anchor)],
                "roles": roles, "unassigned_spans": gaps,
                "status": "parsed_event" if {"subject", "predicate"} <= roles.keys()
                else "partial_event",
                "relation_verification": "not_performed",
            }
            nodes.update(pending)
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"proposal_index": index, "proposal": proposal, "reason": str(error)})
    covered = {
        i for i, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
        if any(e["anchor_span"][0] <= match.start() and match.end() <= e["anchor_span"][1]
               for e in events.values())
    }
    graph = {
        "schema": "pointer-event-graph@1", "inventory_sha256": inventory["sha256"],
        "side": inventory["side"], "prediction_envelope": envelope,
        "nodes": list(nodes.values()), "events": list(events.values()), "failures": failures,
        "reference_integrity": "validated_for_retained_events",
        "coverage": {"raw_tokens": inventory["token_count"], "event_anchor_tokens": len(covered),
                     "meaning": "extraction_coverage_not_fact_completeness"},
        "absence_inference_allowed": False,
    }
    # Prevent caller mutation of predictions from altering a published graph.
    graph = json.loads(json.dumps(graph, ensure_ascii=False, allow_nan=False))
    return {**graph, "sha256": _digest(graph)}


def compile_literal_fields(inventory):
    """Parse Python literal structure without evaluating code or inferring relations.

    Protocol: dict with string keys, lists, finite scalar constants and signed
    numbers only. Calls/names/attributes/expressions/unpacking are rejected.
    Field paths express literal provenance, not predicates or factual truth.
    """
    validate_inventory(inventory)
    if inventory["side"] != "source":
        raise ValueError("literal field graph requires an independent source inventory")
    text = inventory["text"]
    tree = ast.parse(text, mode="eval")
    if not isinstance(tree.body, (ast.Dict, ast.List)):
        raise TypeError("literal source root must be a dict or list")
    lines = text.splitlines(keepends=True)
    starts, cursor = [], 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)

    def raw_span(node):
        positions = []
        for line, byte_column in ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)):
            prefix = lines[line - 1].encode()[:byte_column].decode()
            positions.append(starts[line - 1] + len(prefix))
        return positions

    nodes, edges = [], []

    def visit(node, path):
        span = raw_span(node)
        nid = _digest([inventory["sha256"], path, span])
        entry = {"id": nid, "path": path, "span": span, "raw_literal": text[slice(*span)]}
        if isinstance(node, ast.Dict):
            entry["kind"] = "dict"
            seen = set()
            for key, value in zip(node.keys, node.values, strict=True):
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    raise ValueError("literal dict keys must be strings; no unpacking")
                if key.value in seen:
                    raise ValueError("duplicate literal dict key")
                seen.add(key.value)
                child = visit(value, [*path, key.value])
                edges.append({"parent": nid, "child": child, "kind": "literal_field",
                              "key": key.value, "key_span": raw_span(key)})
        elif isinstance(node, ast.List):
            entry["kind"] = "list"
            for i, value in enumerate(node.elts):
                child = visit(value, [*path, i])
                edges.append({"parent": nid, "child": child, "kind": "literal_index", "index": i})
        else:
            value = None
            if isinstance(node, ast.Constant) and type(node.value) in {str, int, float, bool, type(None)}:
                value = node.value
            elif (
                isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}
                and isinstance(node.operand, ast.Constant) and type(node.operand.value) in {int, float}
            ):
                value = node.operand.value * (-1 if isinstance(node.op, ast.USub) else 1)
            else:
                raise ValueError("non-literal syntax is forbidden")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite literal")
            entry.update(kind="scalar", value=value, value_type=type(value).__name__,
                         epistemic_status="unknown" if value is None else "observed_literal")
        nodes.append(entry)
        return nid

    root = visit(tree.body, [])
    graph = {
        "schema": "literal-field-graph@1", "inventory_sha256": inventory["sha256"],
        "root": root, "nodes": nodes, "edges": edges,
        "reference_integrity": "validated_literal_coordinates",
        "relation_verification": "not_performed",
        "absence_inference_allowed": False,
    }
    return {**graph, "sha256": _digest(graph)}


def validate_pointer_graph(inventory, graph):
    """Recompile rather than trusting a self-reported integrity digest."""
    if graph.get("schema") == "pointer-event-graph@1":
        expected = compile_pointer_events(inventory, graph["prediction_envelope"])
    elif graph.get("schema") == "literal-field-graph@1":
        expected = compile_literal_fields(inventory)
    else:
        raise ValueError("unknown pointer graph schema")
    if graph != expected:
        raise ValueError("pointer graph differs from raw inventory compilation")
