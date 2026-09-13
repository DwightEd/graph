"""Complete source provenance before learning natural constraint ownership.

Containment and source key sets are observed structure, not verified semantic
roles, query requirements, or support labels. Long text and unknown values stay
addressable. Components never silently escape their original source field.
"""

import argparse
import ast
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from next_iteration.surface_graph import (
    _check,
    _sealed,
    source_occurrences,
    surface_graph,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import rows
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "complete-constraint-inventory@1", "max_text_words": None,
    "component_types": "lexical surface proposals plus every word atom; not semantic roles",
    "unknown_value_policy": "retain relation owner with value_status=unknown",
    "bundle_meaning": "observed record/field/context containment; not an asserted joint fact",
    "weekday_domain": "seven named source keys, not proof of a response all-days quantifier",
    "candidate_type_filter": "none at inventory stage", "labels_read": False,
    "model_forwards": 0, "support_labels": False, "absence_inference": False}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def decoded_string_map(raw, start, expected):
    """Map each decoded Python-string character to its exact raw source span.

Unsupported or concatenated literal spellings are rejected. Their complete
field remains in the graph, explicitly without certified component mapping.
"""
    match = re.match(r'''(?i)^([ru]*)("""|\x27\x27\x27|"|\x27)''', raw)
    if match is None or not raw.endswith(match[2]) or ast.literal_eval(raw) != expected:
        raise ValueError("unsupported or inconsistent source string literal")
    prefix, quote = match.groups()
    left, right = match.end(), len(raw) - len(quote)
    decoded, spans, i = [], [], left
    simple = {"\\": "\\", "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}
    while i < right:
        a = i
        if raw[i] != "\\" or "r" in prefix.lower():
            value, i = raw[i], i + 1
        else:
            if i + 1 >= right:
                raise ValueError("incomplete literal escape")
            code = raw[i + 1]
            if code == "\n" or raw[i + 1:i + 3] == "\r\n":
                i += 3 if code == "\r" else 2
                continue
            if code in simple:
                value, i = simple[code], i + 2
            elif code in "xXuU":
                width = {"x": 2, "u": 4, "U": 8}.get(code)
                if width is None:
                    value, i = "\\", i + 1
                else:
                    digits = raw[i + 2:i + 2 + width]
                    if len(digits) != width or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                        raise ValueError("invalid literal Unicode/hex escape")
                    value, i = chr(int(digits, 16)), i + 2 + width
            elif code == "N" and raw[i + 2:i + 3] == "{":
                end = raw.find("}", i + 3, right)
                if end < 0:
                    raise ValueError("incomplete named Unicode escape")
                value, i = unicodedata.lookup(raw[i + 3:end]), end + 1
            elif code in "01234567":
                digits = re.match(r"[0-7]{1,3}", raw[i + 1:right]).group()
                value, i = chr(int(digits, 8)), i + 1 + len(digits)
            else:
                # Python retains an unknown escape as two characters.
                value, i = "\\", i + 1
        if len(value) != 1:
            raise ValueError("non-scalar decoded character")
        decoded.append(value)
        spans.append([start + a, start + i])
    if "".join(decoded) != expected or len(spans) != len(expected):
        raise ValueError("decoded source coordinate reconstruction differs")
    return spans


def _raw_component(mapping, span, value, original):
    a, b = span
    if not 0 <= a < b <= len(mapping):
        raise ValueError("component escapes decoded value")
    raw_span = [mapping[a][0], mapping[b - 1][1]]
    return {"decoded_span": [a, b], "display_text": value[a:b], "raw_span": raw_span,
        "raw_surface": original[slice(*raw_span)], "character_source_spans": mapping[a:b],
        "mapping_status": "exact_decoded_character_to_raw_source"}


def component_inventory(value, mapping, *, original, parent_id, sample_id):
    graph = surface_graph(value, sample_id=sample_id + ":" + parent_id, side="source")
    contexts, components = [], {}
    for base in graph["base_units"]:
        if not base["text"].strip():
            continue
        context = _raw_component(mapping, base["span"], value, original)
        contexts.append({"id": digest([parent_id, "context", base["span"]]), "kind": "text_context",
            "parent_id": parent_id, **context, "semantic_event_status": "unverified"})
    for slot in graph["slots"]:
        span = slot["span"]
        components[tuple(span)] = {"id": digest([parent_id, "component", span]), "kind": "surface_component",
            "parent_id": parent_id, **_raw_component(mapping, span, value, original),
            "surface_type": slot["surface_type"], "proposal_reason": slot["proposal_reason"], "semantic_role": None}
    for token in re.finditer(r"\w+(?:['’\-]\w+)*", value, flags=re.UNICODE):
        span = list(token.span())
        if tuple(span) not in components:
            components[tuple(span)] = {"id": digest([parent_id, "component", span]), "kind": "word_atom",
                "parent_id": parent_id, **_raw_component(mapping, span, value, original),
                "surface_type": "number" if token.group().isdigit() else "word", "proposal_reason": "full_word_inventory", "semantic_role": None}
    return contexts, sorted(components.values(), key=lambda c: (c["decoded_span"], c["kind"]))


def compile_inventory(text, *, source_id, task):
    source = source_occurrences(text, sample_id=source_id, task=task)
    _check(source)
    fields, contexts, components, bundles, edges, mapping_failures = [], [], [], [], [], []
    if source["source_kind"] == "literal_field":
        literal = source["literal_graph"]
        nodes = {n["id"]: n for n in literal["nodes"]}
        by_record = {}
        for occurrence in source["occurrences"]:
            node = nodes[occurrence["occurrence_id"]]
            fid = node["id"]
            field = {"id": fid, "kind": "literal_field_owner", "field_path": node["path"],
                "raw_span": node["span"], "raw_surface": node["raw_literal"],
                "display_text": occurrence["display_surface"], "value_type": node["value_type"],
                "value_status": "unknown" if node["value"] is None else "observed_literal",
                "owner_available": True, "record_id": occurrence["parent_id"],
                "component_mapping_status": "not_a_string", "factual_status": "unverified"}
            fields.append(field)
            by_record.setdefault(field["record_id"], []).append(fid)
            field_contexts, field_components = [], []
            if isinstance(node["value"], str):
                try:
                    mapping = decoded_string_map(node["raw_literal"], node["span"][0], node["value"])
                except (ValueError, SyntaxError, KeyError) as error:
                    field["component_mapping_status"] = "unavailable_preserve_whole_field"
                    mapping_failures.append({"field_id": fid, "reason": str(error)})
                else:
                    field_contexts, field_components = component_inventory(node["value"], mapping,
                        original=text, parent_id=fid, sample_id=str(source_id))
                    field["component_mapping_status"] = "exact"
            contexts.extend(field_contexts)
            components.extend(field_components)
            bundles.append({"id": digest([fid, "field_bundle"]), "kind": "field_provenance_bundle", "owner_id": fid,
                "member_ids": [fid] + [c["id"] for c in field_contexts + field_components],
                "relation_label": " / ".join(str(k) for k in node["path"] if isinstance(k, str)),
                "query_requirement_verified": False, "semantic_joint_fact": False})
        for rid, members in sorted(by_record.items()):
            bundles.append({"id": digest([rid, "record_bundle"]), "kind": "literal_record_provenance_bundle",
                "owner_id": rid, "member_ids": members, "raw_span": nodes[rid]["span"],
                "record_kind": nodes[rid]["kind"], "semantic_joint_fact": False, "query_requirement_verified": False})
        children = {}
        for edge in literal["edges"]:
            if edge["kind"] == "literal_field":
                children.setdefault(edge["parent"], {})[edge["key"]] = edge["child"]
        for rid, keyed in children.items():
            if set(WEEKDAYS) <= set(keyed):
                bundles.append({"id": digest([rid, "weekday_key_domain"]), "kind": "weekday_key_inventory",
                    "owner_id": rid, "members_by_key": {k: keyed[k] for k in WEEKDAYS},
                    "member_ids": [keyed[k] for k in WEEKDAYS], "source_key_count": 7,
                    "query_requirement_verified": False, "semantic_joint_fact": False,
                    "note": "source keys only; typed verifier must establish all-days applicability"})
        structural_graph = literal
    else:
        document_id = digest([source_id, "source_document", text])
        mapping = [[i, i + 1] for i in range(len(text))]
        contexts, components = component_inventory(text, mapping, original=text, parent_id=document_id, sample_id=str(source_id))
        fields = [{"id": document_id, "kind": "source_document_owner", "raw_span": [0, len(text)],
            "display_text": text, "raw_surface": text, "value_status": "observed_text_not_fact_truth", "owner_available": True,
            "field_path": [], "record_id": None, "factual_status": "unverified", "component_mapping_status": "exact"}]
        for context in contexts:
            a, b = context["raw_span"]
            members = [c["id"] for c in components if a <= c["raw_span"][0] < c["raw_span"][1] <= b]
            bundles.append({"id": digest([context["id"], "context_bundle"]), "kind": "text_context_provenance_bundle",
                "owner_id": document_id, "member_ids": [context["id"]] + members,
                "query_requirement_verified": False, "semantic_joint_fact": False})
        structural_graph = source["source_graph"]
    for component in contexts + components:
        edges.append({"from": component["id"], "to": component["parent_id"], "kind": "component_of_source_parent"})
    for bundle in bundles:
        edges.extend({"from": member, "to": bundle["id"], "kind": "member_of_provenance_bundle"} for member in bundle["member_ids"])
    ids = [n["id"] for n in fields + contexts + components + bundles]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source graph inventory node")
    if any(text[slice(*c["raw_span"])] != c["raw_surface"] for c in fields + contexts + components):
        raise ValueError("inventory nodes differ from raw source text")
    return _sealed({"protocol": PROTOCOL, "source_id": str(source_id), "task": task, "text": text,
        "source_kind": source["source_kind"], "original_source_graph_sha256": source["sha256"], "source_structure": structural_graph,
        "fields": fields, "contexts": contexts, "components": components, "bundles": bundles, "edges": edges,
        "mapping_failures": mapping_failures, "coverage_scope": "raw source representation, not semantic fact completeness",
        "labels_read": False, "model_forwards": 0, "semantic_certificate_count": 0})


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh complete source inventory output required")
    input_sha = file_sha256(args.inputs)
    roster = rows(argparse.Namespace(inputs=args.inputs))
    sources, response_refs = {}, []
    for row in roster:
        sid = str(row["source_id"])
        if not sid.isdigit():
            raise ValueError("source IDs must be digits")
        text = row["prompt"][slice(*row["source_span"])]
        value = (row["task"], text)
        if sid in sources and sources[sid] != value:
            raise ValueError("source ID reused with different source text/task")
        sources[sid] = value
        response_refs.append({"response_id": str(row["id"]), "source_id": sid, "source_span_in_prompt": row["source_span"],
            "response_sha256": row["response_sha256"], "official_split": row["official_split"]})
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "route_graph").glob("*.py")) + [Path(__file__).with_name(n) for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "constraint_inventory.py")]
    code = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    settings = {"protocol": PROTOCOL, "input_path": str(args.inputs.resolve()), "input_sha256": input_sha,
        "code_sha256": code, "sources": sorted(sources), "response_ids": [r["response_id"] for r in response_refs]}
    write_json_once(args.output / "settings.json", settings)
    for name, sha in code.items():
        destination = args.output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        if file_sha256(destination) != sha:
            raise ValueError("source inventory code changed during snapshot")
    write_json_once(args.output / "response_source_refs.json", response_refs)
    counts, artifacts = Counter(), {"response_source_refs.json": file_sha256(args.output / "response_source_refs.json")}
    for i, (sid, (task, text)) in enumerate(sorted(sources.items())):
        inventory = compile_inventory(text, source_id=sid, task=task)
        name = f"sources/{sid}.json"
        write_json_once(args.output / name, {"settings_object_digest": digest(settings), "data": inventory})
        artifacts[name] = file_sha256(args.output / name)
        counts["sources"] += 1
        counts["task:" + task] += 1
        for key in ("fields", "contexts", "components", "bundles", "edges", "mapping_failures"):
            counts[key] += len(inventory[key])
        counts["unknown_value_owners"] += sum(f["value_status"] == "unknown" for f in inventory["fields"])
        counts["long_text_fields_over_40words"] += sum(len(f["display_text"].split()) > 40 for f in inventory["fields"])
        counts["weekday_key_inventories"] += sum(b["kind"] == "weekday_key_inventory" for b in inventory["bundles"])
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(counts)), flush=True)
    if file_sha256(args.inputs) != input_sha or any(file_sha256(root / n) != sha or file_sha256(args.output / "executed_code" / n) != sha for n, sha in code.items()):
        raise ValueError("inventory code/input changed during compilation")
    write_json_once(args.output / "summary.json", {"status": "complete", "responses": len(response_refs), "counts": dict(counts),
        "model_forwards": 0, "semantic_certificate_count": 0, "labels_read": False})
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_file_sha256": file_sha256(args.output / "settings.json"), "artifacts": artifacts})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
