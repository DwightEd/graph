"""Source-only masked-field reconstruction views; never hallucination labels.

The first training domain is structured Data2txt. Exact source pointer recovery
is the supervision target; it must not be reported as natural-claim ownership
accuracy or as a factual support/conflict judgment.
"""

import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from next_iteration.surface_graph import _sealed, source_occurrences
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import rows as validated_rows
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "source-rel-mini-data@1", "official_split": "train",
    "domain": "Data2txt_source_fields", "source_validation": "sha256(seed,source_id) modulo5 ==0",
    "seed": 20260913, "queries_per_source": 8, "max_scalar_words": 40,
    "supervision": "original source field pointer reconstruction, not semantic entailment/ownership GT",
    "positive": "the masked field occurrence only; no assumed cross-field fact equivalence",
    "mask": "entire selected scalar value in local parent view, no global homograph deletion",
    "candidate_order": "stable hash independent of source position; IDs not text encoder inputs",
    "pair_loss": "disabled; visible source topology is used as fixed provenance only",
    "labels_read": False}


def _lookup(root, path):
    value = root
    for key in path:
        value = value[key]
    return value


def _field_words(path):
    # Integer list indices are occurrence metadata, never lexical query hints.
    text = " / ".join(str(k).replace("_", " ") for k in path if isinstance(k, str))
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def _value_type(occurrence):
    kind = occurrence["value_type"]
    if kind == "bool":
        return "boolean"
    if kind in {"float", "int"}:
        return "number"
    if kind == "NoneType":
        return "null"
    text = str(occurrence["display_surface"])
    if re.fullmatch(r"\d{1,2}:\d{1,2}-\d{1,2}:\d{1,2}", text):
        return "time_range"
    if re.match(r"\d{4}-\d{2}-\d{2}(?: |$)", text):
        return "date"
    return "text"


def compile_source(source, source_id):
    graph = source_occurrences(source, sample_id=source_id, task="Data2txt")
    if graph["source_kind"] != "literal_field":
        return _sealed({"source_id": source_id, "status": "source_graph_unresolved", "queries": [], "candidates": []})
    root = ast.literal_eval(source)
    root_id = graph["literal_graph"]["root"]
    root_node = next(n for n in graph["literal_graph"]["nodes"] if n["id"] == root_id)
    if root_node["kind"] != "dict" or root_node["path"] != []:
        raise ValueError("source relation training requires the actual dictionary root")
    candidates, query_views = [], {}
    for occurrence in graph["occurrences"]:
        path = occurrence["field_path"]
        if not path or not isinstance(path[-1], str):
            continue
        value = _lookup(root, path)
        if isinstance(value, (list, dict)) or len(str(value).split()) > PROTOCOL["max_scalar_words"]:
            continue
        parent = _lookup(root, path[:-1])
        if not isinstance(parent, dict):
            continue
        # Keep all immediate scalar siblings, including long review text. Nested
        # subrecords remain in the graph and have their own occurrence views.
        local = {str(k): v for k, v in parent.items() if not isinstance(v, (dict, list))}
        if str(path[-1]) not in local:
            raise ValueError("target scalar disappeared from its local parent")
        owner = _field_words(path[:-1]) or "business record"
        name = root.get("name")
        # A name target must not be repeated as a query-side owner descriptor.
        identity = "" if path == ["name"] else ("Business: " + str(name) + "\n" if isinstance(name, str) else "")
        family = _field_words(path)
        candidate_text = (identity + "Record context: " + owner + "\n" + "Source fields: "
            + json.dumps(local, ensure_ascii=False, sort_keys=True) + "\nSelected field: " + family
            + "\nValue: " + json.dumps(value, ensure_ascii=False))
        masked = {**local, str(path[-1]): "<VALUE>"}
        # Query and candidate serialize keys in different orders. Neither gets
        # absolute source offsets/IDs or the original list occurrence index.
        query_text = (identity + "Record context: " + owner + "\n" + "Source fields: "
            + json.dumps(dict(sorted(masked.items(), reverse=True)), ensure_ascii=False)
            + "\nFind the source occurrence for the withheld <VALUE>. Field binding:")
        cid = occurrence["occurrence_id"]
        candidates.append({"id": cid, "text": candidate_text, "family": family,
            "value_type": _value_type(occurrence), "parent_id": occurrence["parent_id"],
            "field_path": path, "source_span": occurrence["span"],
            "value_digest": digest({"type": occurrence["value_type"], "value": value}),
            "epistemic_status": occurrence["epistemic_status"]})
        query_views[cid] = {"text": query_text, "target_value_removed_from_field": True,
            "mask_dependency_scope": "one source field; no alias/equivalence inferred across independent fields",
            "mask_source_spans": [occurrence["span"]], "view_projection": "immediate scalar siblings, no text truncation"}
    candidates.sort(key=lambda c: digest({"seed": PROTOCOL["seed"], "id": c["id"]}))
    queries = []
    for candidate in candidates:
        if candidate["epistemic_status"] != "observed_literal" or candidate["value_type"] == "null":
            continue
        pool = [c for c in candidates if c["value_type"] == candidate["value_type"] and c["epistemic_status"] == "observed_literal"]
        negatives = [c for c in pool if c["id"] != candidate["id"]]
        if not negatives:
            continue
        # Identical masked contexts admit multiple source-pointer answers. Keep
        # their entire positive set rather than teaching an arbitrary index.
        view = query_views[candidate["id"]]
        positive = [c["id"] for c in pool if query_views[c["id"]]["text"] == view["text"]]
        negatives = [c for c in negatives if c["id"] not in positive]
        if not negatives:
            continue
        queries.append({"id": digest({"source": source_id, "view": view["text"], "type": candidate["value_type"]}),
            **view, "value_type": candidate["value_type"], "positive_ids": sorted(positive),
            "candidate_ids": [c["id"] for c in pool], "family": candidate["family"],
            "homograph_negative_ids": [c["id"] for c in negatives if c["value_digest"] == candidate["value_digest"]],
            "other_record_same_field_ids": [c["id"] for c in negatives if c["family"] == candidate["family"] and c["parent_id"] != candidate["parent_id"]],
            "supervision_scope": PROTOCOL["supervision"]})
    queries = list({q["id"]: q for q in queries}.values())
    queries.sort(key=lambda q: digest({"seed": PROTOCOL["seed"], "query": q["id"]}))
    selected = queries[:PROTOCOL["queries_per_source"]]
    split = "source_validation" if int(digest({"seed": PROTOCOL["seed"], "source_id": source_id}), 16) % 5 == 0 else "source_train"
    return _sealed({"source_id": source_id, "status": "compiled", "source_text_digest": digest(source),
        "record_root_id": root_id, "source_graph": graph, "split": split, "candidates": candidates,
        "queries": selected, "query_candidates_before_cap": len(queries), "unselected_query_ids": [q["id"] for q in queries[len(selected):]],
        "protocol": PROTOCOL, "factuality_labels_used": False, "native_effects_used": False})


def prepare(args):
    if args.output.exists():
        raise FileExistsError("source relation data requires a fresh output")
    input_sha = file_sha256(args.inputs)
    sources = {}
    for row in validated_rows(args):
        if row["official_split"] != "train" or row["task"] != "Data2txt":
            continue
        source = row["prompt"][slice(*row["source_span"])]
        sid = str(row["source_id"])
        if not sid.isdigit():
            raise ValueError("source IDs must be digits for immutable output names")
        if sid in sources and sources[sid] != source:
            raise ValueError("same source ID has different original source text")
        sources[sid] = source
    graph_root = Path(__file__).resolve().parents[1]
    paths = list((graph_root / "route_graph").glob("*.py")) + [Path(__file__).with_name(name) for name in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "source_relation_data.py")]
    code_sha = {str(p.relative_to(graph_root)): file_sha256(p) for p in paths}
    settings = {"protocol": PROTOCOL, "input_path": str(args.inputs.resolve()), "input_sha256": input_sha,
        "sources": sorted(sources), "code_sha256": code_sha, "reader_calls": 0, "model_forwards": 0, "labels_read": False}
    write_json_once(args.output / "settings.json", settings)
    for name, sha in code_sha.items():
        destination = args.output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(graph_root / name, destination)
        if file_sha256(destination) != sha:
            raise ValueError("source data code changed during snapshot")
    settings_hash = digest(settings)
    counts, artifacts = Counter(), {}
    for index, (sid, source) in enumerate(sorted(sources.items())):
        data = compile_source(source, sid)
        name = f"sources/{sid}.json"
        write_json_once(args.output / name, {"settings_object_digest": settings_hash, "data": data})
        artifacts[name] = file_sha256(args.output / name)
        counts["sources"] += 1
        counts[data["status"]] += 1
        if data["status"] == "compiled":
            counts[data["split"] + ":sources"] += 1
            counts[data["split"] + ":queries"] += len(data["queries"])
            counts["candidates"] += len(data["candidates"])
            counts["queries_with_homograph_negatives"] += sum(bool(q["homograph_negative_ids"]) for q in data["queries"])
            counts["queries_with_other_record_same_field"] += sum(bool(q["other_record_same_field_ids"]) for q in data["queries"])
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(counts)), flush=True)
    if file_sha256(args.inputs) != input_sha or any(file_sha256(graph_root / p) != sha
            or file_sha256(args.output / "executed_code" / p) != sha for p, sha in code_sha.items()):
        raise ValueError("source reconstruction input/code changed during prepare")
    write_json_once(args.output / "summary.json", {"status": "complete", "counts": dict(counts), "settings_object_digest": settings_hash,
        "labels_read": False, "model_forwards": 0, "scope": "source-pointer training data only; no natural ownership/factuality result"})
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_file_sha256": file_sha256(args.output / "settings.json"), "artifacts": artifacts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
