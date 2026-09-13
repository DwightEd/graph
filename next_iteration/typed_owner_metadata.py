"""Additive record-root correction for immutable typed-hours v1 artifacts."""

import argparse
import json
from pathlib import Path

from next_iteration.surface_graph import _check
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once


def normalize_schedule(schedule):
    _check(schedule)
    graph = schedule["source_graph"]["literal_graph"]
    _check(graph)
    nodes = {n["id"]: n for n in graph["nodes"]}
    root = nodes[graph["root"]]
    if root["kind"] != "dict" or root["path"] != []:
        raise ValueError("typed source has no validated root record")
    names = [n for n in graph["nodes"] if n["path"] == ["name"] and n["kind"] == "scalar"]
    if len(names) != 1:
        raise ValueError("typed business name anchor is not unique")
    original = schedule["root_owner_id"]
    if original not in {root["id"], names[0]["id"]}:
        raise ValueError("unexpected frozen owner ID; no inferred correction")
    return {"source_id": schedule["source_graph"]["sample_id"], "source_schedule_digest": schedule["sha256"],
        "record_root_id": root["id"], "business_name_anchor_id": names[0]["id"],
        "legacy_root_owner_id": original, "correction_needed": original != root["id"],
        "semantics": "legacy constraint_graph.owner is a business-name anchor, not the dict root",
        "unchanged": ["seven_day_constraints", "source_value_spans", "B/A_text", "token_alignment", "source_A_keys", "native_measurements"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("additive correction needs fresh output")
    manifest = json.loads((args.parent / "manifest.json").read_text())
    corrections = []
    for name, sha in manifest["artifacts"].items():
        if not name.startswith("sources/"):
            continue
        path = args.parent / name
        if file_sha256(path) != sha:
            raise ValueError("frozen source artifact changed")
        corrections.append({"source_artifact": name, "source_file_sha256": sha,
            **normalize_schedule(json.loads(path.read_text()))})
    data = {"schema": "typed-owner-metadata-correction@1", "parent": str(args.parent.resolve()),
        "parent_manifest_sha256": file_sha256(args.parent / "manifest.json"), "corrections": corrections,
        "count": len(corrections), "changed_metadata": sum(c["correction_needed"] for c in corrections),
        "labels_read": False, "model_forwards": 0, "frozen_files_modified": False,
        "code_sha256": file_sha256(__file__)}
    write_json_once(args.output, {**data, "object_digest": digest(data)})
    print(json.dumps({k: v for k, v in data.items() if k != "corrections"}))


if __name__ == "__main__":
    main()
