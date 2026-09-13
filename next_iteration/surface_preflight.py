"""Natural-roster structural preflight, explicitly lexical-only and label-free."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from next_iteration.surface_graph import (
    edit_candidate,
    source_occurrences,
    surface_graph,
)
from next_iteration.surface_owner import feature_requests, match_owners
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import write_json_once
from route_graph.soft_graph_runner import rows


def prepare(inputs, output):
    inputs, output = inputs.resolve(), output.resolve()
    input_sha = file_sha256(inputs)
    roster = rows(argparse.Namespace(inputs=inputs))
    paths = [Path(__file__).with_name(name) for name in
             ("__init__.py", "surface_graph.py", "surface_owner.py", "surface_preflight.py", "graph_boundaries.py")]
    paths.extend(Path(__file__).parents[1] / "route_graph" / name for name in
                 ("source_event_graph.py", "frozen_reader.py", "soft_graph_runner.py", "audit_artifacts.py"))
    code = {str(p.resolve()): file_sha256(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "executed_code"
    snapshot.mkdir()
    for p in paths:
        target = snapshot / p.parent.name / p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, target)
        if file_sha256(target) != code[str(p.resolve())]:
            raise ValueError("code changed while freezing preflight")
    counts, statuses, documents, source_cache = Counter(), Counter(), {}, {}
    with (output / "samples.jsonl").open("x") as stream:
        for row in roster:
            text = row["prompt"][slice(*row["source_span"])]
            key = (str(row["source_id"]), row["task"], text)
            if key not in source_cache:
                source_cache[key] = source_occurrences(text, sample_id=row["source_id"], task=row["task"])
            source = source_cache[key]
            response = surface_graph(row["response"], sample_id=row["id"], side="response")
            requests = feature_requests(response, source)
            documents.update((d["document_id"], d) for d in requests["documents"])
            matches = match_owners(response, source, {}, mode="lexical_only_preflight")
            drafts = []
            for match in matches["matches"]:
                counts["slots_with_source_topk"] += bool(match["selected"])
                counts["owner_ambiguous_slots"] += match["owner_ambiguous"]
                counts["compatible_occurrences_in_denominator"] += match["candidate_count"]
                for selected in match["selected"]:
                    draft = edit_candidate(response, source, match["slot_id"], selected["occurrence_id"])
                    statuses[draft["status"]] += 1
                    drafts.append(draft)
            counts["responses"] += 1
            counts["original_words"] += len(row["response"].split())
            counts["base_units"] += len(response["base_units"])
            counts["surface_slots"] += len(response["slots"])
            counts.update("surface_type:" + s["surface_type"] for s in response["slots"])
            stream.write(json.dumps({"response_id": str(row["id"]), "source_id": str(row["source_id"]),
                "response_graph": response, "source_graph": source, "matches": matches, "drafts": drafts,
                "feature_requests": requests}, ensure_ascii=False, allow_nan=False) + "\n")
    write_json_once(output / "documents.json", {"documents": list(documents.values())})
    if input_sha != file_sha256(inputs) or any(file_sha256(p) != sha for p, sha in code.items()):
        raise ValueError("input or live code changed during preflight")
    for p in paths:
        if file_sha256(snapshot / p.parent.name / p.name) != code[str(p.resolve())]:
            raise ValueError("snapshot changed during preflight")
    counts["unique_sources"] = len(source_cache)
    counts["unique_masked_documents_not_yet_encoded"] = len(documents)
    result = {"schema": "surface-owner-preflight@1", "input_path": str(inputs), "input_sha256": input_sha,
        "code_sha256": code, "counts": dict(counts), "draft_statuses": dict(statuses),
        "artifacts": {name: file_sha256(output / name) for name in ("samples.jsonl", "documents.json")},
        "labels_read": False, "reader_calls": 0, "native_forward_calls": 0, "feature_forward_calls": 0,
        "matcher_mode": "lexical_only_preflight", "status": "structural_proposals_not_semantically_validated"}
    write_json_once(output / "manifest.json", result)
    return {k: v for k, v in result.items() if k != "code_sha256"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
