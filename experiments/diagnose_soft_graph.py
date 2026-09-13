"""Read-only, annotation-free graph-run accounting, separate from evaluation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once
from route_graph.soft_graph_energy import positive_dependence


def diagnose(run):
    frozen = json.loads((run / "settings.json").read_text())
    inputs = Path(frozen["input_path"])
    if file_sha256(inputs) != frozen["input_sha256"]:
        raise ValueError("frozen input changed")
    rows = {str(r["id"]): r for r in (json.loads(s) for s in inputs.read_text().splitlines() if s)}
    counters = {k: Counter() for k in ("stages", "extraction", "reader", "candidates", "semantic", "native", "merge")}
    sources, artifact_hashes = {}, {}
    for stage in frozen["protocol"]["phase_order"]:
        for path in sorted((run / stage).glob("*.json")):
            saved = json.loads(path.read_text())
            row = rows[path.stem]
            if (saved["row_sha256"] != digest(row) or saved["settings_sha256"] != digest(frozen)
                    or saved["data_sha256"] != digest(saved["data"])):
                raise ValueError("invalid stage artifact identity")
            for relative, checksum in saved["upstream"].items():
                if file_sha256(run / relative) != checksum:
                    raise ValueError("upstream stage bytes changed")
            artifact_hashes[str(path.relative_to(run))] = file_sha256(path)
            data = saved["data"]
            counters["stages"][stage] += 1
            counters["reader"].update({stage + ":" + k: v for k, v in data.get("reader_outcomes", {}).items()})
            if stage == "A":
                sources[str(row["source_id"])] = data["source_graph"]
                counters["extraction"].update("response_event:" + e["status"] for e in data["response_graph"]["events"])
                counters["extraction"]["response_proposal_failures"] += len(data["response_graph"]["failures"])
                counters["extraction"].update("response_leaf:" + c["extraction_kind"] for c in data["claims"])
                counters["extraction"]["original_words"] += len(row["response"].split())
            elif stage == "B":
                for claim in data["claims"]:
                    c = counters["candidates"]
                    c["leaves"] += 1
                    c["source_edges"] += len(claim["source_terms"])
                    c["source_edges_pruned_before_verifier"] += claim["source_terms_unmeasured"]
                    c["history_edges"] += len(claim["history_terms"])
                    c["history_edges_unsearched"] += claim["history_candidates_unsearched"]
                    c.update("source_kind:" + t["candidate"]["source_kind"] for t in claim["source_terms"])
                    c.update("source_status:" + t["candidate"]["status"] for t in claim["source_terms"])
                    for role in ("source_terms", "history_terms"):
                        c[role + ":pool_has_two_structural_controls"] += sum(len(t["controls"]["candidates"]) >= 2 for t in claim[role])
            elif stage == "C":
                for claim in data["claims"]:
                    c = counters["semantic"]
                    q = claim["q_global"]; best = max(q, key=q.get)
                    c["global_argmax_prediction:" + best] += 1
                    c["global_prediction_ge_08:" + best] += int(q[best] >= .8)
                    c["local_SC_mass_ge_08"] += sum(t["q_local"]["S"] + t["q_local"]["C"] >= .8 for t in claim["source_terms"])
                    c["source_with_two_verified_controls"] += sum(len(t["selected_control_ids"]) == 2 for t in claim["source_terms"])
                    c["history_with_two_verified_controls"] += sum(len(t["selected_control_ids"]) == 2 for t in claim["history_terms"])
                    c["history_reuse_ge_08"] += sum(t["q_relation"]["reuse"] >= .8 for t in claim["history_terms"])
            elif stage == "D":
                c = counters["native"]
                c["forward_calls"] += data["native_forward_calls"]
                for claim in data["claims"]:
                    c["leaves"] += 1
                    c["tokens_processed"] += claim.get("tokens_processed", 0)
                    c["donor_artifact_references"] += len(claim.get("donor_artifacts", []))
                    for role in ("source", "history"):
                        for edge in claim.get(role, {}).values():
                            c[role + ":" + edge["status"]] += 1
                            c[role + ":positive_control_adjusted"] += int(positive_dependence(edge) > 0)
                            c[role + ":negative_origin_delta"] += int(edge.get("delta", 0) < 0)
            elif stage == "merge":
                c = counters["merge"]
                c["words"] += len(data["words"])
                c["regions"] += len(data["graph_regions"])
                c["native_forward_calls"] += data["native_forward_calls"]
                for word in data["words"]:
                    for name, score in word["scores"].items():
                        c[name + ":exactly_neutral"] += int(score == .5)
                        c[name + ":ge_08"] += int(score >= .8)
    source_counts = Counter()
    for source in sources.values():
        source_counts[source["schema"]] += 1
        source_counts["nodes"] += len(source["nodes"])
        source_counts.update("event:" + e["status"] for e in source.get("events", []))
        source_counts["proposal_failures"] += len(source.get("failures", []))
    return {"schema": "soft-graph-no-label-diagnostic@1", "progress_snapshot": json.loads((run / "progress.json").read_text()),
            "counts": {k: dict(v) for k, v in counters.items()}, "unique_source_graphs": dict(source_counts),
            "artifact_hashes": artifact_hashes, "diagnostic_code_sha256": file_sha256(Path(__file__)),
            "settings_file_sha256": file_sha256(run / "settings.json"), "labels_read": False,
            "verification_scope": "stage_identity_and_upstream_hashes; donor_bytes_not_reaudited_by_this_monitor",
            "interpretation": "reader categories are predictions, not factual ground truth or detection metrics"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run)
    if args.output is not None:
        write_json_once(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"}, indent=2))
