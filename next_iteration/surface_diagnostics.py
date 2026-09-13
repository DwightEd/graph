"""Read-only, label-free diagnostics of a completed surface-owner run.

First rejection is an ordered software status, not a causal failure diagnosis.
Therefore also retain every gate's outcome for every actually checked pair.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from next_iteration.surface_graph import edit_candidate
from next_iteration.surface_runner import verify_preflight
from next_iteration.surface_verifier import (
    THRESHOLD,
    validate_target_assessment,
    verification_status,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once
from route_graph.soft_graph_runner import verify_executed_code


def analyze(run, output):
    settings = json.loads((run / "settings.json").read_text())
    verify_executed_code(run, settings)
    verify_preflight(Path(settings["preflight_path"]))
    if json.loads((run / "progress.json").read_text())["status"] != "complete":
        raise ValueError("require completed frozen run")
    parent = Path(settings["preflight_path"])
    samples = [json.loads(line) for line in (parent / "samples.jsonl").read_text().splitlines()]
    roster = [str(s["response_id"]) for s in samples]
    for phase in ("B", "C"):
        if sorted(p.stem for p in (run / phase).glob("*.json")) != sorted(roster):
            raise ValueError("incomplete or extra phase rows")
    counts, failed, passed = Counter(), Counter(), Counter()
    attempts, targets, artifacts = [], [], {}
    criteria = {"target_original": "CN", "edited_base": "S", "selected_occurrence": "S",
                "preservation": "P", "edit_kind": "V"}
    for sample in samples:
        rid = str(sample["response_id"])
        for phase in ("B", "C"):
            path = run / phase / (rid + ".json")
            record = json.loads(path.read_text())
            artifacts[str(path.resolve())] = file_sha256(path)
            if record["settings_sha256"] != digest(settings) or record["data_sha256"] != digest(record["data"]):
                raise ValueError("phase identity differs")
        if record["upstream"]["B_sha256"] != file_sha256(run / "B" / (rid + ".json")):
            raise ValueError("C upstream B differs")
        data = record["data"]
        if data["labels_used"] or data["native_forward_calls"]:
            raise ValueError("unexpected evaluation scope")
        rg, sg = sample["response_graph"], sample["source_graph"]
        slots = {s["slot_id"]: s for s in rg["slots"]}
        bases = {b["base_id"]: b for b in rg["base_units"]}
        if sorted(t["slot_id"] for t in data["target_checks"]) != sorted(slots):
            raise ValueError("target denominator differs")
        for target in data["target_checks"]:
            scores = validate_target_assessment(rg, sg, target, reader_identity=target["reader_identity"])
            slot = slots[target["slot_id"]]
            targets.append({"response_id": rid, "source_id": str(sample["source_id"]),
                "slot_id": slot["slot_id"], "span": slot["span"], "quote": slot["quote"],
                "base": bases[slot["base_id"]]["text"], "scores": scores,
                "reader_record": target["reader_record"],
                "target_CN_gate_passed": scores["C"] + scores["N"] >= THRESHOLD})
        for attempt in data["attempts"]:
            counts["first_status:" + attempt["status"]] += 1
            if "verification" not in attempt:
                continue
            draft, verified = attempt["draft"], attempt["verification"]
            if edit_candidate(rg, sg, draft["slot"]["slot_id"], draft["occurrence"]["occurrence_id"]) != draft:
                raise ValueError("draft differs from actual graphs")
            status = verification_status(draft, verified, reader_identity=verified["reader_identity"])
            if status != attempt["status"]:
                raise ValueError("published status differs from all-gate verification")
            scores = {k: v["scores"] for k, v in verified["checks"].items()}
            gate_pass = {k: sum(scores[k][label] for label in labels) >= THRESHOLD
                         for k, labels in criteria.items()}
            for k, yes in gate_pass.items():
                (passed if yes else failed)[k] += 1
            occurrence = draft["occurrence"]
            attempts.append({"response_id": rid, "source_id": str(sample["source_id"]),
                "slot_id": draft["slot"]["slot_id"], "target": draft["slot"]["quote"],
                "original_base": draft["original_base"], "edited_base": draft["edited_base"],
                "replacement": draft["replacement"], "occurrence_id": occurrence["occurrence_id"],
                "source_parent": occurrence["parent_text"], "field_path": occurrence["field_path"],
                "first_status": status, "gate_pass": gate_pass, "scores": scores,
                "draft_sha256": draft["sha256"], "verification_sha256": verified["sha256"]})
    summary = {"schema": "surface-owner-diagnostics@1", "responses": len(samples),
        "targets": len(targets), "target_CN_gate_passed": sum(t["target_CN_gate_passed"] for t in targets),
        "pairs_checked": len(attempts), "all_five_passed": sum(all(a["gate_pass"].values()) for a in attempts),
        "first_status_counts": dict(counts), "nonexclusive_gate_failures": dict(failed),
        "nonexclusive_gate_passes": dict(passed), "labels_read": False, "native_forwards": 0,
        "scope": "reader predictions and pipeline validity, not accuracy or causal failure attribution",
        "settings_sha256": digest(settings), "input_sha256": settings["input_sha256"],
        "phase_artifacts": artifacts, "diagnostic_code_sha256": file_sha256(__file__)}
    if output.exists():
        raise FileExistsError("diagnostics require a fresh output directory")
    output.mkdir(parents=True)
    write_json_once(output / "targets.json", targets)
    write_json_once(output / "attempts.json", attempts)
    summary["artifacts"] = {n: file_sha256(output / n) for n in ("targets.json", "attempts.json")}
    write_json_once(output / "summary.json", summary)
    return {k: v for k, v in summary.items() if k not in ("phase_artifacts", "artifacts")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))
