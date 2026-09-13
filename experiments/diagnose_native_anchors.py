"""Read immutable A artifacts to locate anchor bottlenecks without annotations."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest


def diagnose(run):
    settings = json.loads((run / "settings.json").read_text())
    input_path = Path(settings["input_path"])
    if file_sha256(input_path) != settings["input_sha256"]:
        raise ValueError("input roster hash differs")
    roster = {
        str(row["id"]): row
        for row in map(json.loads, input_path.read_text().splitlines())
    }
    counts, examples, artifacts = Counter(), [], {}
    by_task = defaultdict(Counter)
    for path in sorted((run / "A").glob("*.json")):
        saved = json.loads(path.read_text())
        row = roster[path.stem]
        data = saved["data"]
        if (
            saved["settings_sha256"] != digest(settings)
            or saved["row_sha256"] != digest(row)
            or saved["content_sha256"] != digest(data)
            or saved["upstream"] != {}
        ):
            raise ValueError("A artifact identity/hash differs")
        artifacts[path.name] = file_sha256(path)
        c = Counter(responses=1, all_words=len(data["words"]))
        frames = defaultdict(list)
        for q in data["questions"]:
            c["questions"] += 1
            c["question_" + q["status"]] += 1
            if "claim_span" not in q:
                continue
            frames[tuple(q["claim_span"])].append(q)
            check = q.get("self_check", {})
            if q["status"] == "invalid_question" and check:
                c["self_failed"] += 1
                if check.get("answer_quote") != q.get("answer_quote"):
                    c["self_answer_span_mismatch"] += 1
        collapsed = []
        for span, questions in frames.items():
            failed_roles, condition_failures = set(), []
            for q in questions:
                if q["status"] != "uncertain":
                    continue
                failed = []
                for branch in ("source_answer", "source_verification"):
                    prediction = q.get(branch, {})
                    # A malformed condition table is an interface failure,
                    # not evidence that a non-target constraint is unsupported.
                    if (
                        prediction.get("atomic_condition_binding", {}).get("valid") is not True
                        or prediction.get("condition_table_valid") is not True
                        or prediction.get("citation_error")
                        or prediction.get("reader_error")
                    ):
                        continue
                    for check in prediction.get("condition_checks", []):
                        if check.get("status") in {"missing", "conflicting", "uncertain"}:
                            failed.append({
                                "branch": branch,
                                "role": check["condition_role"],
                                "status": check["status"],
                            })
                if failed and q.get("atomic_target_role"):
                    failed_roles.add(q["atomic_target_role"])
                    condition_failures.append({"question_id": q["id"], "conditions": failed})
            if len(failed_roles) >= 2:
                collapsed.append(span)
                examples.append({
                    "response_id": row["id"], "source_id": row["source_id"],
                    "task": row["task"], "claim_span": list(span),
                    "claim_quote": row["response"][slice(*span)],
                    "failed_mask_roles": sorted(failed_roles),
                    "observed_condition_failures": condition_failures,
                })
        c["distinct_claim_spans_with_questions"] = len(frames)
        c["multiple_masks_with_non_target_failure_spans"] = len(collapsed)
        for word in data["words"]:
            c["word_" + word["coverage_state"]] += 1
            c["semantic_scored_words"] += int(not word["abstain"])
            # Count the union, not sums of overlapping or repeated frames.
            if any(a < word["end"] and b > word["start"] for a, b in collapsed):
                c["multiple_masks_with_non_target_failure_words"] += 1
        counts.update(c)
        by_task[row["task"]].update(c)
    return {
        "schema": "native-anchor-diagnostic@1",
        "settings_sha256": file_sha256(run / "settings.json"),
        "expected_responses": len(roster),
        "observed_A_artifacts": artifacts,
        "counts": counts,
        "by_task": dict(by_task),
        "multiple_mask_failures": examples,
        "labels_used": False,
        "interpretation": "Reader-reported non-target failures across multiple masks; not ground truth of multi-role errors.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run), ensure_ascii=False, indent=2))
