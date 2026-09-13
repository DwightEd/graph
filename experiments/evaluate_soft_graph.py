"""Join labels only after all immutable graph predictions and measurements finish."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import write_json_once
from route_graph.metrics import binary_detection_metrics
from route_graph.soft_graph_energy import VARIANTS
from route_graph.soft_graph_runner import read_artifact, verify_executed_code


def evaluate(run, labels_path, output):
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads((run / "settings.json").read_text())
    verify_executed_code(run, frozen)
    if json.loads((run / "progress.json").read_text())["status"] != "complete":
        raise ValueError("all predictions must complete before annotation join")
    reference = frozen["evaluation_manifest"]
    if not reference or file_sha256(Path(reference["path"])) != reference["sha256"]:
        raise ValueError("missing or modified evaluation manifest")
    manifest = json.loads(Path(reference["path"]).read_text())
    raw = labels_path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != manifest["labels_sha256"] or
            Path(manifest["labels_path"]).resolve() != labels_path.resolve() or
            manifest["input_sha256"] != frozen["input_sha256"] or
            file_sha256(Path(frozen["input_path"])) != frozen["input_sha256"]):
        raise ValueError("annotation or input frozen bytes differ")
    annotations = {}
    for line in raw.decode().splitlines():
        a = json.loads(line)
        if str(a["id"]) in annotations:
            raise ValueError("duplicate annotation ID")
        annotations[str(a["id"])] = a
    rows = [json.loads(s) for s in Path(frozen["input_path"]).read_text().splitlines() if s]
    members, native_counts, claim_counts = [], Counter(), Counter()
    for row in rows:
        annotation = annotations[str(row["id"])]
        if (annotation["response"] != row["response"] or str(annotation["source_id"]) != str(row["source_id"])
                or annotation["split"] != row["official_split"] or annotation["model"] != row["generator"]):
            raise ValueError("annotation identity differs")
        saved = read_artifact(run, "merge", row, frozen)
        stage_d = read_artifact(run, "D", row, frozen)
        spans = [list(m.span()) for m in re.finditer(r"\S+", row["response"])]
        if [w["span"] for w in saved["words"]] != spans:
            raise ValueError("word denominator missing or reordered")
        for lab in annotation["labels"]:
            if (not 0 <= lab["start"] < lab["end"] <= len(row["response"])
                    or row["response"][lab["start"]:lab["end"]] != lab["text"]):
                raise ValueError("annotation span differs from raw response")
        truth = np.array([any(lab["start"] < b and lab["end"] > a for lab in annotation["labels"]) for a, b in spans])
        scores = {v: np.array([w["scores"][v] for w in saved["words"]]) for v in VARIANTS}
        if any(not np.isfinite(s).all() or np.any(s < 0) or np.any(s > 1) for s in scores.values()):
            raise ValueError("invalid risk score")
        members.append({"id": str(row["id"]), "source_id": str(row["source_id"]), "task": row["task"], "truth": truth, "scores": scores})
        native_counts["forwards"] += stage_d["native_forward_calls"]
        for claim in stage_d["claims"]:
            native_counts["claims_measured"] += int(claim["forward_calls"] > 0)
            native_counts["donor_artifacts"] += len(claim.get("donor_artifacts", []))
            for role in ("source", "history"):
                for edge in claim.get(role, {}).values():
                    native_counts[role + ":" + edge["status"]] += 1
        for claim in saved["claims"]:
            claim_counts["all"] += 1
            claim_counts["source_positive_controlled"] += any(e["w_native_positive_specific"] > 0 for e in claim["source_terms"])
            claim_counts["history_positive_controlled_reuse"] += any(e["eligible"] for e in claim["history_terms"])
    groups = {}
    for task in ["ALL", *sorted({m["task"] for m in members})]:
        subset = [m for m in members if task == "ALL" or m["task"] == task]
        y = np.concatenate([m["truth"] for m in subset])
        ids = np.concatenate([np.full(len(m["truth"]), m["source_id"]) for m in subset])
        metrics = {}
        for variant in VARIANTS:
            s = np.concatenate([m["scores"][variant] for m in subset])
            detected = s >= frozen["protocol"]["score_threshold"]
            tp = int((detected & y).sum())
            metrics[variant] = {"ranking": binary_detection_metrics(y, s, ids, bootstrap=0, seed=20260913, source_balanced=True)
                                if len(np.unique(y)) == 2 else None,
                                "detected_words": int(detected.sum()), "precision": tp / int(detected.sum()) if detected.any() else None,
                                "recall": tp / int(y.sum()) if y.any() else None,
                                "score_equal_half": int((s == .5).sum()), "score_min": float(s.min()), "score_max": float(s.max())}
        groups[task] = {"responses": len(subset), "sources": len(set(ids)), "words": len(y), "annotated_error_words": int(y.sum()), "metrics": metrics}
    result = {"schema": "soft-graph-evaluation@1", "run": str(run.resolve()), "groups": groups,
              "native_counts": dict(native_counts), "claim_counts": dict(claim_counts),
              "settings_sha256": file_sha256(run / "settings.json"), "labels_sha256": hashlib.sha256(raw).hexdigest(),
              "evaluator_sha256": file_sha256(Path(__file__)), "labels_used_stage": "after_complete_evaluation_only",
              "scope": "six_development_sources; no_independent_generalization; observer_replay_not_original_internal_traces",
              "lookback_accuracy": "not_identifiable_without_independent_node_ground_truth",
              "method_claim_supported": "requires_independent_result_to_claim_review"}
    write_json_once(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.labels, args.output), indent=2), flush=True)
