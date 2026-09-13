"""Evaluation-only RAGTruth label join after immutable natural predictions."""

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from next_iteration.grounded_graph_feature_runner import read
from next_iteration.grounded_graph_restoration_features import verify
from next_iteration.grounded_graph_restoration_predict import (
    PROTOCOL as PREDICTION_PROTOCOL,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "grounded-graph-restoration-ragtruth-evaluation@2", "labels_stage": "evaluation only after frozen predictions",
    "subsets": ["all_words", "through_first_error", "post_first_error"], "source_balanced": True,
    "score_direction": "frozen larger is risk, no flips or fitted combinations", "bootstrap_replicates": 0,
    "scope": "reused development panel unless all entries official test; not independent test or semantic-owner/native-route accuracy",
    "multiple_scores": "both predeclared high-risk differences reported separately; no label-driven checkpoint/score flips"}


def ranking(y, scores, source):
    if len(np.unique(y)) < 2:
        return None
    _, inverse = np.unique(source, return_inverse=True)
    weights = 1. / np.bincount(inverse)[inverse]
    return {"auroc": float(roc_auc_score(y, scores, sample_weight=weights)),
        "auprc": float(average_precision_score(y, scores, sample_weight=weights)),
        "error_prevalence": float(np.average(y, weights=weights))}


def evaluate_members(members):
    joined = {key: np.concatenate([m[key] for m in members]) for key in ("y", "source", "available", "first")}
    scores = {k: np.concatenate([m["scores"][k] for m in members]) for k in PREDICTION_PROTOCOL["scores"]}
    output = {}
    for subset, mask in {"all_words": np.ones(len(joined["y"]), bool),
            "through_first_error": joined["first"], "post_first_error": ~joined["first"]}.items():
        output[subset] = {"words": int(mask.sum()), "error_words": int(joined["y"][mask].sum()),
            "unavailable_words": int((~joined["available"][mask]).sum()),
            "metrics": {k: ranking(joined["y"][mask], v[mask], joined["source"][mask]) for k, v in scores.items()}}
    return output


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh evaluation output required")
    settings = read(args.predictions / "settings.json")
    manifest = read(args.predictions / "manifest.json")
    if (manifest["status"] != "complete" or settings["protocol"] != PREDICTION_PROTOCOL
            or file_sha256(args.predictions / "settings.json") != manifest["settings_sha256"]):
        raise ValueError("natural predictions not complete/frozen")
    features = Path(settings["features_path"])
    if file_sha256(features / "manifest.json") != settings["features_manifest_sha256"]:
        raise ValueError("natural prediction feature parent changed")
    _, entries = verify(features, complete=True)
    expected = {"summary.json"} | {f"predictions/{e['id']}.json" for e in entries} | {
        f"predictions/{e['id']}.npz" for e in entries if e["status"] == "available"}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("prediction manifest omitted or added response denominator")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(args.predictions / name) != sha:
            raise ValueError("natural prediction artifact changed")
    for name, sha in settings["code_sha256"].items():
        root = Path(__file__).resolve().parents[1]
        if file_sha256(root / name) != sha or file_sha256(args.predictions / "executed_code" / name) != sha:
            raise ValueError("prediction executed/live code changed")
    population = read(args.population / "settings.json")
    if population.get("labels_used") is not False:
        raise ValueError("canonical population must be label-free measurement")
    completion = (args.population / "COMPLETE").read_text().strip()
    population_progress = read(args.population / "progress.json")
    input_manifest = read(args.population / "input_manifest.json")
    input_sha = file_sha256(args.population / "inputs.jsonl")
    if (completion != "completed=17790 failed=0 total=17790" or population_progress["status"] != "complete"
            or population_progress["failed"] != 0 or population_progress["completed"] != 17790
            or input_sha != input_manifest["sha256"]):
        raise ValueError("evaluation requires completed, bound RAGTruth population")
    population_rows = [json.loads(line) for line in (args.population / "inputs.jsonl").open()]
    roster = {str(r["id"]): r for r in population_rows}
    if len(roster) != 17790 or len(population_rows) != 17790:
        raise ValueError("canonical annotation-free population roster differs")
    for entry in entries:
        if entry["id"] not in roster or digest(roster[entry["id"]]) != entry["row_sha256"]:
            raise ValueError("predicted response does not match canonical frozen population")
    labels_path = Path(population["dataset"]) / "response.jsonl"
    label_sha = population["input_sha256"]["response.jsonl"]
    audit = {"protocol": PROTOCOL, "prediction_manifest_sha256": file_sha256(args.predictions / "manifest.json"),
        "prediction_path": str(args.predictions.resolve()), "labels_path": str(labels_path), "labels_sha256": label_sha,
        "population_parent_sha256": {str((args.population / n).resolve()): file_sha256(args.population / n)
            for n in ("settings.json", "COMPLETE", "progress.json", "input_manifest.json", "inputs.jsonl")},
        "evaluator_sha256": file_sha256(Path(__file__)), "labels_read": False}
    write_json_once(args.output / "settings.json", audit)
    shutil.copyfile(Path(__file__), args.output / "executed_evaluator.py")
    if file_sha256(args.output / "executed_evaluator.py") != audit["evaluator_sha256"]:
        raise ValueError("evaluation code changed before label access")
    if file_sha256(labels_path) != label_sha:
        raise ValueError("RAGTruth annotation bytes differ from frozen population")
    labels = {}
    for line in labels_path.open():
        value = json.loads(line)
        if str(value["id"]) in labels:
            raise ValueError("duplicate annotation ID")
        labels[str(value["id"])] = value
    if not {e["id"] for e in entries} <= set(labels):
        raise ValueError("evaluation annotations omit predicted responses")
    members, groups, overlap = [], defaultdict(list), defaultdict(list)
    for entry in entries:
        record = read(args.predictions / "predictions" / f"{entry['id']}.json")
        annotation = labels[entry["id"]]
        if (record["id"] != entry["id"] or record["row_sha256"] != entry["row_sha256"]
                or str(annotation["source_id"]) != record["source_id"] or annotation["model"] != record["generator"]
                or annotation["split"] != record["official_split"] or annotation["response"] != record["response_text"]):
            raise ValueError("annotation/response/source/generator/split identity differs")
        words = record["words"]
        expected_words = [(list(w.span()), w.group()) for w in re.finditer(r"\S+", annotation["response"])]
        if [(w["span"], w["text"]) for w in words] != expected_words:
            raise ValueError("prediction word roster omits, overlaps or changes exact response words")
        spans = np.array([w["span"] for w in words])
        if not len(spans) or np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > len(annotation["response"])) or np.any(spans[:, 0] >= spans[:, 1]):
            raise ValueError("word coordinates invalid")
        y = np.zeros(len(words), bool)
        for span in annotation["labels"]:
            if not isinstance(span, dict) or not {"start", "end", "text"} <= set(span):
                raise ValueError("annotation label lacks required start/end/text schema")
            if not 0 <= span["start"] < span["end"] <= len(annotation["response"]):
                raise ValueError("annotation span escapes response")
            if span["text"] != annotation["response"][span["start"]:span["end"]]:
                raise ValueError("annotation text differs from its exact response span")
            y |= (spans[:, 0] < span["end"]) & (spans[:, 1] > span["start"])
        first = np.ones(len(y), bool)
        if y.any():
            first[np.flatnonzero(y)[0] + 1:] = False
        member = {"id": record["id"], "y": y, "first": first,
            "source": np.repeat(record["source_id"], len(y)), "available": np.array([w["status"] == "available" for w in words]),
            "scores": {k: np.array([w["scores"][k] for w in words]) for k in PREDICTION_PROTOCOL["scores"]}}
        if any(not np.isfinite(v).all() for v in member["scores"].values()):
            raise ValueError("natural scores nonfinite")
        members.append(member)
        groups[(record["task"], record["generator"], record["official_split"])].append(member)
        overlap[record["source_training_overlap"]].append(member)
    summary = read(args.predictions / "summary.json")
    if summary["responses"] != len(members) or summary["words"] != sum(len(m["y"]) for m in members):
        raise ValueError("evaluation omitted response/word denominator")
    result = {"protocol": PROTOCOL, "responses": len(members), "sources": len(set(np.concatenate([m["source"] for m in members]))),
        "overall": evaluate_members(members), "groups": [{"task": k[0], "generator": k[1], "official_split": k[2],
            "responses": len(v), "subsets": evaluate_members(v)} for k, v in sorted(groups.items())],
        "source_training_overlap": {k: {"responses": len(v), "subsets": evaluate_members(v)} for k, v in overlap.items()},
        "labels_used_only_after_prediction_freeze": True, "owner_accuracy_available": False,
        "annotation_file_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest()}
    if result["annotation_file_sha256"] != label_sha or file_sha256(Path(__file__)) != audit["evaluator_sha256"]:
        raise ValueError("evaluation code/labels changed during analysis")
    write_json_once(args.output / "evaluation.json", result)
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_sha256": file_sha256(args.output / "settings.json"),
        "artifacts": {n: file_sha256(args.output / n) for n in ("executed_evaluator.py", "evaluation.json")}})
    print(json.dumps({"responses": result["responses"], "sources": result["sources"], "overall": result["overall"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
