"""Evaluation-only join of immutable native-audit predictions and RAGTruth spans."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import read_artifact
from route_graph.frozen_reader import digest, write_json_once
from route_graph.metrics import binary_detection_metrics


def metrics(members, name, threshold=0.8):
    labels = np.concatenate([m["labels"] for m in members])
    scores = np.concatenate([m["scores"][name] for m in members])
    abstain = np.concatenate([m["abstain"][name] for m in members])
    source_ids = np.concatenate(
        [np.full(len(m["labels"]), m["source_id"]) for m in members]
    )
    selected = ~abstain
    detected = selected & (scores >= threshold)
    tp = int((detected & labels).sum())
    ranking = (
        binary_detection_metrics(
            labels,
            scores,
            source_ids,
            bootstrap=0,
            seed=20260913,
            source_balanced=True,
        )
        if len(np.unique(labels)) == 2
        else None
    )
    return {
        "all_words": len(labels),
        "annotated_error_words": int(labels.sum()),
        "scored_words": int(selected.sum()),
        "abstained_words": int(abstain.sum()),
        "coverage": float(selected.mean()),
        "annotated_error_coverage": float(selected[labels].mean())
        if labels.any()
        else None,
        "all_words_source_balanced_ranking": ranking,
        "frozen_threshold": threshold,
        "detected_words_at_frozen_threshold": int(detected.sum()),
        "precision_at_frozen_threshold": tp / int(detected.sum())
        if detected.any()
        else None,
        "recall_including_abstentions": tp / int(labels.sum())
        if labels.any()
        else None,
        "covered_error_rate": float(labels[selected].mean())
        if selected.any()
        else None,
    }


def evaluate(run, labels_path, output):
    if output.exists():
        raise FileExistsError(output)
    settings_path = run / "settings.json"
    settings = json.loads(settings_path.read_text())
    if json.loads((run / "progress.json").read_text()).get("status") != "complete":
        raise ValueError("all method predictions must finish before annotation join")
    reference = settings.get("evaluation_manifest")
    if not reference:
        raise ValueError("run lacks a pre-frozen independent annotation manifest")
    manifest_bytes = Path(reference["path"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != reference["sha256"]:
        raise ValueError("pre-frozen evaluation manifest changed")
    manifest = json.loads(manifest_bytes)
    if (
        manifest["input_sha256"] != settings["input_sha256"]
        or Path(manifest["labels_path"]).resolve() != labels_path.resolve()
    ):
        raise ValueError("evaluation manifest roster/label path mismatch")
    expected_labels_hash = manifest["labels_sha256"]
    input_path = Path(settings["input_path"])
    if file_sha256(input_path) != settings["input_sha256"]:
        raise ValueError("frozen roster changed")
    roster = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    # Hash the exact byte buffer subsequently parsed; annotation file contains
    # the full dataset, while this immutable roster is a strict subset.
    label_bytes = labels_path.read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != expected_labels_hash:
        raise ValueError(
            "annotation bytes differ from the independently frozen dataset"
        )
    annotations = {}
    for line in label_bytes.decode().splitlines():
        annotation = json.loads(line)
        key = str(annotation["id"])
        if key in annotations:
            raise ValueError("duplicate annotation ID")
        annotations[key] = annotation
    members, response_rows = [], []
    counters = {
        key: Counter()
        for key in (
            "question_status",
            "event_failures",
            "native_status",
            "mechanisms",
            "position_status",
            "origin_status",
        )
    }
    for row in roster:
        annotation = annotations[str(row["id"])]
        if (
            str(annotation["source_id"]) != row["source_id"]
            or annotation["model"] != row["generator"]
            or annotation["split"] != row["official_split"]
            or hashlib.sha256(annotation["response"].encode()).hexdigest()
            != row["response_sha256"]
        ):
            raise ValueError("annotation source/generator/split/text mismatch")
        result = read_artifact(run, "merge", row, digest(settings))
        anchors = read_artifact(run, "A", row, digest(settings))
        native = read_artifact(run, "D", row, digest(settings))
        expected_words = [
            (m.start(), m.end()) for m in re.finditer(r"\S+", row["response"])
        ]
        actual = [(w["start"], w["end"]) for w in result["words"]]
        semantic_a_offsets = [
            (w["start"], w["end"]) for w in result["semantic_A_words"]
        ]
        if actual != expected_words or semantic_a_offsets != expected_words:
            raise ValueError("full word coverage or offsets differ")
        y = np.zeros(len(actual), dtype=bool)
        for span in annotation["labels"]:
            if not 0 <= span["start"] < span["end"] <= len(row["response"]):
                raise ValueError("invalid annotation span")
            y |= np.array([a < span["end"] and b > span["start"] for a, b in actual])
        scores = {
            "semantic_A": np.array([w["score"] for w in result["semantic_A_words"]]),
            "semantic_C": np.array([w["semantic_score"] for w in result["words"]]),
            "mechanism": np.array([w["mechanism_score"] for w in result["words"]]),
        }
        abstain = {
            "semantic_A": np.array([w["abstain"] for w in result["semantic_A_words"]]),
            "semantic_C": np.array([w["semantic_abstain"] for w in result["words"]]),
            "mechanism": np.array([w["mechanism_abstain"] for w in result["words"]]),
        }
        for values in scores.values():
            if values.shape != y.shape or not np.isfinite(values).all():
                raise ValueError("invalid score coverage/finiteness")
        member = {
            "id": row["id"],
            "source_id": row["source_id"],
            "task": row["task"],
            "labels": y,
            "scores": scores,
            "abstain": abstain,
        }
        members.append(member)
        for question in anchors["questions"]:
            counters["question_status"][question["status"]] += 1
            if question.get("event_error"):
                counters["event_failures"][question["event_error"]] += 1
        for claim in result["graph"]["claim_nodes"]:
            counters["native_status"][claim["native_status"]] += 1
            counters["mechanisms"].update(claim["mechanisms"])
        for validation in native.values():
            counters["position_status"].update(
                p["status"] for p in validation.get("positions", [])
            )
            counters["origin_status"].update(
                p["status"] for p in validation.get("origins", {}).values()
            )
        response_rows.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "task": row["task"],
                "generator": row["generator"],
                "coverage": result["coverage"],
                "forward_calls": result["forward_calls"],
                "native_tokens_processed": result["native_tokens_processed"],
                "annotated_error_words": int(y.sum()),
                "metrics": {
                    name: metrics(
                        [member], name, settings["protocol"]["semantic_threshold"]
                    )
                    for name in scores
                },
            }
        )
    groups = {}
    for task in ["ALL", *sorted({m["task"] for m in members})]:
        selected = (
            members if task == "ALL" else [m for m in members if m["task"] == task]
        )
        groups[task] = {
            "responses": len(selected),
            "sources": len({m["source_id"] for m in selected}),
            "metrics": {
                name: metrics(
                    selected, name, settings["protocol"]["semantic_threshold"]
                )
                for name in selected[0]["scores"]
            },
        }
    write_json_once(
        output,
        {
            "schema": "native-audit/evaluation@1",
            "run": str(run.resolve()),
            "settings_sha256": file_sha256(settings_path),
            "labels_sha256": expected_labels_hash,
            "evaluator_sha256": file_sha256(Path(__file__)),
            "labels_used_stage": "evaluation_only_after_complete",
            "scope": "six_train_sources_exploratory_observer_replay_not_original_generator_trace",
            "confidence_intervals": "not_estimated_from_only_six_sources",
            "unique_lookback_accuracy": "not_identifiable_without_independent_node_ground_truth",
            "groups": groups,
            "failures_and_mechanisms": counters,
            "responses": response_rows,
            "scientific_review": "not_performed",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.labels, args.output)
