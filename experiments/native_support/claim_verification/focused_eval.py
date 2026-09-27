"""Evaluate all development refinement predictions only after they are frozen."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import classification, label_record, summarize
from .run import read_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.output / "focused_protocol.json").read_text())
    completed = json.loads((args.output / "focused_completed.json").read_text())
    if completed["ids"] != protocol["ids"]:
        raise ValueError("Complete frozen roster is required")
    manifest = json.loads((args.output / "manifest.json").read_text())
    records = [row for row in manifest["records"] if row["id"] in protocol["ids"]]
    if len(records) != len(protocol["ids"]) or any(row["group"] != "regression" for row in records):
        raise ValueError("Development evaluation scope mismatch")
    method = "focused_reconstruction"
    for row in records:
        with np.load(args.output / "responses" / row["id"] / "focused_scores.npz") as saved:
            values = saved[method]
            if (not np.array_equal(saved["token_id"], row["token_ids"])
                    or values.shape != (len(row["token_ids"]),) or not np.isfinite(values).all()):
                raise ValueError("Incomplete original-token predictions")
            row["scores"] = {method:values}
    official = {str(row["id"]):row for row in read_jsonl(Path(manifest["dataset"]) / "response.jsonl")}
    for row in records:
        label_record(row, official[row["id"]])
    scores = summarize(records, method)
    labels = np.concatenate([row["labels"][row["valid"]] for row in records])
    values = np.concatenate([row["scores"][method][row["valid"]] for row in records])
    scores["fixed_alarm"] = classification(labels,values)
    diagnostics = [dict(id=row["id"],**span,
        alarm_coverage=float((row["scores"][method][span["tokens"]]>0).mean()))
        for row in records for span in row["spans"]]
    fractions = [span["alarm_coverage"] for span in diagnostics]
    scores["span_detection"] = dict(spans=len(fractions),any_hit=sum(x>0 for x in fractions),
                                     half_coverage_hit=sum(x>=.5 for x in fractions))
    clean = [row for row in records if not row["labels"].any()]
    scores["normal_answers"] = dict(count=len(clean),any_false_alarm=sum(
        bool((row["scores"][method][row["valid"]]>0).any()) for row in clean))
    audits=[json.loads((args.output/"responses"/row["id"]/"focused_audit.json").read_text()) for row in records]
    result=dict(method=method,scope="post-evaluation development only",metrics=scores,
        by_task={task:summarize([row for row in records if row["task"]==task],method)
                 for task in ("QA","Summary","Data2txt")},
        coverage=dict(questions=sum(len(a["questions"]) for a in audits),
            invalid=sum(len(a["invalid"]) for a in audits),covered_tokens=sum(a["covered_tokens"] for a in audits),
            answer_parse_failures=sum(a["answers"] is None for a in audits),
            question_contains_quote=sum(a["question_contains_quote"] for a in audits),
            question_truncations=sum(a["question_raw"]["reached_limit"] for a in audits),
            answer_truncations=sum(a["answer_raw"]["reached_limit"] for a in audits)))
    save_json(args.output/"focused_evaluation.json",result)
    save_json(args.output/"focused_diagnostics.json",diagnostics)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
