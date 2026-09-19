"""Recompute the submitted grounding/binding evidence without loading an LLM."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def grounding_metrics(table):
    rows = []
    groups = [("ALL", table), *list(table.groupby("task"))]
    for task, frame in groups:
        for scope, mask in (
            ("all", np.ones(len(frame), bool)),
            ("previous_gold_0", frame.previous_gold.eq(0)),
            ("previous_gold_1", frame.previous_gold.eq(1)),
        ):
            for score in ("raw_surprise", "head_contrast_surprise", "self_jump"):
                selected = frame[mask & np.isfinite(frame[score])]
                y, values = selected.gold, selected[score]
                measurable = y.nunique() == 2
                rows.append(dict(task=task, scope=scope, score=score,
                                 tokens=len(y), positives=int(y.sum()),
                                 auroc=roc_auc_score(y, values) if measurable else None,
                                 ap=average_precision_score(y, values) if measurable else None))
    return rows


def binding_counts(table):
    tested = table.dropna(subset=["final_condition", "final_value"])
    rows = []
    for key, group in tested.groupby(["case_id", "panel", "side"]):
        for tolerance in (.01, .02, .05):
            weak_condition = group.final_condition.abs() <= tolerance
            strong_value = group.final_value > tolerance
            rows.append(dict(zip(["case_id", "panel", "side"], key),
                             tolerance=tolerance, tested_heads=len(group),
                             value_effect_only=int((weak_condition & strong_value).sum())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = pd.read_csv(args.grounding / "token_scores.csv.gz")
    binding = pd.read_csv(args.flow / "binding_completeness.csv")
    result = dict(
        operation="read-only recomputation of submitted v1 scores, no LLM or retraining",
        answers=int(tokens.id.nunique()), sources=int(tokens.source_id.nunique()),
        grounding=grounding_metrics(tokens), binding_direct_effects=binding_counts(binding),
        interpretation="value-position effect without a direct condition-position effect is not missing semantic binding",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Recomputed {len(tokens)} scored tokens, {len(binding.dropna(subset=['final_condition']))} tested head rows")
    print(args.output)


if __name__ == "__main__":
    main()
