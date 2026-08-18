"""Print the frozen MG-CASF detector and mechanism-expert results.

Usage:
    python -m experiments.causal_setflow.inspect_results \
        experiments/causal_setflow/outputs/v2/smoke/evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric_rows(report: dict):
    sections = []
    for field in ("components", "experts", "diagnostics"):
        value = report.get(field)
        if isinstance(value, dict):
            sections.append((field, value))
    primary = report.get("metrics")
    if isinstance(primary, dict):
        sections.insert(0, ("primary", {"primary": primary}))

    for family, entries in sections:
        for name, metrics in entries.items():
            if not isinstance(metrics, dict):
                continue
            if metrics.get("auroc") is None:
                continue
            yield {
                "family": family,
                "name": name,
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "baseline": metrics.get("auprc_random_baseline", metrics.get("prevalence")),
                "correct_median": metrics.get("correct_median"),
                "hallucination_median": metrics.get("hallucination_median"),
            }


def _fmt(value, digits=5):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json")
    args = parser.parse_args(argv)

    path = Path(args.evaluation_json)
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = list(_metric_rows(report))
    rows.sort(key=lambda row: (row["family"] != "primary", -(row["auroc"] or 0.0)))

    print(f"schema:           {report.get('schema')}")
    print(f"primary_detector: {report.get('primary_detector')}")
    print(f"labels_read:      {report.get('labels_read')}")
    print()
    print(
        f"{'family':16s} {'name':34s} {'AUROC':>9s} {'AUPRC':>9s} "
        f"{'AP/base':>9s} {'correct':>10s} {'hallu':>10s}"
    )
    print("-" * 106)
    for row in rows:
        baseline = row["baseline"]
        lift = (
            None
            if baseline in (None, 0)
            else float(row["auprc"]) / float(baseline)
        )
        print(
            f"{row['family'][:16]:16s} {row['name'][:34]:34s} "
            f"{_fmt(row['auroc']):>9s} {_fmt(row['auprc']):>9s} "
            f"{_fmt(lift, 3):>9s} {_fmt(row['correct_median']):>10s} "
            f"{_fmt(row['hallucination_median']):>10s}"
        )

    claim = report.get("claim_boundary")
    if claim:
        print("\nclaim boundary:")
        print(claim)


if __name__ == "__main__":
    main()
