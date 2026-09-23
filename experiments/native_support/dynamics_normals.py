"""High-score normal tokens, contiguous runs and observed feature contrasts.

Labels define diagnostic groups only. 'Normal' means unmarked by the supplied
annotations, not a new semantic verification of the text.
"""

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from state_audit.storage import write_csv, write_json


def normal_runs(tokens, threshold):
    runs, current = [], []
    for row in tokens:
        selected = row["label"] == 0 and row["state_dynamics"] >= threshold and row["future_query_count"] > 0
        adjacent = current and row["response_id"] == current[-1]["response_id"] and row["target"] == current[-1]["target"] + 1
        if current and (not selected or not adjacent):
            runs.append(run_row(current))
            current = []
        if selected:
            current.append(row)
    if current:
        runs.append(run_row(current))
    return runs


def run_row(rows):
    route = next(name for name in rows[0] if name.endswith("routing_imbalance"))
    names = ("state_dynamics", "state_log_odds", "entropy", route,
             "mean_head_source_read", "mean_head_history_read")
    return {"response_id": rows[0]["response_id"], "source_id": rows[0]["source_id"],
            "start": rows[0]["target"], "end": rows[-1]["target"], "tokens": len(rows),
            "text": "".join(row["token"] for row in rows),
            **{name: float(np.mean([row[name] for row in rows])) for name in names}}


def normal_tables(tokens, threshold=.9):
    feature_names = [name for name in tokens[0] if name.startswith("mean_head_")]
    route = next(name for name in tokens[0] if name.endswith("routing_imbalance"))
    feature_names += ["entropy", route, "state_log_odds", "emission_log_ratio"]
    groups = defaultdict(list)
    for row in tokens:
        if row["state_dynamics"] >= threshold:
            boundary = "no_future" if row["future_query_count"] == 0 else "future_observed"
            groups[(boundary, row["label"])].append(row)
    counts, contrasts = [], []
    for (boundary, label), rows in sorted(groups.items()):
        counts.append({"boundary": boundary, "label": label, "threshold": threshold,
                       "tokens": len(rows), "responses": len({row["response_id"] for row in rows})})
        for name in feature_names:
            values = np.asarray([row[name] for row in rows])
            contrasts.append({"boundary": boundary, "label": label, "feature": name,
                              "tokens": len(rows), "mean": float(values.mean()),
                              "median": float(np.median(values)), "std": float(values.std())})
    selected = [row for row in tokens if row["label"] == 0 and row["state_dynamics"] >= threshold]
    return {"normal_high_counts": counts, "normal_high_features": contrasts,
            "normal_high_answer_groups": answer_groups(tokens, threshold),
            "normal_high_runs": normal_runs(tokens, threshold),
            "normal_high_tokens": sorted(selected, key=lambda row: -row["state_log_odds"])}


def answer_groups(tokens, threshold):
    """Separate normal content in mixed answers from entirely unmarked answers."""
    has_error = defaultdict(bool)
    for row in tokens:
        has_error[row["response_id"]] |= bool(row["label"])
    groups = defaultdict(list)
    for row in tokens:
        if row["state_dynamics"] >= threshold and row["future_query_count"] > 0:
            groups[has_error[row["response_id"]], row["label"]].append(row)
    return [{"evaluated_answer_has_error": marked, "label": label, "tokens": len(rows),
             "responses": len({row["response_id"] for row in rows})}
            for (marked, label), rows in sorted(groups.items())]


def read_tokens(archive):
    with ZipFile(archive) as source, source.open("audit/tokens.csv") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
    integer_fields = ("target", "label", "future_query_count")
    float_fields = [name for name in rows[0] if name.startswith("mean_head_")]
    float_fields += ["state_dynamics", "state_log_odds", "entropy", "emission_log_ratio"]
    float_fields += [name for name in rows[0] if name.endswith("routing_imbalance")]
    for row in rows:
        for name in integer_fields:
            row[name] = int(row[name])
        for name in float_fields:
            row[name] = float(row[name])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tables = normal_tables(read_tokens(args.archive))
    for name, rows in tables.items():
        if rows:
            write_csv(args.output / f"{name}.csv", rows, list(rows[0]))
    summary = {"labels_used": "diagnostic_groups_only", "model_fitting": False,
               "threshold": .9, "threshold_is_diagnostic_not_calibrated": True,
               "counts": tables["normal_high_counts"],
               "interior_answer_groups": tables["normal_high_answer_groups"],
               "interior_normal_runs": len(tables["normal_high_runs"]),
               "max_run_length": max((row["tokens"] for row in tables["normal_high_runs"]), default=0)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
