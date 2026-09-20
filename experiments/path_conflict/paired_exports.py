"""Export cached readouts without activations or a model; expose unmeasured tests."""

import json

import numpy as np
import pandas as pd


def export_record(path):
    """NPZ access is lazy: only the JSON record is read, never message tensors."""
    with np.load(path, allow_pickle=False) as saved:
        record = json.loads(str(saved["record"]))
    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def export_readouts(output):
    rows = []
    for context_path in sorted((output / "pairs").glob("*/*/*/context.json")):
        directory = context_path.parent
        context = json.loads(context_path.read_text())
        baseline = directory / "baseline.npz"
        if baseline.exists():
            export_record(baseline)
        for path in sorted((directory / "worlds").glob("*.npz")):
            export_record(path)
        rows.append(dict(case_id=directory.parents[1].name, side=directory.parent.name,
            phase=directory.name, readout=context["readout"], position=context["position"],
            first_candidate_tokens=len(context["candidates"][0]),
            second_candidate_tokens=len(context["candidates"][1]),
            baseline_record_available=(directory / "baseline.json").exists(),
            world_records_available=len(list((directory / "worlds").glob("*.json")))))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "readout_inventory.csv", index=False)
    available = sum(row["baseline_record_available"] for row in rows)
    return dict(phases=len(rows), baseline_records=available,
                missing_baseline_records=len(rows) - available,
                world_records=sum(row["world_records_available"] for row in rows))


def adaptation_coverage(output, adaptation):
    rows = []
    for path in sorted((output / "pairs").glob("*/plan.json")):
        plan = json.loads(path.read_text())
        layers = {f"L{head['layer']}H{head['head']}_head_total": head["layer"]
                  for head in plan["heads"]}
        eligible = [pair for pair in plan["pairs"] if pair["relation"] == "different_heads"
                    and layers[pair["left"]] != layers[pair["right"]]]
        measured = adaptation[adaptation.case_id == path.parent.name] if not adaptation.empty else adaptation
        if not eligible:
            status = "not_measured_no_cross_layer_pair"
        elif measured.empty:
            status = "not_measured"
        else:
            status = "measured"
        rows.append(dict(case_id=path.parent.name, eligible_cross_layer_pairs=len(eligible),
            measured_pairs=len(measured[["early", "late"]].drop_duplicates()) if not measured.empty else 0,
            measured_rows=len(measured), status=status))
    return pd.DataFrame(rows)
