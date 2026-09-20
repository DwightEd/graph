"""CPU reports: distinguish unmeasured routes, failed controls, and finite effects."""

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_rows(output, filename):
    frames = []
    for path in sorted((output / "cases").glob("*/*/" + filename)):
        frame = pd.read_csv(path)
        context = json.loads((path.parent / "context.json").read_text())
        for key in ("case_id", "source_id", "side"):
            frame[key] = context[key]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def case_coverage(row, output, settings):
    directory = output / "cases" / row["case_id"] / row["side"]
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else []
    selections = [event["selected"]["selection"] for event in plan]
    baseline_path = directory / "baseline_checks.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    route_path = directory / "route_checks.json"
    route = json.loads(route_path.read_text()) if route_path.exists() else {}
    effects = pd.read_csv(directory / "effects.csv") if (directory / "effects.csv").exists() else pd.DataFrame()
    units = sum(bool(unit["sources"]) for event in plan for unit in event["source_units"])
    expected = units * len(settings["doses"]) if settings else None
    return dict(case_id=row["case_id"], source_id=row["source_id"], side=row["side"],
        native_graph_measured=plan_path.exists(), earlier_reanchor_candidates=selections.count("lookback_target_flow"),
        route_capture_available=(directory / "routes.npz").exists(),
        route_checks_status="passed" if route.get("route_checks_ok") else ("failed" if route else "not_measured"),
        matched_controls=selections.count("matched_low_flow"),
        unmatched_candidates=selections.count("lookback_target_flow") - selections.count("matched_low_flow"),
        no_lookback_controls=selections.count("high_flow_without_positive_lookback"),
        direct_read_controls=selections.count("decision_query_control"),
        events_without_relay=sum(event["relay"] is None for event in plan),
        empty_source_units=sum(not unit["sources"] for event in plan for unit in event["source_units"]),
        expected_effect_rows=expected, measured_effect_rows=len(effects),
        baseline_status="passed" if baseline.get("baseline_ok") else ("failed" if baseline else "not_measured"),
        readout_status=row["readout_status"], history_status=row["history_status"])


def role_interactions(effects):
    rows = []
    if effects.empty:
        return pd.DataFrame()
    keys = ["case_id", "source_id", "side", "event", "layer", "head", "receiver", "dose"]
    for values, frame in effects.groupby(keys):
        roles = frame.set_index("source_group")
        required = ["scope", "supported_value", "scope_and_value"]
        if not set(required).issubset(roles.index):
            continue
        scope, value, both = [roles.loc[key] for key in required]
        rows.append(dict(zip(keys, values), scope_support=scope.support, value_support=value.support,
            joint_support=both.support, interaction=scope.full - scope.removed - value.removed + both.removed,
            numeric_ok=bool(roles.loc[required, "numeric_ok"].all())))
    return pd.DataFrame(rows)


def summary_counts(inventory, coverage, effects):
    finite = np.isfinite(effects[["full", "removed", "support"]]).all(axis=1) if not effects.empty else pd.Series(dtype=bool)
    passed = effects.numeric_ok.eq(True) if not effects.empty else pd.Series(dtype=bool)
    return dict(sources=int(inventory.source_id.nunique()), paired_sides=len(inventory),
        native_graphs=int(coverage.native_graph_measured.sum()),
        earlier_reanchor_candidates=int(coverage.earlier_reanchor_candidates.sum()),
        measured_effect_rows=len(effects), finite_effect_rows=int(finite.sum()),
        failed_control_rows=int((~passed).sum()), nonfinite_effect_rows=int((~finite).sum()),
        baseline_checks_passed=int(coverage.baseline_status.eq("passed").sum()),
        baseline_checks_missing=int(coverage.baseline_status.eq("not_measured").sum()),
        baseline_checks_failed=int(coverage.baseline_status.eq("failed").sum()),
        route_checks_failed=int(coverage.route_checks_status.eq("failed").sum()),
        purpose="label_assisted_reanchor_mechanism_audit_not_detector_evaluation")


def write_report(output, summary):
    measured = summary["native_graphs"]
    text = ["# Reanchor audit", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
        "No new native routes or causal effects have been measured." if not measured else
        "Routes are descriptive screening proxies. Only controlled native interventions measure effects.", "",
        "Coverage and missing tests are in coverage.csv; all measured worlds keep their full candidate readouts.",
        "The two source questions are the independent units. Side/head/world rows are not independent examples.",
        "Natural headwear prefixes/wording differ; onion alternatives compare different attribute slots.",
        "Neither pair establishes a truth detector or a clean semantic counterfactual.", "",
        "The graph sink precedes the candidate; sequence margins additionally score forced candidate tails.",
        "Attention and message norms cannot establish whether evidence is applicable or carries a supporting sign.",
        "Complete entry-state restoration is an execution identity. Its gate difference equals the four-world interaction;",
        "do not count them as independent evidence. Relay-message restoration tests a conditional downstream route.",
        "Sparse graphs omit edges without renormalizing the retained attention. Absence of a retained path is not proof of no model path.", ""]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def review_package(output):
    with tarfile.open(output / "reanchor_review.tar.gz", "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            include = path.suffix in (".json", ".csv", ".md", ".html", ".png") or path.name == "routes.npz"
            if path.is_file() and include:
                archive.add(path, arcname=str(path.relative_to(output)))


def report(output, package=True):
    output = Path(output)
    inventory = pd.read_csv(output / "inventory.csv")
    config = output / "audit_config.json"
    settings = json.loads(config.read_text()) if config.exists() else None
    coverage = pd.DataFrame([case_coverage(row, output, settings) for row in inventory.to_dict("records")])
    coverage.to_csv(output / "coverage.csv", index=False)
    effects = read_rows(output, "effects.csv")
    if not effects.empty:
        effects.to_csv(output / "effects.csv", index=False)
        role_interactions(effects).to_csv(output / "role_interactions.csv", index=False)
    summary = summary_counts(inventory, coverage, effects)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(output, summary)
    from .views import write_views

    write_views(output)
    if package:
        review_package(output)
    return summary
