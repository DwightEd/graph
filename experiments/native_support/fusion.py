"""Cached, label-free direct fusion; evaluate ranking and alarm tradeoffs afterward."""

from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)
from tqdm import tqdm

from .comparison import evaluation_headlines
from .comparison_evaluation import evaluate_comparison
from .filter_evaluation import write_deltas
from .fusion_audit import write_fusion_audit
from .fusion_inputs import load_fusion_inputs, load_state_scores
from .fusion_report import write_fusion_report
from .risk_envelope import REFERENCE_QUANTILE, VIEWS, apply_envelope, fit_envelope

DIRECTORY = "risk_fusion_v6"


def fusion_rows(response, scores, methods):
    rows = []
    diagnostics = tuple(f"percentile_{name}" for name in (*VIEWS, "route"))
    for target in range(len(scores["token_id"])):
        values = {name: float(scores[name][target]) for name in (*methods, *diagnostics, "ledger_error")}
        dominant = [view for view in VIEWS if scores[f"dominant_{view}"][target]]
        rows.append({"response_id": response["id"], "source_id": response["source_id"],
                     "target": target, "query": int(scores["query"][target]),
                     "token_id": int(scores["token_id"][target]),
                     "token": response["token_text"][response["prompt_length"] + target],
                     "dominant_views": "|".join(dominant), **values})
    return rows


def fusion_protocol(inputs, thresholds):
    old = inputs["state_protocol"]
    sources = inputs["reference_sources"]
    responses = inputs["reference_responses"]
    methods = {**old["methods"], "percentile_route_state": "percentile_route_state",
               "risk_envelope": "risk_envelope", "instant_envelope": "instant_envelope"}
    return {"purpose": "label_free_directional_risk_fusion", "candidate_method": "risk_envelope",
            "primary_baseline": "route_state", "raw_baseline": old["raw_baseline"], "methods": methods,
            "window": old["window"], "cohort": old["cohort"], "source_regions": old["source_regions"],
            "reference_output": str(inputs["reference_output"]), "reference_mode": "frozen_external_source_disjoint",
            "reference_cohort": inputs["reference_cohort"],
            "reference_source_ids": sorted(set(sources.tolist())),
            "reference_response_ids": list(dict.fromkeys(responses.tolist())), "reference_tokens": len(sources),
            "reference_weighting": "equal_source_then_uniform_token", "views": list(VIEWS),
            "score": "max_of_directional_weighted_reference_midranks", "direction": "higher_is_risk_for_all_views",
            "model_forward_during_scoring": False, "target_state_recomputed": False,
            "labels_used_for_scoring": False, "labels_used_for_reference_or_thresholds": False,
            "gradients": False, "interventions": False, "automatic_method_selection": False,
            "reference_quantile": REFERENCE_QUANTILE, "alarm_thresholds": thresholds,
            "normal_fpr_guarantee": False, "reference_prior_and_quantiles_share_data": True,
            "prediction_alignment": old["prediction_alignment"],
            "comparison_controls": ["route_state", old["raw_baseline"], "percentile_route_state",
                                    "route_mean", "joint_state", "instant_envelope"],
            "mechanism_claim": "none; union of directional excursions, not truth or semantic reanchor",
            "validation_scope": "previously_inspected_cohort_is_development_not_fresh_confirmation"}


def finish_fusion(output, destination, summary, rows, annotations):
    annotations = annotations or output / "annotations.json"
    methods = summary["methods"]
    evaluation = evaluate_comparison(output, destination, annotations, methods, rows)
    pairs = [(summary["candidate_method"], control) for control in summary["comparison_controls"]]
    comparisons = write_deltas(output, destination, annotations, evaluation,
                               summary["raw_baseline"], methods, rows, pairs)
    audit = {"status": "unavailable"}
    if evaluation["status"] == "evaluated":
        audit = write_fusion_audit(output, destination, annotations, methods, rows,
                                  summary["alarm_thresholds"], summary["raw_baseline"])
    summary.update(responses=len({row["response_id"] for row in rows}), scored_tokens=len(rows),
                   max_abs_ledger_error=max(abs(row["ledger_error"]) for row in rows),
                   saturated_fusion_tokens=sum(row["risk_envelope"] >= 1 for row in rows),
                   evaluation=evaluation_headlines(evaluation),
                   evaluation_performed_by_this_stage=evaluation["status"] == "evaluated")
    summary["comparisons"] = {"status": comparisons["status"], "file": "comparisons.json",
                              "all_error": [r for r in comparisons.get("comparisons", []) if r["phase"] == "all_error"]}
    summary["complementarity"] = {"status": audit["status"], "file": "complementarity.json",
                                  "all_error": {name: value["all_error"] for name, value in audit.get("comparisons", {}).items()}}
    write_json(destination / "summary.json", summary)
    write_fusion_report(destination / "report.html", rows, summary, evaluation, comparisons, audit)
    return summary


def run_fusion(output, annotations=None, window=16, reference_output=None):
    inputs = load_fusion_inputs(output, reference_output, window)
    baseline, attention = inputs["baseline"], inputs["attention"]
    sources, responses = inputs["reference_sources"], inputs["reference_responses"]
    distributions, thresholds, reference = fit_envelope(inputs["reference_scores"], sources, baseline, attention)
    destination = output / DIRECTORY / f"w{window}"
    summary = fusion_protocol(inputs, thresholds)
    write_json(destination / "scoring_protocol.json", summary)
    frozen = {f"{view}_{field}": values for view, distribution in distributions.items() for field, values in distribution.items()}
    write_arrays(destination / "reference_distributions.npz", **frozen)
    write_arrays(destination / "reference_scores.npz", source_id=sources, response_id=responses, **reference)
    rows = []
    for index, response in enumerate(tqdm(inputs["settings"]["responses"], desc="direct risk fusion")):
        source = inputs["state_directory"] / "responses" / f"{index:04d}"
        saved = load_state_scores(source, response, inputs["state_protocol"]["methods"])
        saved = apply_envelope(saved, distributions, baseline, attention)
        current = fusion_rows(response, saved, summary["methods"])
        directory = destination / "responses" / f"{index:04d}"
        write_arrays(directory / "scores.npz", **saved)
        write_csv(directory / "tokens.csv", current, list(current[0]))
        rows.extend(current)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    return finish_fusion(output, destination, summary, rows, annotations)


def evaluate_fusion(output, annotations=None, window=16):
    destination = output / DIRECTORY / f"w{window}"
    summary = read_json(destination / "scoring_protocol.json")
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        scores = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(fusion_rows(response, scores, summary["methods"]))
    return finish_fusion(output, destination, summary, rows, annotations)["evaluation"]
