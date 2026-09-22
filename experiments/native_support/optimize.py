"""Freeze cache-only routing estimates before opening any evaluation labels."""

from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .comparison import evaluation_headlines
from .comparison_evaluation import evaluate_comparison
from .filter_evaluation import write_deltas
from .filter_features import cached_features, score_arrays, token_rows
from .filter_report import write_filter_report
from .filtering import BANDWIDTH_FLOOR, REGIONS, filter_scores
from .source_regions import compile_regions, region_mask

DIRECTORY = "route_filter_v3"


def methods_for(regions):
    baseline, attention = "routing_imbalance", "attention_displacement"
    if regions["status"] != "available":
        baseline, attention = "prompt_routing_imbalance", "prompt_attention_displacement"
    names = (baseline, "route_mean", "route_state_filter", "route_pooled_filter", attention, "entropy")
    return baseline, {name: name for name in names}


def score_responses(output, destination, settings, regions, baseline, methods, window):
    rows = []
    for index, response in enumerate(tqdm(settings["responses"], desc="causal routing filter")):
        raw = output / "responses" / f"{index:04d}"
        cache = output / DIRECTORY / "features" / f"{index:04d}.npz"
        features = cached_features(response, raw, cache, region_mask(regions, response))
        filtered = filter_scores(features, baseline, window)
        saved = score_arrays(features, filtered)
        current = token_rows(response, saved, methods)
        directory = destination / "responses" / f"{index:04d}"
        write_arrays(directory / "scores.npz", **saved)
        write_csv(directory / "tokens.csv", current, list(current[0]))
        rows.extend(current)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    return rows


def protocol_summary(settings, regions, baseline, methods, rows, window):
    return {
        "purpose": "label_free_routing_auc_optimization", "primary_baseline": baseline,
        "candidate_method": "route_state_filter", "methods": methods, "window": window,
        "responses": len(settings["responses"]), "scored_tokens": len(rows),
        "model_forward_during_scoring": False, "labels_used_for_scoring": False,
        "gradients": False, "interventions": False, "automatic_method_selection": False,
        "source_regions": regions["status"], "cohort": settings.get("cohort", {"selection": "input_manifest"}),
        "max_abs_ledger_error": max(abs(row["ledger_error"]) for row in rows),
        "state_regions": REGIONS, "head_scope": "all_cached_physical_layers_and_heads",
        "distance": "mean_layer_squared_Hellinger_of_joint_head_region_budget",
        "bandwidth": "causal_median_adjacent_state_distance", "bandwidth_floor": BANDWIDTH_FLOOR,
        "score": "normalized_kernel_average_of_original_route_scores_in_current_and_past_window",
        "prediction_alignment": "query=P+t-1; no current or future answer token in routing input",
        "parameter_selection": "default_window16_from_prior; explicit_cli_window_recorded; no automatic_tuning",
        "mechanism_claim": "none; head-state similarity and ranking gain are empirical questions",
    }


def finish_evaluation(output, destination, summary, rows, annotations):
    annotations = annotations or output / "annotations.json"
    methods = summary["methods"]
    evaluation = evaluate_comparison(output, destination, annotations, methods, rows)
    comparisons = write_deltas(output, destination, annotations, evaluation,
                               summary["primary_baseline"], methods, rows)
    summary["evaluation"] = evaluation_headlines(evaluation)
    summary["evaluation_performed_by_this_stage"] = evaluation["status"] == "evaluated"
    summary["comparisons"] = {"status": comparisons["status"], "file": "comparisons.json",
                              "all_error": [row for row in comparisons.get("comparisons", []) if row["phase"] == "all_error"]}
    write_json(destination / "summary.json", summary)
    write_filter_report(destination / "report.html", rows, summary, evaluation, comparisons)
    return summary


def run_optimization(output, settings, annotations=None, window=16):
    destination = output / DIRECTORY / f"w{window}"
    existing = output / "route_comparison_v2"
    regions_directory = existing if (existing / "source_regions.json").is_file() else output / DIRECTORY
    regions = compile_regions(settings, regions_directory)
    baseline, methods = methods_for(regions)
    with threadpool_limits(limits=1):
        rows = score_responses(output, destination, settings, regions, baseline, methods, window)
    summary = protocol_summary(settings, regions, baseline, methods, rows, window)
    write_json(destination / "scoring_protocol.json", summary)
    return finish_evaluation(output, destination, summary, rows, annotations)


def evaluate_optimization(output, annotations=None, window=16):
    destination = output / DIRECTORY / f"w{window}"
    summary = read_json(destination / "scoring_protocol.json")
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        scores = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(token_rows(response, scores, summary["methods"]))
    result = finish_evaluation(output, destination, summary, rows, annotations)
    return result["evaluation"]
