"""Three observations, one switching-state posterior, existing ranking evaluation."""

import numpy as np
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
from .filter_features import score_arrays
from .state_inputs import (
    COVARIANCE_RIDGE,
    load_features,
    load_reference,
    reference_moments,
)
from .state_report import write_state_report
from .switching import PRIOR_COUNT, switching_filter

DIRECTORY = "joint_state_v4"
DIAGNOSTICS = ("reset_probability", "expected_run_length", "state_route_sd",
               "reset_contribution", "continuation_contribution", "observation_nll")


def score_record(record, prior, window):
    observations = record["observations"]
    mean, covariance = np.asarray(prior["mean"]), np.asarray(prior["covariance"])
    joint = switching_filter(observations, mean, covariance, window)
    route = switching_filter(observations[:, :1], mean[:1], covariance[:1, :1], window)
    scores = {**joint, "joint_state": joint["state_mean"][:, 0],
              "route_state": route["state_mean"][:, 0],
              "instant_state": (PRIOR_COUNT * mean[0] + observations[:, 0]) / (PRIOR_COUNT + 1),
              "route_mean": np.array([observations[max(0, t - window + 1):t + 1, 0].mean()
                                      for t in range(len(observations))])}
    return score_arrays(record["features"], scores)


def token_rows(response, scores, methods):
    rows = []
    for target in range(len(scores["token_id"])):
        values = {name: float(scores[name][target]) for name in (*methods, *DIAGNOSTICS, "ledger_error")}
        rows.append({"response_id": response["id"], "source_id": response["source_id"],
                     "target": target, "query": int(scores["query"][target]),
                     "token_id": int(scores["token_id"][target]),
                     "token": response["token_text"][response["prompt_length"] + target], **values})
    return rows


def protocol(settings, regions, baseline, attention, mode, priors, window, reference_output):
    names = (baseline, "route_mean", "joint_state", "route_state", "instant_state", attention, "entropy")
    return {"purpose": "label_free_joint_observation_switching_state", "candidate_method": "joint_state",
            "primary_baseline": "route_mean", "raw_baseline": baseline,
            "methods": {name: name for name in names}, "window": window,
            "expected_run_prior": window, "hard_run_length_limit": None,
            "observations": [baseline, attention, "log1p_entropy"], "reference_mode": mode,
            "reference_output": str(reference_output) if reference_output else None, "priors_by_source": priors,
            "prior_count": PRIOR_COUNT, "prior_degrees": "dimension+2", "covariance_ridge": COVARIANCE_RIDGE,
            "source_regions": regions["status"], "cohort": settings.get("cohort", {"selection": "input_manifest"}),
            "model_forward_during_scoring": False, "labels_used_for_scoring": False,
            "gradients": False, "interventions": False, "automatic_method_selection": False,
            "prediction_alignment": "query=P+t-1; current target not in model input",
            "score": "posterior_expected_functional_route_mean; not hallucination_probability",
            "state_change": "before_current_observation; first_row_forced_reset_is_initialization",
            "mechanism_claim": "statistical routing regimes; no truth, head synergy or semantic reanchor claim"}


def finish_evaluation(output, destination, summary, rows, annotations):
    annotations = annotations or output / "annotations.json"
    methods = summary["methods"]
    evaluation = evaluate_comparison(output, destination, annotations, methods, rows)
    controls = summary.get("comparison_controls", [summary["raw_baseline"], "route_mean", "route_state", "instant_state"])
    pairs = [(summary["candidate_method"], name) for name in controls]
    comparisons = write_deltas(output, destination, annotations, evaluation,
                               summary["raw_baseline"], methods, rows, pairs)
    summary.update(responses=len({row["response_id"] for row in rows}), scored_tokens=len(rows),
                   max_abs_ledger_error=max(abs(row["ledger_error"]) for row in rows),
                   evaluation=evaluation_headlines(evaluation),
                   evaluation_performed_by_this_stage=evaluation["status"] == "evaluated")
    summary["comparisons"] = {"status": comparisons["status"], "file": "comparisons.json",
                              "all_error": [row for row in comparisons.get("comparisons", []) if row["phase"] == "all_error"]}
    write_json(destination / "summary.json", summary)
    write_state_report(destination / "report.html", rows, summary, evaluation, comparisons)
    return summary


def run_state_model(output, settings, annotations=None, window=16, reference_output=None):
    records, regions, baseline, attention = load_features(output, settings)
    reference, mode = load_reference(reference_output, output, settings, records, baseline)
    sources = dict.fromkeys(r["response"]["source_id"] for r in records)
    priors = {source: reference_moments(reference, source) for source in sources}
    destination = output / DIRECTORY / f"w{window}"
    summary = protocol(settings, regions, baseline, attention, mode, priors, window, reference_output)
    write_json(destination / "scoring_protocol.json", summary)
    rows = []
    with threadpool_limits(limits=1):
        for index, record in enumerate(tqdm(records, desc="joint switching state")):
            response = record["response"]
            scores = score_record(record, priors[response["source_id"]], window)
            current = token_rows(response, scores, summary["methods"])
            directory = destination / "responses" / f"{index:04d}"
            write_arrays(directory / "scores.npz", **scores)
            write_csv(directory / "tokens.csv", current, list(current[0]))
            rows.extend(current)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    return finish_evaluation(output, destination, summary, rows, annotations)


def evaluate_state_model(output, annotations=None, window=16):
    destination = output / DIRECTORY / f"w{window}"
    summary = read_json(destination / "scoring_protocol.json")
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        scores = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(token_rows(response, scores, summary["methods"]))
    return finish_evaluation(output, destination, summary, rows, annotations)["evaluation"]
