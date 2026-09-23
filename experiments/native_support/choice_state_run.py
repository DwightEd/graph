"""CPU-only conditional choice audit and fixed lineage candidate comparison."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
from state_audit.storage import read_json, write_arrays, write_csv, write_json

from .choice_cache import CaptureReader, validate_row
from .choice_state import read_observations, read_state, summarize_states

BASELINES = ("raw_route", "route_offline_mean", "root_route", "root_route_mean",
             "raw_attention", "entropy")
CANDIDATES = ("local_source_deficit", "lineage_source_deficit", "local_opposition",
              "lineage_opposition", "descendant_opposition")
METHODS = {name: name for name in (*BASELINES, *CANDIDATES)}


def protocol(settings, capture):
    return {
        "version": 2, "purpose": "conditional_choice_lineage_candidate_audit",
        "candidate_method": "lineage_source_deficit", "primary_baseline": "raw_route",
        "default_risk": "raw_route; candidate_not_promoted",
        "candidate_development": "source_deficit_added_after_opposition_failed_on_same_four_answer_pilot",
        "risk_definition": "no_positive_source_endpoint_conditional_on_resolved_lineage",
        "unresolved_policy": "report_path_bounds; fully_unresolved_is_unscored",
        "methods": METHODS, "capture": capture, "cohort": settings.get("cohort"),
        "state_axes": ["source_group_plus_unresolved", "support_or_opposition"],
        "read_audit": "all_native_heads_by_current_source_key_group; separate_from_root_state",
        "reanchor_scope": "read_concentration_and_root_support_are_observations_not_semantic_reanchor_labels",
        "state_update": "terminal_mass_plus_positive_history_kernel_times_previous_state",
        "alternatives": "native_probability_conditional_on_not_the_observed_token",
        "probability_reconstruction": "float64_logits_plus_saved_surprisal; log_mass_clamped_only_for_roundoff",
        "uncaptured_alternatives": "absorbed_into_unresolved_without_imputation",
        "history_edges": "complete_positive_input_root_choice_attribution; not_attention_top_k",
        "edge_alignment": "input_key=P+j -> observed_decision=j; includes_predictor_self_j=t-1",
        "negative_history": "terminal_opposition; no_sign_product_across_token_contrasts",
        "candidate_alignment": "actual_ids_local_to_each_decision; never_match_candidate_ranks",
        "ffn": "included_once_in_native_root_attribution; not_an_extra_risk_term",
        "native_probability": "saved_LM_candidate_probability_given_full_observed_prefix",
        "trajectory": "teacher_forced_observer_of_saved_answer; not_original_generator_sampling",
        "lineage_probability": "defined_attribution_path_event; not_LM_transition_or_truth_probability",
        "cross_token_recursion": "detector_hypothesis_not_native_backprop_through_sampling",
        "future_tokens_used": {name: name in ("route_offline_mean", "root_route_mean", "descendant_opposition")
                               for name in METHODS},
        "labels_used_for_scoring": False, "parameter_fitting": False,
        "automatic_method_selection": False, "model_forward_during_scoring": False,
        "interventions": False, "svd": False,
    }


def collect_states(reader, index, response, sources):
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    source_count = len(sources["blocks"])
    states = np.zeros((count, source_count + 4, 2))
    local = np.zeros_like(states)
    history = np.zeros((count, count))
    factors, candidate_states, candidate_edges, ledger, reads = [], [], [], [], []
    for target in range(count):
        path = f"value_transport/capture/{index:04d}/token_{target:06d}.npz"
        row = reader.capture(path)
        validate_row(row, response, target, sources["group_ids"], source_count)
        result = read_state(row, source_count, prompt, states[:target])
        factor, states[target], local[target], candidate_state, edges, history[target, :target] = result
        factor["candidate_ids"] = row["candidate_ids"]
        factor["native_margin"] = row["margin"]
        factor["head_group_attention"] = row["group_attention"]
        factors.append(factor)
        candidate_states.append(candidate_state)
        candidate_edges.append(np.pad(edges, ((0, count - target), (0, 0))))
        ledger.append(np.max(np.abs(row["ledger_error"])))
        reads.append(read_observations(row["group_attention"], source_count))
    details = {name: np.asarray([item[name] for item in factors]) for name in factors[0]}
    details.update(state=states, local_terminal=local, history_kernel=history,
                   candidate_state=np.asarray(candidate_states),
                   candidate_history_kernel=np.asarray(candidate_edges))
    scores = summarize_states(states, local, history, source_count)
    scores.update({name: np.asarray([item[name] for item in reads]) for name in reads[0]})
    scores.update(ledger_error=np.asarray(ledger),
                  unobserved_alternative_mass=details["unobserved_alternative_mass"],
                  tail_roundoff_correction=details["tail_roundoff_correction"],
                  normalizer_roundoff=details["normalizer_roundoff"],
                  choice_reconstruction_error=details["choice_reconstruction_error"])
    return scores, details


def score_response(reader, index, response, destination):
    sources = reader.json(f"value_transport/capture/{index:04d}/sources.json")
    scores, details = collect_states(reader, index, response, sources)
    baseline = reader.arrays(f"value_transport/responses/{index:04d}/scores.npz")
    targets = np.arange(len(scores["lineage_opposition"]))
    expected = {"target": targets, "query": response["prompt_length"] + targets - 1,
                "token_id": np.asarray(response["token_ids"][response["prompt_length"]:])}
    for name, values in expected.items():
        if not np.array_equal(baseline[name], values):
            raise ValueError(f"{response['id']}: saved baseline {name} differs")
    scores.update(expected)
    scores.update({name: baseline[name] for name in BASELINES})
    scores["risk"] = scores["raw_route"]
    directory = destination / "responses" / f"{index:04d}"
    write_arrays(directory / "scores.npz", **scores)
    write_arrays(directory / "state.npz", **details)
    write_json(directory / "sources.json", sources)
    return [{"response_id": response["id"], "source_id": response["source_id"],
             "token": response["token_text"][response["prompt_length"] + target],
             **{name: values[target].item() for name, values in scores.items()}}
            for target in targets]


def evaluate(destination, annotations, rows):
    from .comparison_evaluation import evaluate_comparison
    from .comparison_deltas import write_deltas
    from .choice_state_audit import rank_audit, state_audit

    result = evaluate_comparison(destination, destination, annotations, METHODS, rows)
    pairs = (("lineage_source_deficit", "local_source_deficit"),
             ("lineage_source_deficit", "raw_route"),
             ("lineage_source_deficit", "route_offline_mean"),
             ("lineage_opposition", "local_opposition"),
             ("lineage_opposition", "raw_route"),
             ("lineage_opposition", "route_offline_mean"),
             ("descendant_opposition", "local_opposition"))
    write_deltas(destination, destination, annotations, result, METHODS, rows, pairs)
    if result["status"] == "evaluated":
        rank_audit(destination, annotations, METHODS)
        state_audit(destination, annotations)
    return result


def run(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        capture = reader.json("value_transport/capture_settings.json")
        if capture["model"] != settings["model"] or capture["responses"] != settings["responses"]:
            raise ValueError("capture/settings model or responses differ")
        frozen = protocol(settings, capture)
        write_json(args.output / "settings.json", settings)
        write_json(args.output / "scoring_protocol.json", frozen)
        rows = []
        for index, response in enumerate(settings["responses"]):
            rows.extend(score_response(reader, index, response, args.output))
            print(f"{response['id']}: cached conditional states saved", flush=True)
        write_csv(args.output / "tokens.csv", rows, list(rows[0]))
        # Annotation access begins only after every response has saved its scores.
        annotations = args.output / "annotations.json"
        if args.annotations is not None:
            write_json(annotations, read_json(args.annotations))
        elif reader.exists("annotations.json"):
            write_json(annotations, reader.json("annotations.json"))
    scoring_seconds = perf_counter() - started
    result = evaluate(args.output, annotations, rows)
    metrics = {method: phases["all_error"] for method, phases in result.get("methods", {}).items()}
    summary = {**frozen, "responses": len(settings["responses"]), "scored_tokens": len(rows),
               "scoring_seconds": scoring_seconds,
               "max_state_mass_error": max(row["state_mass_error"] for row in rows),
               "max_abs_tail_roundoff_correction": max(abs(row["tail_roundoff_correction"]) for row in rows),
               "max_normalizer_roundoff": max(row["normalizer_roundoff"] for row in rows),
               "max_abs_ledger_error": max(row["ledger_error"] for row in rows),
               "evaluation_status": result["status"], "all_error": metrics}
    write_json(args.output / "summary.json", summary)
    return summary


def main(argv=None):
    from .choice_state_pack import pack_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "pack"), default="run",
                        help="run scores/evaluates then packs; pack only archives completed results")
    parser.add_argument("--input", required=True, type=Path, help="Existing run directory or transport-pack ZIP")
    parser.add_argument("--output", required=True, type=Path,
                        help="New result directory for run; existing result directory for pack")
    parser.add_argument("--annotations", type=Path, help="Optional token labels, read after scoring")
    parser.add_argument("--archive", type=Path, help="Default: OUTPUT_review.zip beside the result directory")
    args = parser.parse_args(argv)
    if args.stage == "pack":
        pack_results(args.input, args.output, args.archive, args.annotations)
        return
    result = run(args)
    review = pack_results(args.input, args.output, args.archive)
    print(json.dumps({"output": str(args.output), "responses": result["responses"],
                      "scored_tokens": result["scored_tokens"], "all_error": result["all_error"],
                      "review_archive": review}))
