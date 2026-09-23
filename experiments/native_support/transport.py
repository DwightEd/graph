"""Native cache -> source response graph -> budget state -> token evaluation.

Source units are automatic text blocks, not manually labelled evidence types.
The graph estimates message budgets; it does not assign truth to hidden states.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
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
from .dynamics import capture_cohorts, score_rows
from .filter_evaluation import write_deltas
from .source_regions import region_mask
from .token_detection import source_regions
from .transport_graph import build_graph
from .transport_observations import load_observations
from .transport_state import budget_risk, infer_budget

DIRECTORY = "source_transport"
METHODS = ("transport_route", "raw_route", "route_offline_mean", "raw_attention", "entropy")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("capture", "score", "evaluate", "audit", "run"), default="score")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1., help="Fixed graph precision; zero recovers raw routing")
    parser.add_argument("--score-device", default="cpu", help="Tensor graph computation: cpu or cuda:0")
    parser.add_argument("--head-batch", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--annotations", type=Path, help="Evaluation only; scoring never reads labels")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000, help="Audit only: paired source resamples; zero skips intervals")
    parser.add_argument("--device", default="cuda:0", help="Native capture device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--rank", type=int, default=8, help="Capture only: common native response directions")
    parser.add_argument("--choices", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=4)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--resume", action="store_true", help="Resume an identical native capture")
    args = parser.parse_args(argv)
    if not np.isfinite(args.strength) or args.strength < 0:
        parser.error("--strength must be finite and nonnegative")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be nonnegative")
    if min(args.head_batch, args.cpu_threads, args.rank, args.block_tokens,
           args.gradient_batch, args.prefill_chunk_size) < 1 or args.choices < 2:
        parser.error("Budgets must be positive and --choices must be at least two")
    return args


def scoring_protocol(args, settings, capture):
    return {
        "purpose": "annotation_free_source_response_budget_state",
        "candidate_method": "transport_route", "primary_baseline": "raw_route",
        "methods": {name: name for name in METHODS}, "strength": args.strength,
        "score_device": args.score_device, "head_batch": args.head_batch,
        "cpu_threads": args.cpu_threads,
        "source_units": "automatic_prompt_blocks", "semantic_types": "unassigned",
        "manual_evidence_annotations_required": False, "labels_used_for_scoring": False,
        "labels_used_for_training": False, "parameter_fitting": False, "svd": False,
        "future_tokens_used": True, "model_forward_during_scoring": False,
        "interventions": False, "automatic_method_selection": False,
        "risk_definition": "history_minus_source_budget_share_after_graph_conditioning",
        "state_model": "Gaussian_field_on_head_and_source_message_budgets",
        "state_graph": "shared_token_graph; head_identity_retained_before_edge_aggregation",
        "graph": "per_head_native_response_similarity_times_observed_history_attention",
        "edge_alignment": "key=P+j -> query_state=j+1; predictor self excluded",
        "bandwidth": "per_head_offline_median_positive_adjacent_squared_distance",
        "distance_channels": ["source_response_direction", "sqrt_group_attention", "ffn_direction"],
        "distance_weights": [1., 1., 1.], "prediction_alignment": "query=P+t-1",
        "source_scope": capture["source_scope"], "gradient_scope": capture["scope"],
        "readout_scope": "routing_candidate_not_semantic_conflict_probability",
        "missingness": "unavailable_future_is_no_edge_not_zero_observation",
        "cohort": settings.get("cohort", {}),
        "capture": {name: capture[name] for name in
                    ("model", "rank", "choices", "seed", "block_tokens", "dtype")},
    }


def offline_mean(values):
    positions = np.arange(len(values))
    left = np.maximum(positions - 7, 0)
    right = np.minimum(positions + 9, len(values))
    cumulative = np.r_[0., np.cumsum(values)]
    return (cumulative[right] - cumulative[left]) / (right - left)


def score_answer(observation, source_count, args):
    graph = build_graph(observation, source_count, args.score_device, args.head_batch)
    state = infer_budget(observation["routing_budget"], graph["edge_weight"], args.strength)
    raw = observation["raw_routing"]
    risk = budget_risk(state["inferred_budget"], source_count + 1)
    scores = {name: observation[name] for name in ("target", "query", "token_id", "entropy")}
    scores.update(transport_route=risk, raw_route=raw, route_offline_mean=offline_mean(raw),
                  raw_attention=observation["raw_attention"], risk=risk)
    return scores, graph, state


def save_answer(directory, scores, graph, state, observation, blocks):
    write_arrays(directory / "scores.npz", **scores)
    write_arrays(directory / "state.npz", **state, **graph,
                 observed_budget=observation["routing_budget"],
                 future_query_count=observation["future_query_count"],
                 future_observed=observation["future_observed"])
    # Token-local contrasts retain candidate IDs; they are not compared across rows.
    write_arrays(directory / "choices.npz", candidate_ids=observation["candidate_ids"],
                 candidate_logits=observation["candidate_logits"],
                 choice_contrast=observation["choice_contrast"],
                 ffn_choice_contrast=observation["ffn_choice_contrast"])
    write_json(directory / "sources.json", {"blocks": blocks, "semantic_types": "unassigned"})


def score(args, settings):
    source = args.output / "state_dynamics"
    destination = args.output / DIRECTORY
    capture = read_json(source / "capture_settings.json")
    if capture["model"] != settings["model"] or capture["responses"] != settings["responses"]:
        raise ValueError("Native capture differs from current settings; use the matching output directory")
    regions = source_regions(args.output, settings)
    if regions["status"] != capture["source_scope"]:
        raise ValueError("Native source scope differs from source_regions.json")
    protocol = scoring_protocol(args, settings, capture)
    write_json(destination / "scoring_protocol.json", protocol)
    rows, timings = [], []
    for index, response in enumerate(tqdm(settings["responses"], desc="source transport")):
        started = perf_counter()
        observation, blocks = load_observations(
            response, source / "capture" / f"{index:04d}", capture["rank"],
            capture["special_ids"], region_mask(regions, response),
        )
        scores, graph, state = score_answer(observation, len(blocks), args)
        save_answer(destination / "responses" / f"{index:04d}", scores, graph, state, observation, blocks)
        rows.extend(score_rows(response, scores, METHODS))
        timings.append({"response_id": response["id"], "tokens": len(scores["risk"]),
                        "seconds": perf_counter() - started, "edges": int(np.count_nonzero(graph["edge_weight"]))})
        del observation, graph, state
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    write_csv(destination / "timing.csv", timings, list(timings[0]))
    return finish(args.output, args.annotations, protocol, rows)


def finish(output, annotations, protocol, rows):
    from .transport_report import write_report

    destination = output / DIRECTORY
    annotations = annotations or output / "annotations.json"
    methods = protocol["methods"]
    evaluation = evaluate_comparison(output, destination, annotations, methods, rows)
    comparison = write_deltas(output, destination, annotations, evaluation, "raw_route", methods, rows,
                              pairs=[("transport_route", "raw_route"), ("transport_route", "route_offline_mean")])
    summary = {**protocol, "responses": len({row["response_id"] for row in rows}),
               "scored_tokens": len(rows), "evaluation": evaluation_headlines(evaluation),
               "comparisons": comparison}
    write_json(destination / "summary.json", summary)
    write_report(destination, rows, evaluation)
    return summary


def evaluate(args, settings):
    destination = args.output / DIRECTORY
    protocol = read_json(destination / "scoring_protocol.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        scores = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(score_rows(response, scores, protocol["methods"]))
    return finish(args.output, args.annotations, protocol, rows)


def main(argv=None):
    import torch
    from threadpoolctl import threadpool_limits

    args = arguments(argv)
    torch.set_num_threads(args.cpu_threads)
    settings = read_json(args.output / "settings.json")
    with threadpool_limits(limits=args.cpu_threads):
        if args.stage in ("capture", "run"):
            capture_cohorts(args, [(args.output, settings)], build_profiles=False)
        if args.stage in ("score", "run"):
            print(json.dumps(score(args, settings)))
        elif args.stage == "evaluate":
            print(json.dumps(evaluate(args, settings)))
        elif args.stage == "audit":
            from .transport_audit import audit

            print(json.dumps(audit(args.output, args.annotations, args.bootstrap_replicates, args.seed)))


if __name__ == "__main__":
    main()
