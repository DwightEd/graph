"""Capture -> train on disjoint sources -> offline token scoring -> AUROC/AP."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
from state_audit.storage import (
    read_arrays,
    read_json,
    start_stage,
    write_arrays,
    write_csv,
    write_json,
)
from tqdm import tqdm

from .comparison import evaluation_headlines
from .comparison_evaluation import evaluate_comparison
from .dynamics_fit import (
    fit_coordinates,
    fit_model,
    infer,
    make_sequence,
    profile_matrix,
    projection_error,
)
from .dynamics_observations import (
    build_observations,
    partition_prompt,
    project_source_covariance,
)
from .filter_evaluation import write_deltas
from .inputs import validate_tokenizer
from .source_regions import region_mask
from .token_detection import source_regions

DIRECTORY = "state_dynamics"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "capture", "fit", "score", "evaluate", "audit", "run"), default="run")
    parser.add_argument("--output", type=Path, required=True, help="Existing support settings/manifest directory")
    parser.add_argument("--reference-output", type=Path, help="Disjoint natural training sources; labels never read by fit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--profile-rank", type=int, default=16)
    parser.add_argument("--choices", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=4)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--dataset", type=Path, help="prepare: official RAGTruth directory")
    parser.add_argument("--model", help="prepare: observer model path")
    parser.add_argument("--exclude-output", type=Path, help="prepare: previously inspected cohort to exclude")
    parser.add_argument("--task", choices=("QA", "Summary", "Data2txt"), default="QA")
    parser.add_argument("--generator", default="llama-2-7b-chat")
    parser.add_argument("--reference-count", type=int, default=64)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(argv)
    if min(args.rank, args.profile_rank, args.block_tokens, args.gradient_batch,
           args.prefill_chunk_size, args.epochs, args.cpu_threads) < 1 or args.choices < 2 or args.learning_rate <= 0:
        parser.error("Dimensions, budgets and learning rate must be positive; choices >= 2")
    if args.stage in ("fit", "score", "run") and args.reference_output is None:
        parser.error("--reference-output is required for source-disjoint fitting/scoring")
    if args.stage == "prepare" and (args.dataset is None or args.model is None or args.exclude_output is None):
        parser.error("prepare requires --dataset, --model and --exclude-output")
    if min(args.reference_count, args.limit) < 1:
        parser.error("Cohort sizes must be positive")
    return args


def prepare(args):
    from state_audit.dataset.jsonl import read_jsonl
    from transformers import AutoTokenizer

    from .validation import prepare_cohort, select_cohorts

    args.selection_seed = args.seed
    sources = {str(row["source_id"]): row for row in read_jsonl(args.dataset / "source_info.jsonl")}
    groups, excluded = select_cohorts(args, read_jsonl(args.dataset / "response.jsonl"), sources)
    plan = {"method": "offline_native_response_dynamics", "model": args.model,
            "seed": args.seed, "labels_used_for_selection": False, "excluded_source_ids": excluded,
            "groups": {name: [str(row["id"]) for row in rows] for name, rows in groups.items()}}
    start_stage(args.output / "dynamics_plan.json", plan, args.resume)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    for name, rows in groups.items():
        prepare_cohort(args, name, rows, sources, tokenizer, args.model, excluded)
    print(json.dumps({"status": "prepared", **plan}))


def capture_contract(args, settings):
    return {"version": 1, "model": settings["model"], "responses": settings["responses"],
            "rank": args.rank, "choices": args.choices, "seed": args.seed,
            "block_tokens": args.block_tokens, "dtype": args.dtype,
            "scope": "current_query_native_derivative_detached_past_KV",
            "state_coordinates": "fixed_orthonormal_projection_of_native_final_hidden",
            "labels_used": False, "interventions": False}


def prepare_capture(args, output, settings, special_ids):
    destination = output / DIRECTORY
    regions = source_regions(output, settings)
    contract = capture_contract(args, settings)
    contract.update(special_ids=list(special_ids), source_scope=regions["status"])
    start_stage(destination / "capture_settings.json", contract, args.resume)
    jobs = []
    for index, response in enumerate(settings["responses"]):
        directory = destination / "capture" / f"{index:04d}"
        mask = region_mask(regions, response)
        groups, blocks = partition_prompt(response, mask, special_ids, args.block_tokens)
        start_stage(directory / "sources.json", {"blocks": blocks, "prompt_groups": groups.tolist(),
                                                 "source_scope": regions["status"]}, args.resume)
        count = len(response["token_ids"]) - response["prompt_length"]
        pending = [i for i in range(count) if not (directory / f"token_{i:06d}.npz").is_file()]
        jobs.append((response, directory, mask, groups, blocks, pending))
    return jobs


def capture_cohorts(args, cohorts):
    from state_audit.model import load_model
    from transformers import AutoTokenizer

    settings = cohorts[0][1]
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], use_fast=True)
    jobs = [job for output, current in cohorts
            for job in prepare_capture(args, output, current, tokenizer.all_special_ids)]
    if any(job[-1] for job in jobs):
        model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        if args.rank > model.native.config.hidden_size or args.choices > model.native.config.vocab_size:
            raise ValueError("rank/choices exceed the native model dimensions")
        for job in tqdm(jobs, desc="native response trajectories"):
            capture_one(args, model, tokenizer, job)
        del model
    for response, directory, mask, _groups, blocks, _pending in jobs:
        if not (directory / "observations.npz").is_file():
            build_observations(response, directory, len(blocks), args.rank, mask, tokenizer.all_special_ids)


def capture_one(args, model, tokenizer, job):
    import torch
    from state_audit.native_response import iter_response_traces

    response, directory, _mask, groups, blocks, pending = job
    if not pending:
        return
    validate_tokenizer(response, tokenizer)
    if model.native.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.native.device)
    iterator = iter_response_traces(
        model, response["token_ids"], response["prompt_length"], pending, groups, len(blocks),
        tokenizer.all_special_ids, rank=args.rank, choices=args.choices, seed=args.seed,
        gradient_batch=args.gradient_batch, prefill_chunk_size=args.prefill_chunk_size,
    )
    started = perf_counter()
    with closing(iterator):
        for target in tqdm(pending, desc=response["id"], leave=False):
            write_arrays(directory / f"token_{target:06d}.npz", **next(iterator))
    peak = torch.cuda.max_memory_allocated(model.native.device) if model.native.device.type == "cuda" else None
    write_json(directory / "timing.json", {"captured_tokens": len(pending), "wall_seconds": perf_counter() - started,
                                          "gradient_batch": args.gradient_batch, "peak_cuda_bytes": peak})


def observations(output, settings):
    return [read_arrays(output / DIRECTORY / "capture" / f"{index:04d}" / "observations.npz")
            for index in range(len(settings["responses"]))]


def build_sequences(output, values, coordinates):
    capture = read_json(output / DIRECTORY / "capture_settings.json")
    result = []
    for index, observation in enumerate(tqdm(values, desc="source covariance")):
        directory = output / DIRECTORY / "capture" / f"{index:04d}"
        sources = read_json(directory / "sources.json")
        projection = coordinates["source_input"]
        covariance = project_source_covariance(directory, len(observation["state"]), len(sources["blocks"]),
                                               capture["rank"], projection["components"], projection["scale"])
        result.append(make_sequence(observation, coordinates, covariance))
    return result


def save_coordinates(path, coordinates):
    arrays = {}
    for name, value in coordinates.items():
        if isinstance(value, dict):
            arrays.update({f"{name}__{key}": np.asarray(item) for key, item in value.items()})
        else:
            arrays[name] = value
    write_arrays(path, **arrays)


def load_coordinates(path):
    coordinates = {}
    for name, value in read_arrays(path).items():
        if "__" in name:
            group, field = name.split("__")
            coordinates.setdefault(group, {})[field] = value
        else:
            coordinates[name] = value
    return coordinates


def train(args, reference, settings):
    destination = reference / DIRECTORY / "model"
    capture = read_json(reference / DIRECTORY / "capture_settings.json")
    protocol = {"capture": capture, "profile_rank": args.profile_rank, "epochs": args.epochs,
                "learning_rate": args.learning_rate, "labels_used": False,
                "initialization": "route_median_only_initializes_E_H; no_label_selected_direction",
                "source_weighting": "equal", "mode_order": ["evidence_update", "history_dominant"]}
    start_stage(destination / "fit_settings.json", protocol, args.resume)
    if args.resume and (destination / "complete.json").is_file():
        return
    started = perf_counter()
    values = observations(reference, settings)
    sources = [response["source_id"] for response in settings["responses"]]
    coordinates = fit_coordinates(values, sources, capture["rank"], args.profile_rank)
    save_coordinates(destination / "coordinates.npz", coordinates)
    sequences = build_sequences(reference, values, coordinates)
    route = "routing_imbalance" if capture["source_scope"] == "available" else "prompt_routing_imbalance"
    parameters, trace = fit_model(sequences, [obs[route] for obs in values], sources,
                                   args.epochs, args.learning_rate, args.fit_device)
    write_arrays(destination / "parameters.npz", **parameters)
    write_csv(destination / "training.csv", trace, ["epoch", "objective"])
    posteriors = [infer(parameters, sequence)["smoothed"] for sequence in sequences]
    write_json(destination / "complete.json", {"labels_used": False, "wall_seconds": perf_counter() - started,
               "best_objective": min(row["objective"] for row in trace),
               "mode_occupancy": np.concatenate(posteriors).mean(0).tolist(),
               "projection_retained_variance": {name: float(coordinates[name]["retained_variance"])
                                                 for name in ("source_input", "history", "profile")}})


def verify_disjoint(target, reference):
    target_sources = {row["source_id"] for row in target["responses"]}
    reference_sources = {row["source_id"] for row in reference["responses"]}
    if target_sources & reference_sources:
        raise ValueError("Training and scoring sources overlap; provide a disjoint reference cohort")
    if target["model"] != reference["model"]:
        raise ValueError("Training and scoring observer models differ")


def score(args, settings):
    destination = args.output / DIRECTORY
    reference = args.reference_output / DIRECTORY
    contract = read_json(destination / "capture_settings.json")
    trained = read_json(reference / "model" / "fit_settings.json")["capture"]
    for field in ("model", "rank", "choices", "seed", "block_tokens", "dtype", "source_scope", "special_ids"):
        if contract[field] != trained[field]:
            raise ValueError(f"Training/scoring capture mismatch: {field}")
    coordinates = load_coordinates(reference / "model" / "coordinates.npz")
    parameters = read_arrays(reference / "model" / "parameters.npz")
    values = observations(args.output, settings)
    sequences = build_sequences(args.output, values, coordinates)
    prefix = "" if contract["source_scope"] == "available" else "prompt_"
    methods = ("state_dynamics", "state_forward", prefix + "routing_imbalance",
               prefix + "attention_displacement", "route_offline_mean", "entropy", "position")
    protocol = {"purpose": "offline_native_response_dynamics", "primary_method": methods[0],
                "methods": {name: name for name in methods}, "reference_output": str(args.reference_output.resolve()),
                "labels_used_for_scoring": False, "labels_used_for_training": False, "future_tokens_used": True,
                "gradients_during_capture": True, "model_forward_during_scoring": False, "interventions": False,
                "cohort": settings.get("cohort", {}), "source_scope": contract["source_scope"],
                "risk_definition": "history_dominant_mode_posterior_not_truth_probability",
                "mode_structure": "joint_head_profile_emission_and_heteroscedastic_input_driven_dynamics",
                "forward_control": "forward_messages_with_offline_FAI_features; not_an_online_detector",
                "smoothing_control": "route_offline_mean: current, 7 earlier and 8 later rows; truncate at answer edges",
                "gradient_scope": contract["scope"], "automatic_method_selection": False}
    write_json(destination / "scoring_protocol.json", protocol)
    rows, audits, errors = [], [], []
    for index, (response, observation, sequence) in enumerate(zip(settings["responses"], values, sequences)):
        posterior = infer(parameters, sequence, args.fit_device)
        arrays = {name: observation[name] for name in ("target", "query", "token_id", methods[2], methods[3], "entropy")}
        arrays.update(state_dynamics=posterior["smoothed"][:, 1], state_forward=posterior["filtered"][:, 1],
                      mode_posterior=posterior["smoothed"], mode_emission=posterior["emission"],
                      position=np.arange(len(sequence["state"])) / max(len(sequence["state"]) - 1, 1))
        positions = np.arange(len(sequence["state"]))
        left, right = np.maximum(positions - 7, 0), np.minimum(positions + 9, len(positions))
        route_sum = np.r_[0., np.cumsum(arrays[methods[2]])]
        arrays["route_offline_mean"] = (route_sum[right] - route_sum[left]) / (right - left)
        arrays["risk"] = arrays["state_dynamics"]
        write_arrays(destination / "responses" / f"{index:04d}" / "scores.npz", **arrays)
        rows.extend(score_rows(response, arrays, methods))
        audits.extend(observation_rows(response, observation, posterior))
        errors.append({"response_id": response["id"], **{
            name: projection_error(profile_matrix(observation) if name == "profile" else observation[name], coordinates[name])
            for name in ("source_input", "history", "profile")}})
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    write_csv(destination / "observations.csv", audits, list(audits[0]))
    write_json(destination / "projection_error.json", {"meaning": "held_out_standardized_energy_outside_training_subspace",
                                                     "responses": errors})
    return finish(args.output, args.annotations, protocol, rows)


def observation_rows(response, observation, posterior):
    from scipy.special import xlogy

    profile = observation["profile"]
    channels = list(observation["profile_channels"])
    names = ("source_read", "source_read_change", "source_response_entropy",
             "source_direction_coherence", "source_direction_variance", "future_mean_attention", "future_mean_response")
    entropy = -xlogy(posterior["smoothed"], posterior["smoothed"]).sum(-1)
    result = []
    for target, row in enumerate(profile):
        values = {name: float(row[..., channels.index(name)].mean()) for name in names}
        result.append({"response_id": response["id"], "target": target, **values,
                       "mode_entropy": float(entropy[target]), "future_query_count": int(observation["future_query_count"][target]),
                       "risk": float(posterior["smoothed"][target, 1])})
    return result


def score_rows(response, arrays, methods):
    return [{"response_id": response["id"], "source_id": response["source_id"], "target": target,
             "query": int(arrays["query"][target]), "token_id": int(arrays["token_id"][target]),
             "token": response["token_text"][response["prompt_length"] + target],
             **{name: float(arrays[name][target]) for name in (*methods, "risk")}}
            for target in range(len(arrays["token_id"]))]


def finish(output, annotations, protocol, rows):
    from .dynamics_report import write_report

    destination = output / DIRECTORY
    annotations = annotations or output / "annotations.json"
    evaluation = evaluate_comparison(output, destination, annotations, protocol["methods"], rows)
    baseline = list(protocol["methods"])[2]
    comparison = write_deltas(output, destination, annotations, evaluation, baseline, protocol["methods"], rows,
                              pairs=[("state_dynamics", baseline), ("state_dynamics", "route_offline_mean"),
                                     ("state_dynamics", "state_forward")])
    summary = {**protocol, "responses": len({row["response_id"] for row in rows}), "scored_tokens": len(rows),
               "evaluation": evaluation_headlines(evaluation), "comparisons": comparison}
    risks = np.asarray([row["state_dynamics"] for row in rows])
    summary["score_distribution"] = {"min": float(risks.min()), "max": float(risks.max()),
                                     "std": float(risks.std()), "history_mode_mean": float(risks.mean())}
    write_json(destination / "summary.json", summary)
    write_report(output, rows, protocol, evaluation)
    return summary


def main(argv=None):
    import torch
    from threadpoolctl import threadpool_limits

    args = arguments(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.stage == "prepare":
        prepare(args)
        return
    settings = read_json(args.output / "settings.json")
    if args.stage == "audit":
        from .dynamics_audit import audit

        with threadpool_limits(limits=args.cpu_threads):
            print(json.dumps(audit(args.output, args.reference_output, args.annotations)))
        return
    if args.stage == "evaluate":
        protocol = read_json(args.output / DIRECTORY / "scoring_protocol.json")
        rows = [row for index, response in enumerate(settings["responses"])
                for row in score_rows(response, read_arrays(args.output / DIRECTORY / "responses" / f"{index:04d}" / "scores.npz"), protocol["methods"])]
        print(json.dumps(finish(args.output, args.annotations, protocol, rows)))
        return
    cohorts = [(args.output, settings)]
    if args.reference_output is not None:
        reference = read_json(args.reference_output / "settings.json")
        verify_disjoint(settings, reference)
        cohorts.insert(0, (args.reference_output, reference))
    with threadpool_limits(limits=args.cpu_threads):
        if args.stage in ("capture", "run"):
            capture_cohorts(args, cohorts)
        if args.stage in ("fit", "run"):
            train(args, args.reference_output, reference)
        if args.stage in ("score", "run"):
            print(json.dumps(score(args, settings)))


if __name__ == "__main__":
    main()
