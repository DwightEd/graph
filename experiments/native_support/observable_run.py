"""Capture native distribution transport, score a structured reference, and evaluate."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, start_stage, write_arrays, write_csv, write_json

from .choice_cache import CaptureReader
from .observable_pack import pack
from .inputs import validate_tokenizer
from .observable_state import (
    CHANNELS, CONTEXT, ROLES, context_features, observed_transitions, score_queries, state_features,
)

VERSION = 1
METHODS = ("transport_conditional", "transport_context", "observable_route", "raw_route",
           "route_offline_mean", "raw_attention", "entropy")


def capture_protocol(args, settings):
    return dict(version=VERSION, model=settings["model"], responses=settings["responses"],
                rank=args.rank, seed=args.seed, chunk_tokens=args.chunk_tokens, dtype=args.dtype,
                purpose="native_output_observability_conditional_detection_candidate",
                derivative="full_context_native_QK_RMS_SwiGLU; independent_query_VJP",
                metric="full_vocabulary_categorical_Fisher_random_sign_sketch",
                objective="distribution_response; no_selected_token_or_semantic_candidates",
                roles=ROLES, channels=CHANNELS, context=CONTEXT,
                interventions=False, labels_used_for_capture=False,
                compression="fixed_random_output_probes; not_trained_SVD",
                spectral_scope="output_conditioned_pullback_Gram; not_full_J_eigenspectrum",
                prediction_alignment="query=P+t-1; native_causal_mask",
                original_scores_modified=False)


def prepare(args):
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        captured = reader.json("value_transport/capture_settings.json")
        if captured["model"] != settings["model"] or captured["responses"] != settings["responses"]:
            raise ValueError("Input value capture/settings disagree")
        if captured["source_scope"] != "available":
            raise ValueError("Observable source transport requires saved prompt source regions")
        reference_settings = read_json(args.reference / "settings.json") if args.reference else settings
        source_ids = {response["source_id"] for response in reference_settings["responses"]}
        if any(len(source_ids - {response["source_id"]}) < 2 for response in settings["responses"]):
            raise ValueError("Need two other reference sources per answer; supply >=3 sources or --reference")
        start_stage(args.output / "protocol.json", capture_protocol(args, settings), args.resume)
        write_json(args.output / "settings.json", settings)
        for index, response in enumerate(settings["responses"]):
            sources = reader.json(f"value_transport/capture/{index:04d}/sources.json")
            if len(sources["group_ids"]) != len(response["token_ids"]):
                raise ValueError(f"{response['id']}: source groups do not match original tokens")
            write_json(args.output / "responses" / f"{index:04d}" / "sources.json", sources)
    return settings


def capture(args, settings):
    import torch
    from state_audit.model import load_model
    from state_audit.observable_transport import iter_observable_transport

    model, tokenizer = None, None
    for index, response in enumerate(tqdm(settings["responses"], desc="observable answers")):
        directory = args.output / "responses" / f"{index:04d}"
        source = read_json(directory / "sources.json")
        count = len(response["token_ids"]) - response["prompt_length"]
        pending = [target for target in range(count) if not (directory / f"token_{target:06d}.npz").exists()]
        if not pending:
            continue
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        validate_tokenizer(response, tokenizer)
        for offset in tqdm(range(0, len(pending), args.chunk_tokens), desc=response["id"], leave=False):
            targets = pending[offset:offset + args.chunk_tokens]
            started = perf_counter()
            if model.native.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.native.device)
            iterator = iter_observable_transport(model, response["token_ids"], response["prompt_length"],
                targets, source["group_ids"], len(source["blocks"]), args.rank, args.seed)
            with closing(iterator):
                for row in iterator:
                    write_arrays(directory / f"token_{int(row['target']):06d}.npz", **row)
            peak = None
            if model.native.device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(model.native.device)
            write_json(directory / f"timing_{targets[0]:06d}.json", dict(targets=targets,
                seconds=perf_counter() - started, peak_cuda_bytes=peak))


def read_answer(directory, response):
    source_count = len(read_json(directory / "sources.json")["blocks"])
    count = len(response["token_ids"]) - response["prompt_length"]
    features, rows = [], []
    for target in range(count):
        row = read_arrays(directory / f"token_{target:06d}.npz")
        identity = (int(row["target"]), int(row["query"]), int(row["token_id"]))
        position = response["prompt_length"] + target
        expected = (target, position - 1, response["token_ids"][position])
        if identity != expected:
            raise ValueError(f"{response['id']}/{target}: capture alignment differs")
        features.append(state_features(row, source_count))
        rows.append({name: float(row[name].mean()) for name in ("entropy", "raw_route", "raw_attention")})
    feature = {name: np.stack([item[name] for item in features]) for name in features[0]}
    feature.update({name: np.asarray([item[name] for item in rows]) for name in rows[0]})
    feature["context"] = context_features(
        feature["entropy"], np.arange(count), response["prompt_length"], source_count,
    )
    feature.update(source=np.repeat(response["source_id"], count), response=np.repeat(response["id"], count),
                   target=np.arange(count), previous=np.arange(count) - 1)
    return feature


def load_features(root):
    settings = read_json(root / "settings.json")
    return settings, [read_answer(root / "responses" / f"{index:04d}", response)
                      for index, response in enumerate(settings["responses"])]


def join_reference(features, excluded_source):
    selected = [feature for feature in features if feature["source"][0] != excluded_source]
    if len({feature["source"][0] for feature in selected}) < 2:
        raise ValueError("Need two reference sources after source exclusion; use >=3 sources or --reference")
    previous, offset = [], 0
    for feature in selected:
        previous.append(np.where(feature["previous"] >= 0, feature["previous"] + offset, -1))
        offset += len(feature["target"])
    result = {name: np.concatenate([feature[name] for feature in selected]) for name in selected[0]}
    result["previous"] = np.concatenate(previous)
    return result


def window_mean(values):
    positions = np.arange(len(values))
    left, right = np.maximum(positions - 7, 0), np.minimum(positions + 9, len(values))
    cumulative = np.r_[0., np.cumsum(values)]
    return (cumulative[right] - cumulative[left]) / (right - left)


def score(args, settings):
    _, observations = load_features(args.output)
    references = observations
    if args.reference:
        contract = read_json(args.reference / "protocol.json")
        current = read_json(args.output / "protocol.json")
        fields = ("version", "model", "rank", "seed", "dtype", "roles", "channels")
        if any(contract[key] != current[key] for key in fields):
            raise ValueError("Reference must use the same model, probe coordinates and capture protocol")
        _, references = load_features(args.reference)
    protocol = dict(reference=str(args.reference) if args.reference else "source_held_out_input_cohort",
                    neighbors_per_source=args.neighbors, seed=args.seed, labels_used=False,
                    candidate="transport_conditional", default_risk="raw_route; candidate_not_promoted",
                    fitting="unlabelled_reference_context_scales_and_kernel_bandwidth",
                    score="negative_log_conditional_kernel_affinity; not_hallucination_probability",
                    future_tokens_used=True, future_scope="other_reference_answers_and_offline_mean",
                    automatic_method_selection=False)
    start_stage(args.output / "score_settings.json", protocol, args.resume)
    rows, calibrations = [], {}
    jobs = zip(settings["responses"], observations)
    for index, (response, feature) in enumerate(tqdm(jobs, total=len(observations), desc="conditional states")):
        reference = join_reference(references, response["source_id"])
        values, neighbors, calibration = score_queries(
            feature, reference, args.neighbors, args.score_device, args.seed,
        )
        for name in ("observable_route", "raw_route", "raw_attention", "entropy",
                     "source_read_mass", "source_concentration"):
            values[name] = feature[name]
        values["route_offline_mean"] = window_mean(values["raw_route"])
        values.update(observed_transitions(feature))
        values["risk"] = values["raw_route"]
        values["target"] = feature["target"]
        values["token_id"] = np.asarray(response["token_ids"][response["prompt_length"]:])
        save_answer(args.output / "responses" / f"{index:04d}", feature, values, neighbors)
        rows.extend(token_rows(response, values))
        calibrations[response["id"]] = calibration
    write_json(args.output / "reference_fit.json", calibrations)
    write_csv(args.output / "tokens.csv", rows, list(rows[0]))
    return rows


def save_answer(directory, features, scores, neighbors):
    write_arrays(directory / "scores.npz", **scores)
    write_arrays(directory / "state.npz", matrix=features["matrix"], reading=features["reading"],
                 context=features["context"], response_scale=features["response_scale"])
    write_json(directory / "neighbors.json", neighbors)


def token_rows(response, scores):
    rows = []
    for target in range(len(scores["token_id"])):
        row = dict(response_id=response["id"], source_id=response["source_id"], target=target,
                   token=response["token_text"][response["prompt_length"] + target])
        row.update({name: float(scores[name][target]) for name in METHODS})
        rows.append(row)
    return rows


def finish(args, settings, rows):
    from .comparison_evaluation import evaluate_comparison
    from .comparison_deltas import write_deltas
    from .observable_report import write_diagnostics

    write_diagnostics(args.output, settings)
    # Labels become visible only after every score has been saved.
    annotation = args.annotations or args.output / "annotations.json"
    if args.annotations:
        (args.output / "annotations.json").write_bytes(args.annotations.read_bytes())
    elif args.input:
        with closing(CaptureReader(args.input)) as reader:
            if reader.exists("annotations.json"):
                annotation.write_bytes(reader.bytes("annotations.json"))
    methods = {name: name for name in METHODS}
    evaluation = evaluate_comparison(args.output, args.output, annotation, methods, rows)
    controls = ("raw_route", "route_offline_mean", "transport_context")
    pairs = [("transport_conditional", control) for control in controls]
    comparisons = write_deltas(args.output, args.output, annotation, evaluation, methods, rows, pairs)
    summary = dict(purpose="native_observable_transport_detection_candidate",
                   responses=len(settings["responses"]), scored_tokens=len(rows),
                   candidate_method="transport_conditional", primary_baseline="raw_route",
                   evaluation=evaluation, comparisons=comparisons, cohort=settings.get("cohort", {}))
    write_json(args.output / "summary.json", summary)
    return dict(output=str(args.output), scored_tokens=len(rows), evaluation_status=evaluation["status"],
                all_error={name: value["all_error"] for name, value in evaluation.get("methods", {}).items()},
                **pack(args.output, args.archive, args.pack_mode))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Original value-path directory or compact ZIP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Separate observable capture directory; same model/probes")
    parser.add_argument("--stage", choices=("run", "capture", "score", "evaluate", "pack"), default="run")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--pack-mode", choices=("light", "full"), default="light",
                        help="Light review excludes raw token captures/state matrices; no rescoring")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.rank, args.chunk_tokens, args.neighbors, args.cpu_threads) < 1:
        parser.error("rank, chunk-tokens, neighbors and cpu-threads must be positive")
    if args.stage in ("run", "capture") and args.input is None:
        parser.error("--input is required for capture")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps(pack(args.output, args.archive, args.pack_mode)))
        return
    import torch
    torch.set_num_threads(args.cpu_threads)
    if args.stage in ("run", "capture"):
        settings = prepare(args)
        capture(args, settings)
    else:
        settings = read_json(args.output / "settings.json")
    if args.stage == "capture":
        print(json.dumps(dict(status="captured", output=str(args.output))))
        return
    if args.stage in ("run", "score"):
        rows = score(args, settings)
    else:
        rows = []
        for index, response in enumerate(settings["responses"]):
            values = read_arrays(args.output / "responses" / f"{index:04d}" / "scores.npz")
            rows.extend(token_rows(response, values))
    print(json.dumps(finish(args, settings, rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
