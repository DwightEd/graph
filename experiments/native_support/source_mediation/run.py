"""Freeze a finite-interaction test before evaluation; native KV games are separate."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, start_stage, write_arrays, write_json

from ..choice_cache import CaptureReader
from ..evidence_contrast.aggregation import aggregate_scores
from ..evidence_contrast.data import copy_annotations
from ..evidence_contrast.run import pack
from ..evidence_contrast.unit_report import evaluate_units
from .cached import score_cached
from .decomposition import BASE_TOKENS, CACHED_TOKENS, NATIVE_TOKENS, add_unit_scores, native_scores


def prepare(args):
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        input_protocol = reader.json("protocol.json")
        if args.mode == "cache" and input_protocol["version"] not in ("message-carriers-v1", "message-carriers-token-v2"):
            raise ValueError("The cached experiment requires complete carrier v1/v2 finite effects")
        protocol = dict(version="source-mediation-v1", mode=args.mode, input=str(args.input.resolve()),
            input_version=input_protocol["version"], model=args.model or settings["model"],
            primary_candidate="gate_source_shapley" if args.mode == "cache" else "mediated_direct",
            dtype=args.dtype, save_heads=args.save_heads, identity_tolerance=args.identity_tolerance,
            labels_used_for_scoring=False, labels_used_for_training=False, automatic_selection=False,
            score_direction="negative_source_effect; absolute_interactions_are_separate_hypotheses",
            facts="available_source; no_claim_of_semantic_binding_or_truth_identification",
            history_tokens="fixed_observed_text; no_discrete_generation_mediation",
            smoothing="only_independent_saved_unit_mean_controls; no_graph_risk_propagation",
            native_game="prompt_state_and_source_access x strictly_previous_answer_KV_donor",
            native_positions="original_positions; source_keys_blocked_globally_in_null_donor",
            native_queries="independent_batched_queries; own_self_KV_recomputed; no_hybrid_future_keys",
            cache_game="source_present_absent x selected_history_gates_keep_joint_cut",
            coalition="joint_effect_minus_sum_of_single_effects; no_individual_higher_order_terms_identified",
            outcome="observed_token_logp; nonadditivity_includes_logsoftmax_curvature",
            empty_coalition="zero_by_definition; selected_edge_count_marks_unintervened_tokens",
            sham="joint_endpoint_control_only; matched_single_sham_effects_unavailable",
            sources_verified_independent=False)
        start_stage(args.output / "protocol.json", protocol, args.resume)
        start_stage(args.output / "settings.json", settings, args.resume)
        for index, response in enumerate(settings["responses"]):
            directory = f"responses/{index:04d}"
            views = reader.json(directory + "/views.json")
            scores = reader.arrays(directory + "/scores.npz")
            scores.update(aggregate_scores(scores, views))
            if not np.array_equal(scores["token_id"], views["answer_ids"]):
                raise ValueError("Input token identities differ")
            start_stage(args.output / directory / "views.json", views, args.resume)
            write_arrays(args.output / directory / "baselines.npz", **scores)
    return settings, protocol


def capture(args, settings, protocol):
    import torch
    from state_audit.model import load_model
    from ..inputs import validate_tokenizer
    from .native import measure

    torch.set_num_threads(args.cpu_threads)
    model = None
    for index, response in enumerate(tqdm(settings["responses"], desc="native source/history worlds")):
        directory = args.output / "responses" / f"{index:04d}"
        if (directory / "capture_complete.json").exists():
            continue
        if model is None:
            model, tokenizer = load_model(protocol["model"], device=args.device, dtype=protocol["dtype"])
        validate_tokenizer(response, tokenizer)
        started = perf_counter()
        values, heads = measure(model, read_json(directory / "views.json"),
            args.prefill_chunk_size, args.query_chunk_size, protocol["save_heads"])
        write_arrays(directory / "worlds.npz", **values)
        error = float(values["identity_max_error"].max())
        if not np.isfinite(values["worlds"]).all() or not np.isfinite(error) or error > protocol["identity_tolerance"]:
            raise ValueError(f"{response['id']}: native replay identity error {error}; measurements saved, not scored")
        if heads:
            write_arrays(directory / "head_readouts.npz", **heads)
        write_json(directory / "capture_complete.json", dict(seconds=perf_counter()-started,
            identity_max_error=error, query_chunk_size=args.query_chunk_size,
            prefill_chunk_size=args.prefill_chunk_size, device=args.device))


def freeze_scores(args, settings, protocol):
    candidates = CACHED_TOKENS if protocol["mode"] == "cache" else NATIVE_TOKENS
    with closing(CaptureReader(Path(protocol["input"]))) as reader:
        for index, response in enumerate(tqdm(settings["responses"], desc="freeze interaction scores")):
            relative = f"responses/{index:04d}"
            directory = args.output / relative
            views = read_json(directory / "views.json")
            if protocol["mode"] == "cache":
                scores, components, edges = score_cached(reader, relative, views,
                    protocol["input_version"] == "message-carriers-token-v2")
                write_arrays(directory / "head_effects.npz", **edges)
            else:
                read_json(directory / "capture_complete.json")
                values = read_arrays(directory / "worlds.npz")
                if not np.array_equal(values["token_id"], views["answer_ids"]):
                    raise ValueError("Native observation token identities differ")
                scores, components = native_scores(values["worlds"])
            if not all(np.isfinite(value).all() for value in scores.values()):
                raise ValueError("Incomplete/nonfinite candidate scores")
            baseline = read_arrays(directory / "baselines.npz")
            baseline.update(scores)
            add_unit_scores(baseline, views["units"], candidates)
            write_arrays(directory / "scores.npz", **baseline)
            write_arrays(directory / "components.npz", **components)
    write_json(args.output / "coverage.json", dict(status="complete", answers=len(settings["responses"]),
        tokens=sum(len(row["token_ids"])-row["prompt_length"] for row in settings["responses"]),
        labels_used_for_scoring=False, dropped_tokens=0))


def finish(args, settings, protocol):
    from .report import diagnostics

    coverage = read_json(args.output / "coverage.json")
    candidates = CACHED_TOKENS if protocol["mode"] == "cache" else NATIVE_TOKENS
    tokens = (*candidates, *BASE_TOKENS)
    methods = (*tokens, *(name + "_unit_mean" for name in tokens), "raw_route_offline_mean")
    copy_annotations(args.output, Path(protocol["input"]), args.annotations)
    comparisons = [(protocol["primary_candidate"], name) for name in ("raw_route", "source_local_unit_mean")]
    result = evaluate_units(args.output, settings, methods, tokens, comparisons)
    analysis = diagnostics(args.output, settings, protocol)
    write_json(args.output / "diagnostics.json", analysis)
    summary = dict(output=str(args.output), status=result["status"], protocol=protocol, coverage=coverage,
        all_error={name: {key: phases["all_error"][key] for key in ("auroc", "ap")}
                   for name, phases in result.get("methods", {}).items()}, diagnostics=analysis)
    write_json(args.output / "summary.json", summary)
    return dict(**summary, review_archive=pack(args.output))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed contrast/carrier directory or ZIP; cache mode requires carriers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cache", "native"), default="cache")
    parser.add_argument("--stage", choices=("run", "capture", "score", "evaluate", "pack"), default="run")
    parser.add_argument("--model", help="Observer path override, recorded and tokenizer-checked")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-heads", action="store_true", help="Save four [token,layer,head,width] readout arrays")
    parser.add_argument("--identity-tolerance", type=float, default=.05, help="Maximum native diagonal logp error in nats")
    parser.add_argument("--annotations", type=Path, help="Evaluation only, after all scores are saved")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("run", "capture") and args.input is None:
        parser.error("--input is required for run/capture")
    if args.stage == "capture" and args.mode != "native":
        parser.error("capture requires --mode native")
    if min(args.prefill_chunk_size, args.query_chunk_size, args.cpu_threads) < 1:
        parser.error("chunks and threads must be positive")
    if not np.isfinite(args.identity_tolerance) or args.identity_tolerance <= 0:
        parser.error("identity-tolerance must be finite and positive")
    if args.input and args.output.resolve() == args.input.resolve():
        parser.error("Use a separate output; input measurements are immutable")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps(dict(review_archive=pack(args.output))))
        return
    if args.stage in ("run", "capture") or (args.stage == "score" and args.input):
        settings, protocol = prepare(args)
    else:
        settings, protocol = [read_json(args.output / f"{name}.json") for name in ("settings", "protocol")]
    if args.stage in ("run", "capture") and protocol["mode"] == "native":
        capture(args, settings, protocol)
    if args.stage == "capture":
        print(json.dumps(dict(status="captured", output=str(args.output))))
        return
    if args.stage in ("run", "score"):
        freeze_scores(args, settings, protocol)
    if args.stage == "score":
        print(json.dumps(dict(status="scored", output=str(args.output))))
        return
    print(json.dumps(finish(args, settings, protocol), ensure_ascii=False))


if __name__ == "__main__":
    main()
