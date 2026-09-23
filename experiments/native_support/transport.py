"""Native value-path capture -> signed provenance -> token AUROC/AP.

The previous shared-budget graph is retired. Its saved results remain auditable
with --stage audit; all new capture and scores live in value_transport/.
"""

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
from .comparison_deltas import write_deltas
from .comparison_evaluation import evaluate_comparison
from .dynamics_observations import partition_prompt
from .inputs import validate_tokenizer
from .source_regions import region_mask
from .token_detection import source_regions
from .transport_readout import METHODS, score_answer, source_rows, token_rows

DIRECTORY = "value_transport"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("capture", "score", "evaluate", "run", "audit"), default="score")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--choices", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--gradient-batch", type=int, default=1, help="Independent competitor objectives per VJP batch")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--annotations", type=Path, help="Evaluation only")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000, help="Old saved budget audit only")
    parser.add_argument("--seed", type=int, default=37, help="Old saved budget audit only")
    args = parser.parse_args(argv)
    if args.choices < 2 or min(args.block_tokens, args.gradient_batch, args.cpu_threads) < 1:
        parser.error("choices >= 2; block-tokens, gradient-batch and cpu-threads must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be nonnegative")
    return args


def root_groups(response, regions, special_ids, block_tokens):
    groups, blocks = partition_prompt(response, regions, special_ids, block_tokens)
    count = len(blocks)
    groups[groups == count + 2] = count + 1
    groups[groups == count + 3] = count + 2
    tail = np.full(len(response["token_ids"]) - len(groups), count, dtype=int)
    groups = np.r_[groups, tail]
    groups[np.isin(response["token_ids"], special_ids)] = count + 2
    return groups, blocks


def capture_contract(args, settings, regions, special_ids):
    return {"version": 1, "method": "native_value_path_provenance", "model": settings["model"],
        "responses": settings["responses"], "source_scope": regions["status"],
        "choices": args.choices, "block_tokens": args.block_tokens, "dtype": args.dtype,
        "special_ids": list(special_ids), "attention": "native_patterns_fixed_for_attribution",
        "normalization": "native_RMS_scale_fixed_for_attribution",
        "mlp": "SiLU_secant_and_equal_product_relevance",
        "history": "full_prefix_value_paths_connected; discrete_token_choices_are_conditioned_inputs",
        "labels_used": False, "interventions": False}


def capture(args, settings):
    from state_audit.model import load_model
    from transformers import AutoTokenizer

    destination = args.output / DIRECTORY
    regions = source_regions(args.output, settings)
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], use_fast=True)
    contract = capture_contract(args, settings, regions, tokenizer.all_special_ids)
    start_stage(destination / "capture_settings.json", contract, args.resume)
    jobs = []
    for index, response in enumerate(settings["responses"]):
        directory = destination / "capture" / f"{index:04d}"
        groups, blocks = root_groups(response, region_mask(regions, response), tokenizer.all_special_ids, args.block_tokens)
        start_stage(directory / "sources.json", {"blocks": blocks, "group_ids": groups.tolist()}, args.resume)
        count = len(response["token_ids"]) - response["prompt_length"]
        pending = [target for target in range(count) if not (directory / f"token_{target:06d}.npz").is_file()]
        if pending:
            jobs.append((response, directory, groups, blocks, pending))
    if jobs:
        model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        for job in tqdm(jobs, desc="native value paths"):
            capture_answer(args, model, tokenizer, job)


def capture_answer(args, model, tokenizer, job):
    import torch
    from state_audit.value_path_capture import iter_value_paths

    response, directory, groups, blocks, pending = job
    validate_tokenizer(response, tokenizer)
    if model.native.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.native.device)
    started = perf_counter()
    iterator = iter_value_paths(model, response["token_ids"], response["prompt_length"], pending,
        groups, len(blocks) + 3, choices=args.choices, gradient_batch=args.gradient_batch)
    with closing(iterator):
        for target in tqdm(pending, desc=response["id"], leave=False):
            write_arrays(directory / f"token_{target:06d}.npz", **next(iterator))
    peak = torch.cuda.max_memory_allocated(model.native.device) if model.native.device.type == "cuda" else None
    write_json(directory / "timing.json", {"tokens": len(pending), "seconds": perf_counter() - started,
        "peak_cuda_bytes": peak, "gradient_batch": args.gradient_batch, "forward_passes": 1})


def scoring_protocol(settings, contract):
    return {"purpose": "label_free_value_path_provenance_candidate", "candidate_method": "root_route",
        "primary_baseline": "raw_route", "methods": {name: name for name in METHODS},
        "labels_used_for_scoring": False, "labels_used_for_training": False, "parameter_fitting": False,
        "interventions": False, "model_forward_during_scoring": False, "svd": False,
        "automatic_method_selection": False, "manual_evidence_annotations_required": False,
        "prediction_alignment": "query=P+t-1; native causal mask; no target leakage",
        "risk_definition": "signed_history_root_minus_source_root_over_unsigned_root_action",
        "candidate_reduction": "native_probability_weights_conditional_on_selected_alternatives",
        "temporal_control": "current plus 7 earlier and 8 later scores; not a latent truth state",
        "future_tokens_used": "offline_mean_controls_only; native_root_attribution_is_causal",
        "source_units": "automatic_blocks_with_positive_and_negative_token_parts_retained",
        "semantic_types": "unassigned", "capture": contract, "cohort": settings.get("cohort", {}),
        "claim_boundary": "value_path_decomposition; not native_QK_causality_or_semantic_truth"}


def load_answer(directory, response, groups):
    count = len(response["token_ids"]) - response["prompt_length"]
    rows = []
    for target in range(count):
        current = read_arrays(directory / f"token_{target:06d}.npz")
        query = response["prompt_length"] + target - 1
        if (int(current["target"]) != target or int(current["query"]) != query
                or int(current["token_id"]) != response["token_ids"][query + 1]
                or not np.array_equal(current["group_ids"], groups[:query + 1])):
            raise ValueError(f"{response['id']}: value-path cache alignment mismatch at {target}")
        rows.append(current)
    return rows


def score(args, settings):
    destination = args.output / DIRECTORY
    if not (destination / "capture_settings.json").is_file():
        raise ValueError("Value-path cache is absent: run transport --stage capture first. "
                         "Old detached-KV budget caches cannot recover input provenance.")
    contract = read_json(destination / "capture_settings.json")
    if contract["model"] != settings["model"] or contract["responses"] != settings["responses"]:
        raise ValueError("Value-path capture and input settings differ")
    regions = source_regions(args.output, settings)
    if regions["status"] != contract["source_scope"]:
        raise ValueError("Value-path source scope differs from current source regions")
    protocol = scoring_protocol(settings, contract)
    rows, contributions = [], []
    for index, response in enumerate(tqdm(settings["responses"], desc="read value paths")):
        source = destination / "capture" / f"{index:04d}"
        sources = read_json(source / "sources.json")
        arrays = load_answer(source, response, sources["group_ids"])
        count = len(sources["blocks"])
        scores, profile = score_answer(response, arrays, count, region_mask(regions, response))
        directory = destination / "responses" / f"{index:04d}"
        write_arrays(directory / "scores.npz", **scores)
        write_arrays(directory / "source_profile.npz", profile=profile)
        rows.extend(token_rows(response, scores))
        contributions.extend(source_rows(response, arrays, count))
    write_json(destination / "scoring_protocol.json", protocol)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    write_csv(destination / "source_choices.csv", contributions, list(contributions[0]))
    return finish(args, protocol, rows)


def finish(args, protocol, rows):
    from .transport_report import write_report

    destination = args.output / DIRECTORY
    annotations = args.annotations or args.output / "annotations.json"
    methods = protocol["methods"]
    evaluation = evaluate_comparison(args.output, destination, annotations, methods, rows)
    pairs = [("root_route", "raw_route"), ("root_route", "direct_choice_route"),
             ("root_route_mean", "route_offline_mean"), ("root_route_mean", "root_route")]
    comparisons = write_deltas(args.output, destination, annotations, evaluation, methods, rows, pairs)
    summary = {**protocol, "responses": len({row["response_id"] for row in rows}), "scored_tokens": len(rows),
        "max_abs_ledger_error": max(row["ledger_error"] for row in rows),
        "evaluation": evaluation_headlines(evaluation), "comparisons": comparisons}
    write_json(destination / "summary.json", summary)
    write_report(destination, rows, evaluation)
    return summary


def evaluate(args, settings):
    destination = args.output / DIRECTORY
    protocol = read_json(destination / "scoring_protocol.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        scores = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(token_rows(response, scores))
    return finish(args, protocol, rows)


def main(argv=None):
    import torch
    from threadpoolctl import threadpool_limits

    args = arguments(argv)
    torch.set_num_threads(args.cpu_threads)
    settings = read_json(args.output / "settings.json")
    with threadpool_limits(limits=args.cpu_threads):
        if args.stage == "audit":
            from .transport_audit import audit
            result = audit(args.output, args.annotations, args.bootstrap_replicates, args.seed)
        else:
            if args.stage in ("capture", "run"):
                capture(args, settings)
            result = {"status": "captured", "directory": DIRECTORY}
            if args.stage in ("score", "run"):
                result = score(args, settings)
            elif args.stage == "evaluate":
                result = evaluate(args, settings)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
