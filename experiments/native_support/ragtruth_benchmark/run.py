"""All RAGTruth tasks/generators/splits, direct source/routing capture and token metrics."""

import argparse
import json
from pathlib import Path

from state_audit.storage import read_json, start_stage

from .data import TASKS, prepare
from .scoring import FUSIONS, PRIMARY, WEIGHTS, score_all


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", type=Path, help="Official directory containing source_info.jsonl and response.jsonl")
    inputs.add_argument("--cache-input", type=Path, help="Completed contrast/carrier directory or review ZIP; CPU scoring only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    parser.add_argument("--splits", choices=("train", "test"), nargs="+", default=["train", "test"])
    parser.add_argument("--generators", nargs="+", help="Default: all official generators")
    parser.add_argument("--limit", type=int, help="Explicit pilot cap; omitted means every matching answer")
    parser.add_argument("--stage", choices=("run", "prepare", "capture", "score", "select", "evaluate", "pack"), default="run")
    parser.add_argument("--max-unit-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--save-heads", action="store_true", help="Retain layer/head source/history norm and attention sums; extra disk")
    parser.add_argument("--select-on-train", action="store_true", help="Select baseline/fusion on labelled train-development AUROC; not unsupervised")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_unit_tokens, args.window, args.prefill_chunk_size, args.query_chunk_size, args.cpu_threads) < 1:
        parser.error("Token budgets, chunks and cpu-threads must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive or omitted for all answers")
    if not (args.output / "manifest.json").exists() and args.dataset is None and args.cache_input is None:
        parser.error("A new benchmark requires --dataset or --cache-input")
    if args.cache_input and args.output.resolve() == args.cache_input.resolve():
        parser.error("Use a separate output directory; input measurements are immutable")
    return args


def protocol(args):
    return dict(version="ragtruth-source-first-v1", primary_candidate=PRIMARY,
        model=args.model, dataset=str(args.dataset.resolve()) if args.dataset else None,
        cache_input=str(args.cache_input.resolve()) if args.cache_input else None,
        tasks=args.tasks, splits=args.splits, generators=args.generators, limit=args.limit,
        dtype=args.dtype, max_unit_tokens=args.max_unit_tokens, window=args.window, save_heads=args.save_heads,
        source_risk="negative_source_logprob_effect_under_local_history; uniform_saved_unit_mean",
        source_removal="delete_source_token_IDs; positions_recomputed; not_isolated_factual_causality",
        route="unchanged_history_minus_source_message_norm_share; average_layer_after_head_sums",
        candidate_family=dict(zip(FUSIONS, WEIGHTS)), fusion="source_unit_rank + weight*within_unit_route_rank_residual",
        calibration="unlabelled_source_balanced_midCDF_separately_by_task_and_official_split; transductive",
        labels_used_for_fixed_scores=False, classifier_training=False,
        selection="optional_separate_selection.json; train-source-development_labels_only",
        future_tokens_used=True, score_scope="offline_answer_and_unlabelled_task_split_cohort",
        source_regions="prompt_blocks_not_verified_applicability", observer="teacher_forced_may_differ_from_generator",
        native_backwards=0, selected_edge_deletions=0, graph_regularization=False,
        original_results_immutable=True, primary_evaluation="pooled_token_AUROC_per_task_on_official_test")


def load_or_prepare(args):
    path = args.output / "manifest.json"
    if path.exists():
        if args.stage in ("run", "prepare"):
            start_stage(args.output / "protocol.json", protocol(args), args.resume)
        else:
            saved = read_json(args.output / "protocol.json")
            args.window = saved["window"]
            if args.stage == "capture" and (args.dtype != saved["dtype"] or args.save_heads != saved["save_heads"]):
                raise ValueError("Capture dtype/save-heads differs from frozen protocol")
        return read_json(path)
    start_stage(args.output / "protocol.json", protocol(args), args.resume)
    return prepare(args)


def main(argv=None):
    args = arguments(argv)
    from .report import evaluate, pack
    if args.stage == "pack":
        print(json.dumps(dict(review_archive=pack(args.output))))
        return
    manifest = load_or_prepare(args)
    if args.stage == "prepare":
        print(json.dumps(dict(status="prepared", answers=len(manifest["records"]), groups=manifest["groups"])))
        return
    if args.stage in ("run", "capture") and "cache_input" not in manifest:
        from .capture import capture
        capture(args, manifest)
    if args.stage == "capture":
        print(json.dumps(dict(status="captured", output=str(args.output))))
        return
    if args.stage in ("run", "score"):
        coverage = score_all(args, manifest)
        if args.stage == "score":
            print(json.dumps(coverage))
            return
    if args.stage == "select" or (args.stage == "run" and args.select_on_train):
        from .selection import select
        if read_json(args.output / "coverage.json")["status"] != "complete":
            raise ValueError("Freeze all selected token scores before parameter selection")
        selected = select(args, manifest)
        if args.stage == "select":
            print(json.dumps(selected))
            return
    summary = evaluate(args, manifest)
    print(json.dumps(dict(output=str(args.output), status=summary["status"], primary_candidate=PRIMARY,
        test_by_dataset=summary["test_by_dataset"], review_archive=pack(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
