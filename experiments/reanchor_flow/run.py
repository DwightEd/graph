"""Foreground CLI for the complete re-anchor mechanism audit."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze import analyze_split
from .evaluate import evaluate_results

MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/"
    "Meta-Llama-3.1-8B-Instruct"
)
CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def output_root(args) -> Path:
    if args.output:
        return args.output
    root = Path(__file__).resolve().parent / "outputs" / args.model.name / "mechanism_v8"
    return root / "smoke" if args.smoke else root


def selected_splits(args) -> tuple[str, ...]:
    return ("train", "test") if args.split == "all" else (args.split,)


def split_output(args, split: str) -> Path:
    root = output_root(args)
    return root / split if args.split == "all" else root


def load_model(path: Path, device: str, dtype: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=DTYPE[dtype],
        attn_implementation="eager",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def analyze(args) -> dict:
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    limit = 1 if args.smoke and args.limit is None else args.limit
    max_events = 96 if args.smoke and args.max_events is None else args.max_events
    mechanism_limit = (
        1 if args.smoke and args.mechanism_limit == 0 else args.mechanism_limit
    )
    captured = {}
    for split in selected_splits(args):
        counts = analyze_split(
            model,
            tokenizer,
            args.cache / split,
            args.source_info,
            split_output(args, split),
            model_path=str(args.model),
            model_id=args.model.name,
            dtype=args.dtype,
            limit=limit,
            max_events=max_events,
            query_chunk=args.query_chunk,
            route_window=args.route_window,
            future_horizon=args.future_horizon,
            distance_scale=args.distance_scale,
            peak_quantile=args.peak_quantile,
            max_lag=args.max_lag,
            plot_limit=args.plot_limit,
            plot_sample_id=args.plot_sample_id,
            mechanism_limit=mechanism_limit,
        )
        captured[split] = counts
        print(
            f"captured {split} "
            + " ".join(f"{task}={count}" for task, count in counts.items())
        )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return captured


def number(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def interval(summary: dict) -> str:
    low, high = summary["ci95"]
    return f"[{number(low)},{number(high)}]"


def effect(name: str, summary: dict) -> str:
    return f"{name}={number(summary['mean'])} CI={interval(summary)}"


def evaluate(args) -> dict:
    all_reports = {}
    seed_offset = {"test": 0, "train": 1000}
    for split in selected_splits(args):
        reports = evaluate_results(
            split_output(args, split),
            args.cache / split,
            bootstrap=args.bootstrap,
            seed=args.seed + seed_offset[split],
            curve_radius=args.curve_radius,
        )
        all_reports[split] = reports
        print(f"\n=== {split.upper()} ===")
        for task, report in reports.items():
            print_report(task, report)
    return all_reports


def print_report(task: str, report: dict) -> None:
    normal = report["normal"]
    shift = normal["direct_route_shift"]
    transition = normal["internal_transition"]
    onset = report["onset_minus_matched_clean"]
    functional = report["functional"]
    mechanism = report["mechanism"]
    print(
        f"{task:9s} samples={report['samples']} tokens={report['tokens']} "
        f"positives={report['positive_tokens']} prevalence={number(report['prevalence'])} "
        f"functional_pairs={functional['onset_pairs']} "
        f"functional_pair_sources={functional['onset_pair_sources']} "
        f"grouped_samples={mechanism['samples']} "
        f"grouped_sources={mechanism['sources']} "
        f"grouped_pairs={mechanism['onset_pairs']} "
        f"grouped_pair_sources={mechanism['onset_pair_sources']}"
    )
    print(
        "  H0 direct drift: "
        + effect("prompt_slope", shift["prompt_lift_slope"])
        + "  "
        + effect("history_slope", shift["history_lift_slope"])
        + "  "
        + effect(
            "prompt_vs_history",
            shift["conditional_prompt_history_log_odds_slope"],
        )
    )
    print(
        "  H1 transition: "
        + effect("prompt", transition["prompt_delta"])
        + "  "
        + effect("evidence", transition["evidence_delta"])
        + "  "
        + effect("predictor_reuse", transition["predictor_reuse"])
        + "  "
        + effect("emitted_anchor", transition["emitted_token_anchor"])
    )
    print(
        "  H2 onset-clean: "
        + effect("route_change", onset["route_change"])
        + "  "
        + effect("prompt", onset["prompt_delta"])
        + "  "
        + effect("evidence", onset["evidence_delta"])
        + "  "
        + effect("predictor_reuse", onset["predictor_reuse"])
        + "  "
        + effect("emitted_token_anchor", onset["emitted_token_anchor"])
    )
    if functional["samples"]:
        deep = functional["onset_minus_clean"]
        print(
            "  H3 functional context: "
            + effect("entry", deep["evidence_entry"])
            + "  "
            + effect("target_effect", deep["evidence_effect"])
            + "  "
            + effect("distribution_js", deep["context_distribution_js"])
        )
        print(
            "  H3 adoption: "
            + effect("target_logprob_gain", deep["context_target_logprob_gain"])
            + "  "
            + effect("adoption_margin", deep["context_adoption_margin"])
            + "  "
            + effect("target_log_rank", deep["context_target_log_rank"])
        )
    if mechanism["samples"]:
        deep = mechanism["onset_minus_clean"]
        print(
            "  H4 grouped/state: "
            + effect("integration", deep["evidence_prompt_interaction"])
            + "  "
            + effect("history_effect", deep["history_effect"])
            + "  "
            + effect("late_control_loss", deep["evidence_late_control_loss"])
            + "  "
            + effect("readout_gain", deep["evidence_readout_gain"])
        )
    balance = report["matching_balance"]
    print(
        "  onset matching: "
        f"pairs={balance['pairs']} sources={balance['sources']} "
        f"position_gap={number(balance['mean_absolute_relative_position_gap']['mean'])} "
        f"boundary_match={number(balance['boundary_match_fraction']['mean'])} "
        f"token_match={number(balance['token_match_fraction']['mean'])}"
    )
    decisions = report["registered_decisions"]
    print(
        "  decisions: "
        + " ".join(f"{name}={status}" for name, status in decisions.items())
    )


def run_all(args) -> dict:
    analyze(args)
    return evaluate(args)


def add_common(command) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--split", choices=("train", "test", "all"), default="test"
    )
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument("--limit", type=int)
    command.add_argument("--max-events", type=int)
    command.add_argument("--query-chunk", type=int, default=64)
    command.add_argument("--route-window", type=int, default=4)
    command.add_argument("--future-horizon", type=int, default=16)
    command.add_argument("--distance-scale", type=int, default=16)
    command.add_argument("--peak-quantile", type=float, default=0.9)
    command.add_argument("--max-lag", type=int, default=3)
    command.add_argument("--plot-limit", type=int, default=1)
    command.add_argument("--plot-sample-id")
    command.add_argument(
        "--mechanism-limit",
        type=int,
        default=0,
        help="deep grouped-cut samples per task; -1 means every selected sample",
    )
    command.add_argument("--smoke", action="store_true")


def add_evaluation(command) -> None:
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--curve-radius", type=int, default=6)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Complete re-anchor mechanism audit")
    commands = root.add_subparsers(dest="command", required=True)
    analyze_command = commands.add_parser("analyze")
    add_common(analyze_command)
    analyze_command.set_defaults(handler=analyze)
    evaluate_command = commands.add_parser("evaluate")
    add_common(evaluate_command)
    add_evaluation(evaluate_command)
    evaluate_command.set_defaults(handler=evaluate)
    all_command = commands.add_parser("all")
    add_common(all_command)
    add_evaluation(all_command)
    all_command.set_defaults(handler=run_all)
    return root


def validate_args(args) -> None:
    for name in ("query_chunk", "route_window", "future_horizon", "distance_scale", "max_lag"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit", "max_events"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.mechanism_limit < -1:
        raise ValueError("--mechanism-limit must be -1 or non-negative")
    if not 0 < args.peak_quantile < 1:
        raise ValueError("--peak-quantile must lie in (0,1)")
    if args.plot_limit < 0:
        raise ValueError("--plot-limit cannot be negative")
    if hasattr(args, "bootstrap") and args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")
    if hasattr(args, "curve_radius") and args.curve_radius < 1:
        raise ValueError("--curve-radius must be positive")


def main() -> None:
    command_parser = parser()
    args = command_parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        command_parser.error(str(error))
    args.handler(args)


if __name__ == "__main__":
    main()
