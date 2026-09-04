"""Foreground CLI for the re-anchor phenomenon audit."""

from __future__ import annotations

import argparse
import gc
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
    root = (
        Path(__file__).resolve().parent
        / "outputs"
        / args.model.name
        / "phenomenon_v3"
    )
    return root / "smoke" if args.smoke else root


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
    max_events = 24 if args.smoke and args.max_events is None else args.max_events
    counts = analyze_split(
        model,
        tokenizer,
        args.cache / "test",
        args.source_info,
        output_root(args),
        model_path=str(args.model),
        model_id=args.model.name,
        dtype=args.dtype,
        limit=limit,
        plot_limit=args.plot_limit,
        max_events=max_events,
        query_chunk=args.query_chunk,
        min_claim_tokens=args.min_claim_tokens,
        max_claim_tokens=args.max_claim_tokens,
        causal_cuts=args.causal_cuts,
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(counts)
    return counts


def evaluate(args) -> dict:
    reports = evaluate_results(
        output_root(args),
        args.cache / "test",
        bootstrap=args.bootstrap,
        seed=args.seed,
        pre=args.pre_window,
        post=args.post_window,
        curve_low=args.curve_low,
        curve_high=args.curve_high,
    )
    for task, report in reports.items():
        status = report["hypothesis_status"]
        observer = report["observer_hypothesis_status"]
        scope = (
            "generation"
            if report["model_scope"]["generation_claims_allowed"]
            else "observer-only; generation status withheld"
        )
        boundary = report["correct_boundary_vs_within_claim"]["evidence_specificity"]
        missed = report["missed_reanchor_at_claim_boundary"]["exact_boundary_primary"]
        print(
            f"{task:9s} scope={scope}\n"
            f"  reported status={status}\n"
            f"  observer H1={observer['H1_exposure_adjusted_preference_drift']} "
            f"H2={observer['H2_natural_boundary_evidence_specificity']} "
            f"H3={observer['H3_exact_boundary_missed_entry_association']}\n"
            f"  clean boundary-minus-control={boundary['source_mean']} "
            f"CI95={boundary['ci95']} n={boundary['events']}\n"
            f"  hallucinated-minus-clean boundary entry={missed['source_mean']} "
            f"CI95={missed['ci95']} pairs={missed['matched_pairs']}/"
            f"{missed['candidate_hallucinations']} sources={missed['sources']}\n"
            f"  next: {report['recommended_next_step']}"
        )
    return reports


def run_all(args) -> dict:
    analyze(args)
    return evaluate(args)


def add_common(command) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument("--output", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument("--limit", type=int)
    command.add_argument("--plot-limit", type=int, default=3)
    command.add_argument("--max-events", type=int)
    command.add_argument(
        "--query-chunk",
        type=int,
        default=64,
        help="queries per attention matmul; lower this if GPU memory is tight",
    )
    command.add_argument("--min-claim-tokens", type=int, default=2)
    command.add_argument("--max-claim-tokens", type=int, default=96)
    command.add_argument(
        "--causal-cuts",
        action="store_true",
        help="also rerun direct and global evidence-source cuts (three forwards total)",
    )
    command.add_argument("--smoke", action="store_true")


def add_evaluation(command) -> None:
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--pre-window", type=int, default=5)
    command.add_argument("--post-window", type=int, default=3)
    command.add_argument("--curve-low", type=int, default=-5)
    command.add_argument("--curve-high", type=int, default=10)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Layer-resolved claim re-anchor phenomenon audit"
    )
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
    for name in ("limit", "max_events"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.query_chunk < 1:
        raise ValueError("--query-chunk must be positive")
    if args.plot_limit < 0:
        raise ValueError("--plot-limit cannot be negative")
    if args.min_claim_tokens < 1 or args.max_claim_tokens < args.min_claim_tokens:
        raise ValueError("claim token bounds are inconsistent")
    if hasattr(args, "bootstrap"):
        if args.bootstrap < 0:
            raise ValueError("--bootstrap cannot be negative")
        if args.pre_window < 1 or args.post_window < 1:
            raise ValueError("event windows must be positive")
        if args.curve_low >= 0 or args.curve_high < 0:
            raise ValueError("event curve must straddle offset zero")


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
