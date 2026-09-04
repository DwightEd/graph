"""Foreground CLI for prompt-revisit, nonlocal-review and anchor discovery."""

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
        / "rhythm_v5"
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
    max_events = 96 if args.smoke and args.max_events is None else args.max_events
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
        max_events=max_events,
        query_chunk=args.query_chunk,
        route_window=args.route_window,
        future_horizon=args.future_horizon,
        distance_scale=args.distance_scale,
        peak_quantile=args.peak_quantile,
        max_lag=args.max_lag,
        plot_limit=args.plot_limit,
        plot_sample_id=args.plot_sample_id,
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("captured " + " ".join(f"{task}={count}" for task, count in counts.items()))
    return counts


def compact_number(value) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def compact_ci(summary: dict) -> str:
    low, high = summary["ci95"]
    return f"[{compact_number(low)},{compact_number(high)}]"


def coupling_text(name: str, summary: dict) -> str:
    sample = summary["sample_lift"]
    return (
        f"  {name} pooled={compact_number(summary['pooled_rate'])} "
        f"null={compact_number(summary['pooled_null'])} "
        f"lift={compact_number(summary['pooled_lift'])} "
        f"sample_lift={compact_number(sample['mean'])} CI={compact_ci(sample)} "
        f"positive_sources={compact_number(summary['positive_source_fraction'])} "
        f"lag={compact_number(summary['median_anchor_lag'])}"
    )


def evaluate(args) -> dict:
    reports = evaluate_results(
        output_root(args),
        args.cache / "test",
        bootstrap=args.bootstrap,
        seed=args.seed,
        curve_radius=args.curve_radius,
    )
    for task, report in reports.items():
        onset = report["onset_minus_matched_clean"]
        prompt = onset["prompt_delta"]
        nonlocal = onset["nonlocal_delta"]
        anchor = onset["future_influence"]
        print(
            f"{task:9s} samples={report['samples']} tokens={report['tokens']} "
            f"positives={report['positive_tokens']} "
            f"prevalence={compact_number(report['prevalence'])} "
            f"anchors={report['anchor_peaks']}"
        )
        print(coupling_text("prompt->anchor", report["prompt_to_anchor"]))
        print(coupling_text("nonlocal->anchor", report["nonlocal_to_anchor"]))
        print(
            f"  onset-clean prompt={compact_number(prompt['mean'])} CI={compact_ci(prompt)} "
            f"nonlocal={compact_number(nonlocal['mean'])} CI={compact_ci(nonlocal)} "
            f"anchor={compact_number(anchor['mean'])} CI={compact_ci(anchor)} "
            f"pairs={report['onset_pairs']}"
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
    command.add_argument("--max-events", type=int)
    command.add_argument("--query-chunk", type=int, default=64)
    command.add_argument("--route-window", type=int, default=4)
    command.add_argument("--future-horizon", type=int, default=16)
    command.add_argument(
        "--distance-scale",
        type=int,
        default=16,
        help="lag where continuous nonlocal weight saturates; no hard far-token cutoff",
    )
    command.add_argument("--peak-quantile", type=float, default=0.9)
    command.add_argument("--max-lag", type=int, default=3)
    command.add_argument("--plot-limit", type=int, default=1)
    command.add_argument("--plot-sample-id")
    command.add_argument("--smoke", action="store_true")


def add_evaluation(command) -> None:
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--curve-radius", type=int, default=6)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Internal prompt-revisit, nonlocal-review and anchor rhythm audit"
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
    for name in (
        "query_chunk",
        "route_window",
        "future_horizon",
        "distance_scale",
        "max_lag",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit", "max_events"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
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
