"""Foreground CLI for constraint-routing rhythm analysis and evaluation."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .analyze import analyze_split
from .evaluate import evaluate_results

DEFAULT_MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
DEFAULT_SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def output_root(args: argparse.Namespace) -> Path:
    model = getattr(args, "model", DEFAULT_MODEL)
    if args.output:
        return args.output
    root = Path(__file__).resolve().parent / "outputs" / model.name
    return root / "smoke" if getattr(args, "smoke", False) else root


def load_model_and_tokenizer(path: Path, device: str, dtype: str):
    """Load the declared local model without network fallback."""

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        dtype=DTYPES[dtype],
        attn_implementation="eager",
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def smoke_limits(args: argparse.Namespace) -> tuple[int | None, int | None]:
    limit = args.limit
    max_events = args.max_events
    if args.smoke:
        limit = 1 if limit is None else limit
        max_events = 8 if max_events is None else max_events
    return limit, max_events


def analysis_run_config(
    args: argparse.Namespace, limit: int | None, max_events: int | None
) -> dict[str, object]:
    """Return the plain-JSON identity and intended scope of one analysis run."""

    return {
        "model": str(args.model.resolve()),
        "model_id": args.model.name,
        "dataset_root": str((args.cache / "test").resolve()),
        "source_info": str(args.source_info.resolve()),
        "dtype": args.dtype,
        "smoke": bool(args.smoke),
        "limit_per_task": limit,
        "max_events": max_events,
        "audit_limit": args.audit_limit,
        "audit_seed": args.audit_seed,
        "plot_limit": args.plot_limit,
        "head_quantile": args.head_quantile,
        "query_chunk": args.query_chunk,
        "window": args.window,
        "horizon_low": args.horizon_low,
        "horizon_high": args.horizon_high,
        "carrier_quantile": args.carrier_quantile,
        "mass_floor": args.mass_floor,
        "max_carriers": args.max_carriers,
        "split_layer": args.split_layer,
    }


def analyze_command(args: argparse.Namespace) -> dict:
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    limit, max_events = smoke_limits(args)
    counts = analyze_split(
        model,
        tokenizer,
        split_root=args.cache / "test",
        source_info=args.source_info,
        output_root=output_root(args),
        limit=limit,
        audit_limit=args.audit_limit,
        plot_limit=args.plot_limit,
        max_events=max_events,
        head_quantile=args.head_quantile,
        query_chunk=args.query_chunk,
        window=args.window,
        horizon_low=args.horizon_low,
        horizon_high=args.horizon_high,
        carrier_quantile=args.carrier_quantile,
        mass_floor=args.mass_floor,
        max_carriers=args.max_carriers,
        split_layer=args.split_layer,
        model_id=args.model.name,
        audit_seed=args.audit_seed,
        run_config=analysis_run_config(args, limit, max_events),
    )
    del model, tokenizer
    print(counts)
    return counts


def evaluate_command(args: argparse.Namespace) -> dict:
    root = output_root(args)
    return evaluate_results(
        result_root=root / "results",
        dataset_root=args.cache / "test",
        output_root=root / "reports",
        bootstrap=args.bootstrap,
        seed=args.seed,
    )


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def all_command(args: argparse.Namespace) -> dict:
    analyze_command(args)
    release_cuda()
    return evaluate_command(args)


def add_analysis_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    command.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    command.add_argument("--source-info", type=Path, default=DEFAULT_SOURCE_INFO)
    command.add_argument("--output", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    command.add_argument("--limit", type=int)
    command.add_argument("--audit-limit", type=int, default=0)
    command.add_argument("--audit-seed", type=int, default=2026)
    command.add_argument("--plot-limit", type=int, default=4)
    command.add_argument("--max-events", type=int)
    command.add_argument("--head-quantile", type=float, default=0.3)
    command.add_argument("--query-chunk", type=int, default=128)
    command.add_argument("--window", type=int, default=10)
    command.add_argument("--horizon-low", type=int, default=10)
    command.add_argument("--horizon-high", type=int, default=100)
    command.add_argument("--carrier-quantile", type=float, default=0.75)
    command.add_argument("--mass-floor", type=float, default=1e-6)
    command.add_argument("--max-carriers", type=int, default=8)
    command.add_argument("--split-layer", type=int)
    command.add_argument("--smoke", action="store_true")


def add_evaluation_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--bootstrap", type=int, default=400)
    command.add_argument("--seed", type=int, default=2026)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Constraint-routing rhythm analysis and evaluation"
    )
    commands = root.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze", help="freeze label-free rhythm and intervention artifacts"
    )
    add_analysis_arguments(analyze)
    analyze.set_defaults(handler=analyze_command)

    evaluate = commands.add_parser(
        "evaluate", help="open labels and evaluate frozen artifacts"
    )
    evaluate.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    evaluate.add_argument("--output", type=Path)
    add_evaluation_arguments(evaluate)
    evaluate.set_defaults(handler=evaluate_command)

    all_data = commands.add_parser(
        "all", help="analyze the test split, release the model, then evaluate"
    )
    add_analysis_arguments(all_data)
    add_evaluation_arguments(all_data)
    all_data.set_defaults(handler=all_command)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
