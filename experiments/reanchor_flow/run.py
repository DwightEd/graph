"""Foreground CLI for claim-boundary re-anchor flow discovery."""

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
    root = Path(__file__).resolve().parent / "outputs" / args.model.name
    return root / "smoke" if args.smoke else root


def load_model(path: Path, device: str, dtype: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        dtype=DTYPE[dtype],
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
        limit=limit,
        audit_limit=args.audit_limit,
        plot_limit=args.plot_limit,
        max_events=max_events,
        query_chunk=args.query_chunk,
        min_claim_tokens=args.min_claim_tokens,
        max_claim_tokens=args.max_claim_tokens,
        anchor_width=args.anchor_width,
        reread_window=args.reread_window,
        backbone_cover=args.backbone_cover,
        backbone_edges=args.backbone_edges,
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
    )
    for task, report in reports.items():
        primary = report["metrics"][report["primary"]]
        print(
            f"{task:9s} claims={report['claims']} "
            f"positives={report['hallucinated_claims']} "
            f"AUROC={primary['auroc']} AP={primary['average_precision']}"
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
    command.add_argument("--audit-limit", type=int, default=2)
    command.add_argument("--plot-limit", type=int, default=3)
    command.add_argument("--max-events", type=int)
    command.add_argument("--query-chunk", type=int, default=128)
    command.add_argument("--min-claim-tokens", type=int, default=2)
    command.add_argument("--max-claim-tokens", type=int, default=96)
    command.add_argument("--anchor-width", type=int, default=3)
    command.add_argument("--reread-window", type=int, default=5)
    command.add_argument("--backbone-cover", type=float, default=0.8)
    command.add_argument("--backbone-edges", type=int, default=32)
    command.add_argument("--smoke", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="One-model claim re-anchor flow discovery"
    )
    commands = root.add_subparsers(dest="command", required=True)
    analyze_command = commands.add_parser("analyze")
    add_common(analyze_command)
    analyze_command.set_defaults(handler=analyze)

    evaluate_command = commands.add_parser("evaluate")
    add_common(evaluate_command)
    evaluate_command.add_argument("--bootstrap", type=int, default=400)
    evaluate_command.add_argument("--seed", type=int, default=2026)
    evaluate_command.set_defaults(handler=evaluate)

    all_command = commands.add_parser("all")
    add_common(all_command)
    all_command.add_argument("--bootstrap", type=int, default=400)
    all_command.add_argument("--seed", type=int, default=2026)
    all_command.set_defaults(handler=run_all)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
