"""CLI for real-sample teacher-forced mechanism auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit import capture_split
from .evaluate import evaluate_saved


DEFAULT_MODEL = (
    "/share/home/tm902089733300000/a903202310/lys/models/"
    "Meta-Llama-3.1-8B-Instruct"
)


def _capture(args: argparse.Namespace) -> None:
    report = capture_split(
        split_root=args.split_root,
        source_info=args.source_info,
        model_path=args.model,
        output_root=args.output,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        limit=args.limit,
        predictor_chunk=args.predictor_chunk,
        top_k=args.top_k,
        logit_chunk=args.logit_chunk,
        intervention_batch=args.intervention_batch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_saved(
        trace_root=args.traces,
        split_root=args.split_root,
        output=args.output,
        position_bin=args.position_bin,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print("\n=== Three-mechanism audit ===")
    for name, summary in report["summaries"].items():
        effect = summary["position_matched_source_equal_difference"]
        interval = summary["ci95"]
        correct = summary["correct_mean"]
        hallucinated = summary["hallucinated_mean"]
        print(
            f"{name:38s} "
            f"correct={'n/a' if correct is None else f'{correct:.6f}'} "
            f"hallucinated={'n/a' if hallucinated is None else f'{hallucinated:.6f}'} "
            f"delta={effect} CI={interval}"
        )
    print(f"\nFull report: {args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Teacher-forced functional-message audit")
    commands = root.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="extract and save raw model dynamics")
    capture.add_argument("--split-root", type=Path, required=True)
    capture.add_argument("--source-info", type=Path, required=True)
    capture.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    capture.add_argument("--predictor-chunk", type=int, default=64)
    capture.add_argument("--intervention-batch", type=int, choices=(1, 3), default=3)
    capture.add_argument("--top-k", type=int, default=8)
    capture.add_argument("--logit-chunk", type=int, default=64)
    capture.add_argument("--limit", type=int)
    capture.set_defaults(handler=_capture)

    evaluate = commands.add_parser("evaluate", help="compare frozen traces with labels")
    evaluate.add_argument("--traces", type=Path, required=True)
    evaluate.add_argument("--split-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--position-bin", type=int, default=16)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260828)
    evaluate.set_defaults(handler=_evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
