"""CLI for the frozen-model three-mechanism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit import capture_split
from .evaluate import SCORE_ORDER, evaluate_all, plot_saved_sample

DEFAULT_MODEL = (
    "/share/home/tm902089733300000/a903202310/lys/models/"
    "Meta-Llama-3.1-8B-Instruct"
)


def _print_report(report: dict) -> None:
    def ci(interval: list[float | None]) -> str:
        if interval[0] is None:
            return "n/a"
        return f"[{interval[0]:.6f},{interval[1]:.6f}]"

    scope = "ALL-QA" if report["capture_complete"] else "PARTIAL-QA"
    print(f"\n=== {scope} THREE-MECHANISM DETECTION ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['hallucinated_tokens']} "
        f"prevalence={report['prevalence']:.4%} "
        f"capture_complete={report['capture_complete']}"
    )
    for name in SCORE_ORDER:
        result = report["detection"][name]
        role = "PRIMARY" if name == report["primary_score"] else "component"
        if result["auroc"] is None:
            print(f"{role:9s} {name:24s} AUROC=n/a AUPRC=n/a")
            continue
        print(
            f"{role:9s} {name:24s} "
            f"AUROC={result['auroc']:.6f} "
            f"CI={ci(result['auroc_ci95'])} "
            f"AUPRC={result['auprc']:.6f} "
            f"CI={ci(result['auprc_ci95'])} "
            f"lift={result['auprc_lift']:.3f}"
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
    report = evaluate_all(
        inputs=args.input,
        output=args.output,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    _print_report(report)
    print(f"\nreport: {args.output}")
    print(f"token scores: {report['token_scores']}")
    print(f"population figures: {report['figures']}")


def _plot_sample(args: argparse.Namespace) -> None:
    result = plot_saved_sample(
        inputs=args.input,
        sample_id=args.sample_id,
        model_path=args.model,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Frozen-model three-mechanism audit")
    commands = root.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="capture or resume one cache shard")
    capture.add_argument("--split-root", type=Path, required=True)
    capture.add_argument("--source-info", type=Path, required=True)
    capture.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    capture.add_argument("--predictor-chunk", type=int, default=128)
    capture.add_argument("--intervention-batch", type=int, choices=(1, 3), default=3)
    capture.add_argument("--top-k", type=int, default=8)
    capture.add_argument("--logit-chunk", type=int, default=64)
    capture.add_argument("--limit", type=int)
    capture.set_defaults(handler=_capture)

    evaluate = commands.add_parser(
        "evaluate",
        help="pool every physical cache shard and evaluate once",
    )
    evaluate.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("TRACES", "CACHE"),
        required=True,
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260828)
    evaluate.set_defaults(handler=_evaluate)

    sample = commands.add_parser(
        "plot-sample",
        help="render one saved sample without replaying the model",
    )
    sample.add_argument(
        "--input",
        action="append",
        type=Path,
        metavar="TRACES",
        required=True,
    )
    sample.add_argument("--sample-id", required=True)
    sample.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    sample.add_argument("--output", type=Path, required=True)
    sample.set_defaults(handler=_plot_sample)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
