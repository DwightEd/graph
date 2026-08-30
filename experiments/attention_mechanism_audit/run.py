"""CLI for real-sample teacher-forced mechanism auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit import capture_split
from .evaluate import combine_saved, evaluate_saved
from .reporting import render_report, render_sample


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
        trace_level=args.trace_level,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_saved(
        trace_root=args.traces,
        split_root=args.split_root,
        output=args.output,
        model_path=args.model,
        split_name=args.split_name,
        sample_output=args.sample_output,
        figure_output=args.figures,
        position_bin=args.position_bin,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(
        "\n"
        + render_report(
            report,
            all_metrics=args.all_metrics,
            explain=args.explain,
        )
    )
    print(f"\nFull report: {args.output}")


def _combine(args: argparse.Namespace) -> None:
    report = combine_saved(
        inputs=[(name, path) for name, path in args.input],
        output=args.output,
        figure_output=args.figures,
        position_bin=args.position_bin,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print("\n" + render_report(report, all_metrics=args.all_metrics, explain=args.explain))
    print(f"\nFull report: {args.output}")


def _summarize(args: argparse.Namespace) -> None:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(
        render_report(
            report,
            all_metrics=args.all_metrics,
            explain=args.explain,
        )
    )


def _summarize_sample(args: argparse.Namespace) -> None:
    record = json.loads(args.audit.read_text(encoding="utf-8"))
    print(render_sample(record))


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
    capture.add_argument(
        "--trace-level", choices=("mechanism", "raw"), default="mechanism"
    )
    capture.set_defaults(handler=_capture)

    evaluate = commands.add_parser("evaluate", help="compare frozen traces with labels")
    evaluate.add_argument("--traces", type=Path, required=True)
    evaluate.add_argument("--split-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    evaluate.add_argument("--split-name")
    evaluate.add_argument("--sample-output", type=Path)
    evaluate.add_argument("--figures", type=Path)
    evaluate.add_argument("--position-bin", type=int, default=16)
    evaluate.add_argument("--bootstrap", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=20260828)
    evaluate.add_argument("--all-metrics", action="store_true")
    evaluate.add_argument("--explain", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    combine = commands.add_parser(
        "combine", help="recompute all-QA statistics from split evaluation reports"
    )
    combine.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("SPLIT", "REPORT"),
        required=True,
    )
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--figures", type=Path)
    combine.add_argument("--position-bin", type=int, default=16)
    combine.add_argument("--bootstrap", type=int, default=10000)
    combine.add_argument("--seed", type=int, default=20260828)
    combine.add_argument("--all-metrics", action="store_true")
    combine.add_argument("--explain", action="store_true")
    combine.set_defaults(handler=_combine)

    summarize = commands.add_parser(
        "summarize", help="print key results from an existing report without reevaluation"
    )
    summarize.add_argument("--report", type=Path, required=True)
    summarize.add_argument("--all-metrics", action="store_true")
    summarize.add_argument("--explain", action="store_true")
    summarize.set_defaults(handler=_summarize)

    sample = commands.add_parser(
        "summarize-sample", help="print the mechanism evidence for one saved response"
    )
    sample.add_argument("--audit", type=Path, required=True)
    sample.set_defaults(handler=_summarize_sample)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
