"""One foreground entry point for source-resolved functional anchor flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.attention_mechanism_audit.collect import (
    STATE_DIRECTORY as GRAPH_DIRECTORY,
    capture_all,
)
from .evaluate import CONTROLS, PRIMARY, SECONDARY, evaluate
from .pipeline import STATE_DIRECTORY, build_all

MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl"
)
OUTPUT = Path(__file__).resolve().parent / "outputs" / MODEL.name
TASKS = ("QA", "Summary", "Data2txt")


def state_pairs(state_root: Path, cache: Path):
    return [
        (state_root / STATE_DIRECTORY / split, cache / split)
        for split in ("train", "test")
    ]


def display(value):
    return "n/a" if value is None else f"{value:.6f}"


def print_report(report):
    print(f"\n=== {report['task'].upper()} GROUNDED ANCHOR FLOW ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['positives']} "
        f"prevalence={report['prevalence']:.4%}"
    )
    names = (
        PRIMARY,
        f"{PRIMARY}_position_adjusted",
        SECONDARY,
        *CONTROLS[:3],
    )
    for name in names:
        value = report["metrics"][name]
        print(
            f"{name:52s} AUROC={display(value['auroc'])} "
            f"AP={display(value['average_precision'])} tokens={value['tokens']}"
        )
    for name, value in report["paired_capacity_controls"].items():
        print(
            f"paired functional - {name:39s} "
            f"dAUROC={display(value['auroc_difference'])} "
            f"dAP={display(value['average_precision_difference'])}"
        )


def run_all(args):
    graph_root = args.output / "graph"
    capture_all(
        args.cache,
        args.source_info,
        args.model,
        graph_root,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        predictor_batch=args.predictor_batch,
        edge_cover=args.edge_cover,
        edge_budget=args.edge_budget,
        limit=args.limit,
    )
    graph_state = graph_root / GRAPH_DIRECTORY
    build_all(graph_state, args.output)
    for task in TASKS:
        report_path = args.output / "reports" / task.casefold() / "report.json"
        report = evaluate(
            state_pairs(args.output, args.cache),
            task,
            report_path,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        print_report(report)
        print(f"report: {report_path}")


def parser():
    parser = argparse.ArgumentParser(
        description="Source-resolved target-conditioned flow on exact AVWO messages"
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--predictor-batch", type=int, default=8)
    parser.add_argument("--edge-cover", type=float, default=0.95)
    parser.add_argument("--edge-budget", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main():
    run_all(parser().parse_args())


if __name__ == "__main__":
    main()
