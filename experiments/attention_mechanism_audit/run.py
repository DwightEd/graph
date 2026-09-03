"""Foreground entry point for functional message graph construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .collect import STATE_DIRECTORY, capture_all
from .export import export_nodes

MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
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
OUTPUT = Path(__file__).resolve().parent / "outputs" / MODEL.name


def build(args: argparse.Namespace) -> None:
    reports = capture_all(
        args.cache,
        args.source_info,
        args.model,
        args.output,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        predictor_batch=args.predictor_batch,
        edge_cover=args.edge_cover,
        edge_budget=args.edge_budget,
        limit=args.limit,
    )
    print(json.dumps(reports, indent=2))
    print(f"graphs: {args.output / STATE_DIRECTORY}")


def export(args: argparse.Namespace) -> None:
    path = export_nodes(args.state_root, args.output, args.task)
    print(path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Exact AVWO functional message graphs")
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("build")
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument("--output", type=Path, default=OUTPUT)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    command.add_argument("--predictor-batch", type=int, default=8)
    command.add_argument("--edge-cover", type=float, default=0.95)
    command.add_argument("--edge-budget", type=int, default=64)
    command.add_argument("--limit", type=int)
    command.set_defaults(handler=build)

    command = commands.add_parser("export")
    command.add_argument("--state-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--task", choices=("QA", "Summary", "Data2txt"))
    command.set_defaults(handler=export)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
