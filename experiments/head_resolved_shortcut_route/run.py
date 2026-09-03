"""Foreground CLI for the head-resolved shortcut-route audit."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from .collect import STATE_DIRECTORY, capture_all
from .data import TASK_TYPES
from .evaluate import SCORE_ORDER, evaluate_all

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
REPORT_DIRECTORY = "shortcut_route_v1"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def split_roots(cache: Path) -> tuple[Path, Path]:
    """Return the two physical shards used by collection and evaluation."""

    return cache / "train", cache / "test"


def state_inputs(state_root: Path, cache: Path) -> list[tuple[Path, Path]]:
    """Pair every saved artifact shard with its canonical label shard."""

    return [
        (state_root / STATE_DIRECTORY / split.name, split)
        for split in split_roots(cache)
    ]


def _interval(value: Any) -> str:
    if value is None or len(value) != 2 or value[0] is None:
        return "n/a"
    return f"[{value[0]:.6f},{value[1]:.6f}]"


def _print_report(report: Mapping[str, Any]) -> None:
    """Print only the three preregistered support endpoints."""

    prevalence = report.get("prevalence")
    prevalence_text = "n/a" if prevalence is None else f"{prevalence:.4%}"
    scope = (
        "COMPLETE-CAPTURE"
        if report.get("capture_complete", True)
        else "PARTIAL-CAPTURE"
    )
    print(f"\n=== {scope} {report['task_type'].upper()} SHORTCUT-ROUTE ASSOCIATION ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['hallucinated_tokens']} "
        f"prevalence={prevalence_text}"
    )
    for name in SCORE_ORDER:
        metric = report["detection"][name]
        if metric["auroc"] is None:
            print(f"axis      {name:43s} AUROC=n/a AP=n/a")
            continue
        print(
            f"axis      {name:43s} "
            f"AUROC={metric['auroc']:.6f} CI={_interval(metric['auroc_ci95'])} "
            f"AP={metric['average_precision']:.6f} "
            f"CI={_interval(metric['average_precision_ci95'])}"
        )
    print("veto channels are raw audits only; no label-facing direction was selected")
    print("mechanism retention still requires the METHOD structural controls")


def _capture(args: argparse.Namespace) -> dict[str, list[tuple[Path, Path]]]:
    output = (
        args.output or Path(__file__).resolve().parent / "outputs" / args.model.name
    )
    return capture_all(
        split_roots=split_roots(args.cache),
        source_info=args.source_info,
        model_path=args.model,
        output_root=output,
        device=args.device,
        dtype=DTYPES[args.dtype],
        limit=args.limit,
        top_k=args.top_k,
        cover_mass=args.cover_mass,
    )


def _evaluate_tasks(
    *,
    inputs: Iterable[tuple[Path, Path]],
    tasks: Iterable[str],
    report_root: Path,
    bootstrap: int,
    seed: int,
    allow_partial: bool,
) -> dict[str, dict]:
    reports = {}
    physical_inputs = list(inputs)
    for task in tasks:
        destination = report_root / task.casefold() / "report.json"
        report = evaluate_all(
            inputs=physical_inputs,
            task_type=task,
            output=destination,
            bootstrap=bootstrap,
            seed=seed,
            allow_partial=allow_partial,
        )
        reports[task] = report
        _print_report(report)
        print(f"report: {destination}")
        print(f"frozen axes: {report['frozen_axes']}")
    return reports


def _collect_command(args: argparse.Namespace) -> None:
    inputs = _capture(args)
    for task in TASK_TYPES:
        print(f"{task}: {len(inputs[task])} physical shards")


def _evaluate_command(args: argparse.Namespace) -> None:
    tasks = TASK_TYPES if args.task == "all" else (args.task,)
    report_root = args.report_root or args.state_root / REPORT_DIRECTORY
    _evaluate_tasks(
        inputs=state_inputs(args.state_root, args.cache),
        tasks=tasks,
        report_root=report_root,
        bootstrap=args.bootstrap,
        seed=args.seed,
        allow_partial=args.allow_partial,
    )


def _all_command(args: argparse.Namespace) -> None:
    inputs = _capture(args)
    output = (
        args.output or Path(__file__).resolve().parent / "outputs" / args.model.name
    )
    _evaluate_tasks(
        inputs=inputs[TASK_TYPES[0]],
        tasks=TASK_TYPES,
        report_root=output / REPORT_DIRECTORY,
        bootstrap=args.bootstrap,
        seed=args.seed,
        allow_partial=args.limit is not None,
    )


def add_capture_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    command.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    command.add_argument("--source-info", type=Path, default=DEFAULT_SOURCE_INFO)
    command.add_argument("--output", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    command.add_argument("--limit", type=int)
    command.add_argument("--top-k", type=int, default=64)
    command.add_argument("--cover-mass", type=float, default=0.95)


def add_evaluation_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=20260828)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Head-resolved shortcut-route collection and evaluation"
    )
    commands = root.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="capture label-free route artifacts")
    add_capture_arguments(collect)
    collect.set_defaults(handler=_collect_command)

    evaluate = commands.add_parser(
        "evaluate", help="freeze axes, then open labels for one or all tasks"
    )
    evaluate.add_argument(
        "--state-root",
        type=Path,
        required=True,
        help="collection output root containing shortcut_route_state",
    )
    evaluate.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    evaluate.add_argument("--report-root", type=Path)
    evaluate.add_argument("--task", choices=("all", *TASK_TYPES), default="all")
    evaluate.add_argument(
        "--allow-partial",
        action="store_true",
        help="evaluate an explicitly incomplete smoke subset",
    )
    add_evaluation_arguments(evaluate)
    evaluate.set_defaults(handler=_evaluate_command)

    all_data = commands.add_parser(
        "all", help="capture once and evaluate QA, Summary, and Data2txt separately"
    )
    add_capture_arguments(all_data)
    add_evaluation_arguments(all_data)
    all_data.set_defaults(handler=_all_command)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
