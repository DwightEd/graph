"""Command line interface for the attention holonomy mechanism audit."""

from __future__ import annotations

import argparse

from research_dataset import open_research_dataset

from .config import (
    AuditConfig,
    EvaluationConfig,
    GraphConfig,
    ReferenceConfig,
    TransportConfig,
)
from .evaluation import evaluate_scores
from .experiment import fit_reference, score_split


def _config(args) -> AuditConfig:
    return AuditConfig(
        graph=GraphConfig(
            block_rows=args.block_rows,
            censored_fill_ratio=args.censored_fill_ratio,
            max_relay_predecessors=args.max_relay_predecessors,
            max_query_events=args.max_query_events,
        ),
        transport=TransportConfig(
            ridge_alpha=args.ridge_alpha,
            minimum_pairs=args.minimum_pairs,
        ),
        reference=ReferenceConfig(
            calibration_fraction=args.calibration_fraction,
            reservoir_rows=args.reservoir_rows,
            nuisance_ridge_alpha=args.nuisance_ridge_alpha,
            position_degree=args.position_degree,
            seed=args.seed,
        ),
        evaluation=EvaluationConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-type", default="QA")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--censored-fill-ratio", type=float, default=0.5)
    parser.add_argument("--max-relay-predecessors", type=int, default=12)
    parser.add_argument("--max-query-events", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--minimum-pairs", type=int, default=32)
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--reservoir-rows", type=int, default=50_000)
    parser.add_argument("--nuisance-ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--position-degree", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--reference", required=True)
    _common(fit)

    score = subparsers.add_parser("score")
    score.add_argument("--test-split", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--sidecar-dir")
    _common(score)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--test-split", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(args.train_split, device=args.device)
        print(
            fit_reference(
                dataset,
                args.reference,
                config=_config(args),
                task_type=args.task_type,
                limit=args.limit,
            )
        )
    elif args.command == "score":
        dataset = open_research_dataset(args.test_split, device=args.device)
        print(
            score_split(
                dataset,
                args.reference,
                args.output,
                task_type=args.task_type,
                limit=args.limit,
                sidecar_dir=args.sidecar_dir,
            )
        )
    else:
        dataset = open_research_dataset(args.test_split, device=args.device)
        print(
            evaluate_scores(
                dataset,
                args.scores,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )


if __name__ == "__main__":
    main()
