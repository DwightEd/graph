"""CLI for the typed route-grammar detector."""

from __future__ import annotations

import argparse

from research_dataset import open_research_dataset

from .config import (
    AuditConfig,
    CalibrationConfig,
    GrammarConfig,
    GraphConfig,
    PhaseConfig,
)
from .evaluation import evaluate_scores
from .experiment import fit_reference, score_split


def _config(arguments) -> AuditConfig:
    return AuditConfig(
        graph=GraphConfig(
            block_rows=arguments.block_rows,
            recent_lag=arguments.recent_lag,
        ),
        grammar=GrammarConfig(
            alpha=arguments.alpha,
            backoff_tau=arguments.backoff_tau,
        ),
        phase=PhaseConfig(
            cusum_slack=arguments.cusum_slack,
            rupture_decay=arguments.rupture_decay,
            closure_decay=arguments.closure_decay,
        ),
        calibration=CalibrationConfig(
            channel_fraction=arguments.channel_fraction,
            fusion_fraction=arguments.fusion_fraction,
            reservoir_rows=arguments.reservoir_rows,
            topology_min_changed_fraction=arguments.topology_min_changed_fraction,
            seed=arguments.seed,
        ),
    )


def _fit_score_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-type", default="QA")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--recent-lag", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--backoff-tau", type=float, default=32.0)
    parser.add_argument("--cusum-slack", type=float, default=0.5)
    parser.add_argument("--rupture-decay", type=float, default=0.95)
    parser.add_argument("--closure-decay", type=float, default=0.9)
    parser.add_argument("--channel-fraction", type=float, default=0.2)
    parser.add_argument("--fusion-fraction", type=float, default=0.2)
    parser.add_argument("--reservoir-rows", type=int, default=20_000)
    parser.add_argument(
        "--topology-min-changed-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--seed", type=int, default=20260825)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--reference", required=True)
    _fit_score_options(fit)

    score = commands.add_parser("score")
    score.add_argument("--test-split", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    _fit_score_options(score)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--test-split", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260825)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "fit":
        dataset = open_research_dataset(
            arguments.train_split,
            device=arguments.device,
        )
        result = fit_reference(
            dataset,
            arguments.reference,
            config=_config(arguments),
            task_type=arguments.task_type,
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        dataset = open_research_dataset(
            arguments.test_split,
            device=arguments.device,
        )
        result = score_split(
            dataset,
            arguments.reference,
            arguments.output,
            task_type=arguments.task_type,
            limit=arguments.limit,
        )
    else:
        dataset = open_research_dataset(
            arguments.test_split,
            device="cpu",
            retain_embedded_labels=True,
        )
        result = evaluate_scores(
            dataset,
            arguments.scores,
            arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
    print(result)


if __name__ == "__main__":
    main()
