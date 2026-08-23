"""Command-line entry point for the non-neural structure audit."""

from __future__ import annotations

import argparse

from .config import AuditConfig, EvaluationConfig
from .evaluation import StructureEvaluator
from .experiment import StructureAudit
from .protocol import freeze_confirmation, prepare_split_plan


def _add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = AuditConfig()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-type", default="QA")
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument(
        "--causal-position-bins", type=int, default=defaults.causal_position_bins
    )
    parser.add_argument("--recent-tokens", type=int, default=defaults.recent_tokens)
    parser.add_argument(
        "--reference-capacity", type=int, default=defaults.reference_capacity
    )
    parser.add_argument("--null-replicates", type=int, default=defaults.null_replicates)
    parser.add_argument(
        "--layer-shuffle-replicates",
        type=int,
        default=defaults.layer_shuffle_replicates,
    )
    parser.add_argument("--swap-rounds", type=int, default=defaults.swap_rounds)
    parser.add_argument("--lag-bins", type=int, default=defaults.response_lag_bins)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-progress", action="store_true")


def _audit_config(arguments) -> AuditConfig:
    return AuditConfig(
        block_rows=arguments.block_rows,
        causal_position_bins=arguments.causal_position_bins,
        recent_tokens=arguments.recent_tokens,
        reference_capacity=arguments.reference_capacity,
        null_replicates=arguments.null_replicates,
        layer_shuffle_replicates=arguments.layer_shuffle_replicates,
        swap_rounds=arguments.swap_rounds,
        response_lag_bins=arguments.lag_bins,
        random_seed=arguments.seed,
        show_progress=not arguments.no_progress,
    )


def _add_evaluation_arguments(
    parser: argparse.ArgumentParser, *, include_scope: bool = True
) -> None:
    defaults = EvaluationConfig()
    if include_scope:
        parser.add_argument(
            "--scope",
            choices=("smoke", "discovery", "confirmation"),
            default=defaults.scope,
        )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=defaults.bootstrap_replicates,
    )
    parser.add_argument(
        "--permutation-replicates",
        type=int,
        default=defaults.permutation_replicates,
    )
    parser.add_argument("--onset-window", type=int, default=defaults.onset_window)
    parser.add_argument("--cv-folds", type=int, default=defaults.grouped_cv_folds)
    parser.add_argument(
        "--minimum-confirmation-samples",
        type=int,
        default=defaults.minimum_confirmation_samples,
    )
    parser.add_argument(
        "--minimum-positive-responses",
        type=int,
        default=defaults.minimum_positive_responses,
    )
    parser.add_argument(
        "--endpoint-minimum-changed-fraction",
        type=float,
        default=defaults.endpoint_minimum_changed_fraction,
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)


def _evaluation_config(arguments) -> EvaluationConfig:
    return EvaluationConfig(
        scope=arguments.scope,
        bootstrap_replicates=arguments.bootstrap_replicates,
        permutation_replicates=arguments.permutation_replicates,
        onset_window=arguments.onset_window,
        grouped_cv_folds=arguments.cv_folds,
        minimum_confirmation_samples=arguments.minimum_confirmation_samples,
        minimum_positive_responses=arguments.minimum_positive_responses,
        endpoint_minimum_changed_fraction=arguments.endpoint_minimum_changed_fraction,
        random_seed=arguments.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit train-only unlabeled references")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    _add_audit_arguments(fit)

    score = commands.add_parser("score", help="freeze label-free structure scores")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output-dir", required=True)
    _add_audit_arguments(score)

    plan = commands.add_parser("plan", help="freeze discovery/confirmation groups")
    plan.add_argument("--score-dir", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--discovery-fraction", type=float, default=0.5)
    plan.add_argument("--seed", type=int, default=EvaluationConfig().random_seed)

    evaluate = commands.add_parser("evaluate", help="open labels and run gates")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--score-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--tokenizer")
    evaluate.add_argument("--split-plan")
    evaluate.add_argument("--confirmation-plan")
    _add_evaluation_arguments(evaluate)

    freeze = commands.add_parser(
        "freeze-confirmation", help="freeze one post-discovery confirmation run"
    )
    freeze.add_argument("--split-plan", required=True)
    freeze.add_argument("--discovery-evaluation", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--tokenizer", required=True)
    freeze.set_defaults(scope="confirmation")
    _add_evaluation_arguments(freeze, include_scope=False)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "fit":
        StructureAudit(_audit_config(arguments)).fit(
            train_split=arguments.train_split,
            output=arguments.output,
            device=arguments.device,
            limit=arguments.limit,
            task_type=arguments.task_type,
        )
    elif arguments.command == "score":
        StructureAudit(_audit_config(arguments)).score(
            split_root=arguments.split_root,
            reference_path=arguments.reference,
            output_dir=arguments.output_dir,
            device=arguments.device,
            limit=arguments.limit,
            task_type=arguments.task_type,
        )
    elif arguments.command == "plan":
        prepare_split_plan(
            score_dir=arguments.score_dir,
            output=arguments.output,
            discovery_fraction=arguments.discovery_fraction,
            seed=arguments.seed,
        )
    elif arguments.command == "freeze-confirmation":
        freeze_confirmation(
            split_plan=arguments.split_plan,
            discovery_evaluation=arguments.discovery_evaluation,
            output=arguments.output,
            tokenizer_path=arguments.tokenizer,
            config=_evaluation_config(arguments),
        )
    else:
        StructureEvaluator(_evaluation_config(arguments)).run(
            split_root=arguments.split_root,
            score_dir=arguments.score_dir,
            output_dir=arguments.output_dir,
            tokenizer_path=arguments.tokenizer,
            split_plan=arguments.split_plan,
            confirmation_plan=arguments.confirmation_plan,
        )


if __name__ == "__main__":
    main()
