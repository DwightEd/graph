"""CLI for the causal-walk hypothesis validation suite."""

from __future__ import annotations

import argparse

from .config import WalkAuditConfig
from .evaluation import evaluate_walk_audit
from .experiment import fit_walk_audit, score_walk_audit


def _common(parser: argparse.ArgumentParser) -> None:
    defaults = WalkAuditConfig()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-type", default="QA")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--anchor-manifest")
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--max-anchors", type=int, default=defaults.max_anchors)
    parser.add_argument(
        "--prompt-chunk-tokens",
        type=int,
        default=defaults.prompt_chunk_tokens,
    )
    parser.add_argument(
        "--reservoir-rows",
        type=int,
        default=defaults.train_reservoir_rows,
    )
    parser.add_argument("--ridge-alpha", type=float, default=defaults.ridge_alpha)
    parser.add_argument("--horizon", type=int, default=defaults.score_horizon)
    parser.add_argument(
        "--minimum-anchor-mass",
        type=float,
        default=defaults.minimum_anchor_mass,
    )
    parser.add_argument(
        "--anchor-shuffle-replicates",
        type=int,
        default=defaults.anchor_shuffle_replicates,
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
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-progress", action="store_true")


def _config(arguments) -> WalkAuditConfig:
    return WalkAuditConfig(
        block_rows=arguments.block_rows,
        max_anchors=arguments.max_anchors,
        prompt_chunk_tokens=arguments.prompt_chunk_tokens,
        train_reservoir_rows=arguments.reservoir_rows,
        ridge_alpha=arguments.ridge_alpha,
        score_horizon=arguments.horizon,
        minimum_anchor_mass=arguments.minimum_anchor_mass,
        anchor_shuffle_replicates=arguments.anchor_shuffle_replicates,
        bootstrap_replicates=arguments.bootstrap_replicates,
        permutation_replicates=arguments.permutation_replicates,
        random_seed=arguments.seed,
        show_progress=not arguments.no_progress,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit label-free nested Markov models")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output-dir", required=True)
    _common(fit)

    score = commands.add_parser(
        "score",
        help="freeze label-free causal-walk scores",
    )
    score.add_argument("--split-root", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--output-dir", required=True)
    _common(score)

    evaluate = commands.add_parser("evaluate", help="open labels and test H1-H4")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--score-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int)
    evaluate.add_argument("--permutation-replicates", type=int)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "fit":
        fit_walk_audit(
            train_split=arguments.train_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            task_type=arguments.task_type,
            limit=arguments.limit,
            anchor_manifest=arguments.anchor_manifest,
        )
    elif arguments.command == "score":
        score_walk_audit(
            split_root=arguments.split_root,
            model_path=arguments.model,
            output_dir=arguments.output_dir,
            device=arguments.device,
            task_type=arguments.task_type,
            limit=arguments.limit,
            anchor_manifest=arguments.anchor_manifest,
        )
    else:
        evaluate_walk_audit(
            split_root=arguments.split_root,
            score_dir=arguments.score_dir,
            output_dir=arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            permutation_replicates=arguments.permutation_replicates,
        )


if __name__ == "__main__":
    main()
