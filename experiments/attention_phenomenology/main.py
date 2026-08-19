"""Command-line interface for the attention phenomenology audit."""

from __future__ import annotations

import argparse

from .config import PhenomenologyConfig
from .evaluation import evaluate_scores
from .experiment import fit_reference, score_split


def _config(arguments) -> PhenomenologyConfig:
    return PhenomenologyConfig(
        prompt_bins=arguments.prompt_bins,
        rr_lag_bins=arguments.rr_lag_bins,
        lid_neighbors=arguments.lid_neighbors,
        transition_projections=arguments.transition_projections,
        anchor_count=arguments.anchor_count,
        recent_lag_max=arguments.recent_lag_max,
        causal_position_bins=arguments.causal_position_bins,
        block_rows=arguments.block_rows,
        random_seed=arguments.seed,
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-bins", type=int, default=4)
    parser.add_argument("--rr-lag-bins", type=int, default=8)
    parser.add_argument("--lid-neighbors", type=int, default=8)
    parser.add_argument("--transition-projections", type=int, default=12)
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument("--recent-lag-max", type=int, default=4)
    parser.add_argument("--causal-position-bins", type=int, default=10)
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260819)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit unlabeled routing references")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--reservoir-rows", type=int, default=2048)
    fit.add_argument("--limit", type=int)
    _add_config(fit)

    score = subparsers.add_parser("score", help="freeze test mechanism fields")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--limit", type=int)
    score.add_argument("--no-rewire", action="store_true")
    score.add_argument("--detail-sample-id", action="append", default=[])
    _add_config(score)

    evaluate = subparsers.add_parser(
        "evaluate", help="unlock labels for post-hoc hypothesis tests"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--score-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--onset-window", type=int, default=4)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260819)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "fit":
        fit_reference(
            train_split=arguments.train_split,
            output=arguments.output,
            device=arguments.device,
            config=_config(arguments),
            reservoir_rows=arguments.reservoir_rows,
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        score_split(
            split_root=arguments.split_root,
            reference_path=arguments.reference,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            rewire=not arguments.no_rewire,
            detail_sample_ids=tuple(arguments.detail_sample_id),
            limit=arguments.limit,
        )
    else:
        evaluate_scores(
            split_root=arguments.split_root,
            score_dir=arguments.score_dir,
            output_dir=arguments.output_dir,
            onset_window=arguments.onset_window,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
