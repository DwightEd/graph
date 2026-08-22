"""Train and audit a label-free multiplex graph recovery model."""

import argparse

from .config import RecoveryConfig
from .evaluate import evaluate_recovery_scores
from .experiment import score_recovery_split, train_recovery_model


def _config(args) -> RecoveryConfig:
    return RecoveryConfig(
        representation=args.representation,
        hidden_dim=args.hidden_dim,
        channel_mask_rate=args.channel_mask_rate,
        pair_layer_mask_rate=args.pair_layer_mask_rate,
        diagonal_mask_rate=args.diagonal_mask_rate,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        score_rounds=args.score_rounds,
        block_rows=args.block_rows,
        random_seed=args.seed,
        show_progress=not args.no_progress,
    )


def _add_training_options(parser):
    defaults = RecoveryConfig()
    parser.add_argument(
        "--representation",
        choices=("full", "layer_mean", "global_mean"),
        default=defaults.representation,
    )
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument(
        "--channel-mask-rate", type=float, default=defaults.channel_mask_rate
    )
    parser.add_argument(
        "--pair-layer-mask-rate", type=float, default=defaults.pair_layer_mask_rate
    )
    parser.add_argument(
        "--diagonal-mask-rate", type=float, default=defaults.diagonal_mask_rate
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--score-rounds", type=int, default=defaults.score_rounds)
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-progress", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--train-split", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--task-type")
    train.add_argument("--limit", type=int)
    _add_training_options(train)

    score = commands.add_parser("score")
    score.add_argument("--split-root", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--task-type")
    score.add_argument("--limit", type=int)
    score.add_argument("--no-graphs", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260822)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train_recovery_model(
            train_split=args.train_split,
            output_dir=args.output_dir,
            device=args.device,
            config=_config(args),
            task_type=args.task_type,
            limit=args.limit,
        )
    elif args.command == "score":
        score_recovery_split(
            split_root=args.split_root,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            task_type=args.task_type,
            limit=args.limit,
            save_graphs=not args.no_graphs,
        )
    else:
        evaluate_recovery_scores(
            split_root=args.split_root,
            score_path=args.scores,
            output_dir=args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
