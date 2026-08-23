"""CLI for cross-origin routing dynamics audits."""

import argparse

from .dynamics_config import DynamicsConfig
from .dynamics_evaluate import evaluate_dynamics_scores
from .dynamics_experiment import score_dynamics_split, train_dynamics_model


def _config(arguments):
    defaults = DynamicsConfig()
    return DynamicsConfig(
        hidden_dim=arguments.hidden_dim,
        role_dim=defaults.role_dim,
        position_dim=defaults.position_dim,
        lag_bins=defaults.lag_bins,
        dropout=defaults.dropout,
        input_dropout=arguments.input_dropout,
        edge_loss_weight=defaults.edge_loss_weight,
        diagonal_loss_weight=defaults.diagonal_loss_weight,
        support_loss_weight=defaults.support_loss_weight,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=defaults.weight_decay,
        validation_fraction=defaults.validation_fraction,
        patience=defaults.patience,
        score_rounds=arguments.score_rounds,
        block_rows=arguments.block_rows,
        random_seed=arguments.seed,
        show_progress=not arguments.no_progress,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--train-split", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--task-type")
    train.add_argument("--limit", type=int)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--input-dropout", type=float, default=0.1)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--score-rounds", type=int, default=3)
    train.add_argument("--block-rows", type=int, default=8192)
    train.add_argument("--seed", type=int, default=20260823)
    train.add_argument("--no-progress", action="store_true")

    score = commands.add_parser("score")
    score.add_argument("--split-root", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--task-type")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260823)
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        train_dynamics_model(
            train_split=arguments.train_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            task_type=arguments.task_type,
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        score_dynamics_split(
            split_root=arguments.split_root,
            checkpoint_path=arguments.checkpoint,
            output_dir=arguments.output_dir,
            device=arguments.device,
            task_type=arguments.task_type,
            limit=arguments.limit,
        )
    else:
        evaluate_dynamics_scores(
            split_root=arguments.split_root,
            score_path=arguments.scores,
            output_dir=arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
