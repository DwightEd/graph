"""CLI for grounding-sensitive edge refinement and counterfactual scoring."""

from __future__ import annotations

import argparse

from .grounding_config import GroundingGraphConfig
from .grounding_evaluation import evaluate_grounding_scores
from .grounding_experiment import score_grounding_split, train_grounding_model


def _add_config(parser: argparse.ArgumentParser) -> None:
    defaults = GroundingGraphConfig()
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--received-topk", type=int, default=defaults.received_topk)
    parser.add_argument("--edge-mask-rate", type=float, default=defaults.edge_mask_rate)
    parser.add_argument("--perturbation-scale", type=float, default=defaults.perturbation_scale)
    parser.add_argument("--gate-keep-target", type=float, default=defaults.gate_keep_target)
    parser.add_argument("--gate-regularization", type=float, default=defaults.gate_regularization)
    parser.add_argument("--raw-loss-weight", type=float, default=defaults.raw_loss_weight)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--score-rounds", type=int, default=defaults.score_rounds)
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-reuse-memory", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def _config(arguments) -> GroundingGraphConfig:
    defaults = GroundingGraphConfig()
    return GroundingGraphConfig(
        hidden_dim=arguments.hidden_dim,
        layer_embedding_dim=defaults.layer_embedding_dim,
        head_embedding_dim=defaults.head_embedding_dim,
        relation_embedding_dim=defaults.relation_embedding_dim,
        lag_embedding_dim=defaults.lag_embedding_dim,
        response_lag_bins=defaults.response_lag_bins,
        received_topk=arguments.received_topk,
        edge_mask_rate=arguments.edge_mask_rate,
        perturbation_scale=arguments.perturbation_scale,
        gate_keep_target=arguments.gate_keep_target,
        gate_regularization=arguments.gate_regularization,
        raw_loss_weight=arguments.raw_loss_weight,
        reuse_loss_weight=defaults.reuse_loss_weight,
        grounding_loss_weight=defaults.grounding_loss_weight,
        provenance_loss_weight=defaults.provenance_loss_weight,
        use_reuse_memory=not arguments.no_reuse_memory,
        dropout=defaults.dropout,
        bptt_steps=defaults.bptt_steps,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        validation_fraction=arguments.validation_fraction,
        early_stopping_patience=arguments.early_stopping_patience,
        score_rounds=arguments.score_rounds,
        block_rows=arguments.block_rows,
        random_seed=arguments.seed,
        show_progress=not arguments.no_progress,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train label-free edge refinement")
    train.add_argument("--train-split", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--task-type")
    train.add_argument("--limit", type=int)
    _add_config(train)

    score = commands.add_parser("score", help="freeze counterfactual token scores")
    score.add_argument("--split-root", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--task-type")
    score.add_argument("--limit", type=int)
    score.add_argument("--no-embeddings", action="store_true")

    evaluate = commands.add_parser("evaluate", help="open labels after scores freeze")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--onset-window", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        train_grounding_model(
            train_split=arguments.train_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            task_type=arguments.task_type,
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        score_grounding_split(
            split_root=arguments.split_root,
            checkpoint_path=arguments.checkpoint,
            output_dir=arguments.output_dir,
            device=arguments.device,
            task_type=arguments.task_type,
            limit=arguments.limit,
            save_embeddings=not arguments.no_embeddings,
        )
    else:
        evaluate_grounding_scores(
            split_root=arguments.split_root,
            score_path=arguments.scores,
            output_dir=arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            onset_window=arguments.onset_window,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
