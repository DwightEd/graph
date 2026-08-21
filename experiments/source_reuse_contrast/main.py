"""CLI for causal source-reuse predictability experiments."""

from __future__ import annotations

import argparse

from .config import SourceReuseConfig
from .evaluation import evaluate_scores
from .experiment import score_split, train_model
from .predictability import write_predictability_gate


def _add_train_config(parser: argparse.ArgumentParser) -> None:
    defaults = SourceReuseConfig()
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument(
        "--memory-mode",
        choices=("current", "birth", "dynamic"),
        default=defaults.memory_mode,
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--negative-count", type=int, default=defaults.negative_count)
    parser.add_argument(
        "--negative-pool-size", type=int, default=defaults.negative_pool_size
    )
    parser.add_argument(
        "--prompt-position-bins", type=int, default=defaults.prompt_position_bins
    )
    parser.add_argument(
        "--response-lag-bins", type=int, default=defaults.response_lag_bins
    )
    parser.add_argument("--usage-bins", type=int, default=defaults.usage_bins)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--bptt-steps", type=int, default=defaults.bptt_steps)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.early_stopping_patience,
    )
    parser.add_argument("--score-rounds", type=int, default=defaults.score_rounds)
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-progress", action="store_true")


def _config(arguments) -> SourceReuseConfig:
    defaults = SourceReuseConfig()
    return SourceReuseConfig(
        hidden_dim=arguments.hidden_dim,
        layer_embedding_dim=defaults.layer_embedding_dim,
        head_embedding_dim=defaults.head_embedding_dim,
        relation_embedding_dim=defaults.relation_embedding_dim,
        source_bin_embedding_dim=defaults.source_bin_embedding_dim,
        usage_embedding_dim=defaults.usage_embedding_dim,
        prompt_position_bins=arguments.prompt_position_bins,
        response_lag_bins=arguments.response_lag_bins,
        usage_bins=arguments.usage_bins,
        memory_mode=arguments.memory_mode,
        temperature=arguments.temperature,
        negative_count=arguments.negative_count,
        negative_pool_size=arguments.negative_pool_size,
        prompt_position_tolerance=defaults.prompt_position_tolerance,
        response_lag_tolerance=defaults.response_lag_tolerance,
        dropout=arguments.dropout,
        bptt_steps=arguments.bptt_steps,
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


def _score_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("path arguments must use name=path")
        name, path = value.split("=", 1)
        result[name] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train on unlabeled attention graphs")
    train.add_argument("--train-split", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--task-type")
    train.add_argument("--limit", type=int)
    _add_train_config(train)

    score = commands.add_parser("score", help="freeze raw token scores and embeddings")
    score.add_argument("--split-root", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--task-type")
    score.add_argument("--limit", type=int)
    score.add_argument("--no-embeddings", action="store_true")

    gate = commands.add_parser(
        "gate", help="write label-free source-reuse evidence gates"
    )
    gate.add_argument(
        "--training", action="append", required=True, help="name=training.json"
    )
    gate.add_argument(
        "--manifest", action="append", required=True, help="name=manifest.json"
    )
    gate.add_argument("--output", required=True)

    evaluate = commands.add_parser(
        "evaluate", help="unlock labels for model ladder evaluation"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument(
        "--score", action="append", required=True, help="name=path; repeat per mode"
    )
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--onset-window", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=20260820)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        train_model(
            train_split=arguments.train_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            limit=arguments.limit,
            task_type=arguments.task_type,
        )
    elif arguments.command == "score":
        score_split(
            split_root=arguments.split_root,
            checkpoint_path=arguments.checkpoint,
            output_dir=arguments.output_dir,
            device=arguments.device,
            limit=arguments.limit,
            task_type=arguments.task_type,
            save_embeddings=not arguments.no_embeddings,
        )
    elif arguments.command == "gate":
        write_predictability_gate(
            training_paths=_score_mapping(arguments.training),
            manifest_paths=_score_mapping(arguments.manifest),
            output_path=arguments.output,
        )
    else:
        evaluate_scores(
            split_root=arguments.split_root,
            score_paths=_score_mapping(arguments.score),
            output_dir=arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            onset_window=arguments.onset_window,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
