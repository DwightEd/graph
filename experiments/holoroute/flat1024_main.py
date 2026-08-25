"""Command-line entrypoint for the flat-1024 HoloRoute baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from research_dataset import open_research_dataset

from .evaluation import evaluate_scores
from .flat1024 import Flat1024Config
from .flat1024_experiment import score_flat_split, train_flat_reference


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Flat all-layer attention baseline without graph adjacency"
    )
    command = root.add_subparsers(dest="command", required=True)

    train = command.add_parser("train")
    train.add_argument("--train-split", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--density", required=True)
    train.add_argument("--task-type", default="QA")
    train.add_argument("--limit", type=int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int)
    train.add_argument("--hidden-dim", type=int)

    score = command.add_parser("score")
    score.add_argument("--test-split", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--density", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--task-type", default="QA")
    score.add_argument("--limit", type=int)
    score.add_argument("--device", default="cpu")

    evaluate = command.add_parser("evaluate")
    evaluate.add_argument("--test-split", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260825)
    evaluate.add_argument("--device", default="cpu")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        dataset = open_research_dataset(args.train_split, device=args.device)
        config = Flat1024Config()
        if args.epochs is not None:
            config = replace(config, train=replace(config.train, epochs=args.epochs))
        if args.hidden_dim is not None:
            config = replace(config, model=replace(config.model, hidden_dim=args.hidden_dim))
        report = train_flat_reference(
            dataset,
            args.checkpoint,
            args.density,
            config=config,
            task_type=args.task_type,
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(args.test_split, device=args.device)
        report = score_flat_split(
            dataset,
            args.checkpoint,
            args.density,
            args.output,
            task_type=args.task_type,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(args.test_split, device=args.device)
        report = evaluate_scores(
            dataset,
            args.scores,
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
