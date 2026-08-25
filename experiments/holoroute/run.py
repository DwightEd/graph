"""Command line interface for HoloRoute and flat-1024."""

import argparse
import json
from dataclasses import replace

from research_dataset import open_research_dataset

from .config import HoloRouteConfig
from .evaluate import evaluate
from .pipeline import score_flat, score_holoroute, train_flat, train_holoroute


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HoloRoute attention event graph")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "flat-train"):
        command = commands.add_parser(name)
        command.add_argument("--train", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--reference", required=True)
        command.add_argument("--task", default="QA")
        command.add_argument("--limit", type=int)
        command.add_argument("--epochs", type=int)
        command.add_argument("--device", default="cpu")
        command.add_argument("--hidden", type=int, default=96)

    for name in ("score", "flat-score"):
        command = commands.add_parser(name)
        command.add_argument("--test", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--reference", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--task", default="QA")
        command.add_argument("--limit", type=int)
        command.add_argument("--device", default="cpu")

    command = commands.add_parser("evaluate")
    command.add_argument("--test", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=500)
    command.add_argument("--seed", type=int, default=20260825)
    command.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    config = HoloRouteConfig()
    if getattr(arguments, "epochs", None) is not None:
        config = replace(config, train=replace(config.train, epochs=arguments.epochs))

    if arguments.command == "train":
        dataset = open_research_dataset(arguments.train, device=arguments.device)
        report = train_holoroute(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            config,
            arguments.task,
            arguments.limit,
        )
    elif arguments.command == "flat-train":
        dataset = open_research_dataset(arguments.train, device=arguments.device)
        report = train_flat(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            config,
            arguments.task,
            arguments.limit,
            arguments.hidden,
        )
    elif arguments.command == "score":
        dataset = open_research_dataset(arguments.test, device=arguments.device)
        report = score_holoroute(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            arguments.output,
            arguments.task,
            arguments.limit,
        )
    elif arguments.command == "flat-score":
        dataset = open_research_dataset(arguments.test, device=arguments.device)
        report = score_flat(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            arguments.output,
            arguments.task,
            arguments.limit,
        )
    else:
        dataset = open_research_dataset(arguments.test, device=arguments.device)
        report = evaluate(
            dataset,
            arguments.scores,
            arguments.output,
            arguments.bootstrap,
            arguments.seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
