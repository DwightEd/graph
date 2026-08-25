"""Command-line entry points for the GroundedRoute experiment."""

from __future__ import annotations

import argparse

from research_dataset import open_research_dataset

from .config import GRAPH_VARIANTS, TrainConfig
from .detection import PCAKNNConfig
from .evaluate import evaluate
from .pipeline import build, detect, encode, fit


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GroundedRoute causal token-graph representation learning"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("build")
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)

    command = commands.add_parser("fit")
    command.add_argument("--spec", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--gradient-clip", type=float, default=1.0)
    command.add_argument("--validation-fraction", type=float, default=0.15)
    command.add_argument("--detector-fraction", type=float, default=0.20)
    command.add_argument("--variant", choices=GRAPH_VARIANTS, default="real")
    command.add_argument("--minimum-changed-fraction", type=float, default=0.01)
    command.add_argument("--seed", type=int, default=20260825)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("encode")
    command.add_argument("--spec", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--scope", choices=("all", "calibration"), required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument("--variant", choices=GRAPH_VARIANTS)

    command = commands.add_parser("detect")
    command.add_argument("--calibration", required=True)
    command.add_argument("--test", required=True)
    command.add_argument("--reference", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--components", type=int, default=32)
    command.add_argument("--neighbors", type=int, default=20)
    command.add_argument("--max-reference", type=int, default=20_000)
    command.add_argument("--seed", type=int, default=20260825)

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
    if arguments.command == "build":
        report = build(
            arguments.data,
            arguments.output,
            task=arguments.task,
            limit=arguments.limit,
        )
    elif arguments.command == "fit":
        report = fit(
            arguments.spec,
            arguments.checkpoint,
            device=arguments.device,
            train_config=TrainConfig(
                epochs=arguments.epochs,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                gradient_clip=arguments.gradient_clip,
                validation_fraction=arguments.validation_fraction,
                detector_fraction=arguments.detector_fraction,
                seed=arguments.seed,
            ),
            variant=arguments.variant,
            minimum_changed_fraction=arguments.minimum_changed_fraction,
        )
    elif arguments.command == "encode":
        report = encode(
            arguments.spec,
            arguments.checkpoint,
            arguments.output,
            scope=arguments.scope,
            device=arguments.device,
            variant=arguments.variant,
        )
    elif arguments.command == "detect":
        report = detect(
            arguments.calibration,
            arguments.test,
            arguments.reference,
            arguments.scores,
            config=PCAKNNConfig(
                components=arguments.components,
                neighbors=arguments.neighbors,
                max_reference=arguments.max_reference,
                seed=arguments.seed,
            ),
        )
    else:
        dataset = open_research_dataset(
            arguments.test,
            device=arguments.device,
            retain_embedded_labels=True,
        )
        report = evaluate(
            dataset,
            arguments.scores,
            arguments.output,
            bootstrap_replicates=arguments.bootstrap,
            seed=arguments.seed,
        )
    print_report(arguments.command, report)


def print_report(command: str, report: dict[str, object]) -> None:
    print(f"{command} completed")
    for name in (
        "spec",
        "checkpoint",
        "embeddings",
        "reference",
        "scores",
        "evaluation",
        "samples",
        "nodes",
        "train_loss",
        "best_validation_loss",
        "parameter_count",
        "variant",
        "changed_fraction",
        "auroc",
        "auprc",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
