"""Command line interface for P-Cut."""

import argparse

from research_dataset import open_research_dataset

from .config import MethodConfig
from .evaluate import evaluate
from .pipeline import fit_reference, score_dataset


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-provenance cut detector")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit")
    fit.add_argument("--train", required=True)
    fit.add_argument("--checkpoint", required=True)
    fit.add_argument("--reference", required=True)
    fit.add_argument("--task", default="QA")
    fit.add_argument("--limit", type=int)
    fit.add_argument("--device", default="cpu")

    score = commands.add_parser("score")
    score.add_argument("--test", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--task", default="QA")
    score.add_argument("--limit", type=int)
    score.add_argument("--device", default="cpu")

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--test", required=True)
    evaluation.add_argument("--scores", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--bootstrap", type=int, default=500)
    evaluation.add_argument("--seed", type=int, default=20260826)
    evaluation.add_argument("--device", default="cpu")
    return parser


def print_report(command: str, report: dict[str, object]) -> None:
    print(f"{command} completed")
    for name in ("checkpoint", "reference", "scores", "graphs", "samples", "tokens"):
        if name in report:
            print(f"{name}: {report[name]}")


def main() -> None:
    arguments = command_line().parse_args()
    config = MethodConfig()

    if arguments.command == "fit":
        dataset = open_research_dataset(arguments.train, device=arguments.device)
        report = fit_reference(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            config,
            arguments.task,
            arguments.limit,
        )
    elif arguments.command == "score":
        dataset = open_research_dataset(arguments.test, device=arguments.device)
        report = score_dataset(
            dataset,
            arguments.checkpoint,
            arguments.reference,
            arguments.output,
            arguments.task,
            arguments.limit,
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
            arguments.bootstrap,
            arguments.seed,
        )

    print_report(arguments.command, report)


if __name__ == "__main__":
    main()
