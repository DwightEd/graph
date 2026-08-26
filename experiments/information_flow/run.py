"""Command-line entry points for the information-flow audit."""

import argparse

from .config import FlowConfig
from .evaluate import evaluate
from .extract import extract


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer-wise attention information-flow node embeddings"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("extract")
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--scope", choices=("calibration", "all"), required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--sketch-dim", type=int, default=32)
    command.add_argument("--residual-weight", type=float, default=1.0)
    command.add_argument("--unresolved", choices=("self", "renormalize"), default="self")
    command.add_argument("--calibration-fraction", type=float, default=0.20)
    command.add_argument("--seed", type=int, default=20260827)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("evaluate")
    command.add_argument("--calibration", required=True)
    command.add_argument("--test", required=True)
    command.add_argument("--test-root", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument("--folds", type=int, default=5)
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--bootstrap", type=int, default=1_000)
    command.add_argument("--seeds", nargs="+", type=int, default=[20260827])
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "extract":
        report = extract(
            arguments.data,
            arguments.output,
            task=arguments.task,
            scope=arguments.scope,
            limit=arguments.limit,
            device=arguments.device,
            config=FlowConfig(
                sketch_dim=arguments.sketch_dim,
                residual_weight=arguments.residual_weight,
                unresolved=arguments.unresolved,
                calibration_fraction=arguments.calibration_fraction,
                seed=arguments.seed,
            ),
        )
    else:
        report = evaluate(
            arguments.calibration,
            arguments.test,
            arguments.test_root,
            arguments.output,
            device=arguments.device,
            folds=arguments.folds,
            epochs=arguments.epochs,
            bootstrap=arguments.bootstrap,
            seeds=tuple(arguments.seeds),
        )

    print(f"{arguments.command} completed")
    for name in ("scope", "samples", "tokens", "edges", "report"):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
