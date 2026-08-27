"""Command-line entry points for the directed route-hypergraph experiment."""

import argparse

from experiments.grounded_route.config import GRAPH_VARIANTS
from experiments.grounded_route.detection import PCAKNNConfig

from .config import LearningConfig, ModelConfig, TrainConfig
from .pipeline import detect, encode, evaluate, fit


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directed attention-row hypergraph")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("fit")
    command.add_argument("--train", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--rows-per-graph", type=int, default=256)
    command.add_argument("--layout-rows-per-batch", type=int, default=64)
    command.add_argument("--layout-min-mass", type=float, default=1e-4)
    command.add_argument("--layout-max-elements", type=int, default=8_000_000)
    command.add_argument(
        "--layout-max-work-elements",
        type=int,
        default=250_000_000,
    )
    command.add_argument(
        "--layout-order",
        choices=("ordered", "reverse"),
        default="ordered",
    )
    command.add_argument("--incidence-dropout", type=float, default=0.15)
    command.add_argument("--head-dropout", type=float, default=0.05)
    command.add_argument("--flow-weight", type=float, default=0.5)
    command.add_argument("--layout-weight", type=float, default=0.25)
    command.add_argument("--residual-weight", type=float, default=1.0)
    command.add_argument("--variant", choices=GRAPH_VARIANTS, default="real")
    command.add_argument("--seed", type=int, default=20260827)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("encode")
    command.add_argument("--data", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--scope", choices=("calibration", "all"), required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("detect")
    command.add_argument("--calibration", required=True)
    command.add_argument("--test", required=True)
    command.add_argument("--reference", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--components", type=int, default=32)
    command.add_argument("--neighbors", type=int, default=20)
    command.add_argument("--max-reference", type=int, default=20_000)
    command.add_argument("--seed", type=int, default=20260827)

    command = commands.add_parser("evaluate")
    command.add_argument("--test", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=500)
    command.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "fit":
        report = fit(
            arguments.train,
            arguments.checkpoint,
            task=arguments.task,
            limit=arguments.limit,
            device=arguments.device,
            variant=arguments.variant,
            model_config=ModelConfig(residual_weight=arguments.residual_weight),
            learning_config=LearningConfig(
                rows_per_graph=arguments.rows_per_graph,
                layout_rows_per_batch=arguments.layout_rows_per_batch,
                layout_min_mass=arguments.layout_min_mass,
                layout_max_elements=arguments.layout_max_elements,
                layout_max_work_elements=arguments.layout_max_work_elements,
                layout_order=arguments.layout_order,
                incidence_dropout=arguments.incidence_dropout,
                head_dropout=arguments.head_dropout,
                flow_weight=arguments.flow_weight,
                layout_weight=arguments.layout_weight,
            ),
            train_config=TrainConfig(epochs=arguments.epochs, seed=arguments.seed),
        )
    elif arguments.command == "encode":
        report = encode(
            arguments.data,
            arguments.checkpoint,
            arguments.output,
            scope=arguments.scope,
            task=arguments.task,
            limit=arguments.limit,
            device=arguments.device,
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
        report = evaluate(
            arguments.test,
            arguments.scores,
            arguments.output,
            bootstrap_replicates=arguments.bootstrap,
            seed=arguments.seed,
        )

    print(f"{arguments.command} completed")
    for name in (
        "checkpoint",
        "embeddings",
        "reference",
        "scores",
        "evaluation",
        "samples",
        "calibration_samples",
        "nodes",
        "edges",
        "best_validation_loss",
        "parameter_count",
        "auroc",
        "auprc",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
