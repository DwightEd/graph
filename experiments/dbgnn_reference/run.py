"""CLI for the copied-code DBGNN reference."""

import argparse

from experiments.grounded_route.detection import PCAKNNConfig

from .config import DBGNNConfig, HIGHER_ORDER_MODES
from .pipeline import detect, encode, evaluate, fit
from .upstream import ENCODERS


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original-code DBGNN node embeddings")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("fit")
    command.add_argument("--train-index", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--encoder", choices=ENCODERS, default="dbgnn")
    command.add_argument("--hidden-dim", type=int, default=64)
    command.add_argument("--embedding-dim", type=int, default=64)
    command.add_argument("--dropout", type=float, default=0.1)
    command.add_argument("--delta-layers", type=int, default=1)
    command.add_argument(
        "--higher-order-mode",
        choices=HIGHER_ORDER_MODES,
        default="causal",
    )
    command.add_argument("--edge-drop-fraction", type=float, default=0.15)
    command.add_argument("--positives-per-graph", type=int, default=4096)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--validation-fraction", type=float, default=0.15)
    command.add_argument("--detector-fraction", type=float, default=0.20)
    command.add_argument("--seed", type=int, default=20260826)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("encode")
    command.add_argument("--index", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--scope", choices=("calibration", "all"), required=True)
    command.add_argument("--device", default="cpu")

    command = commands.add_parser("detect")
    command.add_argument("--calibration", required=True)
    command.add_argument("--test", required=True)
    command.add_argument("--reference", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--components", type=int, default=32)
    command.add_argument("--neighbors", type=int, default=20)
    command.add_argument("--max-reference", type=int, default=20_000)
    command.add_argument("--seed", type=int, default=20260826)

    command = commands.add_parser("evaluate")
    command.add_argument("--test", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=500)
    command.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "fit":
        report = fit(
            arguments.train_index,
            arguments.checkpoint,
            config=DBGNNConfig(
                encoder=arguments.encoder,
                hidden_dim=arguments.hidden_dim,
                embedding_dim=arguments.embedding_dim,
                dropout=arguments.dropout,
                delta_layers=arguments.delta_layers,
                higher_order_mode=arguments.higher_order_mode,
                edge_drop_fraction=arguments.edge_drop_fraction,
                positives_per_graph=arguments.positives_per_graph,
                epochs=arguments.epochs,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                validation_fraction=arguments.validation_fraction,
                detector_fraction=arguments.detector_fraction,
                seed=arguments.seed,
            ),
            device=arguments.device,
        )
    elif arguments.command == "encode":
        report = encode(
            arguments.index,
            arguments.checkpoint,
            arguments.output,
            scope=arguments.scope,
            device=arguments.device,
        )
    elif arguments.command == "detect":
        report = detect(
            arguments.calibration,
            arguments.test,
            arguments.reference,
            arguments.scores,
            detector_config=PCAKNNConfig(
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
        "encoder",
        "higher_order_mode",
        "samples",
        "calibration_samples",
        "nodes",
        "first_order_edges",
        "higher_order_edges",
        "best_validation_loss",
        "positive_pairs",
        "eligible_pairs",
        "auroc",
        "auprc",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
