"""Command-line entry points for the directed route-hypergraph experiment."""

import argparse

from experiments.grounded_route.config import GRAPH_VARIANTS
from experiments.grounded_route.detection import PCAKNNConfig

from .config import LearningConfig, ModelConfig, TrainConfig
from .pipeline import detect, encode, evaluate, fit

ENDPOINT_RECOVERY_VARIANTS = tuple(
    variant for variant in GRAPH_VARIANTS if variant != "no_message"
)


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directed attention-row hypergraph")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("fit")
    command.add_argument("--train", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--positive-edges-per-graph", type=int, default=4096)
    command.add_argument("--holdout-fraction", type=float, default=0.15)
    command.add_argument("--negative-count", type=int, default=1)
    command.add_argument("--negative-attempt-factor", type=int, default=8)
    command.add_argument("--layout-rows-per-graph", type=int, default=32)
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
    command.add_argument("--incidence-dropout", type=float, default=0.0)
    command.add_argument("--head-dropout", type=float, default=0.0)
    command.add_argument("--flow-weight", type=float, default=0.0)
    command.add_argument("--layout-weight", type=float, default=0.0)
    command.add_argument("--variance-weight", type=float, default=0.05)
    command.add_argument("--residual-weight", type=float, default=1.0)
    command.add_argument("--slot-dim", type=int, default=16)
    command.add_argument("--edge-hidden-dim", type=int, default=64)
    command.add_argument(
        "--latent-mode",
        choices=("deterministic", "vae"),
        default="deterministic",
    )
    command.add_argument(
        "--vae-export",
        choices=("mean", "mean_logvar"),
        default="mean_logvar",
    )
    command.add_argument("--kl-weight", type=float, default=1e-3)
    command.add_argument("--kl-free-bits", type=float, default=1e-2)
    command.add_argument("--kl-warmup-epochs", type=int, default=4)
    command.add_argument(
        "--variant",
        choices=ENDPOINT_RECOVERY_VARIANTS,
        default="real",
    )
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
            model_config=ModelConfig(
                slot_dim=arguments.slot_dim,
                edge_hidden_dim=arguments.edge_hidden_dim,
                residual_weight=arguments.residual_weight,
                latent_mode=arguments.latent_mode,
                vae_export=arguments.vae_export,
            ),
            learning_config=LearningConfig(
                positive_edges_per_graph=arguments.positive_edges_per_graph,
                holdout_fraction=arguments.holdout_fraction,
                negative_count=arguments.negative_count,
                negative_attempt_factor=arguments.negative_attempt_factor,
                layout_rows_per_graph=arguments.layout_rows_per_graph,
                layout_rows_per_batch=arguments.layout_rows_per_batch,
                layout_min_mass=arguments.layout_min_mass,
                layout_max_elements=arguments.layout_max_elements,
                layout_max_work_elements=arguments.layout_max_work_elements,
                layout_order=arguments.layout_order,
                incidence_dropout=arguments.incidence_dropout,
                head_dropout=arguments.head_dropout,
                flow_weight=arguments.flow_weight,
                layout_weight=arguments.layout_weight,
                variance_weight=arguments.variance_weight,
                kl_weight=arguments.kl_weight,
                kl_free_bits=arguments.kl_free_bits,
                kl_warmup_epochs=arguments.kl_warmup_epochs,
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
