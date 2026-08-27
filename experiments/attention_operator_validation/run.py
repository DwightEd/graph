"""Command-line entry points for attention hypernetwork mechanism validation."""

from __future__ import annotations

import argparse

from .pipeline import evaluate_features, extract_features, extract_operators


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pair-specific attention operator-code validation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("operators")
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cpu")
    command.add_argument(
        "--load-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    command.add_argument(
        "--compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    command.add_argument("--block-heads", type=int, default=4)
    command.add_argument("--basis-dir")
    command.add_argument("--trust-remote-code", action="store_true")

    command = commands.add_parser("features")
    command.add_argument("--data", required=True)
    command.add_argument("--operators", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument(
        "--imputation",
        choices=("zero", "floor", "midpoint", "excess"),
        default="zero",
    )
    command.add_argument("--seed", type=int, default=20260828)

    command = commands.add_parser("evaluate")
    command.add_argument("--data", required=True)
    command.add_argument("--features", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=500)
    command.add_argument("--cv-folds", type=int, default=5)
    command.add_argument("--seed", type=int, default=20260828)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "operators":
        report = extract_operators(
            arguments.model,
            arguments.output,
            device=arguments.device,
            load_dtype=arguments.load_dtype,
            compute_dtype=arguments.compute_dtype,
            block_heads=arguments.block_heads,
            trust_remote_code=arguments.trust_remote_code,
            basis_dir=arguments.basis_dir,
        )
    elif arguments.command == "features":
        report = extract_features(
            arguments.data,
            arguments.operators,
            arguments.output,
            task=arguments.task,
            limit=arguments.limit,
            imputation=arguments.imputation,
            seed=arguments.seed,
        )
    else:
        report = evaluate_features(
            arguments.data,
            arguments.features,
            arguments.output,
            bootstrap_replicates=arguments.bootstrap,
            cv_folds=arguments.cv_folds,
            seed=arguments.seed,
        )

    print(f"{arguments.command} completed")
    for name in (
        "operators",
        "features",
        "evaluation",
        "sha256",
        "operator_sha256",
        "model_path",
        "architecture",
        "layers",
        "heads",
        "kv_heads",
        "head_dim",
        "samples",
        "feature_count",
        "positive_answers",
        "prevalence",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
