"""Command-line entry points for the frozen attention-mechanism audit."""

from __future__ import annotations

import argparse


DTYPE_CHOICES = ("float32", "float16", "bfloat16")
DEFAULT_GRADIENT_PROBES = 8
DEFAULT_ATTRIBUTION_SEED = 20260828


def command_line() -> argparse.ArgumentParser:
    """Build the parser without importing Torch or Transformers.

    Keeping heavyweight imports behind the selected command makes ``--help``
    useful on login nodes and in lightweight CI environments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Audit grounding drift, routing dispersion, and counterfactual "
            "evidence bypass in a frozen causal language model"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser(
        "roles",
        help="reconstruct a label-free prompt-role index",
    )
    command.add_argument("--data", required=True)
    command.add_argument("--source-info", required=True)
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--trust-remote-code", action="store_true")

    command = commands.add_parser(
        "capture",
        help="freeze label-free answer and token mechanism trajectories",
    )
    command.add_argument("--data", required=True)
    command.add_argument("--roles", required=True)
    command.add_argument("--source-info", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--torch-dtype", choices=DTYPE_CHOICES, default="float16")
    command.add_argument("--task", default="QA")
    command.add_argument("--limit", type=int)
    command.add_argument("--vocab-chunk-size", type=int, default=4096)
    command.add_argument(
        "--gradient-probes",
        type=int,
        default=DEFAULT_GRADIENT_PROBES,
        help="Rademacher probes for each token-diagonal attribution Jacobian",
    )
    command.add_argument(
        "--attribution-seed",
        type=int,
        default=DEFAULT_ATTRIBUTION_SEED,
        help="deterministic local seed for attribution probes",
    )
    command.add_argument(
        "--role-null-bin-width",
        type=int,
        default=32,
        help="distance-to-response bin width for the stratified role null",
    )
    command.add_argument("--trust-remote-code", action="store_true")

    command = commands.add_parser(
        "evaluate",
        help="open labels only after the mechanism artifact is frozen",
    )
    command.add_argument("--data", required=True)
    command.add_argument("--artifact", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--folds", type=int, default=5)
    command.add_argument("--seed", type=int, default=20260828)
    return parser


def _print_report(command: str, report: dict[str, object]) -> None:
    print(f"{command} completed")
    for name in (
        "roles",
        "artifact",
        "evaluation",
        "sha256",
        "samples",
        "sources",
        "tokens",
        "positive_answers",
        "prevalence",
        "labels_used",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "roles":
        from .pipeline import build_roles

        report = build_roles(
            arguments.data,
            arguments.source_info,
            arguments.tokenizer,
            arguments.output,
            task=arguments.task,
            limit=arguments.limit,
            trust_remote_code=arguments.trust_remote_code,
        )
    elif arguments.command == "capture":
        from .pipeline import capture_mechanisms

        report = capture_mechanisms(
            arguments.data,
            arguments.roles,
            arguments.source_info,
            arguments.model,
            arguments.output,
            device=arguments.device,
            torch_dtype=arguments.torch_dtype,
            task=arguments.task,
            limit=arguments.limit,
            vocab_chunk_size=arguments.vocab_chunk_size,
            gradient_probes=arguments.gradient_probes,
            attribution_seed=arguments.attribution_seed,
            role_null_bin_width=arguments.role_null_bin_width,
            trust_remote_code=arguments.trust_remote_code,
        )
    else:
        from .evaluate import evaluate_artifact

        report = evaluate_artifact(
            arguments.data,
            arguments.artifact,
            arguments.output,
            bootstrap_replicates=arguments.bootstrap,
            cv_folds=arguments.folds,
            seed=arguments.seed,
        )
    _print_report(arguments.command, report)


if __name__ == "__main__":
    main()
