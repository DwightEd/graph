"""Command-line entry point for information-flow node embeddings."""

import argparse

from .encode import encode_bundle


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append identity-preserving route deltas to frozen GCN nodes"
    )
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-mode", choices=("sketch", "mean"), default="sketch")
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    report = encode_bundle(
        arguments.source_index,
        arguments.output,
        mode=arguments.head_mode,
        checkpoints=arguments.checkpoints,
        seed=arguments.seed,
        device=arguments.device,
        limit=arguments.limit,
    )
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
