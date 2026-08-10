"""Command-line entry point for attention extraction and graph building."""

import argparse

from build import GRAPH_KINDS, BuildConfig, GraphDatasetBuilder
from extract import AttentionExtractor, ExtractionConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create label-free attention caches and graphs."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract_parser = subcommands.add_parser("extract")
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--dataset-path", required=True)
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--split", required=True, choices=("train", "test"))
    extract_parser.add_argument("--generator-model", default="llama-2-7b-chat")
    extract_parser.add_argument("--task-type", default="all")
    extract_parser.add_argument("--floor", type=float, default=0.01)
    extract_parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--limit", type=int)

    build_parser = subcommands.add_parser("build")
    build_parser.add_argument("--cache-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--kind", choices=GRAPH_KINDS, default="relation_topk_channels")
    build_parser.add_argument("--tau", type=float, default=0.05)
    build_parser.add_argument("--k-prompt", type=int, default=8)
    build_parser.add_argument("--k-history", type=int, default=8)
    build_parser.add_argument("--device", default="cuda")
    build_parser.add_argument("--limit", type=int)

    arguments = parser.parse_args(argv)
    if arguments.command == "extract":
        config = ExtractionConfig(
            model_path=arguments.model_path,
            dataset_path=arguments.dataset_path,
            output_dir=arguments.output_dir,
            split=arguments.split,
            generator_model=arguments.generator_model,
            task_type=arguments.task_type,
            floor=arguments.floor,
            dtype=arguments.dtype,
            device=arguments.device,
            limit=arguments.limit,
        )
        AttentionExtractor(config).run()
        print(f"extract: {arguments.output_dir}")
        return

    config = BuildConfig(
        cache_dir=arguments.cache_dir,
        output_dir=arguments.output_dir,
        kind=arguments.kind,
        tau=arguments.tau,
        k_prompt=arguments.k_prompt,
        k_history=arguments.k_history,
        device=arguments.device,
        limit=arguments.limit,
    )
    summary = GraphDatasetBuilder(config).run()
    print(f"build {summary['kind']}: {summary['count']} graphs")


if __name__ == "__main__":
    main()
